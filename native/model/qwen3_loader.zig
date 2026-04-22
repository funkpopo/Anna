const std = @import("std");
const cfg = @import("qwen3_config.zig");
const engine = @import("qwen3_engine.zig");
const types = @import("../runtime/types.zig");
const safetensors = @import("../weights/safetensors.zig");
const xpu = @import("../xpu/opencl_backend.zig");

pub const LoadOptions = struct {
    backend: types.RuntimeBackend = .cpu,
};

pub const QwenTextRuntime = struct {
    runtime_allocator: std.mem.Allocator,
    weights_arena: *std.heap.ArenaAllocator,
    backend: types.RuntimeBackend,
    xpu_runtime: ?*xpu.Runtime,
    config: cfg.QwenTextConfig,
    store: safetensors.SafeTensorStore,
    engine: engine.TokenEngine,

    pub fn load(runtime_allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !QwenTextRuntime {
        return try loadWithOptions(runtime_allocator, io, model_dir, .{});
    }

    pub fn loadWithOptions(runtime_allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, options: LoadOptions) !QwenTextRuntime {
        const weights_arena = try runtime_allocator.create(std.heap.ArenaAllocator);
        errdefer runtime_allocator.destroy(weights_arena);
        weights_arena.* = std.heap.ArenaAllocator.init(runtime_allocator);
        errdefer weights_arena.deinit();
        const weights_allocator = weights_arena.allocator();
        var xpu_runtime: ?*xpu.Runtime = null;
        errdefer if (xpu_runtime) |runtime| {
            runtime.deinit();
            runtime_allocator.destroy(runtime);
        };
        switch (options.backend) {
            .cpu => {},
            .xpu_opencl => {
                const runtime = try runtime_allocator.create(xpu.Runtime);
                runtime.* = try xpu.Runtime.init(runtime_allocator);
                xpu_runtime = runtime;
            },
        }

        var config = try cfg.loadQwenTextConfigFromModelDir(weights_allocator, io, model_dir);
        errdefer config.deinit(weights_allocator);
        var store = try safetensors.SafeTensorStore.openModelDir(weights_allocator, io, model_dir);
        errdefer store.deinit();

        const embeddings = try loadTensorF32(
            weights_allocator,
            runtime_allocator,
            &store,
            io,
            "model.language_model.embed_tokens.weight",
            &.{ config.vocab_size, config.hidden_size },
        );
        const norm_weight = try loadTensorF32(
            weights_allocator,
            runtime_allocator,
            &store,
            io,
            "model.language_model.norm.weight",
            &.{config.hidden_size},
        );

        const layers = try weights_allocator.alloc(engine.DecoderLayer, config.num_hidden_layers);
        var loaded_layers: usize = 0;
        errdefer if (xpu_runtime) |runtime| {
            for (layers[0..loaded_layers]) |*layer| deinitDecoderLayerXpu(layer, runtime);
        };
        for (0..config.num_hidden_layers) |layer_idx| {
            layers[layer_idx] = try loadDecoderLayer(weights_allocator, runtime_allocator, io, &store, config, layer_idx, if (xpu_runtime) |runtime| runtime else null);
            loaded_layers += 1;
        }

        const lm_head = try loadLmHead(
            weights_allocator,
            runtime_allocator,
            io,
            &store,
            config,
            embeddings,
            if (xpu_runtime) |runtime| runtime else null,
        );
        errdefer if (xpu_runtime) |runtime| {
            var lm_head_copy = lm_head;
            lm_head_copy.deinitXpu(runtime);
        };

        const states = try initLayerStates(runtime_allocator, config, layers);
        errdefer {
            for (states) |*state| state.deinit();
            runtime_allocator.free(states);
        }

        return .{
            .runtime_allocator = runtime_allocator,
            .weights_arena = weights_arena,
            .backend = options.backend,
            .xpu_runtime = xpu_runtime,
            .config = config,
            .store = store,
            .engine = .{
                .allocator = runtime_allocator,
                .hidden_size = config.hidden_size,
                .embeddings = embeddings,
                .vocab_size = config.vocab_size,
                .layers = layers,
                .norm_weight = norm_weight,
                .lm_head = lm_head,
                .states = states,
            },
        };
    }

    pub fn deinit(self: *QwenTextRuntime) void {
        self.engine.deinitStates();
        if (self.xpu_runtime) |runtime| {
            for (self.engine.layers) |*layer| deinitDecoderLayerXpu(layer, runtime);
            self.engine.lm_head.deinitXpu(runtime);
            runtime.deinit();
            self.runtime_allocator.destroy(runtime);
            self.xpu_runtime = null;
        }
        self.store.deinit();
        self.config.deinit(self.weights_arena.allocator());
        self.weights_arena.deinit();
        self.runtime_allocator.destroy(self.weights_arena);
        self.* = undefined;
    }

    pub fn reset(self: *QwenTextRuntime) !void {
        self.engine.deinitStates();
        self.engine.states = try self.spawnStates();
        self.engine.position = 0;
    }

    pub fn spawnStates(self: *QwenTextRuntime) ![]engine.LayerState {
        return try initLayerStates(self.runtime_allocator, self.config, self.engine.layers);
    }

    pub fn spawnEngine(self: *QwenTextRuntime) !engine.TokenEngine {
        return .{
            .allocator = self.runtime_allocator,
            .hidden_size = self.engine.hidden_size,
            .embeddings = self.engine.embeddings,
            .vocab_size = self.engine.vocab_size,
            .layers = self.engine.layers,
            .norm_weight = self.engine.norm_weight,
            .lm_head = self.engine.lm_head,
            .states = try self.spawnStates(),
            .position = 0,
        };
    }

    pub fn isEosToken(self: *const QwenTextRuntime, token_id: u32) bool {
        return token_id == self.config.eos_token_id;
    }

    pub fn decodeTokenAlloc(self: *QwenTextRuntime, token_id: u32) ![]f32 {
        return try self.engine.forwardTokenAlloc(token_id);
    }

    pub fn forwardPromptAlloc(self: *QwenTextRuntime, token_ids: []const u32) ![]f32 {
        if (token_ids.len == 0) return error.EmptyPrompt;
        try self.reset();

        var last_logits: ?[]f32 = null;
        errdefer if (last_logits) |logits| self.runtime_allocator.free(logits);

        for (token_ids) |token_id| {
            if (last_logits) |previous| self.runtime_allocator.free(previous);
            last_logits = try self.decodeTokenAlloc(token_id);
        }
        return last_logits.?;
    }
};

fn initLayerStates(
    allocator: std.mem.Allocator,
    config: cfg.QwenTextConfig,
    layers: []const engine.DecoderLayer,
) ![]engine.LayerState {
    const states = try allocator.alloc(engine.LayerState, layers.len);
    errdefer allocator.free(states);

    var initialized: usize = 0;
    errdefer {
        for (states[0..initialized]) |*state| state.deinit();
    }

    for (layers, 0..) |layer, idx| {
        states[idx] = switch (layer.attention) {
            .full => .{
                .full = engine.FullAttentionState.init(
                    allocator,
                    config.num_key_value_heads,
                    config.head_dim,
                ),
            },
            .linear => .{
                .linear = try engine.LinearAttentionState.init(
                    allocator,
                    linearConvDim(config),
                    config.linear_conv_kernel_dim,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                ),
            },
        };
        initialized += 1;
    }
    return states;
}

fn loadDecoderLayer(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    io: std.Io,
    store: *const safetensors.SafeTensorStore,
    config: cfg.QwenTextConfig,
    layer_idx: usize,
    xpu_runtime: ?*xpu.Runtime,
) !engine.DecoderLayer {
    const base = try std.fmt.allocPrint(temp_allocator, "model.language_model.layers.{d}", .{layer_idx});
    defer temp_allocator.free(base);

    const input_norm_name = try std.fmt.allocPrint(temp_allocator, "{s}.input_layernorm.weight", .{base});
    defer temp_allocator.free(input_norm_name);
    const post_norm_name = try std.fmt.allocPrint(temp_allocator, "{s}.post_attention_layernorm.weight", .{base});
    defer temp_allocator.free(post_norm_name);

    const attention: engine.Attention = switch (config.layer_types[layer_idx]) {
        .full_attention => .{ .full = try loadFullAttention(weights_allocator, temp_allocator, io, store, config, base, xpu_runtime) },
        .linear_attention => .{ .linear = try loadLinearAttention(weights_allocator, temp_allocator, io, store, config, base, xpu_runtime) },
    };

    const feed_forward: engine.FeedForward = if (config.usesSparseMoe(layer_idx))
        .{ .moe = try loadMoeBlock(weights_allocator, temp_allocator, io, store, config, base, xpu_runtime) }
    else
        .{ .mlp = try loadMlp(weights_allocator, temp_allocator, io, store, config, base, config.intermediate_size, xpu_runtime) };

    return .{
        .attention = attention,
        .feed_forward = feed_forward,
        .input_norm_weight = try loadTensorF32(
            weights_allocator,
            temp_allocator,
            store,
            io,
            input_norm_name,
            &.{config.hidden_size},
        ),
        .post_norm_weight = try loadTensorF32(
            weights_allocator,
            temp_allocator,
            store,
            io,
            post_norm_name,
            &.{config.hidden_size},
        ),
        .eps = config.rms_norm_eps,
    };
}

fn loadFullAttention(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    io: std.Io,
    store: *const safetensors.SafeTensorStore,
    config: cfg.QwenTextConfig,
    layer_base: []const u8,
    xpu_runtime: ?*xpu.Runtime,
) !engine.FullAttention {
    const prefix = try std.fmt.allocPrint(temp_allocator, "{s}.self_attn", .{layer_base});
    defer temp_allocator.free(prefix);

    const q_norm_name = try std.fmt.allocPrint(temp_allocator, "{s}.q_norm.weight", .{prefix});
    defer temp_allocator.free(q_norm_name);
    const k_norm_name = try std.fmt.allocPrint(temp_allocator, "{s}.k_norm.weight", .{prefix});
    defer temp_allocator.free(k_norm_name);
    const q_proj_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.q_proj", .{prefix});
    defer temp_allocator.free(q_proj_prefix);
    const k_proj_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.k_proj", .{prefix});
    defer temp_allocator.free(k_proj_prefix);
    const v_proj_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.v_proj", .{prefix});
    defer temp_allocator.free(v_proj_prefix);
    const o_proj_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.o_proj", .{prefix});
    defer temp_allocator.free(o_proj_prefix);

    return .{
        .q_proj = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            q_proj_prefix,
            config.hidden_size,
            config.num_attention_heads * config.head_dim * 2,
            xpu_runtime,
        ),
        .k_proj = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            k_proj_prefix,
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            xpu_runtime,
        ),
        .v_proj = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            v_proj_prefix,
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            xpu_runtime,
        ),
        .o_proj = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            o_proj_prefix,
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            xpu_runtime,
        ),
        .q_norm_weight = try loadTensorF32(
            weights_allocator,
            temp_allocator,
            store,
            io,
            q_norm_name,
            &.{config.head_dim},
        ),
        .k_norm_weight = try loadTensorF32(
            weights_allocator,
            temp_allocator,
            store,
            io,
            k_norm_name,
            &.{config.head_dim},
        ),
        .num_heads = config.num_attention_heads,
        .num_kv_heads = config.num_key_value_heads,
        .head_dim = config.head_dim,
        .rope_theta = config.rope.rope_theta,
        .partial_rotary_factor = config.rope.partial_rotary_factor,
        .eps = config.rms_norm_eps,
    };
}

fn loadLinearAttention(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    io: std.Io,
    store: *const safetensors.SafeTensorStore,
    config: cfg.QwenTextConfig,
    layer_base: []const u8,
    xpu_runtime: ?*xpu.Runtime,
) !engine.LinearAttention {
    const prefix = try std.fmt.allocPrint(temp_allocator, "{s}.linear_attn", .{layer_base});
    defer temp_allocator.free(prefix);

    const conv_weight_name = try std.fmt.allocPrint(temp_allocator, "{s}.conv1d.weight", .{prefix});
    defer temp_allocator.free(conv_weight_name);
    const dt_bias_name = try std.fmt.allocPrint(temp_allocator, "{s}.dt_bias", .{prefix});
    defer temp_allocator.free(dt_bias_name);
    const a_log_name = try std.fmt.allocPrint(temp_allocator, "{s}.A_log", .{prefix});
    defer temp_allocator.free(a_log_name);
    const norm_name = try std.fmt.allocPrint(temp_allocator, "{s}.norm.weight", .{prefix});
    defer temp_allocator.free(norm_name);
    const in_proj_qkv_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.in_proj_qkv", .{prefix});
    defer temp_allocator.free(in_proj_qkv_prefix);
    const in_proj_z_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.in_proj_z", .{prefix});
    defer temp_allocator.free(in_proj_z_prefix);
    const in_proj_b_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.in_proj_b", .{prefix});
    defer temp_allocator.free(in_proj_b_prefix);
    const in_proj_a_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.in_proj_a", .{prefix});
    defer temp_allocator.free(in_proj_a_prefix);
    const out_proj_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.out_proj", .{prefix});
    defer temp_allocator.free(out_proj_prefix);

    return .{
        .in_proj_qkv = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            in_proj_qkv_prefix,
            config.hidden_size,
            linearConvDim(config),
            xpu_runtime,
        ),
        .in_proj_z = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            in_proj_z_prefix,
            config.hidden_size,
            config.linear_num_value_heads * config.linear_value_head_dim,
            xpu_runtime,
        ),
        .in_proj_b = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            in_proj_b_prefix,
            config.hidden_size,
            config.linear_num_value_heads,
            xpu_runtime,
        ),
        .in_proj_a = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            in_proj_a_prefix,
            config.hidden_size,
            config.linear_num_value_heads,
            xpu_runtime,
        ),
        .out_proj = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            out_proj_prefix,
            config.linear_num_value_heads * config.linear_value_head_dim,
            config.hidden_size,
            xpu_runtime,
        ),
        .conv_weight = try loadConvWeight(
            weights_allocator,
            temp_allocator,
            store,
            io,
            conv_weight_name,
            linearConvDim(config),
            config.linear_conv_kernel_dim,
        ),
        .dt_bias = try loadTensorF32(
            weights_allocator,
            temp_allocator,
            store,
            io,
            dt_bias_name,
            &.{config.linear_num_value_heads},
        ),
        .a_log = try loadTensorF32(
            weights_allocator,
            temp_allocator,
            store,
            io,
            a_log_name,
            &.{config.linear_num_value_heads},
        ),
        .norm_weight = try loadTensorF32(
            weights_allocator,
            temp_allocator,
            store,
            io,
            norm_name,
            &.{config.linear_value_head_dim},
        ),
        .num_k_heads = config.linear_num_key_heads,
        .num_v_heads = config.linear_num_value_heads,
        .head_k_dim = config.linear_key_head_dim,
        .head_v_dim = config.linear_value_head_dim,
        .kernel = config.linear_conv_kernel_dim,
        .eps = config.rms_norm_eps,
    };
}

fn loadMlp(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    io: std.Io,
    store: *const safetensors.SafeTensorStore,
    config: cfg.QwenTextConfig,
    layer_base: []const u8,
    intermediate_size: usize,
    xpu_runtime: ?*xpu.Runtime,
) !engine.Mlp {
    const prefix = try std.fmt.allocPrint(temp_allocator, "{s}.mlp", .{layer_base});
    defer temp_allocator.free(prefix);
    return try loadMlpPrefix(weights_allocator, temp_allocator, io, store, config, prefix, intermediate_size, xpu_runtime);
}

fn loadMlpPrefix(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    io: std.Io,
    store: *const safetensors.SafeTensorStore,
    config: cfg.QwenTextConfig,
    prefix: []const u8,
    intermediate_size: usize,
    xpu_runtime: ?*xpu.Runtime,
) !engine.Mlp {
    const gate_proj_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.gate_proj", .{prefix});
    defer temp_allocator.free(gate_proj_prefix);
    const up_proj_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.up_proj", .{prefix});
    defer temp_allocator.free(up_proj_prefix);
    const down_proj_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.down_proj", .{prefix});
    defer temp_allocator.free(down_proj_prefix);
    return .{
        .gate_proj = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            gate_proj_prefix,
            config.hidden_size,
            intermediate_size,
            xpu_runtime,
        ),
        .up_proj = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            up_proj_prefix,
            config.hidden_size,
            intermediate_size,
            xpu_runtime,
        ),
        .down_proj = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            down_proj_prefix,
            intermediate_size,
            config.hidden_size,
            xpu_runtime,
        ),
    };
}

fn loadMoeBlock(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    io: std.Io,
    store: *const safetensors.SafeTensorStore,
    config: cfg.QwenTextConfig,
    layer_base: []const u8,
    xpu_runtime: ?*xpu.Runtime,
) !engine.MoeBlock {
    const prefix = try std.fmt.allocPrint(temp_allocator, "{s}.mlp", .{layer_base});
    defer temp_allocator.free(prefix);
    const gate_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.gate", .{prefix});
    defer temp_allocator.free(gate_prefix);
    const shared_gate_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.shared_expert_gate", .{prefix});
    defer temp_allocator.free(shared_gate_prefix);
    const experts = try weights_allocator.alloc(engine.Mlp, config.num_experts);
    for (0..config.num_experts) |expert_idx| {
        const expert_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.experts.{d}", .{ prefix, expert_idx });
        defer temp_allocator.free(expert_prefix);
        experts[expert_idx] = try loadMlpPrefix(
            weights_allocator,
            temp_allocator,
            io,
            store,
            config,
            expert_prefix,
            config.moe_intermediate_size,
            xpu_runtime,
        );
    }

    const shared_expert_prefix = try std.fmt.allocPrint(temp_allocator, "{s}.shared_expert", .{prefix});
    defer temp_allocator.free(shared_expert_prefix);
    return .{
        .gate = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            gate_prefix,
            config.hidden_size,
            config.num_experts,
            xpu_runtime,
        ),
        .experts = experts,
        .shared_expert = try loadMlpPrefix(
            weights_allocator,
            temp_allocator,
            io,
            store,
            config,
            shared_expert_prefix,
            config.shared_expert_intermediate_size,
            xpu_runtime,
        ),
        .shared_expert_gate = try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            shared_gate_prefix,
            config.hidden_size,
            1,
            xpu_runtime,
        ),
        .top_k = config.num_experts_per_tok,
        .norm_topk_prob = config.norm_topk_prob,
    };
}

fn loadLmHead(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    io: std.Io,
    store: *const safetensors.SafeTensorStore,
    config: cfg.QwenTextConfig,
    embeddings: []const f32,
    xpu_runtime: ?*xpu.Runtime,
) !engine.Linear {
    const lm_head_prefix = "lm_head";
    const lm_head_weight = "lm_head.weight";
    if (store.has(lm_head_weight)) {
        return try loadLinear(
            weights_allocator,
            temp_allocator,
            io,
            store,
            lm_head_prefix,
            config.hidden_size,
            config.vocab_size,
            xpu_runtime,
        );
    }
    if (config.tie_word_embeddings) {
        var dense: engine.DenseLinear = .{
            .in_features = config.hidden_size,
            .out_features = config.vocab_size,
            .weight = embeddings,
        };
        if (xpu_runtime) |runtime| {
            dense.xpu_weights = try runtime.createDenseWeights(embeddings, null, config.vocab_size, config.hidden_size);
        }
        return .{ .dense = dense };
    }
    return error.MissingLmHeadWeights;
}

fn loadLinear(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    io: std.Io,
    store: *const safetensors.SafeTensorStore,
    prefix: []const u8,
    in_features: usize,
    out_features: usize,
    xpu_runtime: ?*xpu.Runtime,
) !engine.Linear {
    const weight_name = try std.fmt.allocPrint(temp_allocator, "{s}.weight", .{prefix});
    defer temp_allocator.free(weight_name);
    if (store.has(weight_name)) {
        const bias_name = try std.fmt.allocPrint(temp_allocator, "{s}.bias", .{prefix});
        defer temp_allocator.free(bias_name);
        const weight = try loadTensorF32(
            weights_allocator,
            temp_allocator,
            store,
            io,
            weight_name,
            &.{ out_features, in_features },
        );
        const bias = try loadOptionalTensorF32(
            weights_allocator,
            temp_allocator,
            store,
            io,
            bias_name,
            &.{out_features},
        );
        var dense: engine.DenseLinear = .{
            .in_features = in_features,
            .out_features = out_features,
            .weight = weight,
            .bias = bias,
        };
        if (xpu_runtime) |runtime| {
            dense.xpu_weights = try runtime.createDenseWeights(weight, bias, out_features, in_features);
        }
        return .{ .dense = dense };
    }

    const qweight_name = try std.fmt.allocPrint(temp_allocator, "{s}.qweight", .{prefix});
    defer temp_allocator.free(qweight_name);
    const qzeros_name = try std.fmt.allocPrint(temp_allocator, "{s}.qzeros", .{prefix});
    defer temp_allocator.free(qzeros_name);
    const scales_name = try std.fmt.allocPrint(temp_allocator, "{s}.scales", .{prefix});
    defer temp_allocator.free(scales_name);

    if (!store.has(qweight_name) or !store.has(qzeros_name) or !store.has(scales_name)) {
        return error.TensorNotFound;
    }

    const qweight = try loadTensorI32(
        weights_allocator,
        temp_allocator,
        store,
        io,
        qweight_name,
        &.{ (in_features + 7) / 8, out_features },
    );
    const group_count = try inferFirstDim(store, io, temp_allocator, scales_name);
    const qzeros = try loadTensorI32(
        weights_allocator,
        temp_allocator,
        store,
        io,
        qzeros_name,
        &.{ group_count, (out_features + 7) / 8 },
    );
    const scales = try loadTensorF32(
        weights_allocator,
        temp_allocator,
        store,
        io,
        scales_name,
        &.{ group_count, out_features },
    );
    const bias_name = try std.fmt.allocPrint(temp_allocator, "{s}.bias", .{prefix});
    defer temp_allocator.free(bias_name);

    const bias = try loadOptionalTensorF32(
        weights_allocator,
        temp_allocator,
        store,
        io,
        bias_name,
        &.{out_features},
    );
    const group_size = @max(@as(usize, 1), (in_features + group_count - 1) / group_count);
    var autoround: engine.AutoRoundLinear = .{
        .in_features = in_features,
        .out_features = out_features,
        .group_size = group_size,
        .qweight = qweight,
        .qzeros = qzeros,
        .scales = scales,
        .bias = bias,
    };
    if (xpu_runtime) |runtime| {
        autoround.xpu_weights = try runtime.createAutoRoundWeights(qweight, qzeros, scales, bias, out_features, in_features, group_size);
    }
    return .{ .autoround = autoround };
}

fn loadTensorF32(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    store: *const safetensors.SafeTensorStore,
    io: std.Io,
    name: []const u8,
    expected_shape: []const usize,
) ![]const f32 {
    const view = try store.readTensorView(io, name, temp_allocator);
    defer temp_allocator.free(view.bytes);
    try expectShape(view.shape, expected_shape);

    const out = try weights_allocator.alloc(f32, view.numel());
    for (out, 0..) |*value, idx| value.* = view.f32At(idx);
    return out;
}

fn loadOptionalTensorF32(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    store: *const safetensors.SafeTensorStore,
    io: std.Io,
    name: []const u8,
    expected_shape: []const usize,
) !?[]const f32 {
    if (!store.has(name)) return null;
    return try loadTensorF32(weights_allocator, temp_allocator, store, io, name, expected_shape);
}

fn loadTensorI32(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    store: *const safetensors.SafeTensorStore,
    io: std.Io,
    name: []const u8,
    expected_shape: []const usize,
) ![]const i32 {
    const view = try store.readTensorView(io, name, temp_allocator);
    defer temp_allocator.free(view.bytes);
    try expectShape(view.shape, expected_shape);
    if (view.dtype != .i32) return error.InvalidTensorType;

    const out = try weights_allocator.alloc(i32, view.numel());
    for (out, 0..) |*value, idx| value.* = view.i32At(idx);
    return out;
}

fn loadConvWeight(
    weights_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    store: *const safetensors.SafeTensorStore,
    io: std.Io,
    name: []const u8,
    conv_dim: usize,
    kernel: usize,
) ![]const f32 {
    const view = try store.readTensorView(io, name, temp_allocator);
    defer temp_allocator.free(view.bytes);

    const accepted_shape_2 = [_]usize{ conv_dim, kernel };
    const accepted_shape_3 = [_]usize{ conv_dim, 1, kernel };
    const matches = shapeEquals(view.shape, &accepted_shape_2) or shapeEquals(view.shape, &accepted_shape_3);
    if (!matches) return error.ShapeMismatch;

    const out = try weights_allocator.alloc(f32, conv_dim * kernel);
    for (out, 0..) |*value, idx| value.* = view.f32At(idx);
    return out;
}

fn inferFirstDim(
    store: *const safetensors.SafeTensorStore,
    io: std.Io,
    temp_allocator: std.mem.Allocator,
    name: []const u8,
) !usize {
    const view = try store.readTensorView(io, name, temp_allocator);
    defer temp_allocator.free(view.bytes);
    if (view.shape.len == 0) return error.ShapeMismatch;
    return view.shape[0];
}

fn expectShape(actual: []const usize, expected: []const usize) !void {
    if (!shapeEquals(actual, expected)) return error.ShapeMismatch;
}

fn shapeEquals(actual: []const usize, expected: []const usize) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |left, right| {
        if (left != right) return false;
    }
    return true;
}

fn linearConvDim(config: cfg.QwenTextConfig) usize {
    return config.linear_num_key_heads * config.linear_key_head_dim * 2 +
        config.linear_num_value_heads * config.linear_value_head_dim;
}

fn deinitDecoderLayerXpu(layer: *const engine.DecoderLayer, runtime: *xpu.Runtime) void {
    switch (layer.attention) {
        .full => |*attention| {
            attention.q_proj.releaseXpu(runtime);
            attention.k_proj.releaseXpu(runtime);
            attention.v_proj.releaseXpu(runtime);
            attention.o_proj.releaseXpu(runtime);
        },
        .linear => |*attention| {
            attention.in_proj_qkv.releaseXpu(runtime);
            attention.in_proj_z.releaseXpu(runtime);
            attention.in_proj_b.releaseXpu(runtime);
            attention.in_proj_a.releaseXpu(runtime);
            attention.out_proj.releaseXpu(runtime);
        },
    }

    switch (layer.feed_forward) {
        .mlp => |*mlp| deinitMlpXpu(mlp, runtime),
        .moe => |*moe| {
            moe.gate.releaseXpu(runtime);
            for (moe.experts) |*expert| deinitMlpXpu(expert, runtime);
            deinitMlpXpu(&moe.shared_expert, runtime);
            moe.shared_expert_gate.releaseXpu(runtime);
        },
    }
}

fn deinitMlpXpu(mlp: *const engine.Mlp, runtime: *xpu.Runtime) void {
    mlp.gate_proj.releaseXpu(runtime);
    mlp.up_proj.releaseXpu(runtime);
    mlp.down_proj.releaseXpu(runtime);
}

test "qwen runtime loads tied embeddings and produces logits" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config =
        \\{
        \\  "text_config": {
        \\    "hidden_size": 2,
        \\    "intermediate_size": 4,
        \\    "num_hidden_layers": 0,
        \\    "num_attention_heads": 1,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 2,
        \\    "vocab_size": 3,
        \\    "tie_word_embeddings": true,
        \\    "layer_types": []
        \\  }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "config.json",
        .data = config,
    });

    const header =
        \\{"model.language_model.embed_tokens.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]},"model.language_model.norm.weight":{"dtype":"F32","shape":[2],"data_offsets":[24,32]}}
    ;
    var bytes = std.array_list.Managed(u8).init(std.testing.allocator);
    defer bytes.deinit();
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try bytes.appendSlice(&len_buf);
    try bytes.appendSlice(header);
    const values = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        0.0, 0.0,
    };
    for (values) |value| {
        var raw: [4]u8 = undefined;
        std.mem.writeInt(u32, &raw, @bitCast(value), .little);
        try bytes.appendSlice(&raw);
    }
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "model.safetensors",
        .data = bytes.items,
    });

    const model_dir = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(model_dir);

    var runtime = try QwenTextRuntime.load(std.testing.allocator, std.testing.io, model_dir);
    defer runtime.deinit();

    const logits = try runtime.forwardPromptAlloc(&.{1});
    defer std.testing.allocator.free(logits);

    try std.testing.expectEqual(@as(usize, 3), logits.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4142135), logits[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4142135), logits[2], 1e-5);
}
