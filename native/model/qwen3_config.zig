const std = @import("std");

pub const LayerType = enum {
    linear_attention,
    full_attention,
};

pub const RopeParameters = struct {
    rope_theta: f32 = 10_000_000.0,
    partial_rotary_factor: f32 = 0.25,
};

pub const QwenTextConfig = struct {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    vocab_size: usize,
    eos_token_id: u32,
    pad_token_id: u32,
    tie_word_embeddings: bool,
    rms_norm_eps: f32,
    decoder_sparse_step: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    layer_types: []LayerType,
    mlp_only_layers: []const usize,
    rope: RopeParameters,

    pub fn deinit(self: *QwenTextConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.layer_types);
        allocator.free(self.mlp_only_layers);
        self.* = undefined;
    }

    pub fn numKeyValueGroups(self: QwenTextConfig) usize {
        return self.num_attention_heads / self.num_key_value_heads;
    }

    pub fn isMoeModel(self: QwenTextConfig) bool {
        return self.num_experts > 0 and self.num_experts_per_tok > 0;
    }

    pub fn usesSparseMoe(self: QwenTextConfig, layer_idx: usize) bool {
        if (!self.isMoeModel()) return false;
        for (self.mlp_only_layers) |mlp_only_layer| {
            if (mlp_only_layer == layer_idx) return false;
        }
        return (layer_idx + 1) % @max(@as(usize, 1), self.decoder_sparse_step) == 0;
    }
};

const RootConfig = struct {
    text_config: TextConfigJson,
};

const TextConfigJson = struct {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: ?usize = null,
    head_dim: ?usize = null,
    linear_conv_kernel_dim: usize = 4,
    linear_key_head_dim: usize = 128,
    linear_value_head_dim: usize = 128,
    linear_num_key_heads: usize = 16,
    linear_num_value_heads: usize = 16,
    vocab_size: usize,
    eos_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,
    tie_word_embeddings: bool = true,
    rms_norm_eps: f32 = 1e-6,
    decoder_sparse_step: usize = 1,
    moe_intermediate_size: ?usize = null,
    shared_expert_intermediate_size: ?usize = null,
    num_experts: usize = 0,
    num_experts_per_tok: usize = 0,
    norm_topk_prob: bool = true,
    layer_types: ?[]const []const u8 = null,
    mlp_only_layers: ?[]const usize = null,
    full_attention_interval: usize = 4,
    rope_parameters: ?RopeJson = null,
};

const RopeJson = struct {
    rope_theta: f32 = 10_000_000.0,
    partial_rotary_factor: f32 = 0.25,
};

pub fn loadQwenTextConfigFromModelDir(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !QwenTextConfig {
    var dir = try openModelDir(io, model_dir);
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "config.json", allocator, .limited(8 << 20));
    defer allocator.free(raw);
    return parseQwenTextConfig(allocator, raw);
}

pub fn parseQwenTextConfig(allocator: std.mem.Allocator, raw: []const u8) !QwenTextConfig {
    var parsed = try std.json.parseFromSlice(RootConfig, allocator, raw, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const source = parsed.value.text_config;

    const layer_types = try allocator.alloc(LayerType, source.num_hidden_layers);
    errdefer allocator.free(layer_types);
    if (source.layer_types) |raw_layer_types| {
        if (raw_layer_types.len != source.num_hidden_layers) return error.InvalidQwenConfig;
        for (raw_layer_types, 0..) |value, idx| {
            layer_types[idx] = parseLayerType(value) orelse return error.InvalidQwenConfig;
        }
    } else {
        for (layer_types, 0..) |*layer_type, idx| {
            layer_type.* = if ((idx + 1) % source.full_attention_interval == 0) .full_attention else .linear_attention;
        }
    }

    const rope: RopeJson = source.rope_parameters orelse .{};
    const eos = source.eos_token_id orelse 248044;
    const mlp_only_layers = try allocator.dupe(usize, source.mlp_only_layers orelse &.{});
    errdefer allocator.free(mlp_only_layers);
    return .{
        .hidden_size = source.hidden_size,
        .intermediate_size = source.intermediate_size,
        .num_hidden_layers = source.num_hidden_layers,
        .num_attention_heads = source.num_attention_heads,
        .num_key_value_heads = source.num_key_value_heads orelse source.num_attention_heads,
        .head_dim = source.head_dim orelse (source.hidden_size / source.num_attention_heads),
        .linear_conv_kernel_dim = source.linear_conv_kernel_dim,
        .linear_key_head_dim = source.linear_key_head_dim,
        .linear_value_head_dim = source.linear_value_head_dim,
        .linear_num_key_heads = source.linear_num_key_heads,
        .linear_num_value_heads = source.linear_num_value_heads,
        .vocab_size = source.vocab_size,
        .eos_token_id = eos,
        .pad_token_id = source.pad_token_id orelse eos,
        .tie_word_embeddings = source.tie_word_embeddings,
        .rms_norm_eps = source.rms_norm_eps,
        .decoder_sparse_step = source.decoder_sparse_step,
        .moe_intermediate_size = source.moe_intermediate_size orelse source.intermediate_size,
        .shared_expert_intermediate_size = source.shared_expert_intermediate_size orelse source.intermediate_size,
        .num_experts = source.num_experts,
        .num_experts_per_tok = if (source.num_experts_per_tok == 0 and source.num_experts > 0) 8 else source.num_experts_per_tok,
        .norm_topk_prob = source.norm_topk_prob,
        .layer_types = layer_types,
        .mlp_only_layers = mlp_only_layers,
        .rope = .{
            .rope_theta = rope.rope_theta,
            .partial_rotary_factor = rope.partial_rotary_factor,
        },
    };
}

fn parseLayerType(value: []const u8) ?LayerType {
    if (std.mem.eql(u8, value, "linear_attention")) return .linear_attention;
    if (std.mem.eql(u8, value, "full_attention")) return .full_attention;
    return null;
}

fn openModelDir(io: std.Io, path: []const u8) !std.Io.Dir {
    if (std.fs.path.isAbsolute(path)) {
        return try std.Io.Dir.openDirAbsolute(io, path, .{});
    }
    return try std.Io.Dir.cwd().openDir(io, path, .{});
}

test "qwen text config parses hybrid layer types" {
    const raw =
        \\{"text_config":{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":4,
        \\"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"vocab_size":32,
        \\"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]}}
    ;
    var config = try parseQwenTextConfig(std.testing.allocator, raw);
    defer config.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 8), config.hidden_size);
    try std.testing.expectEqual(LayerType.linear_attention, config.layer_types[0]);
    try std.testing.expectEqual(LayerType.full_attention, config.layer_types[3]);
}
