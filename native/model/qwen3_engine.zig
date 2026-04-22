const std = @import("std");
const cfg = @import("qwen3_config.zig");
const tensor = @import("../tensor/tensor.zig");

pub const DenseLinear = struct {
    in_features: usize,
    out_features: usize,
    weight: []const f32,
    bias: ?[]const f32 = null,

    pub fn forwardInto(self: DenseLinear, out: []f32, input: []const f32) void {
        std.debug.assert(input.len == self.in_features);
        std.debug.assert(out.len == self.out_features);
        for (out, 0..) |*dst, row| {
            var sum: f32 = if (self.bias) |bias| bias[row] else 0.0;
            const weights = self.weight[row * self.in_features ..][0..self.in_features];
            for (weights, input) |w, x| sum += w * x;
            dst.* = sum;
        }
    }
};

pub const AutoRoundLinear = struct {
    in_features: usize,
    out_features: usize,
    group_size: usize,
    qweight: []const i32,
    qzeros: []const i32,
    scales: []const f32,
    bias: ?[]const f32 = null,

    pub fn forwardInto(self: AutoRoundLinear, out: []f32, input: []const f32) void {
        std.debug.assert(input.len == self.in_features);
        std.debug.assert(out.len == self.out_features);
        for (out, 0..) |*dst, row| {
            var sum: f32 = if (self.bias) |bias| bias[row] else 0.0;
            for (input, 0..) |x, col| {
                sum += x * self.dequant(row, col);
            }
            dst.* = sum;
        }
    }

    pub fn dequant(self: AutoRoundLinear, out_index: usize, in_index: usize) f32 {
        const packed_in = (self.in_features + 7) / 8;
        const packed_out = (self.out_features + 7) / 8;
        const qword = self.qweight[(in_index / 8) * self.out_features + out_index];
        const qvalue: i32 = @intCast((qword >> @intCast((in_index % 8) * 4)) & 0xF);
        const group = @min(in_index / self.group_size, (self.in_features + self.group_size - 1) / self.group_size - 1);
        const zword = self.qzeros[group * packed_out + out_index / 8];
        const zero: i32 = @intCast(((zword >> @intCast((out_index % 8) * 4)) & 0xF) + 1);
        const scale = self.scales[group * self.out_features + out_index];
        _ = packed_in;
        return @as(f32, @floatFromInt(qvalue - zero)) * scale;
    }
};

pub const Linear = union(enum) {
    dense: DenseLinear,
    autoround: AutoRoundLinear,

    pub fn inFeatures(self: Linear) usize {
        return switch (self) {
            .dense => |linear| linear.in_features,
            .autoround => |linear| linear.in_features,
        };
    }

    pub fn outFeatures(self: Linear) usize {
        return switch (self) {
            .dense => |linear| linear.out_features,
            .autoround => |linear| linear.out_features,
        };
    }

    pub fn forwardAlloc(self: Linear, allocator: std.mem.Allocator, input: []const f32) ![]f32 {
        const out = try allocator.alloc(f32, self.outFeatures());
        self.forwardInto(out, input);
        return out;
    }

    pub fn forwardInto(self: Linear, out: []f32, input: []const f32) void {
        switch (self) {
            .dense => |linear| linear.forwardInto(out, input),
            .autoround => |linear| linear.forwardInto(out, input),
        }
    }
};

pub const Mlp = struct {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,

    pub fn forwardAlloc(self: Mlp, allocator: std.mem.Allocator, hidden: []const f32) ![]f32 {
        const gate = try self.gate_proj.forwardAlloc(allocator, hidden);
        defer allocator.free(gate);
        const up = try self.up_proj.forwardAlloc(allocator, hidden);
        defer allocator.free(up);
        for (gate, up) |*g, u| g.* = tensor.silu(g.*) * u;
        return try self.down_proj.forwardAlloc(allocator, gate);
    }
};

pub const MoeBlock = struct {
    gate: Linear,
    experts: []const Mlp,
    shared_expert: Mlp,
    shared_expert_gate: Linear,
    top_k: usize,
    norm_topk_prob: bool = true,

    pub fn forwardAlloc(self: MoeBlock, allocator: std.mem.Allocator, hidden: []const f32) ![]f32 {
        const router_logits = try self.gate.forwardAlloc(allocator, hidden);
        defer allocator.free(router_logits);
        const selected = try topK(allocator, router_logits, self.top_k);
        defer allocator.free(selected);

        const weights = try allocator.alloc(f32, selected.len);
        defer allocator.free(weights);
        var max_logit = -std.math.inf(f32);
        for (selected) |idx| max_logit = @max(max_logit, router_logits[idx]);
        var sum: f32 = 0.0;
        for (weights, selected) |*weight, idx| {
            weight.* = @exp(router_logits[idx] - max_logit);
            sum += weight.*;
        }
        if (self.norm_topk_prob and sum > 0.0) {
            for (weights) |*weight| weight.* /= sum;
        }

        const out = try allocator.alloc(f32, hidden.len);
        @memset(out, 0.0);
        for (selected, weights) |expert_idx, weight| {
            const expert_out = try self.experts[expert_idx].forwardAlloc(allocator, hidden);
            defer allocator.free(expert_out);
            for (out, expert_out) |*dst, value| dst.* += value * weight;
        }
        const shared = try self.shared_expert.forwardAlloc(allocator, hidden);
        defer allocator.free(shared);
        const shared_gate = try self.shared_expert_gate.forwardAlloc(allocator, hidden);
        defer allocator.free(shared_gate);
        const gate_value = tensor.sigmoid(shared_gate[0]);
        for (out, shared) |*dst, value| dst.* += value * gate_value;
        return out;
    }
};

pub const FeedForward = union(enum) {
    mlp: Mlp,
    moe: MoeBlock,

    pub fn forwardAlloc(self: FeedForward, allocator: std.mem.Allocator, hidden: []const f32) ![]f32 {
        return switch (self) {
            .mlp => |mlp| mlp.forwardAlloc(allocator, hidden),
            .moe => |moe| moe.forwardAlloc(allocator, hidden),
        };
    }
};

pub const FullAttentionState = struct {
    allocator: std.mem.Allocator,
    num_kv_heads: usize,
    head_dim: usize,
    keys: std.array_list.Managed(f32),
    values: std.array_list.Managed(f32),
    len: usize = 0,

    pub fn init(allocator: std.mem.Allocator, num_kv_heads: usize, head_dim: usize) FullAttentionState {
        return .{
            .allocator = allocator,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .keys = std.array_list.Managed(f32).init(allocator),
            .values = std.array_list.Managed(f32).init(allocator),
        };
    }

    pub fn deinit(self: *FullAttentionState) void {
        self.keys.deinit();
        self.values.deinit();
    }

    fn append(self: *FullAttentionState, key: []const f32, value: []const f32) !void {
        try self.keys.appendSlice(key);
        try self.values.appendSlice(value);
        self.len += 1;
    }

    fn keyAt(self: *const FullAttentionState, pos: usize, kv_head: usize) []const f32 {
        const offset = (pos * self.num_kv_heads + kv_head) * self.head_dim;
        return self.keys.items[offset..][0..self.head_dim];
    }

    fn valueAt(self: *const FullAttentionState, pos: usize, kv_head: usize) []const f32 {
        const offset = (pos * self.num_kv_heads + kv_head) * self.head_dim;
        return self.values.items[offset..][0..self.head_dim];
    }
};

pub const FullAttention = struct {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm_weight: []const f32,
    k_norm_weight: []const f32,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32 = 10_000_000.0,
    partial_rotary_factor: f32 = 0.25,
    eps: f32 = 1e-6,

    pub fn forwardStepAlloc(self: FullAttention, allocator: std.mem.Allocator, hidden: []const f32, state: *FullAttentionState, position: usize) ![]f32 {
        const projected_q = try self.q_proj.forwardAlloc(allocator, hidden);
        defer allocator.free(projected_q);
        const projected_k = try self.k_proj.forwardAlloc(allocator, hidden);
        defer allocator.free(projected_k);
        const projected_v = try self.v_proj.forwardAlloc(allocator, hidden);
        defer allocator.free(projected_v);

        const q_values = projected_q[0 .. self.num_heads * self.head_dim];
        const gate = projected_q[self.num_heads * self.head_dim ..][0 .. self.num_heads * self.head_dim];

        var q = try allocator.dupe(f32, q_values);
        defer allocator.free(q);
        var k = try allocator.dupe(f32, projected_k);
        defer allocator.free(k);
        for (0..self.num_heads) |head| {
            tensor.rmsNormQwen(q[head * self.head_dim ..][0..self.head_dim], q[head * self.head_dim ..][0..self.head_dim], self.q_norm_weight, self.eps);
            applyRope(q[head * self.head_dim ..][0..self.head_dim], position, self.rope_theta, self.partial_rotary_factor);
        }
        for (0..self.num_kv_heads) |head| {
            tensor.rmsNormQwen(k[head * self.head_dim ..][0..self.head_dim], k[head * self.head_dim ..][0..self.head_dim], self.k_norm_weight, self.eps);
            applyRope(k[head * self.head_dim ..][0..self.head_dim], position, self.rope_theta, self.partial_rotary_factor);
        }
        try state.append(k, projected_v);

        var attn_out = try allocator.alloc(f32, self.num_heads * self.head_dim);
        defer allocator.free(attn_out);
        const groups = self.num_heads / self.num_kv_heads;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const scores = try allocator.alloc(f32, state.len);
        defer allocator.free(scores);
        for (0..self.num_heads) |head| {
            const kv_head = head / groups;
            const query = q[head * self.head_dim ..][0..self.head_dim];
            for (scores, 0..) |*score, pos| {
                score.* = tensor.dot(query, state.keyAt(pos, kv_head)) * scale;
            }
            tensor.stableSoftmaxInPlace(scores);
            const head_out = attn_out[head * self.head_dim ..][0..self.head_dim];
            @memset(head_out, 0.0);
            for (scores, 0..) |score, pos| {
                const value = state.valueAt(pos, kv_head);
                for (head_out, value) |*dst, v| dst.* += score * v;
            }
        }
        for (attn_out, gate) |*value, g| value.* *= tensor.sigmoid(g);
        return try self.o_proj.forwardAlloc(allocator, attn_out);
    }
};

pub const LinearAttentionState = struct {
    allocator: std.mem.Allocator,
    conv_state: []f32,
    recurrent_state: []f32,

    pub fn init(allocator: std.mem.Allocator, conv_dim: usize, kernel: usize, heads: usize, key_dim: usize, value_dim: usize) !LinearAttentionState {
        const conv_state = try allocator.alloc(f32, conv_dim * kernel);
        errdefer allocator.free(conv_state);
        const recurrent_state = try allocator.alloc(f32, heads * key_dim * value_dim);
        @memset(conv_state, 0.0);
        @memset(recurrent_state, 0.0);
        return .{ .allocator = allocator, .conv_state = conv_state, .recurrent_state = recurrent_state };
    }

    pub fn deinit(self: *LinearAttentionState) void {
        self.allocator.free(self.conv_state);
        self.allocator.free(self.recurrent_state);
    }
};

pub const LinearAttention = struct {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    conv_weight: []const f32,
    dt_bias: []const f32,
    a_log: []const f32,
    norm_weight: []const f32,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    kernel: usize,
    eps: f32 = 1e-6,

    pub fn convDim(self: LinearAttention) usize {
        return self.num_k_heads * self.head_k_dim * 2 + self.num_v_heads * self.head_v_dim;
    }

    pub fn forwardStepAlloc(self: LinearAttention, allocator: std.mem.Allocator, hidden: []const f32, state: *LinearAttentionState) ![]f32 {
        const mixed_input = try self.in_proj_qkv.forwardAlloc(allocator, hidden);
        defer allocator.free(mixed_input);
        const z = try self.in_proj_z.forwardAlloc(allocator, hidden);
        defer allocator.free(z);
        const b = try self.in_proj_b.forwardAlloc(allocator, hidden);
        defer allocator.free(b);
        const a = try self.in_proj_a.forwardAlloc(allocator, hidden);
        defer allocator.free(a);

        const conv_dim = self.convDim();
        var mixed = try allocator.alloc(f32, conv_dim);
        defer allocator.free(mixed);
        for (0..conv_dim) |channel| {
            const row = state.conv_state[channel * self.kernel ..][0..self.kernel];
            if (self.kernel > 1) std.mem.copyForwards(f32, row[0 .. self.kernel - 1], row[1..self.kernel]);
            row[self.kernel - 1] = mixed_input[channel];
            var sum: f32 = 0.0;
            for (row, self.conv_weight[channel * self.kernel ..][0..self.kernel]) |value, weight| {
                sum += value * weight;
            }
            mixed[channel] = tensor.silu(sum);
        }

        const key_dim = self.num_k_heads * self.head_k_dim;
        const value_dim = self.num_v_heads * self.head_v_dim;
        const raw_q = mixed[0..key_dim];
        const raw_k = mixed[key_dim..][0..key_dim];
        const raw_v = mixed[key_dim * 2 ..][0..value_dim];
        const repeat = self.num_v_heads / self.num_k_heads;

        var core = try allocator.alloc(f32, value_dim);
        defer allocator.free(core);
        for (0..self.num_v_heads) |head| {
            const source_head = head / repeat;
            const q = try allocator.dupe(f32, raw_q[source_head * self.head_k_dim ..][0..self.head_k_dim]);
            defer allocator.free(q);
            const k = try allocator.dupe(f32, raw_k[source_head * self.head_k_dim ..][0..self.head_k_dim]);
            defer allocator.free(k);
            tensor.l2NormalizeInPlace(q, 1e-6);
            tensor.l2NormalizeInPlace(k, 1e-6);
            const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_k_dim)));
            for (q) |*value| value.* *= q_scale;
            const value = raw_v[head * self.head_v_dim ..][0..self.head_v_dim];
            const beta = tensor.sigmoid(b[head]);
            const g = -@exp(self.a_log[head]) * tensor.softplus(a[head] + self.dt_bias[head]);
            const decay = @exp(g);
            const recurrent = state.recurrent_state[head * self.head_k_dim * self.head_v_dim ..][0 .. self.head_k_dim * self.head_v_dim];
            for (recurrent) |*entry| entry.* *= decay;
            var delta = try allocator.alloc(f32, self.head_v_dim);
            defer allocator.free(delta);
            @memset(delta, 0.0);
            for (0..self.head_v_dim) |v_idx| {
                var kv_mem: f32 = 0.0;
                for (0..self.head_k_dim) |k_idx| {
                    kv_mem += recurrent[k_idx * self.head_v_dim + v_idx] * k[k_idx];
                }
                delta[v_idx] = (value[v_idx] - kv_mem) * beta;
            }
            for (0..self.head_k_dim) |k_idx| {
                for (0..self.head_v_dim) |v_idx| {
                    recurrent[k_idx * self.head_v_dim + v_idx] += k[k_idx] * delta[v_idx];
                }
            }
            const head_core = core[head * self.head_v_dim ..][0..self.head_v_dim];
            @memset(head_core, 0.0);
            for (0..self.head_v_dim) |v_idx| {
                for (0..self.head_k_dim) |k_idx| {
                    head_core[v_idx] += recurrent[k_idx * self.head_v_dim + v_idx] * q[k_idx];
                }
            }
        }

        var normed = try allocator.alloc(f32, value_dim);
        defer allocator.free(normed);
        for (0..self.num_v_heads) |head| {
            tensor.rmsNormGated(
                normed[head * self.head_v_dim ..][0..self.head_v_dim],
                core[head * self.head_v_dim ..][0..self.head_v_dim],
                z[head * self.head_v_dim ..][0..self.head_v_dim],
                self.norm_weight,
                self.eps,
            );
        }
        return try self.out_proj.forwardAlloc(allocator, normed);
    }
};

pub const Attention = union(enum) {
    full: FullAttention,
    linear: LinearAttention,

    pub fn forwardStepAlloc(self: Attention, allocator: std.mem.Allocator, hidden: []const f32, state: *LayerState, position: usize) ![]f32 {
        return switch (self) {
            .full => |attention| attention.forwardStepAlloc(allocator, hidden, &state.full, position),
            .linear => |attention| attention.forwardStepAlloc(allocator, hidden, &state.linear),
        };
    }
};

pub const LayerState = union(enum) {
    full: FullAttentionState,
    linear: LinearAttentionState,

    pub fn deinit(self: *LayerState) void {
        switch (self.*) {
            .full => |*state| state.deinit(),
            .linear => |*state| state.deinit(),
        }
    }
};

pub const DecoderLayer = struct {
    attention: Attention,
    feed_forward: FeedForward,
    input_norm_weight: []const f32,
    post_norm_weight: []const f32,
    eps: f32 = 1e-6,

    pub fn forwardStepAlloc(self: DecoderLayer, allocator: std.mem.Allocator, hidden: []const f32, state: *LayerState, position: usize) ![]f32 {
        const normed = try allocator.alloc(f32, hidden.len);
        defer allocator.free(normed);
        tensor.rmsNormQwen(normed, hidden, self.input_norm_weight, self.eps);
        const attn = try self.attention.forwardStepAlloc(allocator, normed, state, position);
        defer allocator.free(attn);

        const residual = try allocator.alloc(f32, hidden.len);
        errdefer allocator.free(residual);
        for (residual, hidden, attn) |*dst, h, a| dst.* = h + a;

        tensor.rmsNormQwen(normed, residual, self.post_norm_weight, self.eps);
        const ff = try self.feed_forward.forwardAlloc(allocator, normed);
        defer allocator.free(ff);
        for (residual, ff) |*dst, value| dst.* += value;
        return residual;
    }
};

pub const TokenEngine = struct {
    allocator: std.mem.Allocator,
    hidden_size: usize,
    embeddings: []const f32,
    vocab_size: usize,
    layers: []const DecoderLayer,
    norm_weight: []const f32,
    lm_head: Linear,
    states: []LayerState,
    position: usize = 0,

    pub fn deinitStates(self: *TokenEngine) void {
        for (self.states) |*state| state.deinit();
        self.allocator.free(self.states);
        self.states = &.{};
    }

    pub fn forwardTokenAlloc(self: *TokenEngine, token_id: u32) ![]f32 {
        if (token_id >= self.vocab_size) return error.TokenOutOfRange;
        var hidden = try self.allocator.dupe(f32, self.embeddings[@as(usize, token_id) * self.hidden_size ..][0..self.hidden_size]);
        errdefer self.allocator.free(hidden);

        for (self.layers, 0..) |layer, idx| {
            const next = try layer.forwardStepAlloc(self.allocator, hidden, &self.states[idx], self.position);
            self.allocator.free(hidden);
            hidden = next;
        }

        const normed = try self.allocator.alloc(f32, self.hidden_size);
        defer self.allocator.free(normed);
        tensor.rmsNormQwen(normed, hidden, self.norm_weight, 1e-6);
        self.allocator.free(hidden);
        self.position += 1;
        return try self.lm_head.forwardAlloc(self.allocator, normed);
    }
};

fn applyRope(values: []f32, position: usize, theta: f32, partial_rotary_factor: f32) void {
    var rotary_dim: usize = @intFromFloat(@as(f32, @floatFromInt(values.len)) * partial_rotary_factor);
    rotary_dim = @max(@as(usize, 2), rotary_dim - (rotary_dim % 2));
    rotary_dim = @min(rotary_dim, values.len - (values.len % 2));
    const half = rotary_dim / 2;
    for (0..half) |idx| {
        const inv_freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(idx * 2)) / @as(f32, @floatFromInt(rotary_dim)));
        const angle = @as(f32, @floatFromInt(position)) * inv_freq;
        const cos = @cos(angle);
        const sin = @sin(angle);
        const x1 = values[idx];
        const x2 = values[idx + half];
        values[idx] = x1 * cos - x2 * sin;
        values[idx + half] = x2 * cos + x1 * sin;
    }
}

fn topK(allocator: std.mem.Allocator, values: []const f32, k: usize) ![]usize {
    const count = @min(k, values.len);
    const selected = try allocator.alloc(usize, count);
    var used = try allocator.alloc(bool, values.len);
    defer allocator.free(used);
    @memset(used, false);
    for (selected) |*slot| {
        var best: ?usize = null;
        for (values, 0..) |value, idx| {
            if (used[idx]) continue;
            if (best == null or value > values[best.?]) best = idx;
        }
        slot.* = best.?;
        used[slot.*] = true;
    }
    return selected;
}

test "autoround int4 matvec dequantizes autogptq packing" {
    const qweight = [_]i32{0x00000021};
    const qzeros = [_]i32{0x00000000};
    const scales = [_]f32{0.5};
    const linear: AutoRoundLinear = .{
        .in_features = 2,
        .out_features = 1,
        .group_size = 128,
        .qweight = &qweight,
        .qzeros = &qzeros,
        .scales = &scales,
    };
    const input = [_]f32{ 2.0, 4.0 };
    var out: [1]f32 = undefined;
    linear.forwardInto(&out, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[0], 1e-6);
}

test "mlp forward uses silu gated product" {
    const gate_w = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const up_w = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const down_w = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const mlp: Mlp = .{
        .gate_proj = .{ .dense = .{ .in_features = 2, .out_features = 2, .weight = &gate_w } },
        .up_proj = .{ .dense = .{ .in_features = 2, .out_features = 2, .weight = &up_w } },
        .down_proj = .{ .dense = .{ .in_features = 2, .out_features = 2, .weight = &down_w } },
    };
    const out = try mlp.forwardAlloc(std.testing.allocator, &.{ 1.0, 2.0 });
    defer std.testing.allocator.free(out);
    try std.testing.expectApproxEqAbs(tensor.silu(1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(tensor.silu(2.0) * 2.0, out[1], 1e-6);
}

test "linear attention decode updates recurrent state" {
    const identity2 = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const ones1 = [_]f32{1.0};
    const conv_w = [_]f32{ 1.0, 1.0, 1.0 };
    var state = try LinearAttentionState.init(std.testing.allocator, 3, 1, 1, 1, 1);
    defer state.deinit();

    const attention: LinearAttention = .{
        .in_proj_qkv = .{ .dense = .{ .in_features = 2, .out_features = 3, .weight = &.{ 1.0, 0.0, 1.0, 0.0, 0.0, 1.0 } } },
        .in_proj_z = .{ .dense = .{ .in_features = 2, .out_features = 1, .weight = &.{ 0.0, 1.0 } } },
        .in_proj_b = .{ .dense = .{ .in_features = 2, .out_features = 1, .weight = &.{ 0.0, 0.0 } } },
        .in_proj_a = .{ .dense = .{ .in_features = 2, .out_features = 1, .weight = &.{ 0.0, 0.0 } } },
        .out_proj = .{ .dense = .{ .in_features = 1, .out_features = 2, .weight = &identity2 } },
        .conv_weight = &conv_w,
        .dt_bias = &.{0.0},
        .a_log = &.{0.0},
        .norm_weight = &ones1,
        .num_k_heads = 1,
        .num_v_heads = 1,
        .head_k_dim = 1,
        .head_v_dim = 1,
        .kernel = 1,
    };
    const out = try attention.forwardStepAlloc(std.testing.allocator, &.{ 1.0, 2.0 }, &state);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqual(@as(usize, 2), out.len);
    try std.testing.expect(state.recurrent_state[0] != 0.0);
}

test "full attention step appends kv cache and returns hidden output" {
    const q_w = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
    };
    const kv_w = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const o_w = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const norm = [_]f32{ 0.0, 0.0 };
    var state = FullAttentionState.init(std.testing.allocator, 1, 2);
    defer state.deinit();
    const attention: FullAttention = .{
        .q_proj = .{ .dense = .{ .in_features = 2, .out_features = 4, .weight = &q_w } },
        .k_proj = .{ .dense = .{ .in_features = 2, .out_features = 2, .weight = &kv_w } },
        .v_proj = .{ .dense = .{ .in_features = 2, .out_features = 2, .weight = &kv_w } },
        .o_proj = .{ .dense = .{ .in_features = 2, .out_features = 2, .weight = &o_w } },
        .q_norm_weight = &norm,
        .k_norm_weight = &norm,
        .num_heads = 1,
        .num_kv_heads = 1,
        .head_dim = 2,
        .partial_rotary_factor = 1.0,
    };
    const out = try attention.forwardStepAlloc(std.testing.allocator, &.{ 1.0, 0.0 }, &state, 0);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqual(@as(usize, 1), state.len);
    try std.testing.expectEqual(@as(usize, 2), out.len);
}
