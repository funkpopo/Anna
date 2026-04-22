const std = @import("std");

pub const DType = enum {
    f32,
    f16,
    bf16,
    i32,
    i64,
    u8,
    bool,

    pub fn fromSafetensors(value: []const u8) !DType {
        if (std.mem.eql(u8, value, "F32")) return .f32;
        if (std.mem.eql(u8, value, "F16")) return .f16;
        if (std.mem.eql(u8, value, "BF16")) return .bf16;
        if (std.mem.eql(u8, value, "I32")) return .i32;
        if (std.mem.eql(u8, value, "I64")) return .i64;
        if (std.mem.eql(u8, value, "U8")) return .u8;
        if (std.mem.eql(u8, value, "BOOL")) return .bool;
        return error.UnsupportedDType;
    }

    pub fn elementSize(self: DType) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16 => 2,
            .i64 => 8,
            .u8, .bool => 1,
        };
    }
};

pub const TensorView = struct {
    dtype: DType,
    shape: []const usize,
    bytes: []const u8,

    pub fn numel(self: TensorView) usize {
        var total: usize = 1;
        for (self.shape) |dim| total *= dim;
        return total;
    }

    pub fn byteLen(self: TensorView) usize {
        return self.numel() * self.dtype.elementSize();
    }

    pub fn f32At(self: TensorView, index: usize) f32 {
        return readElementF32(self.dtype, self.bytes[index * self.dtype.elementSize() ..]);
    }

    pub fn i32At(self: TensorView, index: usize) i32 {
        const start = index * 4;
        return @bitCast(std.mem.readInt(u32, self.bytes[start..][0..4], .little));
    }
};

pub const F32Tensor = struct {
    allocator: std.mem.Allocator,
    shape: []usize,
    data: []f32,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !F32Tensor {
        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);
        var count: usize = 1;
        for (shape) |dim| count *= dim;
        const data = try allocator.alloc(f32, count);
        @memset(data, 0.0);
        return .{ .allocator = allocator, .shape = owned_shape, .data = data };
    }

    pub fn fromView(allocator: std.mem.Allocator, view: TensorView) !F32Tensor {
        const tensor = try init(allocator, view.shape);
        for (tensor.data, 0..) |*value, idx| value.* = view.f32At(idx);
        return tensor;
    }

    pub fn deinit(self: *F32Tensor) void {
        self.allocator.free(self.shape);
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn rows(self: F32Tensor) usize {
        return if (self.shape.len > 0) self.shape[0] else 0;
    }

    pub fn cols(self: F32Tensor) usize {
        return if (self.shape.len > 1) self.shape[1] else 0;
    }
};

pub fn readElementF32(dtype: DType, bytes: []const u8) f32 {
    return switch (dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[0..4], .little)),
        .f16 => @as(f32, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[0..2], .little))))),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[0..2], .little)) << 16),
        .i32 => @floatFromInt(@as(i32, @bitCast(std.mem.readInt(u32, bytes[0..4], .little)))),
        .i64 => @floatFromInt(@as(i64, @bitCast(std.mem.readInt(u64, bytes[0..8], .little)))),
        .u8 => @floatFromInt(bytes[0]),
        .bool => if (bytes[0] == 0) 0.0 else 1.0,
    };
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

pub fn softplus(x: f32) f32 {
    if (x > 20.0) return x;
    if (x < -20.0) return @exp(x);
    return @log(1.0 + @exp(x));
}

pub fn dot(left: []const f32, right: []const f32) f32 {
    std.debug.assert(left.len == right.len);
    var sum: f32 = 0.0;
    for (left, right) |l, r| sum += l * r;
    return sum;
}

pub fn rmsNormQwen(out: []f32, input: []const f32, weight: []const f32, eps: f32) void {
    std.debug.assert(out.len == input.len);
    std.debug.assert(weight.len == input.len);
    var mean_square: f32 = 0.0;
    for (input) |value| mean_square += value * value;
    mean_square /= @floatFromInt(input.len);
    const inv = 1.0 / @sqrt(mean_square + eps);
    for (out, input, weight) |*dst, value, w| {
        dst.* = value * inv * (1.0 + w);
    }
}

pub fn rmsNormGated(out: []f32, input: []const f32, gate: []const f32, weight: []const f32, eps: f32) void {
    std.debug.assert(out.len == input.len);
    std.debug.assert(gate.len == input.len);
    std.debug.assert(weight.len == input.len);
    var mean_square: f32 = 0.0;
    for (input) |value| mean_square += value * value;
    mean_square /= @floatFromInt(input.len);
    const inv = 1.0 / @sqrt(mean_square + eps);
    for (out, input, gate, weight) |*dst, value, g, w| {
        dst.* = w * value * inv * silu(g);
    }
}

pub fn l2NormalizeInPlace(values: []f32, eps: f32) void {
    var square_sum: f32 = 0.0;
    for (values) |value| square_sum += value * value;
    const inv = 1.0 / @sqrt(square_sum + eps);
    for (values) |*value| value.* *= inv;
}

pub fn stableSoftmaxInPlace(values: []f32) void {
    var max_value = -std.math.inf(f32);
    for (values) |value| max_value = @max(max_value, value);
    var sum: f32 = 0.0;
    for (values) |*value| {
        value.* = @exp(value.* - max_value);
        sum += value.*;
    }
    const inv = 1.0 / sum;
    for (values) |*value| value.* *= inv;
}

test "dtype conversion reads bf16 and f16" {
    const bf16_one = [_]u8{ 0x80, 0x3f };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), readElementF32(.bf16, &bf16_one), 1e-6);

    const f16_one = [_]u8{ 0x00, 0x3c };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), readElementF32(.f16, &f16_one), 1e-3);
}

test "qwen rms norm uses one plus learned weight" {
    const input = [_]f32{ 3.0, 4.0 };
    const weight = [_]f32{ 0.0, 0.0 };
    var output: [2]f32 = undefined;
    rmsNormQwen(&output, &input, &weight, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8485281), output[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1313708), output[1], 1e-5);
}
