const builtin = @import("builtin");
const std = @import("std");

comptime {
    if (builtin.os.tag != .windows) @compileError("sycl_backend currently supports Windows only");
}

extern "kernel32" fn LoadLibraryW(lpLibFileName: [*:0]const u16) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetProcAddress(hModule: ?*anyopaque, lpProcName: [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn FreeLibrary(hLibModule: ?*anyopaque) callconv(.winapi) i32;

const SyclRuntimeHandle = ?*anyopaque;
const SyclBufferHandle = ?*anyopaque;
const SyclDenseHandle = ?*anyopaque;
const SyclAutoRoundHandle = ?*anyopaque;

const ANNA_SYCL_SUCCESS: c_int = 0;
const ANNA_SYCL_BACKEND_UNAVAILABLE: c_int = 1;
const ANNA_SYCL_DEVICE_NOT_FOUND: c_int = 2;
const ANNA_SYCL_INVALID_ARGUMENT: c_int = 3;
const ANNA_SYCL_ALLOCATION_FAILED: c_int = 4;

const Api = struct {
    anna_sycl_last_error_message: *const fn () callconv(.c) ?[*:0]const u8,
    anna_sycl_runtime_create: *const fn (*SyclRuntimeHandle) callconv(.c) c_int,
    anna_sycl_runtime_destroy: *const fn (SyclRuntimeHandle) callconv(.c) void,
    anna_sycl_runtime_platform_name: *const fn (SyclRuntimeHandle) callconv(.c) ?[*:0]const u8,
    anna_sycl_runtime_device_name: *const fn (SyclRuntimeHandle) callconv(.c) ?[*:0]const u8,
    anna_sycl_runtime_vendor_name: *const fn (SyclRuntimeHandle) callconv(.c) ?[*:0]const u8,
    anna_sycl_dense_create: *const fn (SyclRuntimeHandle, [*]const f32, usize, ?[*]const f32, usize, usize, usize, *SyclDenseHandle) callconv(.c) c_int,
    anna_sycl_dense_destroy: *const fn (SyclDenseHandle) callconv(.c) void,
    anna_sycl_autoround_create: *const fn (SyclRuntimeHandle, [*]const i32, usize, [*]const i32, usize, [*]const f32, usize, ?[*]const f32, usize, usize, usize, usize, *SyclAutoRoundHandle) callconv(.c) c_int,
    anna_sycl_autoround_destroy: *const fn (SyclAutoRoundHandle) callconv(.c) void,
    anna_sycl_buffer_upload_f32: *const fn (SyclRuntimeHandle, [*]const f32, usize, *SyclBufferHandle) callconv(.c) c_int,
    anna_sycl_buffer_alloc_f32: *const fn (SyclRuntimeHandle, usize, *SyclBufferHandle) callconv(.c) c_int,
    anna_sycl_buffer_destroy: *const fn (SyclBufferHandle) callconv(.c) void,
    anna_sycl_buffer_read_f32: *const fn (SyclRuntimeHandle, SyclBufferHandle, [*]f32, usize) callconv(.c) c_int,
    anna_sycl_buffer_write_f32: *const fn (SyclRuntimeHandle, SyclBufferHandle, [*]const f32, usize) callconv(.c) c_int,
    anna_sycl_dense_run: *const fn (SyclRuntimeHandle, SyclDenseHandle, SyclBufferHandle, SyclBufferHandle) callconv(.c) c_int,
    anna_sycl_autoround_run: *const fn (SyclRuntimeHandle, SyclAutoRoundHandle, SyclBufferHandle, SyclBufferHandle) callconv(.c) c_int,
    anna_sycl_silu_mul_inplace: *const fn (SyclRuntimeHandle, SyclBufferHandle, SyclBufferHandle) callconv(.c) c_int,
    anna_sycl_qwen_rmsnorm_rope_inplace: *const fn (SyclRuntimeHandle, SyclBufferHandle, SyclBufferHandle, usize, usize, usize, f32, usize, f32) callconv(.c) c_int,
};

pub const F32Buffer = struct {
    runtime: *Runtime,
    handle: SyclBufferHandle,
    len: usize,
};

pub const DenseLinearWeights = struct {
    runtime: *Runtime,
    in_features: usize,
    out_features: usize,
    handle: SyclDenseHandle,
};

pub const AutoRoundLinearWeights = struct {
    runtime: *Runtime,
    in_features: usize,
    out_features: usize,
    group_size: usize,
    handle: SyclAutoRoundHandle,
};

pub const Runtime = struct {
    allocator: std.mem.Allocator,
    module: ?*anyopaque,
    api: Api,
    handle: SyclRuntimeHandle,
    platform_name: []u8,
    device_name: []u8,
    vendor_name: []u8,

    pub fn init(allocator: std.mem.Allocator) !Runtime {
        const module = loadSyclModule() orelse return error.SyclBackendUnavailable;
        errdefer _ = FreeLibrary(module);
        const api = try loadApi(module);

        var runtime_handle: SyclRuntimeHandle = null;
        try checkStatus(&api, api.anna_sycl_runtime_create(&runtime_handle));
        errdefer api.anna_sycl_runtime_destroy(runtime_handle);

        return .{
            .allocator = allocator,
            .module = module,
            .api = api,
            .handle = runtime_handle,
            .platform_name = try dupeCString(allocator, api.anna_sycl_runtime_platform_name(runtime_handle)),
            .device_name = try dupeCString(allocator, api.anna_sycl_runtime_device_name(runtime_handle)),
            .vendor_name = try dupeCString(allocator, api.anna_sycl_runtime_vendor_name(runtime_handle)),
        };
    }

    pub fn deinit(self: *Runtime) void {
        self.api.anna_sycl_runtime_destroy(self.handle);
        _ = FreeLibrary(self.module);
        self.allocator.free(self.platform_name);
        self.allocator.free(self.device_name);
        self.allocator.free(self.vendor_name);
        self.* = undefined;
    }

    pub fn createDenseWeights(self: *Runtime, weight: []const f32, bias: ?[]const f32, out_features: usize, in_features: usize) !DenseLinearWeights {
        var handle: SyclDenseHandle = null;
        const bias_ptr: ?[*]const f32 = if (bias) |values| values.ptr else null;
        const bias_len: usize = if (bias) |values| values.len else 0;
        try self.check(self.api.anna_sycl_dense_create(
            self.handle,
            weight.ptr,
            weight.len,
            bias_ptr,
            bias_len,
            out_features,
            in_features,
            &handle,
        ));
        return .{
            .runtime = self,
            .in_features = in_features,
            .out_features = out_features,
            .handle = handle,
        };
    }

    pub fn releaseDenseWeights(self: *Runtime, weights: *const DenseLinearWeights) void {
        _ = self;
        weights.runtime.api.anna_sycl_dense_destroy(weights.handle);
    }

    pub fn createAutoRoundWeights(
        self: *Runtime,
        qweight: []const i32,
        qzeros: []const i32,
        scales: []const f32,
        bias: ?[]const f32,
        out_features: usize,
        in_features: usize,
        group_size: usize,
    ) !AutoRoundLinearWeights {
        var handle: SyclAutoRoundHandle = null;
        const bias_ptr: ?[*]const f32 = if (bias) |values| values.ptr else null;
        const bias_len: usize = if (bias) |values| values.len else 0;
        try self.check(self.api.anna_sycl_autoround_create(
            self.handle,
            qweight.ptr,
            qweight.len,
            qzeros.ptr,
            qzeros.len,
            scales.ptr,
            scales.len,
            bias_ptr,
            bias_len,
            out_features,
            in_features,
            group_size,
            &handle,
        ));
        return .{
            .runtime = self,
            .in_features = in_features,
            .out_features = out_features,
            .group_size = group_size,
            .handle = handle,
        };
    }

    pub fn releaseAutoRoundWeights(self: *Runtime, weights: *const AutoRoundLinearWeights) void {
        _ = self;
        weights.runtime.api.anna_sycl_autoround_destroy(weights.handle);
    }

    pub fn uploadF32(self: *Runtime, values: []const f32) !F32Buffer {
        var handle: SyclBufferHandle = null;
        try self.check(self.api.anna_sycl_buffer_upload_f32(self.handle, values.ptr, values.len, &handle));
        return .{ .runtime = self, .handle = handle, .len = values.len };
    }

    pub fn allocF32(self: *Runtime, len: usize) !F32Buffer {
        var handle: SyclBufferHandle = null;
        try self.check(self.api.anna_sycl_buffer_alloc_f32(self.handle, len, &handle));
        return .{ .runtime = self, .handle = handle, .len = len };
    }

    pub fn releaseF32(self: *Runtime, buffer: *F32Buffer) void {
        _ = self;
        buffer.runtime.api.anna_sycl_buffer_destroy(buffer.handle);
        buffer.* = undefined;
    }

    pub fn readF32(self: *Runtime, buffer: *const F32Buffer, out: []f32) !void {
        std.debug.assert(buffer.len == out.len);
        try self.check(self.api.anna_sycl_buffer_read_f32(self.handle, buffer.handle, out.ptr, out.len));
    }

    pub fn writeF32(self: *Runtime, buffer: *const F32Buffer, values: []const f32) !void {
        std.debug.assert(buffer.len == values.len);
        try self.check(self.api.anna_sycl_buffer_write_f32(self.handle, buffer.handle, values.ptr, values.len));
    }

    pub fn runDense(self: *Runtime, weights: *const DenseLinearWeights, out: []f32, input: []const f32) !void {
        std.debug.assert(weights.in_features == input.len);
        std.debug.assert(weights.out_features == out.len);

        var input_buffer = try self.uploadF32(input);
        defer self.releaseF32(&input_buffer);
        var output_buffer = try self.allocF32(out.len);
        defer self.releaseF32(&output_buffer);
        try self.runDenseToBuffer(weights, &output_buffer, &input_buffer);
        try self.readF32(&output_buffer, out);
    }

    pub fn runAutoRound(self: *Runtime, weights: *const AutoRoundLinearWeights, out: []f32, input: []const f32) !void {
        std.debug.assert(weights.in_features == input.len);
        std.debug.assert(weights.out_features == out.len);

        var input_buffer = try self.uploadF32(input);
        defer self.releaseF32(&input_buffer);
        var output_buffer = try self.allocF32(out.len);
        defer self.releaseF32(&output_buffer);
        try self.runAutoRoundToBuffer(weights, &output_buffer, &input_buffer);
        try self.readF32(&output_buffer, out);
    }

    pub fn runDenseToBuffer(self: *Runtime, weights: *const DenseLinearWeights, out: *const F32Buffer, input: *const F32Buffer) !void {
        std.debug.assert(weights.in_features == input.len);
        std.debug.assert(weights.out_features == out.len);
        try self.check(self.api.anna_sycl_dense_run(self.handle, weights.handle, input.handle, out.handle));
    }

    pub fn runAutoRoundToBuffer(self: *Runtime, weights: *const AutoRoundLinearWeights, out: *const F32Buffer, input: *const F32Buffer) !void {
        std.debug.assert(weights.in_features == input.len);
        std.debug.assert(weights.out_features == out.len);
        try self.check(self.api.anna_sycl_autoround_run(self.handle, weights.handle, input.handle, out.handle));
    }

    pub fn siluMulInPlace(self: *Runtime, gate: *const F32Buffer, up: *const F32Buffer) !void {
        std.debug.assert(gate.len == up.len);
        try self.check(self.api.anna_sycl_silu_mul_inplace(self.handle, gate.handle, up.handle));
    }

    pub fn qwenRmsNormRopeInPlace(
        self: *Runtime,
        values: *const F32Buffer,
        weight: *const F32Buffer,
        head_count: usize,
        head_dim: usize,
        position: usize,
        theta: f32,
        rotary_dim: usize,
        eps: f32,
    ) !void {
        std.debug.assert(head_count * head_dim <= values.len);
        std.debug.assert(weight.len == head_dim);
        std.debug.assert(rotary_dim <= head_dim);
        std.debug.assert(rotary_dim % 2 == 0);
        try self.check(self.api.anna_sycl_qwen_rmsnorm_rope_inplace(
            self.handle,
            values.handle,
            weight.handle,
            head_count,
            head_dim,
            position,
            theta,
            rotary_dim,
            eps,
        ));
    }

    fn check(self: *Runtime, status: c_int) !void {
        return checkStatus(&self.api, status);
    }
};

fn loadSyclModule() ?*anyopaque {
    return LoadLibraryW(std.unicode.utf8ToUtf16LeStringLiteral("anna-xpu-backend.dll")) orelse
        LoadLibraryW(std.unicode.utf8ToUtf16LeStringLiteral("zig-out\\bin\\anna-xpu-backend.dll"));
}

fn loadApi(module: ?*anyopaque) !Api {
    return .{
        .anna_sycl_last_error_message = loadSymbol(@FieldType(Api, "anna_sycl_last_error_message"), module, "anna_sycl_last_error_message") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_runtime_create = loadSymbol(@FieldType(Api, "anna_sycl_runtime_create"), module, "anna_sycl_runtime_create") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_runtime_destroy = loadSymbol(@FieldType(Api, "anna_sycl_runtime_destroy"), module, "anna_sycl_runtime_destroy") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_runtime_platform_name = loadSymbol(@FieldType(Api, "anna_sycl_runtime_platform_name"), module, "anna_sycl_runtime_platform_name") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_runtime_device_name = loadSymbol(@FieldType(Api, "anna_sycl_runtime_device_name"), module, "anna_sycl_runtime_device_name") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_runtime_vendor_name = loadSymbol(@FieldType(Api, "anna_sycl_runtime_vendor_name"), module, "anna_sycl_runtime_vendor_name") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_dense_create = loadSymbol(@FieldType(Api, "anna_sycl_dense_create"), module, "anna_sycl_dense_create") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_dense_destroy = loadSymbol(@FieldType(Api, "anna_sycl_dense_destroy"), module, "anna_sycl_dense_destroy") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_autoround_create = loadSymbol(@FieldType(Api, "anna_sycl_autoround_create"), module, "anna_sycl_autoround_create") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_autoround_destroy = loadSymbol(@FieldType(Api, "anna_sycl_autoround_destroy"), module, "anna_sycl_autoround_destroy") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_buffer_upload_f32 = loadSymbol(@FieldType(Api, "anna_sycl_buffer_upload_f32"), module, "anna_sycl_buffer_upload_f32") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_buffer_alloc_f32 = loadSymbol(@FieldType(Api, "anna_sycl_buffer_alloc_f32"), module, "anna_sycl_buffer_alloc_f32") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_buffer_destroy = loadSymbol(@FieldType(Api, "anna_sycl_buffer_destroy"), module, "anna_sycl_buffer_destroy") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_buffer_read_f32 = loadSymbol(@FieldType(Api, "anna_sycl_buffer_read_f32"), module, "anna_sycl_buffer_read_f32") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_buffer_write_f32 = loadSymbol(@FieldType(Api, "anna_sycl_buffer_write_f32"), module, "anna_sycl_buffer_write_f32") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_dense_run = loadSymbol(@FieldType(Api, "anna_sycl_dense_run"), module, "anna_sycl_dense_run") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_autoround_run = loadSymbol(@FieldType(Api, "anna_sycl_autoround_run"), module, "anna_sycl_autoround_run") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_silu_mul_inplace = loadSymbol(@FieldType(Api, "anna_sycl_silu_mul_inplace"), module, "anna_sycl_silu_mul_inplace") orelse return error.MissingSyclBackendSymbol,
        .anna_sycl_qwen_rmsnorm_rope_inplace = loadSymbol(@FieldType(Api, "anna_sycl_qwen_rmsnorm_rope_inplace"), module, "anna_sycl_qwen_rmsnorm_rope_inplace") orelse return error.MissingSyclBackendSymbol,
    };
}

fn loadSymbol(comptime T: type, module: ?*anyopaque, name: [*:0]const u8) ?T {
    const ptr = GetProcAddress(module, name) orelse return null;
    return @ptrCast(ptr);
}

fn checkStatus(api: *const Api, status: c_int) !void {
    return switch (status) {
        ANNA_SYCL_SUCCESS => {},
        ANNA_SYCL_BACKEND_UNAVAILABLE => error.SyclBackendUnavailable,
        ANNA_SYCL_DEVICE_NOT_FOUND => error.XpuDeviceNotFound,
        ANNA_SYCL_INVALID_ARGUMENT => error.InvalidArgument,
        ANNA_SYCL_ALLOCATION_FAILED => error.OutOfMemory,
        else => {
            if (api.anna_sycl_last_error_message()) |message| {
                std.log.err("SYCL backend call failed: {s}", .{std.mem.sliceTo(message, 0)});
            }
            return error.SyclRuntimeError;
        },
    };
}

fn dupeCString(allocator: std.mem.Allocator, value: ?[*:0]const u8) ![]u8 {
    const resolved = value orelse return try allocator.dupe(u8, "");
    return try allocator.dupe(u8, std.mem.sliceTo(resolved, 0));
}

test "sycl dense linear matches cpu reference" {
    var runtime = Runtime.init(std.testing.allocator) catch |err| switch (err) {
        error.SyclBackendUnavailable, error.MissingSyclBackendSymbol, error.XpuDeviceNotFound, error.SyclRuntimeError => return error.SkipZigTest,
        else => return err,
    };
    defer runtime.deinit();

    const weight = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
    };
    const bias = [_]f32{ 0.5, -1.0 };
    var handle = try runtime.createDenseWeights(&weight, &bias, 2, 2);
    defer runtime.releaseDenseWeights(&handle);

    const input = [_]f32{ 5.0, 7.0 };
    var out: [2]f32 = undefined;
    try runtime.runDense(&handle, &out, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 19.5), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), out[1], 1e-4);
}

test "sycl autoround linear matches cpu reference" {
    var runtime = Runtime.init(std.testing.allocator) catch |err| switch (err) {
        error.SyclBackendUnavailable, error.MissingSyclBackendSymbol, error.XpuDeviceNotFound, error.SyclRuntimeError => return error.SkipZigTest,
        else => return err,
    };
    defer runtime.deinit();

    const qweight = [_]i32{0x00000021};
    const qzeros = [_]i32{0x00000000};
    const scales = [_]f32{0.5};
    var handle = try runtime.createAutoRoundWeights(&qweight, &qzeros, &scales, null, 1, 2, 128);
    defer runtime.releaseAutoRoundWeights(&handle);

    const input = [_]f32{ 2.0, 4.0 };
    var out: [1]f32 = undefined;
    try runtime.runAutoRound(&handle, &out, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[0], 1e-4);
}

test "sycl silu mul in-place matches cpu reference" {
    var runtime = Runtime.init(std.testing.allocator) catch |err| switch (err) {
        error.SyclBackendUnavailable, error.MissingSyclBackendSymbol, error.XpuDeviceNotFound, error.SyclRuntimeError => return error.SkipZigTest,
        else => return err,
    };
    defer runtime.deinit();

    const gate = [_]f32{ 1.0, 2.0 };
    const up = [_]f32{ 3.0, 4.0 };
    var gate_buffer = try runtime.uploadF32(&gate);
    defer runtime.releaseF32(&gate_buffer);
    var up_buffer = try runtime.uploadF32(&up);
    defer runtime.releaseF32(&up_buffer);
    try runtime.siluMulInPlace(&gate_buffer, &up_buffer);

    var out: [2]f32 = undefined;
    try runtime.readF32(&gate_buffer, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 2.1931758), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0463767), out[1], 1e-4);
}

test "sycl qwen rmsnorm rope in-place matches cpu reference" {
    var runtime = Runtime.init(std.testing.allocator) catch |err| switch (err) {
        error.SyclBackendUnavailable, error.MissingSyclBackendSymbol, error.XpuDeviceNotFound, error.SyclRuntimeError => return error.SkipZigTest,
        else => return err,
    };
    defer runtime.deinit();

    const weight = [_]f32{ 0.1, -0.2, 0.0, 0.3 };
    const input = [_]f32{
        1.0, -2.0, 0.5,  3.0,
        4.0, -1.5, 2.0,  -0.25,
        8.0, 9.0,  10.0, 11.0,
    };
    var expected = input;
    applyQwenRmsNormRopeCpu(expected[0..8], &weight, 2, 4, 3.0, 10_000.0, 4, 1e-6);

    var value_buffer = try runtime.uploadF32(&input);
    defer runtime.releaseF32(&value_buffer);
    var weight_buffer = try runtime.uploadF32(&weight);
    defer runtime.releaseF32(&weight_buffer);
    try runtime.qwenRmsNormRopeInPlace(&value_buffer, &weight_buffer, 2, 4, 3, 10_000.0, 4, 1e-6);

    var out: [12]f32 = undefined;
    try runtime.readF32(&value_buffer, &out);
    for (expected, out) |exp, actual| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-4);
    }
}

fn applyQwenRmsNormRopeCpu(values: []f32, weight: []const f32, head_count: usize, head_dim: usize, position: f32, theta: f32, rotary_dim: usize, eps: f32) void {
    std.debug.assert(values.len == head_count * head_dim);
    std.debug.assert(weight.len == head_dim);
    for (0..head_count) |head| {
        const head_values = values[head * head_dim ..][0..head_dim];
        var mean_square: f32 = 0.0;
        for (head_values) |value| mean_square += value * value;
        mean_square /= @floatFromInt(head_dim);
        const inv = 1.0 / @sqrt(mean_square + eps);
        for (head_values, weight) |*dst, w| dst.* *= inv * (1.0 + w);
        const half = rotary_dim / 2;
        for (0..half) |idx| {
            const inv_freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(idx * 2)) / @as(f32, @floatFromInt(rotary_dim)));
            const angle = position * inv_freq;
            const c = @cos(angle);
            const s = @sin(angle);
            const x1 = head_values[idx];
            const x2 = head_values[idx + half];
            head_values[idx] = x1 * c - x2 * s;
            head_values[idx + half] = x2 * c + x1 * s;
        }
    }
}
