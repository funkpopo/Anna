const builtin = @import("builtin");
const std = @import("std");

comptime {
    if (builtin.os.tag != .windows) @compileError("opencl_backend currently supports Windows only");
}

const ClPlatform = opaque {};
const ClDevice = opaque {};
const ClContext = opaque {};
const ClCommandQueue = opaque {};
const ClProgram = opaque {};
const ClKernel = opaque {};
const ClMem = opaque {};

const cl_platform_id = ?*ClPlatform;
const cl_device_id = ?*ClDevice;
const cl_context = ?*ClContext;
const cl_command_queue = ?*ClCommandQueue;
const cl_program = ?*ClProgram;
const cl_kernel = ?*ClKernel;
const cl_mem = ?*ClMem;
const cl_int = i32;
const cl_uint = u32;
const cl_ulong = u64;
const cl_bool = cl_uint;
const cl_device_type = cl_ulong;
const cl_mem_flags = cl_ulong;
const cl_platform_info = cl_uint;
const cl_device_info = cl_uint;
const cl_program_build_info = cl_uint;
const cl_context_properties = isize;

extern "kernel32" fn LoadLibraryW(lpLibFileName: [*:0]const u16) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetProcAddress(hModule: ?*anyopaque, lpProcName: [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn FreeLibrary(hLibModule: ?*anyopaque) callconv(.winapi) i32;

const CL_SUCCESS: cl_int = 0;
const CL_DEVICE_NOT_FOUND: cl_int = -1;
const CL_BUILD_PROGRAM_FAILURE: cl_int = -11;

const CL_FALSE: cl_bool = 0;
const CL_TRUE: cl_bool = 1;

const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;

const CL_PLATFORM_NAME: cl_platform_info = 0x0902;
const CL_DEVICE_NAME: cl_device_info = 0x102B;
const CL_DEVICE_VENDOR: cl_device_info = 0x102C;
const CL_PROGRAM_BUILD_LOG: cl_program_build_info = 0x1183;

const CL_MEM_READ_ONLY: cl_mem_flags = 1 << 2;
const CL_MEM_WRITE_ONLY: cl_mem_flags = 1 << 1;
const CL_MEM_READ_WRITE: cl_mem_flags = 1 << 0;
const CL_MEM_COPY_HOST_PTR: cl_mem_flags = 1 << 5;

const Api = struct {
    clGetPlatformIDs: *const fn (cl_uint, ?[*]cl_platform_id, ?*cl_uint) callconv(.c) cl_int,
    clGetPlatformInfo: *const fn (cl_platform_id, cl_platform_info, usize, ?*anyopaque, ?*usize) callconv(.c) cl_int,
    clGetDeviceIDs: *const fn (cl_platform_id, cl_device_type, cl_uint, ?[*]cl_device_id, ?*cl_uint) callconv(.c) cl_int,
    clGetDeviceInfo: *const fn (cl_device_id, cl_device_info, usize, ?*anyopaque, ?*usize) callconv(.c) cl_int,
    clCreateContext: *const fn (?*const cl_context_properties, cl_uint, [*]const cl_device_id, ?*const anyopaque, ?*anyopaque, ?*cl_int) callconv(.c) cl_context,
    clReleaseContext: *const fn (cl_context) callconv(.c) cl_int,
    clCreateCommandQueue: *const fn (cl_context, cl_device_id, cl_ulong, ?*cl_int) callconv(.c) cl_command_queue,
    clReleaseCommandQueue: *const fn (cl_command_queue) callconv(.c) cl_int,
    clCreateProgramWithSource: *const fn (cl_context, cl_uint, [*]const [*]const u8, ?[*]const usize, ?*cl_int) callconv(.c) cl_program,
    clBuildProgram: *const fn (cl_program, cl_uint, ?[*]const cl_device_id, ?[*:0]const u8, ?*const anyopaque, ?*anyopaque) callconv(.c) cl_int,
    clReleaseProgram: *const fn (cl_program) callconv(.c) cl_int,
    clGetProgramBuildInfo: *const fn (cl_program, cl_device_id, cl_program_build_info, usize, ?*anyopaque, ?*usize) callconv(.c) cl_int,
    clCreateKernel: *const fn (cl_program, [*:0]const u8, ?*cl_int) callconv(.c) cl_kernel,
    clReleaseKernel: *const fn (cl_kernel) callconv(.c) cl_int,
    clCreateBuffer: *const fn (cl_context, cl_mem_flags, usize, ?*anyopaque, ?*cl_int) callconv(.c) cl_mem,
    clReleaseMemObject: *const fn (cl_mem) callconv(.c) cl_int,
    clSetKernelArg: *const fn (cl_kernel, cl_uint, usize, ?*const anyopaque) callconv(.c) cl_int,
    clEnqueueNDRangeKernel: *const fn (cl_command_queue, cl_kernel, cl_uint, ?[*]const usize, [*]const usize, ?[*]const usize, cl_uint, ?*const anyopaque, ?*anyopaque) callconv(.c) cl_int,
    clEnqueueReadBuffer: *const fn (cl_command_queue, cl_mem, cl_bool, usize, usize, ?*anyopaque, cl_uint, ?*const anyopaque, ?*anyopaque) callconv(.c) cl_int,
};

pub const DenseLinearWeights = struct {
    runtime: *Runtime,
    in_features: usize,
    out_features: usize,
    weight: cl_mem,
    bias: ?cl_mem = null,
};

pub const AutoRoundLinearWeights = struct {
    runtime: *Runtime,
    in_features: usize,
    out_features: usize,
    group_size: usize,
    qweight: cl_mem,
    qzeros: cl_mem,
    scales: cl_mem,
    bias: ?cl_mem = null,
};

pub const Runtime = struct {
    allocator: std.mem.Allocator,
    module: ?*anyopaque,
    api: Api,
    platform: cl_platform_id,
    device: cl_device_id,
    context: cl_context,
    queue: cl_command_queue,
    program: cl_program,
    dense_kernel: cl_kernel,
    autoround_kernel: cl_kernel,
    zero_bias_buffer: cl_mem,
    platform_name: []u8,
    device_name: []u8,
    vendor_name: []u8,

    pub fn init(allocator: std.mem.Allocator) !Runtime {
        const module = LoadLibraryW(std.unicode.utf8ToUtf16LeStringLiteral("OpenCL.dll")) orelse return error.OpenClUnavailable;
        errdefer _ = FreeLibrary(module);
        const api = try loadApi(module);

        const selected = try pickIntelGpu(allocator, &api);
        errdefer {
            allocator.free(selected.platform_name);
            allocator.free(selected.device_name);
            allocator.free(selected.vendor_name);
        }

        var status: cl_int = 0;
        const context = api.clCreateContext(
            null,
            1,
            @ptrCast(&selected.device),
            null,
            null,
            &status,
        );
        try check(status);
        errdefer _ = api.clReleaseContext(context);

        const queue = api.clCreateCommandQueue(context, selected.device, 0, &status);
        try check(status);
        errdefer _ = api.clReleaseCommandQueue(queue);

        const source = kernel_source;
        const sources = [_][*]const u8{source.ptr};
        const lengths = [_]usize{source.len};
        const program = api.clCreateProgramWithSource(context, 1, &sources, &lengths, &status);
        try check(status);
        errdefer _ = api.clReleaseProgram(program);

        status = api.clBuildProgram(program, 1, @ptrCast(&selected.device), null, null, null);
        if (status != CL_SUCCESS) {
            const log = buildLog(allocator, &api, program, selected.device) catch null;
            defer if (log) |value| allocator.free(value);
            if (log) |value| {
                std.log.err("OpenCL program build failed: {s}", .{value});
            } else {
                std.log.err("OpenCL program build failed with status {d}", .{status});
            }
            return error.OpenClBuildFailed;
        }

        const dense_kernel = api.clCreateKernel(program, "dense_linear", &status);
        try check(status);
        errdefer _ = api.clReleaseKernel(dense_kernel);

        const autoround_kernel = api.clCreateKernel(program, "autoround_linear", &status);
        try check(status);
        errdefer _ = api.clReleaseKernel(autoround_kernel);

        const zero_bias: [1]f32 = .{0.0};
        const zero_bias_buffer = try createBuffer(&api, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std.mem.sliceAsBytes(&zero_bias), zero_bias[0..].ptr);
        errdefer _ = api.clReleaseMemObject(zero_bias_buffer);

        return .{
            .allocator = allocator,
            .module = module,
            .api = api,
            .platform = selected.platform,
            .device = selected.device,
            .context = context,
            .queue = queue,
            .program = program,
            .dense_kernel = dense_kernel,
            .autoround_kernel = autoround_kernel,
            .zero_bias_buffer = zero_bias_buffer,
            .platform_name = selected.platform_name,
            .device_name = selected.device_name,
            .vendor_name = selected.vendor_name,
        };
    }

    pub fn deinit(self: *Runtime) void {
        _ = self.api.clReleaseMemObject(self.zero_bias_buffer);
        _ = self.api.clReleaseKernel(self.autoround_kernel);
        _ = self.api.clReleaseKernel(self.dense_kernel);
        _ = self.api.clReleaseProgram(self.program);
        _ = self.api.clReleaseCommandQueue(self.queue);
        _ = self.api.clReleaseContext(self.context);
        _ = FreeLibrary(self.module);
        self.allocator.free(self.platform_name);
        self.allocator.free(self.device_name);
        self.allocator.free(self.vendor_name);
        self.* = undefined;
    }

    pub fn createDenseWeights(self: *Runtime, weight: []const f32, bias: ?[]const f32, out_features: usize, in_features: usize) !DenseLinearWeights {
        return .{
            .runtime = self,
            .in_features = in_features,
            .out_features = out_features,
            .weight = try createBuffer(&self.api, self.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std.mem.sliceAsBytes(weight), weight.ptr),
            .bias = if (bias) |values|
                try createBuffer(&self.api, self.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std.mem.sliceAsBytes(values), values.ptr)
            else
                null,
        };
    }

    pub fn releaseDenseWeights(self: *Runtime, weights: *const DenseLinearWeights) void {
        _ = self.api.clReleaseMemObject(weights.weight);
        if (weights.bias) |buffer| _ = self.api.clReleaseMemObject(buffer);
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
        return .{
            .runtime = self,
            .in_features = in_features,
            .out_features = out_features,
            .group_size = group_size,
            .qweight = try createBuffer(&self.api, self.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std.mem.sliceAsBytes(qweight), qweight.ptr),
            .qzeros = try createBuffer(&self.api, self.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std.mem.sliceAsBytes(qzeros), qzeros.ptr),
            .scales = try createBuffer(&self.api, self.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std.mem.sliceAsBytes(scales), scales.ptr),
            .bias = if (bias) |values|
                try createBuffer(&self.api, self.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std.mem.sliceAsBytes(values), values.ptr)
            else
                null,
        };
    }

    pub fn releaseAutoRoundWeights(self: *Runtime, weights: *const AutoRoundLinearWeights) void {
        _ = self.api.clReleaseMemObject(weights.qweight);
        _ = self.api.clReleaseMemObject(weights.qzeros);
        _ = self.api.clReleaseMemObject(weights.scales);
        if (weights.bias) |buffer| _ = self.api.clReleaseMemObject(buffer);
    }

    pub fn runDense(self: *Runtime, weights: *const DenseLinearWeights, out: []f32, input: []const f32) !void {
        std.debug.assert(weights.in_features == input.len);
        std.debug.assert(weights.out_features == out.len);

        const input_buffer = try createBuffer(&self.api, self.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std.mem.sliceAsBytes(input), input.ptr);
        defer _ = self.api.clReleaseMemObject(input_buffer);
        const output_buffer = try createEmptyBuffer(&self.api, self.context, CL_MEM_WRITE_ONLY, out.len * @sizeOf(f32));
        defer _ = self.api.clReleaseMemObject(output_buffer);

        const bias_buffer = weights.bias orelse self.zero_bias_buffer;
        var has_bias: cl_int = if (weights.bias != null) 1 else 0;
        var in_features: cl_int = @intCast(weights.in_features);
        try setKernelArg(&self.api, self.dense_kernel, 0, cl_mem, &weights.weight);
        try setKernelArg(&self.api, self.dense_kernel, 1, cl_mem, &bias_buffer);
        try setKernelArg(&self.api, self.dense_kernel, 2, cl_int, &has_bias);
        try setKernelArg(&self.api, self.dense_kernel, 3, cl_int, &in_features);
        try setKernelArg(&self.api, self.dense_kernel, 4, cl_mem, &input_buffer);
        try setKernelArg(&self.api, self.dense_kernel, 5, cl_mem, &output_buffer);

        const global = [_]usize{weights.out_features};
        try check(self.api.clEnqueueNDRangeKernel(self.queue, self.dense_kernel, 1, null, &global, null, 0, null, null));
        try check(self.api.clEnqueueReadBuffer(self.queue, output_buffer, CL_TRUE, 0, out.len * @sizeOf(f32), out.ptr, 0, null, null));
    }

    pub fn runAutoRound(self: *Runtime, weights: *const AutoRoundLinearWeights, out: []f32, input: []const f32) !void {
        std.debug.assert(weights.in_features == input.len);
        std.debug.assert(weights.out_features == out.len);

        const input_buffer = try createBuffer(&self.api, self.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std.mem.sliceAsBytes(input), input.ptr);
        defer _ = self.api.clReleaseMemObject(input_buffer);
        const output_buffer = try createEmptyBuffer(&self.api, self.context, CL_MEM_WRITE_ONLY, out.len * @sizeOf(f32));
        defer _ = self.api.clReleaseMemObject(output_buffer);

        const bias_buffer = weights.bias orelse self.zero_bias_buffer;
        const packed_in: cl_int = @intCast((weights.in_features + 7) / 8);
        const packed_out: cl_int = @intCast((weights.out_features + 7) / 8);
        const group_count: cl_int = @intCast((weights.in_features + weights.group_size - 1) / weights.group_size);
        var has_bias: cl_int = if (weights.bias != null) 1 else 0;
        var in_features: cl_int = @intCast(weights.in_features);
        var out_features: cl_int = @intCast(weights.out_features);
        var group_size: cl_int = @intCast(weights.group_size);
        try setKernelArg(&self.api, self.autoround_kernel, 0, cl_mem, &weights.qweight);
        try setKernelArg(&self.api, self.autoround_kernel, 1, cl_mem, &weights.qzeros);
        try setKernelArg(&self.api, self.autoround_kernel, 2, cl_mem, &weights.scales);
        try setKernelArg(&self.api, self.autoround_kernel, 3, cl_mem, &bias_buffer);
        try setKernelArg(&self.api, self.autoround_kernel, 4, cl_int, &has_bias);
        try setKernelArg(&self.api, self.autoround_kernel, 5, cl_int, &in_features);
        try setKernelArg(&self.api, self.autoround_kernel, 6, cl_int, &out_features);
        try setKernelArg(&self.api, self.autoround_kernel, 7, cl_int, &group_size);
        try setKernelArg(&self.api, self.autoround_kernel, 8, cl_int, &packed_in);
        try setKernelArg(&self.api, self.autoround_kernel, 9, cl_int, &packed_out);
        try setKernelArg(&self.api, self.autoround_kernel, 10, cl_int, &group_count);
        try setKernelArg(&self.api, self.autoround_kernel, 11, cl_mem, &input_buffer);
        try setKernelArg(&self.api, self.autoround_kernel, 12, cl_mem, &output_buffer);

        const global = [_]usize{weights.out_features};
        try check(self.api.clEnqueueNDRangeKernel(self.queue, self.autoround_kernel, 1, null, &global, null, 0, null, null));
        try check(self.api.clEnqueueReadBuffer(self.queue, output_buffer, CL_TRUE, 0, out.len * @sizeOf(f32), out.ptr, 0, null, null));
    }
};

const SelectedDevice = struct {
    platform: cl_platform_id,
    device: cl_device_id,
    platform_name: []u8,
    device_name: []u8,
    vendor_name: []u8,
};

fn loadApi(module: ?*anyopaque) !Api {
    return .{
        .clGetPlatformIDs = loadSymbol(@FieldType(Api, "clGetPlatformIDs"), module, "clGetPlatformIDs") orelse return error.MissingOpenClSymbol,
        .clGetPlatformInfo = loadSymbol(@FieldType(Api, "clGetPlatformInfo"), module, "clGetPlatformInfo") orelse return error.MissingOpenClSymbol,
        .clGetDeviceIDs = loadSymbol(@FieldType(Api, "clGetDeviceIDs"), module, "clGetDeviceIDs") orelse return error.MissingOpenClSymbol,
        .clGetDeviceInfo = loadSymbol(@FieldType(Api, "clGetDeviceInfo"), module, "clGetDeviceInfo") orelse return error.MissingOpenClSymbol,
        .clCreateContext = loadSymbol(@FieldType(Api, "clCreateContext"), module, "clCreateContext") orelse return error.MissingOpenClSymbol,
        .clReleaseContext = loadSymbol(@FieldType(Api, "clReleaseContext"), module, "clReleaseContext") orelse return error.MissingOpenClSymbol,
        .clCreateCommandQueue = loadSymbol(@FieldType(Api, "clCreateCommandQueue"), module, "clCreateCommandQueue") orelse return error.MissingOpenClSymbol,
        .clReleaseCommandQueue = loadSymbol(@FieldType(Api, "clReleaseCommandQueue"), module, "clReleaseCommandQueue") orelse return error.MissingOpenClSymbol,
        .clCreateProgramWithSource = loadSymbol(@FieldType(Api, "clCreateProgramWithSource"), module, "clCreateProgramWithSource") orelse return error.MissingOpenClSymbol,
        .clBuildProgram = loadSymbol(@FieldType(Api, "clBuildProgram"), module, "clBuildProgram") orelse return error.MissingOpenClSymbol,
        .clReleaseProgram = loadSymbol(@FieldType(Api, "clReleaseProgram"), module, "clReleaseProgram") orelse return error.MissingOpenClSymbol,
        .clGetProgramBuildInfo = loadSymbol(@FieldType(Api, "clGetProgramBuildInfo"), module, "clGetProgramBuildInfo") orelse return error.MissingOpenClSymbol,
        .clCreateKernel = loadSymbol(@FieldType(Api, "clCreateKernel"), module, "clCreateKernel") orelse return error.MissingOpenClSymbol,
        .clReleaseKernel = loadSymbol(@FieldType(Api, "clReleaseKernel"), module, "clReleaseKernel") orelse return error.MissingOpenClSymbol,
        .clCreateBuffer = loadSymbol(@FieldType(Api, "clCreateBuffer"), module, "clCreateBuffer") orelse return error.MissingOpenClSymbol,
        .clReleaseMemObject = loadSymbol(@FieldType(Api, "clReleaseMemObject"), module, "clReleaseMemObject") orelse return error.MissingOpenClSymbol,
        .clSetKernelArg = loadSymbol(@FieldType(Api, "clSetKernelArg"), module, "clSetKernelArg") orelse return error.MissingOpenClSymbol,
        .clEnqueueNDRangeKernel = loadSymbol(@FieldType(Api, "clEnqueueNDRangeKernel"), module, "clEnqueueNDRangeKernel") orelse return error.MissingOpenClSymbol,
        .clEnqueueReadBuffer = loadSymbol(@FieldType(Api, "clEnqueueReadBuffer"), module, "clEnqueueReadBuffer") orelse return error.MissingOpenClSymbol,
    };
}

fn loadSymbol(comptime T: type, module: ?*anyopaque, name: [*:0]const u8) ?T {
    const ptr = GetProcAddress(module, name) orelse return null;
    return @ptrCast(ptr);
}

fn pickIntelGpu(allocator: std.mem.Allocator, api: *const Api) !SelectedDevice {
    var platform_count: cl_uint = 0;
    try check(api.clGetPlatformIDs(0, null, &platform_count));
    if (platform_count == 0) return error.OpenClUnavailable;

    const platforms = try allocator.alloc(cl_platform_id, platform_count);
    defer allocator.free(platforms);
    try check(api.clGetPlatformIDs(platform_count, platforms.ptr, null));

    for (platforms) |platform| {
        var device_count: cl_uint = 0;
        const count_status = api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, null, &device_count);
        if (count_status == CL_DEVICE_NOT_FOUND or device_count == 0) continue;
        try check(count_status);

        const devices = try allocator.alloc(cl_device_id, device_count);
        defer allocator.free(devices);
        try check(api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.ptr, null));

        const platform_name = try queryPlatformString(allocator, api, platform, CL_PLATFORM_NAME);
        defer allocator.free(platform_name);

        for (devices) |device| {
            const vendor_name = try queryDeviceString(allocator, api, device, CL_DEVICE_VENDOR);
            defer allocator.free(vendor_name);
            const device_name = try queryDeviceString(allocator, api, device, CL_DEVICE_NAME);
            defer allocator.free(device_name);
            if (!isIntelDevice(vendor_name, device_name)) continue;
            return .{
                .platform = platform,
                .device = device,
                .platform_name = try allocator.dupe(u8, platform_name),
                .device_name = try allocator.dupe(u8, device_name),
                .vendor_name = try allocator.dupe(u8, vendor_name),
            };
        }
    }

    return error.XpuDeviceNotFound;
}

fn isIntelDevice(vendor_name: []const u8, device_name: []const u8) bool {
    return std.ascii.indexOfIgnoreCase(vendor_name, "intel") != null or
        std.ascii.indexOfIgnoreCase(device_name, "arc") != null;
}

fn queryPlatformString(allocator: std.mem.Allocator, api: *const Api, platform: cl_platform_id, info: cl_platform_info) ![]u8 {
    var size: usize = 0;
    try check(api.clGetPlatformInfo(platform, info, 0, null, &size));
    const buffer = try allocator.alloc(u8, size);
    errdefer allocator.free(buffer);
    try check(api.clGetPlatformInfo(platform, info, size, buffer.ptr, null));
    return trimCBuffer(allocator, buffer);
}

fn queryDeviceString(allocator: std.mem.Allocator, api: *const Api, device: cl_device_id, info: cl_device_info) ![]u8 {
    var size: usize = 0;
    try check(api.clGetDeviceInfo(device, info, 0, null, &size));
    const buffer = try allocator.alloc(u8, size);
    errdefer allocator.free(buffer);
    try check(api.clGetDeviceInfo(device, info, size, buffer.ptr, null));
    return trimCBuffer(allocator, buffer);
}

fn trimCBuffer(allocator: std.mem.Allocator, buffer: []u8) ![]u8 {
    const trimmed = std.mem.sliceTo(buffer, 0);
    defer allocator.free(buffer);
    return try allocator.dupe(u8, trimmed);
}

fn buildLog(allocator: std.mem.Allocator, api: *const Api, program: cl_program, device: cl_device_id) ![]u8 {
    var size: usize = 0;
    try check(api.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, null, &size));
    const buffer = try allocator.alloc(u8, size);
    errdefer allocator.free(buffer);
    try check(api.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, buffer.ptr, null));
    defer allocator.free(buffer);
    return try allocator.dupe(u8, std.mem.sliceTo(buffer, 0));
}

fn createBuffer(api: *const Api, context: cl_context, flags: cl_mem_flags, bytes: []const u8, host_ptr: ?*const anyopaque) !cl_mem {
    var status: cl_int = 0;
    const buffer = api.clCreateBuffer(context, flags, bytes.len, @constCast(host_ptr), &status);
    try check(status);
    return buffer;
}

fn createEmptyBuffer(api: *const Api, context: cl_context, flags: cl_mem_flags, byte_len: usize) !cl_mem {
    var status: cl_int = 0;
    const buffer = api.clCreateBuffer(context, flags, byte_len, null, &status);
    try check(status);
    return buffer;
}

fn setKernelArg(api: *const Api, kernel: cl_kernel, index: cl_uint, comptime T: type, value: *const T) !void {
    try check(api.clSetKernelArg(kernel, index, @sizeOf(T), @ptrCast(value)));
}

fn check(status: cl_int) !void {
    if (status == CL_SUCCESS) return;
    std.log.err("OpenCL call failed with status {d}", .{status});
    return error.OpenClCallFailed;
}

const kernel_source: [:0]const u8 =
    \\__kernel void dense_linear(
    \\    __global const float* weight,
    \\    __global const float* bias,
    \\    int has_bias,
    \\    int in_features,
    \\    __global const float* input,
    \\    __global float* output) {
    \\  const int row = (int)get_global_id(0);
    \\  float sum = has_bias ? bias[row] : 0.0f;
    \\  const int row_offset = row * in_features;
    \\  for (int col = 0; col < in_features; ++col) {
    \\    sum += weight[row_offset + col] * input[col];
    \\  }
    \\  output[row] = sum;
    \\}
    \\
    \\__kernel void autoround_linear(
    \\    __global const int* qweight,
    \\    __global const int* qzeros,
    \\    __global const float* scales,
    \\    __global const float* bias,
    \\    int has_bias,
    \\    int in_features,
    \\    int out_features,
    \\    int group_size,
    \\    int packed_in,
    \\    int packed_out,
    \\    int group_count,
    \\    __global const float* input,
    \\    __global float* output) {
    \\  const int row = (int)get_global_id(0);
    \\  if (row >= out_features) {
    \\    return;
    \\  }
    \\  float sum = has_bias ? bias[row] : 0.0f;
    \\  const int zero_pack = row / 8;
    \\  const int zero_shift = (row % 8) * 4;
    \\  for (int pack = 0; pack < packed_in; ++pack) {
    \\    const uint qword = (uint)qweight[pack * out_features + row];
    \\    const int base_col = pack * 8;
    \\    const int lanes = min(8, in_features - base_col);
    \\    for (int lane = 0; lane < lanes; ++lane) {
    \\      const int col = base_col + lane;
    \\      const int group = min(col / group_size, group_count - 1);
    \\      const uint zword = (uint)qzeros[group * packed_out + zero_pack];
    \\      const int zero = (int)((zword >> zero_shift) & 15u) + 1;
    \\      const int qvalue = (int)((qword >> (lane * 4)) & 15u);
    \\      const float scale = scales[group * out_features + row];
    \\      sum += input[col] * (float)(qvalue - zero) * scale;
    \\    }
    \\  }
    \\  output[row] = sum;
    \\}
;

test "opencl dense linear matches cpu reference" {
    var runtime = Runtime.init(std.testing.allocator) catch |err| switch (err) {
        error.OpenClUnavailable, error.XpuDeviceNotFound, error.OpenClBuildFailed => return error.SkipZigTest,
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

test "opencl autoround linear matches cpu reference" {
    var runtime = Runtime.init(std.testing.allocator) catch |err| switch (err) {
        error.OpenClUnavailable, error.XpuDeviceNotFound, error.OpenClBuildFailed => return error.SkipZigTest,
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
