const std = @import("std");
const tensor = @import("../tensor/tensor.zig");

pub const TensorMeta = struct {
    name: []const u8,
    dtype: tensor.DType,
    shape: []const usize,
    data_start: usize,
    data_end: usize,

    pub fn byteLen(self: TensorMeta) usize {
        return self.data_end - self.data_start;
    }

    pub fn expectedByteLen(self: TensorMeta) usize {
        var count: usize = 1;
        for (self.shape) |dim| count *= dim;
        return count * self.dtype.elementSize();
    }
};

pub const SafeTensorsFile = struct {
    allocator: std.mem.Allocator,
    path: []const u8,
    header_len: usize,
    tensors: std.StringHashMap(TensorMeta),

    pub fn open(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !SafeTensorsFile {
        var file = try openFile(io, path);
        defer file.close(io);

        var len_buf: [8]u8 = undefined;
        const read_len = try file.readPositionalAll(io, &len_buf, 0);
        if (read_len != 8) return error.InvalidSafeTensorsFile;
        const header_len: usize = @intCast(std.mem.readInt(u64, &len_buf, .little));
        if (header_len == 0 or header_len > (512 << 20)) return error.InvalidSafeTensorsHeader;

        const header = try allocator.alloc(u8, header_len);
        defer allocator.free(header);
        if (try file.readPositionalAll(io, header, 8) != header_len) return error.InvalidSafeTensorsHeader;

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidSafeTensorsHeader;

        var tensors = std.StringHashMap(TensorMeta).init(allocator);
        errdefer {
            var it = tensors.iterator();
            while (it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.shape);
            }
            tensors.deinit();
        }

        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            const meta = try parseTensorMeta(allocator, entry.key_ptr.*, entry.value_ptr.*);
            try tensors.put(meta.name, meta);
        }

        return .{
            .allocator = allocator,
            .path = try allocator.dupe(u8, path),
            .header_len = header_len,
            .tensors = tensors,
        };
    }

    pub fn deinit(self: *SafeTensorsFile) void {
        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.shape);
        }
        self.tensors.deinit();
        self.allocator.free(self.path);
        self.* = undefined;
    }

    pub fn get(self: *const SafeTensorsFile, name: []const u8) ?TensorMeta {
        return self.tensors.get(name);
    }

    pub fn readTensorBytes(self: *const SafeTensorsFile, io: std.Io, name: []const u8, allocator: std.mem.Allocator) ![]u8 {
        const meta = self.get(name) orelse return error.TensorNotFound;
        if (meta.byteLen() != meta.expectedByteLen()) return error.InvalidSafeTensorsTensor;
        var file = try openFile(io, self.path);
        defer file.close(io);
        const bytes = try allocator.alloc(u8, meta.byteLen());
        errdefer allocator.free(bytes);
        const offset = 8 + self.header_len + meta.data_start;
        if (try file.readPositionalAll(io, bytes, offset) != bytes.len) return error.InvalidSafeTensorsTensor;
        return bytes;
    }

    pub fn readTensorView(self: *const SafeTensorsFile, io: std.Io, name: []const u8, allocator: std.mem.Allocator) !tensor.TensorView {
        const meta = self.get(name) orelse return error.TensorNotFound;
        const bytes = try self.readTensorBytes(io, name, allocator);
        return .{
            .dtype = meta.dtype,
            .shape = meta.shape,
            .bytes = bytes,
        };
    }
};

pub const ResolvedTensor = struct {
    shard_index: usize,
    meta: TensorMeta,
};

pub const SafeTensorStore = struct {
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    shards: []SafeTensorsFile,

    pub fn openModelDir(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !SafeTensorStore {
        const shard_paths = try collectShardPaths(allocator, io, model_dir);
        defer {
            for (shard_paths) |path| allocator.free(path);
            allocator.free(shard_paths);
        }

        const shards = try allocator.alloc(SafeTensorsFile, shard_paths.len);
        errdefer allocator.free(shards);

        var opened: usize = 0;
        errdefer {
            for (shards[0..opened]) |*shard| shard.deinit();
        }

        for (shard_paths, 0..) |path, idx| {
            shards[idx] = try SafeTensorsFile.open(allocator, io, path);
            opened += 1;
        }

        return .{
            .allocator = allocator,
            .model_dir = try allocator.dupe(u8, model_dir),
            .shards = shards,
        };
    }

    pub fn deinit(self: *SafeTensorStore) void {
        for (self.shards) |*shard| shard.deinit();
        self.allocator.free(self.shards);
        self.allocator.free(self.model_dir);
        self.* = undefined;
    }

    pub fn find(self: *const SafeTensorStore, name: []const u8) ?ResolvedTensor {
        for (self.shards, 0..) |shard, shard_index| {
            if (shard.get(name)) |meta| {
                return .{
                    .shard_index = shard_index,
                    .meta = meta,
                };
            }
        }
        return null;
    }

    pub fn has(self: *const SafeTensorStore, name: []const u8) bool {
        return self.find(name) != null;
    }

    pub fn readTensorBytes(self: *const SafeTensorStore, io: std.Io, name: []const u8, allocator: std.mem.Allocator) ![]u8 {
        const resolved = self.find(name) orelse return error.TensorNotFound;
        return self.shards[resolved.shard_index].readTensorBytes(io, name, allocator);
    }

    pub fn readTensorView(self: *const SafeTensorStore, io: std.Io, name: []const u8, allocator: std.mem.Allocator) !tensor.TensorView {
        const resolved = self.find(name) orelse return error.TensorNotFound;
        return self.shards[resolved.shard_index].readTensorView(io, name, allocator);
    }
};

fn openFile(io: std.Io, path: []const u8) !std.Io.File {
    if (std.fs.path.isAbsolute(path)) {
        return try std.Io.Dir.openFileAbsolute(io, path, .{});
    }
    return try std.Io.Dir.cwd().openFile(io, path, .{});
}

fn collectShardPaths(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) ![][]const u8 {
    const index_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors.index.json" });
    defer allocator.free(index_path);
    if (try fileExists(io, index_path)) {
        return try collectShardPathsFromIndex(allocator, io, model_dir, index_path);
    }

    const direct_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    errdefer allocator.free(direct_path);
    if (try fileExists(io, direct_path)) {
        const paths = try allocator.alloc([]const u8, 1);
        errdefer allocator.free(paths);
        paths[0] = direct_path;
        return paths;
    }
    allocator.free(direct_path);
    return error.NoSafeTensorsWeights;
}

fn collectShardPathsFromIndex(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, index_path: []const u8) ![][]const u8 {
    _ = index_path;
    var dir = try openModelDir(io, model_dir);
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 << 20));
    defer allocator.free(raw);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafeTensorsIndex;
    const weight_map = parsed.value.object.get("weight_map") orelse return error.InvalidSafeTensorsIndex;
    if (weight_map != .object) return error.InvalidSafeTensorsIndex;

    var paths = std.array_list.Managed([]const u8).init(allocator);
    errdefer {
        for (paths.items) |path| allocator.free(path);
        paths.deinit();
    }

    var it = weight_map.object.iterator();
    while (it.next()) |entry| {
        const shard_name_value = entry.value_ptr.*;
        if (shard_name_value != .string) return error.InvalidSafeTensorsIndex;
        const shard_name = shard_name_value.string;
        const full_path = try std.fs.path.join(allocator, &.{ model_dir, shard_name });
        if (containsPath(paths.items, full_path)) {
            allocator.free(full_path);
            continue;
        }
        try paths.append(full_path);
    }

    return try paths.toOwnedSlice();
}

fn containsPath(paths: []const []const u8, candidate: []const u8) bool {
    for (paths) |path| {
        if (std.mem.eql(u8, path, candidate)) return true;
    }
    return false;
}

fn fileExists(io: std.Io, path: []const u8) !bool {
    var file = openFile(io, path) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    file.close(io);
    return true;
}

fn openModelDir(io: std.Io, path: []const u8) !std.Io.Dir {
    if (std.fs.path.isAbsolute(path)) {
        return try std.Io.Dir.openDirAbsolute(io, path, .{});
    }
    return try std.Io.Dir.cwd().openDir(io, path, .{});
}

fn parseTensorMeta(allocator: std.mem.Allocator, name: []const u8, value: std.json.Value) !TensorMeta {
    if (value != .object) return error.InvalidSafeTensorsHeader;
    const dtype_value = value.object.get("dtype") orelse return error.InvalidSafeTensorsHeader;
    if (dtype_value != .string) return error.InvalidSafeTensorsHeader;
    const dtype = try tensor.DType.fromSafetensors(dtype_value.string);

    const shape_value = value.object.get("shape") orelse return error.InvalidSafeTensorsHeader;
    if (shape_value != .array) return error.InvalidSafeTensorsHeader;
    const shape = try allocator.alloc(usize, shape_value.array.items.len);
    errdefer allocator.free(shape);
    for (shape, shape_value.array.items) |*dim, raw| {
        dim.* = @intCast(try jsonInt(raw));
    }

    const offsets_value = value.object.get("data_offsets") orelse return error.InvalidSafeTensorsHeader;
    if (offsets_value != .array or offsets_value.array.items.len != 2) return error.InvalidSafeTensorsHeader;
    const data_start: usize = @intCast(try jsonInt(offsets_value.array.items[0]));
    const data_end: usize = @intCast(try jsonInt(offsets_value.array.items[1]));
    if (data_end < data_start) return error.InvalidSafeTensorsHeader;

    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    const meta: TensorMeta = .{
        .name = owned_name,
        .dtype = dtype,
        .shape = shape,
        .data_start = data_start,
        .data_end = data_end,
    };
    if (meta.byteLen() != meta.expectedByteLen()) return error.InvalidSafeTensorsTensor;
    return meta;
}

fn jsonInt(value: std.json.Value) !i64 {
    return switch (value) {
        .integer => |integer| integer,
        .number_string => |text| try std.fmt.parseInt(i64, text, 10),
        else => error.InvalidSafeTensorsHeader,
    };
}

test "safetensors header and tensor bytes are parsed" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header =
        \\{"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}
    ;
    var bytes = std.array_list.Managed(u8).init(std.testing.allocator);
    defer bytes.deinit();
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try bytes.appendSlice(&len_buf);
    try bytes.appendSlice(header);
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    for (values) |value| {
        var value_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &value_bytes, @bitCast(value), .little);
        try bytes.appendSlice(&value_bytes);
    }

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = bytes.items });
    const relative = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/model.safetensors", .{tmp.sub_path});
    defer std.testing.allocator.free(relative);

    var file = try SafeTensorsFile.open(std.testing.allocator, std.testing.io, relative);
    defer file.deinit();
    const meta = file.get("x").?;
    try std.testing.expectEqual(tensor.DType.f32, meta.dtype);
    try std.testing.expectEqual(@as(usize, 2), meta.shape[0]);

    const raw = try file.readTensorBytes(std.testing.io, "x", std.testing.allocator);
    defer std.testing.allocator.free(raw);
    const view: tensor.TensorView = .{ .dtype = meta.dtype, .shape = meta.shape, .bytes = raw };
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), view.f32At(2), 1e-6);
}

test "safetensors store resolves indexed shards" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeSingleF32Shard(&tmp, "shard-a.safetensors", "x", 7.0);
    try writeSingleF32Shard(&tmp, "shard-b.safetensors", "y", 9.0);
    const index =
        \\{"weight_map":{"x":"shard-a.safetensors","y":"shard-b.safetensors"}}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = index });

    const model_dir = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(model_dir);
    var store = try SafeTensorStore.openModelDir(std.testing.allocator, std.testing.io, model_dir);
    defer store.deinit();

    try std.testing.expectEqual(@as(usize, 2), store.shards.len);
    const bytes = try store.readTensorBytes(std.testing.io, "y", std.testing.allocator);
    defer std.testing.allocator.free(bytes);
    const meta = store.find("y").?.meta;
    const view: tensor.TensorView = .{ .dtype = meta.dtype, .shape = meta.shape, .bytes = bytes };
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), view.f32At(0), 1e-6);
}

fn writeSingleF32Shard(tmp: *std.testing.TmpDir, sub_path: []const u8, name: []const u8, value: f32) !void {
    const header = try std.fmt.allocPrint(
        std.testing.allocator,
        "{{\"{s}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}}}",
        .{name},
    );
    defer std.testing.allocator.free(header);

    var bytes = std.array_list.Managed(u8).init(std.testing.allocator);
    defer bytes.deinit();
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try bytes.appendSlice(&len_buf);
    try bytes.appendSlice(header);
    var value_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &value_bytes, @bitCast(value), .little);
    try bytes.appendSlice(&value_bytes);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = sub_path, .data = bytes.items });
}
