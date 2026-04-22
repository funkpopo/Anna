const std = @import("std");

pub fn resolveModelDir(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) ![]const u8 {
    const resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, model_dir, allocator);
    defer allocator.free(resolved);
    return try allocator.dupe(u8, resolved);
}

pub fn resolveModelName(allocator: std.mem.Allocator, model_name: ?[]const u8, model_dir: []const u8) ![]const u8 {
    if (model_name) |value| {
        if (value.len > 0) {
            return try allocator.dupe(u8, value);
        }
    }
    return try allocator.dupe(u8, model_dir);
}

test "resolve model dir keeps explicit path" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var created = try tmp.dir.createDirPathOpen(std.testing.io, "models/Qwen/Qwen3.5-2B", .{});
    defer created.close(std.testing.io);

    const tmp_relative = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/models/Qwen/Qwen3.5-2B", .{tmp.sub_path});
    defer std.testing.allocator.free(tmp_relative);
    const relative = try resolveModelDir(std.testing.allocator, std.testing.io, tmp_relative);
    defer std.testing.allocator.free(relative);

    const resolved = try resolveModelDir(std.testing.allocator, std.testing.io, relative);
    defer std.testing.allocator.free(resolved);

    try std.testing.expectEqualStrings(relative, resolved);
}

test "resolve model name prefers explicit name" {
    const resolved = try resolveModelName(std.testing.allocator, "qwen3.5", "C:/models/Qwen3.5-2B");
    defer std.testing.allocator.free(resolved);
    try std.testing.expectEqualStrings("qwen3.5", resolved);
}

test "resolve model name defaults to full path" {
    const resolved = try resolveModelName(std.testing.allocator, null, "C:/models/Qwen3.5-2B");
    defer std.testing.allocator.free(resolved);
    try std.testing.expectEqualStrings("C:/models/Qwen3.5-2B", resolved);
}
