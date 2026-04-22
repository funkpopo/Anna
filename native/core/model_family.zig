const std = @import("std");
const types = @import("../runtime/types.zig");

pub const ModelFamilyInfo = struct {
    model_family: types.SupportedModelFamily,
    model_type: []const u8,
    architectures: []const []const u8,
};

pub const ModelArtifactSummary = struct {
    has_config_json: bool = false,
    has_generation_config: bool = false,
    has_tokenizer_json: bool = false,
    has_processor_config: bool = false,
    has_quantization_config: bool = false,
    safetensor_files: usize = 0,
    gguf_model_files: usize = 0,
    gguf_mmproj_files: usize = 0,
};

pub const ModelManifest = struct {
    model_dir: []const u8,
    model_name: []const u8,
    family: ?ModelFamilyInfo,
    artifacts: ModelArtifactSummary,
};

const ConfigFile = struct {
    model_type: ?[]const u8 = null,
    architectures: ?[]const []const u8 = null,
};

pub fn inspectModelFamily(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !ModelFamilyInfo {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer dir.close(io);

    const raw = try dir.readFileAlloc(io, "config.json", allocator, .limited(1 << 20));
    defer allocator.free(raw);

    var parsed = try std.json.parseFromSlice(ConfigFile, allocator, raw, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const raw_type = parsed.value.model_type orelse "";
    const model_type = try allocator.dupe(u8, raw_type);
    var architectures = std.array_list.Managed([]const u8).init(allocator);
    defer architectures.deinit();

    if (parsed.value.architectures) |values| {
        for (values) |value| {
            try architectures.append(try allocator.dupe(u8, value));
        }
    }

    const family: types.SupportedModelFamily = if (std.mem.eql(u8, raw_type, "qwen3_tts"))
        .qwen3_tts
    else if (std.mem.eql(u8, raw_type, "gemma4"))
        .gemma4
    else
        .qwen3_5_text;

    return .{
        .model_family = family,
        .model_type = model_type,
        .architectures = try architectures.toOwnedSlice(),
    };
}

pub fn inspectModelManifest(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, model_name: []const u8) !ModelManifest {
    const artifacts = try scanArtifacts(io, model_dir);
    var family: ?ModelFamilyInfo = null;
    if (artifacts.has_config_json) {
        family = try inspectModelFamily(allocator, io, model_dir);
    }
    return .{
        .model_dir = try allocator.dupe(u8, model_dir),
        .model_name = try allocator.dupe(u8, model_name),
        .family = family,
        .artifacts = artifacts,
    };
}

fn scanArtifacts(io: std.Io, model_dir: []const u8) !ModelArtifactSummary {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);

    var iter = dir.iterate();
    var summary: ModelArtifactSummary = .{};
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file) continue;
        const name = entry.name;
        if (std.mem.eql(u8, name, "config.json")) summary.has_config_json = true;
        if (std.mem.eql(u8, name, "generation_config.json")) summary.has_generation_config = true;
        if (std.mem.eql(u8, name, "tokenizer.json")) summary.has_tokenizer_json = true;
        if (std.mem.eql(u8, name, "processor_config.json")) summary.has_processor_config = true;
        if (std.mem.eql(u8, name, "quantization_config.json")) summary.has_quantization_config = true;
        if (std.mem.endsWith(u8, name, ".safetensors")) summary.safetensor_files += 1;
        if (std.mem.endsWith(u8, name, ".gguf")) {
            if (std.mem.startsWith(u8, name, "mmproj-")) {
                summary.gguf_mmproj_files += 1;
            } else {
                summary.gguf_model_files += 1;
            }
        }
    }
    return summary;
}

test "inspect model family detects qwen3.5 text" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "config.json",
        .data = "{\"model_type\":\"qwen3_5\"}",
    });

    const relative = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(relative);
    const dir_path = try testRealPath(std.testing.allocator, relative);
    defer std.testing.allocator.free(dir_path);

    const info = try inspectModelFamily(std.testing.allocator, std.testing.io, dir_path);
    defer std.testing.allocator.free(info.model_type);
    for (info.architectures) |value| std.testing.allocator.free(value);
    std.testing.allocator.free(info.architectures);

    try std.testing.expectEqual(types.SupportedModelFamily.qwen3_5_text, info.model_family);
    try std.testing.expectEqualStrings("qwen3_5", info.model_type);
}

test "inspect model family detects qwen3_tts" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "config.json",
        .data = "{\"model_type\":\"qwen3_tts\",\"architectures\":[\"Qwen3TTSForConditionalGeneration\"]}",
    });

    const relative = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(relative);
    const dir_path = try testRealPath(std.testing.allocator, relative);
    defer std.testing.allocator.free(dir_path);

    const info = try inspectModelFamily(std.testing.allocator, std.testing.io, dir_path);
    defer std.testing.allocator.free(info.model_type);
    for (info.architectures) |value| std.testing.allocator.free(value);
    std.testing.allocator.free(info.architectures);

    try std.testing.expectEqual(types.SupportedModelFamily.qwen3_tts, info.model_family);
    try std.testing.expectEqualStrings("qwen3_tts", info.model_type);
    try std.testing.expectEqual(@as(usize, 1), info.architectures.len);
}

test "inspect model family detects gemma4" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "config.json",
        .data = "{\"model_type\":\"gemma4\",\"architectures\":[\"Gemma4ForConditionalGeneration\"]}",
    });

    const relative = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(relative);
    const dir_path = try testRealPath(std.testing.allocator, relative);
    defer std.testing.allocator.free(dir_path);

    const info = try inspectModelFamily(std.testing.allocator, std.testing.io, dir_path);
    defer std.testing.allocator.free(info.model_type);
    for (info.architectures) |value| std.testing.allocator.free(value);
    std.testing.allocator.free(info.architectures);

    try std.testing.expectEqual(types.SupportedModelFamily.gemma4, info.model_family);
    try std.testing.expectEqualStrings("gemma4", info.model_type);
}

fn testRealPath(allocator: std.mem.Allocator, relative: []const u8) ![]const u8 {
    const raw = try std.Io.Dir.cwd().realPathFileAlloc(std.testing.io, relative, allocator);
    defer allocator.free(raw);
    return try allocator.dupe(u8, raw);
}
