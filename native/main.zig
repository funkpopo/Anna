const std = @import("std");
const anna = @import("anna_native");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const io = init.io;

    var stderr_buffer: [1024]u8 = undefined;
    var stderr_file_writer: std.Io.File.Writer = .init(.stderr(), io, &stderr_buffer);
    const stderr_writer = &stderr_file_writer.interface;

    if (args.len < 2) {
        try printUsage(stderr_writer);
        try stderr_writer.flush();
        return;
    }

    const command = args[1];
    if (std.mem.eql(u8, command, "inspect")) {
        try runInspect(arena, io, args[2..]);
        return;
    }

    try stderr_writer.print("unknown subcommand: {s}\n\n", .{command});
    try printUsage(stderr_writer);
    try stderr_writer.flush();
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\anna-native inspect --model-dir <path> [serve options]
        \\  Validates native serve settings, resolves model metadata, and prints the
        \\  route surface that the Zig control plane will expose.
        \\
    );
}

fn runInspect(allocator: std.mem.Allocator, io: std.Io, raw_args: []const []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const settings = try anna.config.parseServeArgs(arena, raw_args);
    const resolved_dir = try anna.model_path.resolveModelDir(arena, io, settings.model_dir);
    const resolved_name = try anna.model_path.resolveModelName(arena, settings.model_id, resolved_dir);
    const manifest = try anna.model_family.inspectModelManifest(arena, io, resolved_dir, resolved_name);
    const routes = anna.app.defaultRoutes();
    const safety_policy = anna.config.buildSafetyPolicy(settings);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const writer = &stdout_file_writer.interface;
    try writer.writeAll("Anna native inspect\n");
    try writer.print("Model dir: {s}\n", .{manifest.model_dir});
    try writer.print("Model id:  {s}\n", .{manifest.model_name});
    if (manifest.family) |family| {
        try writer.print("Family:    {s} ({s})\n", .{ @tagName(family.model_family), family.model_type });
    } else {
        try writer.writeAll("Family:    unresolved (missing native model-family metadata)\n");
    }
    try writer.print(
        "Artifacts: config={any} generation={any} tokenizer={any} processor={any} quant={any} safetensors={d} gguf={d} mmproj={d}\n",
        .{
            manifest.artifacts.has_config_json,
            manifest.artifacts.has_generation_config,
            manifest.artifacts.has_tokenizer_json,
            manifest.artifacts.has_processor_config,
            manifest.artifacts.has_quantization_config,
            manifest.artifacts.safetensor_files,
            manifest.artifacts.gguf_model_files,
            manifest.artifacts.gguf_mmproj_files,
        },
    );
    if (safety_policy) |policy| {
        try writer.print(
            "Safety:    min_free={d} reserve={d} usage_ratio={d:.2} factor={d:.2}\n",
            .{
                policy.min_free_bytes,
                policy.reserve_margin_bytes,
                policy.max_estimated_usage_ratio,
                policy.generation_memory_safety_factor,
            },
        );
    } else {
        try writer.writeAll("Safety:    default engine policy\n");
    }
    try anna.app.writeRouteAnnouncement(writer, routes, settings.host, settings.port);
    try writer.flush();
}
