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
    if (std.mem.eql(u8, command, "serve")) {
        try runServe(arena, io, args[2..]);
        return;
    }
    if (std.mem.eql(u8, command, "inspect")) {
        try runInspect(arena, io, args[2..]);
        return;
    }
    if (std.mem.eql(u8, command, "eval-token")) {
        try runEvalToken(arena, io, args[2..]);
        return;
    }
    if (std.mem.eql(u8, command, "generate")) {
        try runGenerate(arena, io, args[2..]);
        return;
    }
    if (std.mem.eql(u8, command, "chat-json")) {
        try runChatJson(arena, io, args[2..]);
        return;
    }
    if (std.mem.eql(u8, command, "completion-json")) {
        try runCompletionJson(arena, io, args[2..]);
        return;
    }

    try stderr_writer.print("unknown subcommand: {s}\n\n", .{command});
    try printUsage(stderr_writer);
    try stderr_writer.flush();
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\anna-native serve --model-dir <path> [serve options]
        \\  Starts the native Zig HTTP service with OpenAI-compatible routes.
        \\  Optional: --backend cpu|xpu
        \\
        \\anna-native inspect --model-dir <path> [serve options]
        \\  Validates native serve settings, resolves model metadata, and prints the
        \\  route surface that the Zig control plane will expose.
        \\
        \\anna-native eval-token --model-dir <path> --tokens <id0,id1,...> [--top-k <n>]
        \\  Loads the native Qwen token runtime, runs prompt tokens through Zig-only
        \\  decode, and prints the highest logits for the last position.
        \\  Optional: --backend cpu|xpu
        \\
        \\anna-native generate --model-dir <path> --prompt <text> [sampling options]
        \\  Runs native tokenizer + scheduler + token runtime and prints generated text.
        \\  Optional: --backend cpu|xpu
        \\
        \\anna-native chat-json --model-dir <path> --user <text> [--system <text>] [sampling options]
        \\  Runs native chat generation and prints an OpenAI-compatible chat response JSON.
        \\  Optional: --backend cpu|xpu
        \\
        \\anna-native completion-json --model-dir <path> --prompt <text> [sampling options]
        \\  Runs native text generation and prints an OpenAI-compatible completion JSON.
        \\  Optional: --backend cpu|xpu
        \\
    );
}

fn runServe(allocator: std.mem.Allocator, io: std.Io, raw_args: []const []const u8) !void {
    const settings = try anna.config.parseServeArgs(allocator, raw_args);
    var server = try anna.http_server.AnnaHttpServer.init(std.heap.page_allocator, io, settings);
    defer server.deinit();
    try server.serve();
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
    try writer.print("Backend:   {s}\n", .{@tagName(settings.backend)});
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

const EvalSettings = struct {
    model_dir: []const u8,
    tokens: []const u32,
    top_k: usize = 8,
    backend: anna.types.RuntimeBackend = .cpu,
};

fn runEvalToken(allocator: std.mem.Allocator, io: std.Io, raw_args: []const []const u8) !void {
    const settings = try parseEvalArgs(allocator, raw_args);
    const runtime_allocator = std.heap.page_allocator;

    var runtime = try anna.qwen3_loader.QwenTextRuntime.loadWithOptions(runtime_allocator, io, settings.model_dir, .{ .backend = settings.backend });
    defer runtime.deinit();

    const logits = try runtime.forwardPromptAlloc(settings.tokens);
    defer runtime_allocator.free(logits);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const writer = &stdout_file_writer.interface;

    try writer.writeAll("Anna native eval-token\n");
    try writer.print("Model dir: {s}\n", .{settings.model_dir});
    try writer.print("Prompt tokens: {d}\n", .{settings.tokens.len});
    try writer.print("Last position: {d}\n", .{runtime.engine.position});
    try writer.print("EOS token: {d}\n", .{runtime.config.eos_token_id});
    try writer.writeAll("Top logits:\n");
    try writeTopLogits(writer, allocator, logits, settings.top_k);
    try writer.flush();
}

fn parseEvalArgs(allocator: std.mem.Allocator, args: []const []const u8) !EvalSettings {
    var model_dir: []const u8 = "";
    var tokens_text: ?[]const u8 = null;
    var top_k: usize = 8;
    var backend: anna.types.RuntimeBackend = .cpu;

    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--model-dir")) {
            model_dir = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--tokens")) {
            tokens_text = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            top_k = try parsePositiveInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--backend")) {
            backend = try parseBackend(try nextValue(args, &index));
        } else {
            return error.UnknownFlag;
        }
    }

    if (model_dir.len == 0) return error.MissingModelDir;
    const tokens = try parseTokenIds(allocator, tokens_text orelse return error.MissingTokens);
    if (tokens.len == 0) return error.MissingTokens;
    return .{
        .model_dir = model_dir,
        .tokens = tokens,
        .top_k = top_k,
        .backend = backend,
    };
}

fn parseTokenIds(allocator: std.mem.Allocator, raw: []const u8) ![]const u32 {
    var values = std.array_list.Managed(u32).init(allocator);
    defer values.deinit();

    var parts = std.mem.splitScalar(u8, raw, ',');
    while (parts.next()) |part| {
        const token = std.mem.trim(u8, part, " \t\r\n");
        if (token.len == 0) continue;
        try values.append(@intCast(try std.fmt.parseInt(u32, token, 10)));
    }
    return try values.toOwnedSlice();
}

fn writeTopLogits(writer: anytype, allocator: std.mem.Allocator, logits: []const f32, top_k: usize) !void {
    const count = @min(top_k, logits.len);
    if (count == 0) return;

    const used = try allocator.alloc(bool, logits.len);
    defer allocator.free(used);
    @memset(used, false);

    for (0..count) |_| {
        var best_idx: ?usize = null;
        for (logits, 0..) |value, idx| {
            if (used[idx]) continue;
            if (best_idx == null or value > logits[best_idx.?]) best_idx = idx;
        }
        const selected = best_idx.?;
        used[selected] = true;
        try writer.print("  token={d} logit={d:.6}\n", .{ selected, logits[selected] });
    }
}

fn nextValue(args: []const []const u8, index: *usize) ![]const u8 {
    index.* += 1;
    if (index.* >= args.len) return error.MissingValue;
    return args[index.*];
}

fn parsePositiveInt(text: []const u8) !usize {
    const value = std.fmt.parseInt(usize, text, 10) catch return error.InvalidInteger;
    if (value == 0) return error.InvalidInteger;
    return value;
}

const GenerateSettings = struct {
    model_dir: []const u8,
    prompt: []const u8,
    system: ?[]const u8 = null,
    max_new_tokens: ?usize = null,
    temperature: f32 = 0.7,
    top_p: f32 = 0.95,
    top_k: usize = 50,
    repetition_penalty: f32 = 1.0,
    backend: anna.types.RuntimeBackend = .cpu,
};

fn runGenerate(allocator: std.mem.Allocator, io: std.Io, raw_args: []const []const u8) !void {
    const settings = try parseGenerateArgs(raw_args);
    var service = try anna.qwen_text_service.NativeQwenTextService.loadWithOptions(allocator, io, settings.model_dir, settings.model_dir, .{ .backend = settings.backend });
    defer service.deinit();

    const result = try service.generateTextAlloc(settings.prompt, generationConfig(settings));
    defer allocator.free(result.text);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const writer = &stdout_file_writer.interface;
    try writer.writeAll(result.text);
    try writer.writeByte('\n');
    try writer.flush();
}

fn runChatJson(allocator: std.mem.Allocator, io: std.Io, raw_args: []const []const u8) !void {
    const settings = try parseGenerateArgs(raw_args);
    var service = try anna.qwen_text_service.NativeQwenTextService.loadWithOptions(allocator, io, settings.model_dir, settings.model_dir, .{ .backend = settings.backend });
    defer service.deinit();

    var messages = std.array_list.Managed(anna.qwen3_tokenizer.ChatMessage).init(allocator);
    defer messages.deinit();
    if (settings.system) |system| try messages.append(.{ .role = .system, .content = system });
    try messages.append(.{ .role = .user, .content = settings.prompt });

    const result = try service.generateChatAlloc(messages.items, generationConfig(settings), true);
    defer allocator.free(result.text);
    if (result.reasoning_text) |reasoning| allocator.free(reasoning);
    defer {
        for (result.tool_calls) |tool_call| {
            allocator.free(tool_call.name);
            allocator.free(tool_call.arguments_json);
        }
        allocator.free(result.tool_calls);
    }

    const payload = try anna.openai.chatResponsePayload(allocator, "chatcmpl-native", 0, service.model_id, result);
    defer allocator.free(payload);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const writer = &stdout_file_writer.interface;
    try writer.writeAll(payload);
    try writer.writeByte('\n');
    try writer.flush();
}

fn runCompletionJson(allocator: std.mem.Allocator, io: std.Io, raw_args: []const []const u8) !void {
    const settings = try parseGenerateArgs(raw_args);
    var service = try anna.qwen_text_service.NativeQwenTextService.loadWithOptions(allocator, io, settings.model_dir, settings.model_dir, .{ .backend = settings.backend });
    defer service.deinit();

    const result = try service.generateTextAlloc(settings.prompt, generationConfig(settings));
    defer allocator.free(result.text);

    const payload = try anna.openai.completionResponsePayload(allocator, "cmpl-native", 0, service.model_id, result);
    defer allocator.free(payload);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const writer = &stdout_file_writer.interface;
    try writer.writeAll(payload);
    try writer.writeByte('\n');
    try writer.flush();
}

fn parseGenerateArgs(args: []const []const u8) !GenerateSettings {
    var settings: GenerateSettings = .{ .model_dir = "", .prompt = "" };
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--model-dir")) {
            settings.model_dir = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            settings.prompt = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--user")) {
            settings.prompt = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--system")) {
            settings.system = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--max-new-tokens")) {
            settings.max_new_tokens = try parsePositiveInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            settings.temperature = try parseFloat32(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            settings.top_p = try parseFloat32(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            settings.top_k = try parsePositiveInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--repetition-penalty")) {
            settings.repetition_penalty = try parseFloat32(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--backend")) {
            settings.backend = try parseBackend(try nextValue(args, &index));
        } else {
            return error.UnknownFlag;
        }
    }
    if (settings.model_dir.len == 0) return error.MissingModelDir;
    if (settings.prompt.len == 0) return error.MissingPrompt;
    return settings;
}

fn generationConfig(settings: GenerateSettings) anna.types.GenerationConfig {
    return .{
        .max_new_tokens = settings.max_new_tokens,
        .temperature = settings.temperature,
        .top_p = settings.top_p,
        .top_k = settings.top_k,
        .repetition_penalty = settings.repetition_penalty,
    };
}

fn parseFloat32(text: []const u8) !f32 {
    return std.fmt.parseFloat(f32, text) catch error.InvalidFloat;
}

fn parseBackend(text: []const u8) !anna.types.RuntimeBackend {
    if (std.mem.eql(u8, text, "cpu")) return .cpu;
    if (std.mem.eql(u8, text, "xpu")) return .xpu;
    return error.InvalidBackend;
}
