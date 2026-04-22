const std = @import("std");
const types = @import("../runtime/types.zig");

pub const ServeSettings = struct {
    model_dir: []const u8,
    model_id: ?[]const u8 = null,
    device: []const u8 = "auto",
    dtype: []const u8 = "auto",
    compile_mode: []const u8 = "none",
    compile_fullgraph: bool = false,
    prefill_chunk_size: usize = 0,
    prompt_cache_size: usize = 0,
    prompt_cache_max_tokens: usize = 0,
    profile_runtime: bool = false,
    kv_cache_quantization: []const u8 = "none",
    kv_cache_quant_bits: u8 = 4,
    kv_cache_residual_len: usize = 128,
    default_max_completion_tokens: ?usize = null,
    default_enable_thinking: bool = true,
    reasoning_format: []const u8 = "deepseek",
    offload_mode: []const u8 = "auto",
    offload_vision: bool = false,
    expert_quant: []const u8 = "auto",
    weight_quant: []const u8 = "auto",
    resident_expert_layers: ?usize = null,
    resident_expert_layer_indices: ?[]const i32 = null,
    cached_experts_per_layer: ?usize = null,
    min_free_memory_mib: ?u64 = null,
    reserve_memory_mib: ?u64 = null,
    max_estimated_usage_ratio: ?f64 = null,
    generation_memory_safety_factor: ?f64 = null,
    scheduler_max_batch_size: usize = 1,
    scheduler_batch_wait_ms: f64 = 2.0,
    metrics_log_interval_seconds: f64 = 10.0,
    host: []const u8 = "127.0.0.1",
    port: u16 = 8000,
    log_level: []const u8 = "info",
};

pub const ParseError = error{
    MissingModelDir,
    MissingValue,
    InvalidInteger,
    InvalidFloat,
    InvalidRatio,
    InvalidSafetyFactor,
    InvalidChoice,
    UnknownFlag,
    OutOfMemory,
};

pub fn parseResidentExpertLayerIndices(allocator: std.mem.Allocator, value: ?[]const u8) !?[]const i32 {
    const raw = value orelse return null;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) {
        return try allocator.dupe(i32, &.{});
    }

    var values = std.array_list.Managed(i32).init(allocator);
    defer values.deinit();

    var parts = std.mem.splitScalar(u8, trimmed, ',');
    while (parts.next()) |part| {
        const token = std.mem.trim(u8, part, " \t\r\n");
        if (token.len == 0) continue;
        const parsed = std.fmt.parseInt(i32, token, 10) catch return error.InvalidInteger;
        try values.append(parsed);
    }
    return try values.toOwnedSlice();
}

pub fn buildSafetyPolicy(settings: ServeSettings) ?types.RuntimeSafetyPolicy {
    if (settings.min_free_memory_mib == null and
        settings.reserve_memory_mib == null and
        settings.max_estimated_usage_ratio == null and
        settings.generation_memory_safety_factor == null)
    {
        return null;
    }

    const defaults = types.RuntimeSafetyPolicy{};
    return .{
        .min_free_bytes = (settings.min_free_memory_mib orelse (defaults.min_free_bytes >> 20)) << 20,
        .reserve_margin_bytes = (settings.reserve_memory_mib orelse (defaults.reserve_margin_bytes >> 20)) << 20,
        .max_estimated_usage_ratio = settings.max_estimated_usage_ratio orelse defaults.max_estimated_usage_ratio,
        .generation_memory_safety_factor = settings.generation_memory_safety_factor orelse defaults.generation_memory_safety_factor,
    };
}

pub fn parseServeArgs(allocator: std.mem.Allocator, args: []const []const u8) ParseError!ServeSettings {
    var settings: ServeSettings = .{ .model_dir = "" };
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--model-dir")) {
            settings.model_dir = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--model-name")) {
            settings.model_id = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--device")) {
            settings.device = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--dtype")) {
            settings.dtype = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--compile-mode")) {
            const value = try nextValue(args, &index);
            try expectChoice(value, &.{ "none", "default", "reduce-overhead", "max-autotune" });
            settings.compile_mode = value;
        } else if (std.mem.eql(u8, arg, "--compile-fullgraph")) {
            settings.compile_fullgraph = true;
        } else if (std.mem.eql(u8, arg, "--prefill-chunk-size")) {
            settings.prefill_chunk_size = try parseNonNegativeInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--prompt-cache-size")) {
            settings.prompt_cache_size = try parseNonNegativeInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--prompt-cache-max-tokens")) {
            settings.prompt_cache_max_tokens = try parseNonNegativeInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--profile-runtime")) {
            settings.profile_runtime = true;
        } else if (std.mem.eql(u8, arg, "--kv-cache-quantization")) {
            const value = try nextValue(args, &index);
            try expectChoice(value, &.{ "none", "turboquant" });
            settings.kv_cache_quantization = value;
        } else if (std.mem.eql(u8, arg, "--kv-cache-quant-bits")) {
            const value = try parsePositiveInt(try nextValue(args, &index));
            if (value != 3 and value != 4) return error.InvalidChoice;
            settings.kv_cache_quant_bits = @intCast(value);
        } else if (std.mem.eql(u8, arg, "--kv-cache-residual-len")) {
            settings.kv_cache_residual_len = try parsePositiveInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--enable-thinking")) {
            settings.default_enable_thinking = true;
        } else if (std.mem.eql(u8, arg, "--disable-thinking")) {
            settings.default_enable_thinking = false;
        } else if (std.mem.eql(u8, arg, "--max-completion-tokens")) {
            settings.default_max_completion_tokens = try parsePositiveInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--reasoning-format")) {
            const value = try nextValue(args, &index);
            try expectChoice(value, &.{ "none", "deepseek" });
            settings.reasoning_format = value;
        } else if (std.mem.eql(u8, arg, "--offload-mode")) {
            const value = try nextValue(args, &index);
            try expectChoice(value, &.{ "auto", "none", "experts" });
            settings.offload_mode = value;
        } else if (std.mem.eql(u8, arg, "--offload-vision")) {
            settings.offload_vision = true;
        } else if (std.mem.eql(u8, arg, "--expert-quant")) {
            const value = try nextValue(args, &index);
            try expectChoice(value, &.{ "auto", "none", "int4" });
            settings.expert_quant = value;
        } else if (std.mem.eql(u8, arg, "--weight-quant")) {
            const value = try nextValue(args, &index);
            try expectChoice(value, &.{ "auto", "none", "int4" });
            settings.weight_quant = value;
        } else if (std.mem.eql(u8, arg, "--resident-expert-layers")) {
            settings.resident_expert_layers = try parsePositiveInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--resident-expert-layer-indices")) {
            settings.resident_expert_layer_indices = try parseResidentExpertLayerIndices(allocator, try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--cached-experts-per-layer")) {
            settings.cached_experts_per_layer = try parseNonNegativeInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--min-free-memory-mib")) {
            settings.min_free_memory_mib = try parseNonNegativeInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--reserve-memory-mib")) {
            settings.reserve_memory_mib = try parseNonNegativeInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--max-estimated-usage-ratio")) {
            settings.max_estimated_usage_ratio = try parseRatio(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--generation-memory-safety-factor")) {
            settings.generation_memory_safety_factor = try parseSafetyFactor(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--scheduler-max-batch-size")) {
            settings.scheduler_max_batch_size = try parsePositiveInt(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--scheduler-batch-wait-ms")) {
            settings.scheduler_batch_wait_ms = try parseNonNegativeFloat(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--metrics-log-interval-seconds")) {
            settings.metrics_log_interval_seconds = try parseNonNegativeFloat(try nextValue(args, &index));
        } else if (std.mem.eql(u8, arg, "--host")) {
            settings.host = try nextValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--port")) {
            settings.port = @intCast(try parsePositiveInt(try nextValue(args, &index)));
        } else if (std.mem.eql(u8, arg, "--log-level")) {
            settings.log_level = try nextValue(args, &index);
        } else {
            return error.UnknownFlag;
        }
    }

    if (settings.model_dir.len == 0) {
        return error.MissingModelDir;
    }
    return settings;
}

fn nextValue(args: []const []const u8, index: *usize) ParseError![]const u8 {
    index.* += 1;
    if (index.* >= args.len) return error.MissingValue;
    return args[index.*];
}

fn parseNonNegativeInt(text: []const u8) ParseError!u64 {
    return std.fmt.parseInt(u64, text, 10) catch error.InvalidInteger;
}

fn parsePositiveInt(text: []const u8) ParseError!usize {
    const value = std.fmt.parseInt(usize, text, 10) catch return error.InvalidInteger;
    if (value == 0) return error.InvalidInteger;
    return value;
}

fn parseNonNegativeFloat(text: []const u8) ParseError!f64 {
    const value = std.fmt.parseFloat(f64, text) catch return error.InvalidFloat;
    if (value < 0.0) return error.InvalidFloat;
    return value;
}

fn parseRatio(text: []const u8) ParseError!f64 {
    const value = std.fmt.parseFloat(f64, text) catch return error.InvalidFloat;
    if (!(value > 0.0 and value <= 1.0)) return error.InvalidRatio;
    return value;
}

fn parseSafetyFactor(text: []const u8) ParseError!f64 {
    const value = std.fmt.parseFloat(f64, text) catch return error.InvalidFloat;
    if (value < 1.0) return error.InvalidSafetyFactor;
    return value;
}

fn expectChoice(value: []const u8, choices: []const []const u8) ParseError!void {
    for (choices) |choice| {
        if (std.mem.eql(u8, choice, value)) return;
    }
    return error.InvalidChoice;
}

test "build safety policy uses custom serve overrides" {
    const settings: ServeSettings = .{
        .model_dir = "dummy",
        .min_free_memory_mib = 256,
        .reserve_memory_mib = 128,
        .max_estimated_usage_ratio = 0.95,
        .generation_memory_safety_factor = 1.25,
    };

    const policy = buildSafetyPolicy(settings).?;
    try std.testing.expectEqual(@as(u64, 256 << 20), policy.min_free_bytes);
    try std.testing.expectEqual(@as(u64, 128 << 20), policy.reserve_margin_bytes);
    try std.testing.expectEqual(@as(f64, 0.95), policy.max_estimated_usage_ratio);
    try std.testing.expectEqual(@as(f64, 1.25), policy.generation_memory_safety_factor);
}

test "serve parser accepts memory guard arguments" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const args = [_][]const u8{
        "--model-dir",
        "model",
        "--disable-thinking",
        "--max-completion-tokens",
        "1024",
        "--reasoning-format",
        "deepseek",
        "--offload-vision",
        "--weight-quant",
        "int4",
        "--min-free-memory-mib",
        "256",
        "--reserve-memory-mib",
        "128",
        "--max-estimated-usage-ratio",
        "0.95",
        "--generation-memory-safety-factor",
        "1.25",
        "--kv-cache-quantization",
        "turboquant",
        "--kv-cache-quant-bits",
        "4",
        "--kv-cache-residual-len",
        "96",
        "--metrics-log-interval-seconds",
        "3.5",
    };

    const parsed = try parseServeArgs(arena_state.allocator(), &args);
    try std.testing.expect(parsed.default_enable_thinking == false);
    try std.testing.expectEqual(@as(?usize, 1024), parsed.default_max_completion_tokens);
    try std.testing.expectEqualStrings("deepseek", parsed.reasoning_format);
    try std.testing.expect(parsed.offload_vision);
    try std.testing.expectEqualStrings("int4", parsed.weight_quant);
    try std.testing.expectEqual(@as(?u64, 256), parsed.min_free_memory_mib);
    try std.testing.expectEqual(@as(?u64, 128), parsed.reserve_memory_mib);
    try std.testing.expectEqual(@as(?f64, 0.95), parsed.max_estimated_usage_ratio);
    try std.testing.expectEqual(@as(?f64, 1.25), parsed.generation_memory_safety_factor);
    try std.testing.expectEqualStrings("turboquant", parsed.kv_cache_quantization);
    try std.testing.expectEqual(@as(u8, 4), parsed.kv_cache_quant_bits);
    try std.testing.expectEqual(@as(usize, 96), parsed.kv_cache_residual_len);
    try std.testing.expectEqual(@as(f64, 3.5), parsed.metrics_log_interval_seconds);
}

test "serve parser defaults to direct generation" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const args = [_][]const u8{ "--model-dir", "model" };
    const parsed = try parseServeArgs(arena_state.allocator(), &args);
    try std.testing.expectEqual(@as(usize, 1), parsed.scheduler_max_batch_size);
    try std.testing.expectEqual(@as(f64, 10.0), parsed.metrics_log_interval_seconds);
}
