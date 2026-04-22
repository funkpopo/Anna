const std = @import("std");
const loader = @import("../model/qwen3_loader.zig");
const model_engine = @import("../model/qwen3_engine.zig");
const tokenizer_mod = @import("../tokenizer/qwen3_tokenizer.zig");
const scheduler_mod = @import("scheduler.zig");
const sampler = @import("../sampling/sampler.zig");
const types = @import("types.zig");

const Session = struct {
    engine: model_engine.TokenEngine,
    pending_token: ?u32 = null,
    last_logits: ?[]f32 = null,
    generation_config: types.GenerationConfig,
    repetition_history: std.array_list.Managed(u32),

    fn init(allocator: std.mem.Allocator, engine: model_engine.TokenEngine, generation_config: types.GenerationConfig, prompt_tokens: []const u32) !Session {
        var repetition_history = std.array_list.Managed(u32).init(allocator);
        errdefer repetition_history.deinit();
        var seen = std.AutoHashMap(u32, void).init(allocator);
        defer seen.deinit();
        for (prompt_tokens) |token_id| {
            if ((try seen.getOrPut(token_id)).found_existing) continue;
            try repetition_history.append(token_id);
        }
        return .{
            .engine = engine,
            .generation_config = generation_config,
            .repetition_history = repetition_history,
        };
    }

    fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        if (self.last_logits) |logits| allocator.free(logits);
        self.engine.deinitStates();
        self.repetition_history.deinit();
        self.* = undefined;
    }
};

pub const NativeQwenTextService = struct {
    allocator: std.mem.Allocator,
    runtime: loader.QwenTextRuntime,
    tokenizer: tokenizer_mod.QwenTokenizer,
    model_id: []const u8,
    sessions: std.AutoHashMap(usize, Session),
    next_session_id: usize = 1,
    prng: std.Random.DefaultPrng,
    prefill_chunk_size: usize = 0,
    default_max_completion_tokens: usize = 256,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, model_id: []const u8) !NativeQwenTextService {
        return try loadWithOptions(allocator, io, model_dir, model_id, .{});
    }

    pub fn loadWithOptions(
        allocator: std.mem.Allocator,
        io: std.Io,
        model_dir: []const u8,
        model_id: []const u8,
        options: loader.LoadOptions,
    ) !NativeQwenTextService {
        return .{
            .allocator = allocator,
            .runtime = try loader.QwenTextRuntime.loadWithOptions(allocator, io, model_dir, options),
            .tokenizer = try tokenizer_mod.QwenTokenizer.loadFromModelDir(allocator, io, model_dir),
            .model_id = try allocator.dupe(u8, model_id),
            .sessions = std.AutoHashMap(usize, Session).init(allocator),
            .prng = std.Random.DefaultPrng.init(0x5eed1234),
        };
    }

    pub fn deinit(self: *NativeQwenTextService) void {
        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.sessions.deinit();
        self.allocator.free(self.model_id);
        self.tokenizer.deinit();
        self.runtime.deinit();
        self.* = undefined;
    }

    pub fn schedulerEngine(self: *NativeQwenTextService) scheduler_mod.Engine {
        return .{
            .ctx = self,
            .vtable = &.{
                .prefill_chunk_batch = prefillChunkBatch,
                .decode_batch = decodeBatch,
                .decode_tokens = decodeTokens,
                .is_eos_token = isEosToken,
                .release_state = releaseState,
            },
        };
    }

    pub fn generateTextAlloc(self: *NativeQwenTextService, prompt: []const u8, config: types.GenerationConfig) !types.TextGenerationResult {
        const prompt_tokens = try self.tokenizer.encodeAlloc(self.allocator, prompt);
        defer self.allocator.free(prompt_tokens);
        return try self.generatePromptTokensAlloc(prompt_tokens, config);
    }

    pub fn generateChatAlloc(
        self: *NativeQwenTextService,
        messages: []const tokenizer_mod.ChatMessage,
        config: types.GenerationConfig,
        enable_thinking: bool,
    ) !types.TextGenerationResult {
        const rendered = try self.tokenizer.renderMessagesAlloc(self.allocator, messages, true, enable_thinking);
        defer self.allocator.free(rendered);
        const raw = try self.generateTextAlloc(rendered, config);

        const split = self.tokenizer.splitAssistantReasoning(raw.text, enable_thinking);
        const extracted = try self.tokenizer.extractToolCallsAlloc(self.allocator, split.content);
        defer self.allocator.free(extracted.text);

        const final_text = try self.allocator.dupe(u8, extracted.text);
        self.allocator.free(raw.text);
        var finish_reason = raw.finish_reason;
        if (extracted.tool_calls.len > 0 and std.mem.eql(u8, raw.finish_reason, "stop")) {
            finish_reason = "tool_calls";
        }
        return .{
            .text = final_text,
            .finish_reason = finish_reason,
            .prompt_tokens = raw.prompt_tokens,
            .completion_tokens = raw.completion_tokens,
            .reasoning_text = if (split.reasoning) |reasoning| try self.allocator.dupe(u8, reasoning) else null,
            .tool_calls = extracted.tool_calls,
            .perf = raw.perf,
        };
    }

    pub fn generatePromptTokensAlloc(self: *NativeQwenTextService, prompt_tokens: []const u32, config: types.GenerationConfig) !types.TextGenerationResult {
        var request_config = config;
        if (request_config.max_new_tokens == null) request_config.max_new_tokens = self.default_max_completion_tokens;

        const prepared: scheduler_mod.PreparedInputs = .{
            .prompt_tokens = prompt_tokens,
            .generation_config = request_config,
        };
        const request: scheduler_mod.RequestSpec = .{
            .prepared = prepared,
            .config = request_config,
        };

        var scheduler = scheduler_mod.BatchScheduler{
            .allocator = self.allocator,
            .engine = self.schedulerEngine(),
            .prefill_chunk_size = self.prefill_chunk_size,
        };

        const results = try scheduler.run(&.{request});
        defer self.allocator.free(results);
        return results[0];
    }
};

fn prefillChunkBatch(
    ctx: *anyopaque,
    prompts: []const scheduler_mod.PreparedInputs,
    prior_states: ?[]const usize,
    start_idx: usize,
    end_idx: usize,
    allocator: std.mem.Allocator,
) !scheduler_mod.PrefillBatchResult {
    const self: *NativeQwenTextService = @ptrCast(@alignCast(ctx));
    const handles = try allocator.alloc(usize, prompts.len);
    errdefer allocator.free(handles);
    const next_tokens = try allocator.alloc(?u32, prompts.len);
    errdefer allocator.free(next_tokens);
    @memset(next_tokens, null);

    for (prompts, 0..) |prompt, idx| {
        const handle = if (prior_states) |existing| existing[idx] else try createSession(self, prompt);
        handles[idx] = handle;
        const session = self.sessions.getPtr(handle).?;
        if (end_idx > prompt.prompt_tokens.len) return error.InvalidPromptSlice;

        const token_slice = prompt.prompt_tokens[start_idx..end_idx];
        for (token_slice) |token_id| {
            if (session.last_logits) |previous| self.allocator.free(previous);
            session.last_logits = try session.engine.forwardTokenAlloc(token_id);
        }

        if (end_idx == prompt.prompt_tokens.len) {
            const logits = session.last_logits orelse return error.EmptyPrompt;
            var rng = self.prng.random();
            const sampled = try sampler.sampleNextToken(
                self.allocator,
                logits,
                session.repetition_history.items,
                session.generation_config.temperature,
                session.generation_config.top_p,
                session.generation_config.top_k,
                session.generation_config.repetition_penalty,
                &rng,
            );
            try session.repetition_history.append(sampled);
            session.pending_token = sampled;
            next_tokens[idx] = sampled;
        }
    }

    return .{ .state_handles = handles, .next_tokens = next_tokens };
}

fn decodeBatch(ctx: *anyopaque, state_handles: []const usize, allocator: std.mem.Allocator) !scheduler_mod.DecodeBatchResult {
    const self: *NativeQwenTextService = @ptrCast(@alignCast(ctx));
    const handles = try allocator.dupe(usize, state_handles);
    errdefer allocator.free(handles);
    const next_tokens = try allocator.alloc(?u32, state_handles.len);
    errdefer allocator.free(next_tokens);

    for (state_handles, 0..) |handle, idx| {
        const session = self.sessions.getPtr(handle) orelse return error.UnknownStateHandle;
        const pending = session.pending_token orelse {
            next_tokens[idx] = null;
            continue;
        };

        if (session.last_logits) |previous| self.allocator.free(previous);
        session.last_logits = try session.engine.forwardTokenAlloc(pending);
        const logits = session.last_logits.?;
        var rng = self.prng.random();
        const sampled = try sampler.sampleNextToken(
            self.allocator,
            logits,
            session.repetition_history.items,
            session.generation_config.temperature,
            session.generation_config.top_p,
            session.generation_config.top_k,
            session.generation_config.repetition_penalty,
            &rng,
        );
        try session.repetition_history.append(sampled);
        session.pending_token = sampled;
        next_tokens[idx] = sampled;
    }

    return .{ .state_handles = handles, .next_tokens = next_tokens };
}

fn isEosToken(ctx: *anyopaque, token_id: u32) bool {
    const self: *NativeQwenTextService = @ptrCast(@alignCast(ctx));
    return self.runtime.isEosToken(token_id) or self.tokenizer.isEosToken(token_id);
}

fn decodeTokens(ctx: *const anyopaque, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
    const self: *const NativeQwenTextService = @ptrCast(@alignCast(ctx));
    return try self.tokenizer.decodeAlloc(allocator, token_ids, false);
}

fn releaseState(ctx: *anyopaque, state_handle: usize) void {
    const self: *NativeQwenTextService = @ptrCast(@alignCast(ctx));
    const removed = self.sessions.fetchRemove(state_handle) orelse return;
    var session = removed.value;
    session.deinit(self.allocator);
}

fn createSession(self: *NativeQwenTextService, prompt: scheduler_mod.PreparedInputs) !usize {
    const config = prompt.generation_config orelse return error.MissingGenerationConfig;
    var session = try Session.init(
        self.allocator,
        try self.runtime.spawnEngine(),
        config,
        prompt.prompt_tokens,
    );
    errdefer session.deinit(self.allocator);
    const handle = self.next_session_id;
    self.next_session_id += 1;
    try self.sessions.put(handle, session);
    return handle;
}

test "native qwen text service generates from tiny tokenizer and runtime" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "text_config": {
        \\    "hidden_size": 2,
        \\    "intermediate_size": 4,
        \\    "num_hidden_layers": 0,
        \\    "num_attention_heads": 1,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 2,
        \\    "vocab_size": 3,
        \\    "tie_word_embeddings": true,
        \\    "eos_token_id": 2,
        \\    "layer_types": []
        \\  }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = config_json });

    const tokenizer_json =
        \\{
        \\  "added_tokens": [{"id":2,"content":"<|endoftext|>","special":true}],
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"A":0, "B":1, "<|endoftext|>":2},
        \\    "merges": []
        \\  }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.json", .data = tokenizer_json });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer_config.json", .data = "{}" });

    const header =
        \\{"model.language_model.embed_tokens.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]},"model.language_model.norm.weight":{"dtype":"F32","shape":[2],"data_offsets":[24,32]}}
    ;
    var bytes = std.array_list.Managed(u8).init(std.testing.allocator);
    defer bytes.deinit();
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try bytes.appendSlice(&len_buf);
    try bytes.appendSlice(header);
    const values = [_]f32{
        2.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
    };
    for (values) |value| {
        var raw: [4]u8 = undefined;
        std.mem.writeInt(u32, &raw, @bitCast(value), .little);
        try bytes.appendSlice(&raw);
    }
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = bytes.items });

    const model_dir = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(model_dir);
    var service = try NativeQwenTextService.load(std.testing.allocator, std.testing.io, model_dir, "tiny");
    defer service.deinit();

    const result = try service.generateTextAlloc("A", .{ .max_new_tokens = 1, .temperature = 0.0, .top_p = 1.0, .top_k = 0 });
    defer std.testing.allocator.free(result.text);
    try std.testing.expectEqualStrings("A", result.text);
    try std.testing.expectEqual(@as(usize, 1), result.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 1), result.completion_tokens);
}
