const std = @import("std");
const metrics_mod = @import("service_metrics.zig");
const streaming = @import("streaming.zig");
const types = @import("types.zig");

pub const PreparedInputs = struct {
    prompt_tokens: []const u32,
    generation_config: ?types.GenerationConfig = null,
};

pub const RequestSpec = struct {
    prepared: PreparedInputs,
    config: types.GenerationConfig,
};

pub const PrefillBatchResult = struct {
    state_handles: []usize,
    next_tokens: []?u32,
};

pub const DecodeBatchResult = struct {
    state_handles: []usize,
    next_tokens: []?u32,
};

pub const Engine = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        prefill_chunk_batch: *const fn (ctx: *anyopaque, prompts: []const PreparedInputs, prior_states: ?[]const usize, start_idx: usize, end_idx: usize, allocator: std.mem.Allocator) anyerror!PrefillBatchResult,
        decode_batch: *const fn (ctx: *anyopaque, state_handles: []const usize, allocator: std.mem.Allocator) anyerror!DecodeBatchResult,
        decode_tokens: streaming.DecodeTokensFn,
        is_eos_token: *const fn (ctx: *anyopaque, token_id: u32) bool,
        release_state: *const fn (ctx: *anyopaque, state_handle: usize) void,
    };
};

pub const BatchScheduler = struct {
    allocator: std.mem.Allocator,
    engine: Engine,
    prefill_chunk_size: usize = 0,
    metrics: ?*metrics_mod.AnnaServiceMetrics = null,

    pub fn run(self: *BatchScheduler, requests: []const RequestSpec) ![]types.TextGenerationResult {
        const results = try self.allocator.alloc(types.TextGenerationResult, requests.len);
        errdefer self.allocator.free(results);

        if (self.metrics) |metrics| {
            for (requests) |_| metrics.recordRequestSubmitted(true);
            metrics.recordRequestsStartedFromQueue(requests.len);
        }

        var handled = std.array_list.Managed(bool).init(self.allocator);
        defer handled.deinit();
        try handled.resize(requests.len);
        @memset(handled.items, false);

        var i: usize = 0;
        while (i < requests.len) : (i += 1) {
            if (handled.items[i]) continue;
            const prompt_len = requests[i].prepared.prompt_tokens.len;
            var group_indexes = std.array_list.Managed(usize).init(self.allocator);
            defer group_indexes.deinit();

            try group_indexes.append(i);
            handled.items[i] = true;
            var j = i + 1;
            while (j < requests.len) : (j += 1) {
                if (!handled.items[j] and requests[j].prepared.prompt_tokens.len == prompt_len) {
                    handled.items[j] = true;
                    try group_indexes.append(j);
                }
            }

            try self.runSameLengthGroup(requests, group_indexes.items, results);
        }

        return results;
    }

    fn runSameLengthGroup(self: *BatchScheduler, requests: []const RequestSpec, group_indexes: []const usize, results: []types.TextGenerationResult) !void {
        var arena_state = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var group_prompts = try arena.alloc(PreparedInputs, group_indexes.len);
        for (group_indexes, 0..) |source_index, idx| {
            group_prompts[idx] = requests[source_index].prepared;
        }

        const prompt_length = group_prompts[0].prompt_tokens.len;
        var state_handles: ?[]const usize = null;
        var prefill_result: ?PrefillBatchResult = null;

        if (self.prefill_chunk_size > 0 and prompt_length > self.prefill_chunk_size) {
            var start_idx: usize = 0;
            while (start_idx < prompt_length) : (start_idx += self.prefill_chunk_size) {
                const end_idx = @min(prompt_length, start_idx + self.prefill_chunk_size);
                prefill_result = try self.engine.vtable.prefill_chunk_batch(self.engine.ctx, group_prompts, state_handles, start_idx, end_idx, arena);
                state_handles = prefill_result.?.state_handles;
            }
        } else {
            prefill_result = try self.engine.vtable.prefill_chunk_batch(self.engine.ctx, group_prompts, state_handles, 0, prompt_length, arena);
        }

        if (self.metrics) |metrics| {
            var total_prompt_tokens: usize = 0;
            for (group_indexes) |request_index| total_prompt_tokens += requests[request_index].prepared.prompt_tokens.len;
            metrics.recordPromptTokens(total_prompt_tokens);
        }

        const decoder: streaming.TokenDecoder = .{
            .ctx = self.engine.ctx,
            .decode_tokens = self.engine.vtable.decode_tokens,
        };
        const state_slice = prefill_result.?.state_handles;
        const next_tokens = prefill_result.?.next_tokens;

        var active = try arena.alloc(bool, group_indexes.len);
        @memset(active, true);
        var assemblers = try arena.alloc(streaming.IncrementalTextAssembler, group_indexes.len);

        for (group_indexes, 0..) |request_index, idx| {
            assemblers[idx] = streaming.IncrementalTextAssembler.init(arena, decoder, requests[request_index].config.stop_strings);
        }
        defer {
            for (assemblers) |*assembler| assembler.deinit();
        }

        var texts = try arena.alloc(std.array_list.Managed(u8), group_indexes.len);
        for (texts) |*text| text.* = std.array_list.Managed(u8).init(arena);

        var completion_tokens = try arena.alloc(usize, group_indexes.len);
        @memset(completion_tokens, 0);

        var decode_states = state_slice;
        var remaining_active = group_indexes.len;
        var current_tokens = next_tokens;

        while (remaining_active > 0) {
            var next_decode_input = std.array_list.Managed(usize).init(arena);
            defer next_decode_input.deinit();
            var active_map = std.array_list.Managed(usize).init(arena);
            defer active_map.deinit();

            for (group_indexes, 0..) |request_index, local_idx| {
                if (!active[local_idx]) continue;
                const token = current_tokens[local_idx] orelse {
                    active[local_idx] = false;
                    remaining_active -= 1;
                    continue;
                };
                if (self.engine.vtable.is_eos_token(self.engine.ctx, token)) {
                    active[local_idx] = false;
                    remaining_active -= 1;
                    continue;
                }

                completion_tokens[local_idx] += 1;
                if (self.metrics) |metrics| metrics.recordGenerationTokens(1);

                const feed = try assemblers[local_idx].feedToken(token);
                if (feed.delta.len > 0) try texts[local_idx].appendSlice(feed.delta);
                if (feed.stopped or limitReached(requests[request_index].config.max_new_tokens, completion_tokens[local_idx])) {
                    active[local_idx] = false;
                    remaining_active -= 1;
                    continue;
                }

                try next_decode_input.append(decode_states[local_idx]);
                try active_map.append(local_idx);
            }

            if (next_decode_input.items.len == 0) break;
            const decode_result = try self.engine.vtable.decode_batch(self.engine.ctx, next_decode_input.items, arena);
            for (decode_result.state_handles, 0..) |state_handle, idx| {
                const local_idx = active_map.items[idx];
                decode_states[local_idx] = state_handle;
            }

            var next_tokens_full = try arena.alloc(?u32, group_indexes.len);
            @memset(next_tokens_full, null);
            for (decode_result.next_tokens, 0..) |token, idx| {
                const local_idx = active_map.items[idx];
                next_tokens_full[local_idx] = token;
            }
            current_tokens = next_tokens_full;
        }

        for (group_indexes, 0..) |request_index, local_idx| {
            const flush = try assemblers[local_idx].flush();
            if (flush.delta.len > 0) try texts[local_idx].appendSlice(flush.delta);

            const owned_text = try self.allocator.dupe(u8, texts[local_idx].items);
            results[request_index] = .{
                .text = owned_text,
                .finish_reason = if (limitReached(requests[request_index].config.max_new_tokens, completion_tokens[local_idx])) "length" else "stop",
                .prompt_tokens = requests[request_index].prepared.prompt_tokens.len,
                .completion_tokens = completion_tokens[local_idx],
            };
            if (self.metrics) |metrics| metrics.recordRequestFinished(true);
        }
        for (state_slice) |handle| self.engine.vtable.release_state(self.engine.ctx, handle);
    }
};

fn limitReached(limit: ?usize, current: usize) bool {
    return if (limit) |value| current >= value else false;
}

test "scheduler batches same length requests" {
    const Harness = struct {
        prefill_batch_sizes: std.array_list.Managed(usize),
        decode_batch_sizes: std.array_list.Managed(usize),

        fn init(allocator: std.mem.Allocator) @This() {
            return .{
                .prefill_batch_sizes = std.array_list.Managed(usize).init(allocator),
                .decode_batch_sizes = std.array_list.Managed(usize).init(allocator),
            };
        }

        fn deinit(self: *@This()) void {
            self.prefill_batch_sizes.deinit();
            self.decode_batch_sizes.deinit();
        }

        fn prefill(ctx: *anyopaque, prompts: []const PreparedInputs, _: ?[]const usize, start_idx: usize, end_idx: usize, allocator: std.mem.Allocator) !PrefillBatchResult {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            _ = start_idx;
            _ = end_idx;
            try self.prefill_batch_sizes.append(prompts.len);
            const handles = try allocator.alloc(usize, prompts.len);
            const tokens = try allocator.alloc(?u32, prompts.len);
            for (prompts, 0..) |_, idx| {
                handles[idx] = idx + 1;
                tokens[idx] = if (idx == 0) 1 else 2;
            }
            return .{ .state_handles = handles, .next_tokens = tokens };
        }

        fn decode(ctx: *anyopaque, states: []const usize, allocator: std.mem.Allocator) !DecodeBatchResult {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            try self.decode_batch_sizes.append(states.len);
            const handles = try allocator.dupe(usize, states);
            const tokens = try allocator.alloc(?u32, states.len);
            for (tokens) |*token| token.* = 9;
            return .{ .state_handles = handles, .next_tokens = tokens };
        }

        fn decodeTokens(_: *const anyopaque, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
            var buffer = std.array_list.Managed(u8).init(allocator);
            for (token_ids) |token| {
                try buffer.appendSlice(switch (token) {
                    1 => "A",
                    2 => "B",
                    else => "",
                });
            }
            return buffer.toOwnedSlice();
        }

        fn isEos(_: *anyopaque, token_id: u32) bool {
            return token_id == 9;
        }

        fn release(_: *anyopaque, _: usize) void {}
    };

    var harness = Harness.init(std.testing.allocator);
    defer harness.deinit();
    var metrics = metrics_mod.AnnaServiceMetrics{};
    var batch = BatchScheduler{
        .allocator = std.testing.allocator,
        .engine = .{
            .ctx = &harness,
            .vtable = &.{
                .prefill_chunk_batch = Harness.prefill,
                .decode_batch = Harness.decode,
                .decode_tokens = Harness.decodeTokens,
                .is_eos_token = Harness.isEos,
                .release_state = Harness.release,
            },
        },
        .metrics = &metrics,
    };

    const requests = [_]RequestSpec{
        .{ .prepared = .{ .prompt_tokens = &.{ 4, 5 } }, .config = .{ .max_new_tokens = 2, .temperature = 0.0, .top_p = 1.0, .top_k = 0 } },
        .{ .prepared = .{ .prompt_tokens = &.{ 6, 7 } }, .config = .{ .max_new_tokens = 2, .temperature = 0.0, .top_p = 1.0, .top_k = 0 } },
    };

    const results = try batch.run(&requests);
    defer {
        for (results) |result| std.testing.allocator.free(result.text);
        std.testing.allocator.free(results);
    }

    try std.testing.expectEqualStrings("A", results[0].text);
    try std.testing.expectEqualStrings("B", results[1].text);
    try std.testing.expectEqual(@as(usize, 1), harness.prefill_batch_sizes.items.len);
    try std.testing.expectEqual(@as(usize, 2), harness.prefill_batch_sizes.items[0]);
    try std.testing.expectEqual(@as(usize, 1), harness.decode_batch_sizes.items.len);
    try std.testing.expectEqual(@as(usize, 2), harness.decode_batch_sizes.items[0]);

    const snapshot = metrics.snapshot();
    try std.testing.expectEqual(@as(usize, 2), snapshot.requests_started_total);
    try std.testing.expectEqual(@as(usize, 2), snapshot.requests_completed_total);
    try std.testing.expectEqual(@as(usize, 0), snapshot.requests_failed_total);
    try std.testing.expectEqual(@as(usize, 4), snapshot.prompt_tokens_total);
    try std.testing.expectEqual(@as(usize, 2), snapshot.generation_tokens_total);
}
