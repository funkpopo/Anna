const std = @import("std");
const types = @import("../runtime/types.zig");

pub fn errorPayload(allocator: std.mem.Allocator, err: types.AnnaEngineError) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("error");
    try jw.beginObject();
    try jw.objectField("message");
    try jw.write(err.message);
    try jw.objectField("type");
    try jw.write(err.error_type);
    try jw.objectField("code");
    if (err.code) |code| try jw.write(code) else try jw.write(null);
    try jw.endObject();
    try jw.endObject();
    return try out.toOwnedSlice();
}

pub fn sseErrorFrame(allocator: std.mem.Allocator, err: types.AnnaEngineError) ![]u8 {
    const payload = try errorPayload(allocator, err);
    defer allocator.free(payload);
    return try std.fmt.allocPrint(allocator, "event: error\ndata: {s}\n\n", .{payload});
}

pub fn sseDataFrame(allocator: std.mem.Allocator, payload: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "data: {s}\n\n", .{payload});
}

pub fn sseDoneFrame(allocator: std.mem.Allocator) ![]u8 {
    return try allocator.dupe(u8, "data: [DONE]\n\n");
}

pub fn chatResponsePayload(allocator: std.mem.Allocator, response_id: []const u8, created: i64, model: []const u8, result: types.TextGenerationResult) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("id");
    try jw.write(response_id);
    try jw.objectField("object");
    try jw.write("chat.completion");
    try jw.objectField("created");
    try jw.write(created);
    try jw.objectField("model");
    try jw.write(model);
    try jw.objectField("choices");
    try jw.beginArray();
    try jw.beginObject();
    try jw.objectField("index");
    try jw.write(@as(u32, 0));
    try jw.objectField("message");
    try writeChatMessage(&jw, result);
    try jw.objectField("finish_reason");
    try jw.write(result.finish_reason);
    try jw.endObject();
    try jw.endArray();
    try jw.objectField("usage");
    try writeUsage(&jw, result.prompt_tokens, result.completion_tokens);
    try jw.endObject();
    return try out.toOwnedSlice();
}

pub fn completionResponsePayload(allocator: std.mem.Allocator, response_id: []const u8, created: i64, model: []const u8, result: types.TextGenerationResult) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("id");
    try jw.write(response_id);
    try jw.objectField("object");
    try jw.write("text_completion");
    try jw.objectField("created");
    try jw.write(created);
    try jw.objectField("model");
    try jw.write(model);
    try jw.objectField("choices");
    try jw.beginArray();
    try jw.beginObject();
    try jw.objectField("index");
    try jw.write(@as(u32, 0));
    try jw.objectField("text");
    try jw.write(result.text);
    try jw.objectField("finish_reason");
    try jw.write(result.finish_reason);
    try jw.endObject();
    try jw.endArray();
    try jw.objectField("usage");
    try writeUsage(&jw, result.prompt_tokens, result.completion_tokens);
    try jw.endObject();
    return try out.toOwnedSlice();
}

pub fn chatChunkPayload(
    allocator: std.mem.Allocator,
    response_id: []const u8,
    created: i64,
    model: []const u8,
    role: ?[]const u8,
    event: types.StreamEvent,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("id");
    try jw.write(response_id);
    try jw.objectField("object");
    try jw.write("chat.completion.chunk");
    try jw.objectField("created");
    try jw.write(created);
    try jw.objectField("model");
    try jw.write(model);
    try jw.objectField("choices");
    try jw.beginArray();
    try jw.beginObject();
    try jw.objectField("index");
    try jw.write(@as(u32, 0));
    try jw.objectField("delta");
    try jw.beginObject();
    if (role) |value| {
        try jw.objectField("role");
        try jw.write(value);
    }
    if (event.text.len > 0) {
        try jw.objectField("content");
        try jw.write(event.text);
    }
    if (event.reasoning_text) |reasoning_text| {
        try jw.objectField("reasoning_content");
        try jw.write(reasoning_text);
    }
    if (event.tool_calls.len > 0) {
        try jw.objectField("tool_calls");
        try writeToolCalls(&jw, event.tool_calls);
    }
    try jw.endObject();
    try jw.objectField("finish_reason");
    if (event.finish_reason) |finish_reason| try jw.write(finish_reason) else try jw.write(null);
    try jw.endObject();
    try jw.endArray();
    try jw.endObject();
    return try out.toOwnedSlice();
}

pub fn completionChunkPayload(
    allocator: std.mem.Allocator,
    response_id: []const u8,
    created: i64,
    model: []const u8,
    text: []const u8,
    finish_reason: ?[]const u8,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("id");
    try jw.write(response_id);
    try jw.objectField("object");
    try jw.write("text_completion");
    try jw.objectField("created");
    try jw.write(created);
    try jw.objectField("model");
    try jw.write(model);
    try jw.objectField("choices");
    try jw.beginArray();
    try jw.beginObject();
    try jw.objectField("index");
    try jw.write(@as(u32, 0));
    try jw.objectField("text");
    try jw.write(text);
    try jw.objectField("finish_reason");
    if (finish_reason) |value| try jw.write(value) else try jw.write(null);
    try jw.endObject();
    try jw.endArray();
    try jw.endObject();
    return try out.toOwnedSlice();
}

fn writeChatMessage(jw: *std.json.Stringify, result: types.TextGenerationResult) !void {
    try jw.beginObject();
    try jw.objectField("role");
    try jw.write("assistant");
    try jw.objectField("content");
    if (result.tool_calls.len > 0 and result.text.len == 0) {
        try jw.write(null);
    } else {
        try jw.write(result.text);
    }
    if (result.reasoning_text) |reasoning_text| {
        try jw.objectField("reasoning_content");
        try jw.write(reasoning_text);
    }
    if (result.tool_calls.len > 0) {
        try jw.objectField("tool_calls");
        try writeToolCalls(jw, result.tool_calls);
    }
    try jw.endObject();
}

fn writeToolCalls(jw: *std.json.Stringify, tool_calls: []const types.ToolCall) !void {
    try jw.beginArray();
    for (tool_calls) |tool_call| {
        try jw.beginObject();
        try jw.objectField("id");
        if (tool_call.id) |id| try jw.write(id) else try jw.write(null);
        try jw.objectField("type");
        try jw.write("function");
        try jw.objectField("function");
        try jw.beginObject();
        try jw.objectField("name");
        try jw.write(tool_call.name);
        try jw.objectField("arguments");
        try jw.write(tool_call.arguments_json);
        try jw.endObject();
        try jw.endObject();
    }
    try jw.endArray();
}

fn writeUsage(jw: *std.json.Stringify, prompt_tokens: usize, completion_tokens: usize) !void {
    try jw.beginObject();
    try jw.objectField("prompt_tokens");
    try jw.write(prompt_tokens);
    try jw.objectField("completion_tokens");
    try jw.write(completion_tokens);
    try jw.objectField("total_tokens");
    try jw.write(prompt_tokens + completion_tokens);
    try jw.endObject();
}

test "chat completion returns openai tool calls payload" {
    const result: types.TextGenerationResult = .{
        .text = "",
        .reasoning_text = "先查天气。",
        .tool_calls = &.{
            .{
                .id = "call_123",
                .name = "get_weather",
                .arguments_json = "{\"location\":\"Shanghai\"}",
            },
        },
        .finish_reason = "tool_calls",
        .prompt_tokens = 4,
        .completion_tokens = 2,
    };
    const payload = try chatResponsePayload(std.testing.allocator, "chatcmpl-1", 123, "fake-model", result);
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"finish_reason\":\"tool_calls\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"content\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"reasoning_content\":\"先查天气。\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"name\":\"get_weather\"") != null);
}

test "sse error frame contains event and code" {
    const frame = try sseErrorFrame(std.testing.allocator, .{
        .message = "Insufficient free XPU memory before generation.",
        .status_code = 503,
        .error_type = "server_error",
        .code = "insufficient_device_memory",
    });
    defer std.testing.allocator.free(frame);

    try std.testing.expect(std.mem.indexOf(u8, frame, "event: error") != null);
    try std.testing.expect(std.mem.indexOf(u8, frame, "\"code\":\"insufficient_device_memory\"") != null);
}

test "chat chunk payload encodes reasoning delta" {
    const payload = try chatChunkPayload(std.testing.allocator, "chatcmpl-1", 123, "fake-model", "assistant", .{
        .reasoning_text = "先想一下",
    });
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"object\":\"chat.completion.chunk\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"role\":\"assistant\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"reasoning_content\":\"先想一下\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"finish_reason\":null") != null);
}

test "completion chunk payload encodes finish reason" {
    const payload = try completionChunkPayload(std.testing.allocator, "cmpl-1", 123, "fake-model", "", "length");
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"object\":\"text_completion\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"finish_reason\":\"length\"") != null);
}

test "sse done frame ends stream" {
    const frame = try sseDoneFrame(std.testing.allocator);
    defer std.testing.allocator.free(frame);

    try std.testing.expectEqualStrings("data: [DONE]\n\n", frame);
}
