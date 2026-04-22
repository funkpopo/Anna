const std = @import("std");
const app = @import("../app.zig");
const openai = @import("../api/openai.zig");
const config_mod = @import("../core/config.zig");
const model_path = @import("../core/model_path.zig");
const tokenizer_mod = @import("../tokenizer/qwen3_tokenizer.zig");
const qwen_text_service = @import("qwen_text_service.zig");
const streaming = @import("streaming.zig");
const types = @import("types.zig");

const max_request_body_bytes: usize = 64 << 20;
const json_content_type = "application/json; charset=utf-8";
const html_content_type = "text/html; charset=utf-8";
const sse_content_type = "text/event-stream; charset=utf-8";
const cors_allow_headers = "authorization, content-type";
const cors_allow_origin = "*";

const Route = enum {
    openapi,
    docs,
    healthz,
    models,
    chat_completions,
    completions,
    audio_speech,
};

const ChatRequest = struct {
    model: ?[]const u8 = null,
    messages: []const tokenizer_mod.ChatMessage,
    config: types.GenerationConfig,
    enable_thinking: bool,
    stream: bool = true,
};

const CompletionRequest = struct {
    model: ?[]const u8 = null,
    prompt: []const u8,
    config: types.GenerationConfig,
    stream: bool = true,
};

const CommonRequestOptions = struct {
    stream: bool = true,
};

const CompletionStreamContext = struct {
    allocator: std.mem.Allocator,
    response: *std.http.BodyWriter,
    response_id: []const u8,
    created: i64,
    model: []const u8,
};

const ChatStreamEmitter = struct {
    allocator: std.mem.Allocator,
    tokenizer: *const tokenizer_mod.QwenTokenizer,
    response: *std.http.BodyWriter,
    response_id: []const u8,
    created: i64,
    model: []const u8,
    enable_thinking: bool,
    raw_text: std.array_list.Managed(u8),
    pending: std.array_list.Managed(u8),
    in_reasoning: bool,
    tool_call_mode: bool = false,

    fn init(
        allocator: std.mem.Allocator,
        tokenizer: *const tokenizer_mod.QwenTokenizer,
        response: *std.http.BodyWriter,
        response_id: []const u8,
        created: i64,
        model: []const u8,
        enable_thinking: bool,
    ) ChatStreamEmitter {
        return .{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .response = response,
            .response_id = response_id,
            .created = created,
            .model = model,
            .enable_thinking = enable_thinking,
            .raw_text = std.array_list.Managed(u8).init(allocator),
            .pending = std.array_list.Managed(u8).init(allocator),
            .in_reasoning = enable_thinking,
        };
    }

    fn deinit(self: *ChatStreamEmitter) void {
        self.raw_text.deinit();
        self.pending.deinit();
    }

    fn onRawDelta(ctx: *anyopaque, delta: []const u8) !void {
        const self: *ChatStreamEmitter = @ptrCast(@alignCast(ctx));
        try self.feedRawDelta(delta);
    }

    fn feedRawDelta(self: *ChatStreamEmitter, delta: []const u8) !void {
        if (delta.len == 0) return;
        try self.raw_text.appendSlice(delta);
        if (self.tool_call_mode) return;
        try self.pending.appendSlice(delta);
        if (self.in_reasoning) {
            try self.processReasoning();
        } else {
            try self.processContent();
        }
    }

    fn finish(self: *ChatStreamEmitter, allocator: std.mem.Allocator) !struct {
        finish_reason: []const u8,
        tool_calls: []const types.ToolCall,
    } {
        if (!self.tool_call_mode) {
            if (self.in_reasoning) {
                if (self.pending.items.len > 0) {
                    try self.emitChunk(.{ .reasoning_text = self.pending.items });
                    self.pending.clearRetainingCapacity();
                }
            } else {
                if (std.mem.indexOf(u8, self.pending.items, "<tool_call>")) |index| {
                    if (index > 0) try self.emitChunk(.{ .text = self.pending.items[0..index] });
                    self.tool_call_mode = true;
                    self.pending.clearRetainingCapacity();
                } else if (self.pending.items.len > 0) {
                    try self.emitChunk(.{ .text = self.pending.items });
                    self.pending.clearRetainingCapacity();
                }
            }
        }

        const split = self.tokenizer.splitAssistantReasoning(self.raw_text.items, self.enable_thinking);
        const extracted = try self.tokenizer.extractToolCallsAlloc(allocator, split.content);
        allocator.free(extracted.text);
        return .{
            .finish_reason = if (extracted.tool_calls.len > 0) "tool_calls" else "stop",
            .tool_calls = extracted.tool_calls,
        };
    }

    fn processReasoning(self: *ChatStreamEmitter) !void {
        if (std.mem.indexOf(u8, self.pending.items, "</think>")) |index| {
            if (index > 0) try self.emitChunk(.{ .reasoning_text = self.pending.items[0..index] });
            const after_close = index + "</think>".len;
            const remaining = std.mem.trimStart(u8, self.pending.items[after_close..], "\r\n");
            self.pending.clearRetainingCapacity();
            if (remaining.len > 0) try self.pending.appendSlice(remaining);
            self.in_reasoning = false;
            if (self.pending.items.len > 0) try self.processContent();
            return;
        }

        const hold_back = "</think>".len - 1;
        const safe_len = utf8BoundaryBefore(self.pending.items, self.pending.items.len -| hold_back);
        if (safe_len > 0) {
            try self.emitChunk(.{ .reasoning_text = self.pending.items[0..safe_len] });
            shiftLeft(&self.pending, safe_len);
        }
    }

    fn processContent(self: *ChatStreamEmitter) !void {
        if (std.mem.indexOf(u8, self.pending.items, "<tool_call>")) |index| {
            if (index > 0) try self.emitChunk(.{ .text = self.pending.items[0..index] });
            self.tool_call_mode = true;
            self.pending.clearRetainingCapacity();
            return;
        }

        const hold_back = "<tool_call>".len - 1;
        const safe_len = utf8BoundaryBefore(self.pending.items, self.pending.items.len -| hold_back);
        if (safe_len > 0) {
            try self.emitChunk(.{ .text = self.pending.items[0..safe_len] });
            shiftLeft(&self.pending, safe_len);
        }
    }

    fn emitChunk(self: *ChatStreamEmitter, event: types.StreamEvent) !void {
        if (event.text.len == 0 and event.reasoning_text == null and event.tool_calls.len == 0 and event.finish_reason == null) return;
        const payload = try openai.chatChunkPayload(self.allocator, self.response_id, self.created, self.model, null, event);
        defer self.allocator.free(payload);
        try writeSsePayload(self.allocator, self.response, payload);
    }
};

pub const AnnaHttpServer = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    settings: config_mod.ServeSettings,
    service: qwen_text_service.NativeQwenTextService,
    service_mutex: std.Io.Mutex = .init,
    next_request_id: u64 = 1,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        settings: config_mod.ServeSettings,
    ) !AnnaHttpServer {
        const resolved_dir = try model_path.resolveModelDir(allocator, io, settings.model_dir);
        defer allocator.free(resolved_dir);

        const resolved_name = try model_path.resolveModelName(allocator, settings.model_id, resolved_dir);
        defer allocator.free(resolved_name);

        var service = try qwen_text_service.NativeQwenTextService.loadWithOptions(
            allocator,
            io,
            resolved_dir,
            resolved_name,
            .{ .backend = settings.backend },
        );
        errdefer service.deinit();

        service.prefill_chunk_size = settings.prefill_chunk_size;
        if (settings.default_max_completion_tokens) |value| {
            service.default_max_completion_tokens = value;
        }

        return .{
            .allocator = allocator,
            .io = io,
            .settings = settings,
            .service = service,
        };
    }

    pub fn deinit(self: *AnnaHttpServer) void {
        self.service.deinit();
        self.* = undefined;
    }

    pub fn serve(self: *AnnaHttpServer) !void {
        const address = try std.Io.net.IpAddress.parse(self.settings.host, self.settings.port);
        var tcp_server = try address.listen(self.io, .{ .reuse_address = true });
        defer tcp_server.deinit(self.io);
        var group: std.Io.Group = .init;
        defer group.cancel(self.io);

        try self.writeStartupAnnouncement();

        while (true) {
            var stream = try tcp_server.accept(self.io);
            group.concurrent(self.io, serveConnectionTask, .{ self, stream }) catch |err| {
                try self.writeServerLog("unable to spawn connection task: {s}\n", .{@errorName(err)});
                stream.close(self.io);
            };
        }
    }

    fn writeStartupAnnouncement(self: *AnnaHttpServer) !void {
        var stdout_buffer: [4096]u8 = undefined;
        var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), self.io, &stdout_buffer);
        const writer = &stdout_file_writer.interface;
        try app.writeRouteAnnouncement(writer, app.defaultRoutes(), self.settings.host, self.settings.port);
        try writer.print("Model id: {s}\n", .{self.service.model_id});
        try writer.print("Backend:  {s}\n", .{@tagName(self.settings.backend)});
        try writer.flush();
    }

    fn writeServerLog(self: *AnnaHttpServer, comptime fmt: []const u8, args: anytype) !void {
        var stderr_buffer: [2048]u8 = undefined;
        var stderr_file_writer: std.Io.File.Writer = .init(.stderr(), self.io, &stderr_buffer);
        const writer = &stderr_file_writer.interface;
        try writer.print(fmt, args);
        try writer.flush();
    }

    fn serveConnection(self: *AnnaHttpServer, stream: std.Io.net.Stream) !void {
        defer {
            var copy = stream;
            copy.close(self.io);
        }

        var send_buffer: [4096]u8 = undefined;
        var recv_buffer: [8192]u8 = undefined;
        var connection_reader = stream.reader(self.io, &recv_buffer);
        var connection_writer = stream.writer(self.io, &send_buffer);
        var server: std.http.Server = .init(&connection_reader.interface, &connection_writer.interface);

        var request = server.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return err,
        };

        try self.handleRequest(&request);
    }

    fn serveConnectionTask(self: *AnnaHttpServer, stream: std.Io.net.Stream) void {
        self.serveConnection(stream) catch |err| {
            self.writeServerLog("request handling failed: {s}\n", .{@errorName(err)}) catch {};
        };
    }

    fn handleRequest(self: *AnnaHttpServer, request: *std.http.Server.Request) !void {
        const route = routeFromTarget(request.head.target) orelse {
            return self.respondRequestError(
                request,
                .not_found,
                "route_not_found",
                "The requested route does not exist on this Anna native service.",
                null,
            );
        };

        if (request.head.method == .OPTIONS) {
            return self.respond(request, .no_content, "", "text/plain; charset=utf-8", allowMethods(route), true, null);
        }
        if (!routeAllowsMethod(route, request.head.method)) {
            return self.respondRequestError(
                request,
                .method_not_allowed,
                "method_not_allowed",
                "The HTTP method is not allowed for this route.",
                allowMethods(route),
            );
        }

        var arena_state = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        switch (route) {
            .healthz => {
                const body = try healthzPayload(arena, self.service.model_id, self.settings.backend);
                return self.respondJson(request, .ok, body, allowMethods(route));
            },
            .models => {
                const body = try modelsPayload(arena, self.service.model_id, self.createdSeconds());
                return self.respondJson(request, .ok, body, allowMethods(route));
            },
            .openapi => {
                const body = try openapiPayload(arena, self.settings.host, self.settings.port, self.service.model_id);
                return self.respondJson(request, .ok, body, allowMethods(route));
            },
            .docs => {
                const body = try docsHtmlPayload(arena, self.settings.host, self.settings.port, self.service.model_id);
                return self.respond(request, .ok, body, html_content_type, allowMethods(route), true, null);
            },
            .audio_speech => {
                return self.respondRequestError(
                    request,
                    .not_implemented,
                    "speech_not_implemented",
                    "Native /v1/audio/speech is not implemented in the Zig runtime yet.",
                    allowMethods(route),
                );
            },
            .completions => return try self.handleCompletions(request, arena),
            .chat_completions => return try self.handleChatCompletions(request, arena),
        }
    }

    fn handleCompletions(self: *AnnaHttpServer, request: *std.http.Server.Request, arena: std.mem.Allocator) !void {
        const body = readJsonRequestBody(arena, request) catch |err| {
            return self.respondRequestError(request, requestErrorStatus(err), requestErrorCode(err), requestErrorMessage(err), allowMethods(.completions));
        };

        const parsed = parseCompletionRequest(arena, body) catch |err| {
            return self.respondRequestError(request, requestErrorStatus(err), requestErrorCode(err), requestErrorMessage(err), allowMethods(.completions));
        };

        if (self.modelSelectionError(parsed.model)) |engine_error| {
            return self.respondAnnaError(request, engine_error, allowMethods(.completions));
        }

        if (parsed.stream) {
            return try self.handleCompletionsStreaming(request, arena, parsed);
        }

        self.service_mutex.lockUncancelable(self.io);
        defer self.service_mutex.unlock(self.io);
        const result = self.service.generateTextAlloc(parsed.prompt, parsed.config) catch |err| {
            const engine_error = try generationError(arena, err);
            return self.respondAnnaError(request, engine_error, allowMethods(.completions));
        };
        defer self.freeGenerationResult(result);

        const response_id = try self.nextResponseId(arena, "cmpl-native");
        const payload = try openai.completionResponsePayload(arena, response_id, self.createdSeconds(), self.service.model_id, result);
        try self.respondJson(request, .ok, payload, allowMethods(.completions));
    }

    fn handleChatCompletions(self: *AnnaHttpServer, request: *std.http.Server.Request, arena: std.mem.Allocator) !void {
        const body = readJsonRequestBody(arena, request) catch |err| {
            return self.respondRequestError(request, requestErrorStatus(err), requestErrorCode(err), requestErrorMessage(err), allowMethods(.chat_completions));
        };

        const parsed = parseChatRequest(arena, body, self.settings.default_enable_thinking) catch |err| {
            return self.respondRequestError(request, requestErrorStatus(err), requestErrorCode(err), requestErrorMessage(err), allowMethods(.chat_completions));
        };

        if (self.modelSelectionError(parsed.model)) |engine_error| {
            return self.respondAnnaError(request, engine_error, allowMethods(.chat_completions));
        }

        if (parsed.stream) {
            return try self.handleChatCompletionsStreaming(request, arena, parsed);
        }

        self.service_mutex.lockUncancelable(self.io);
        defer self.service_mutex.unlock(self.io);
        const result = self.service.generateChatAlloc(parsed.messages, parsed.config, parsed.enable_thinking) catch |err| {
            const engine_error = try generationError(arena, err);
            return self.respondAnnaError(request, engine_error, allowMethods(.chat_completions));
        };
        defer self.freeGenerationResult(result);

        const response_id = try self.nextResponseId(arena, "chatcmpl-native");
        const payload = try openai.chatResponsePayload(arena, response_id, self.createdSeconds(), self.service.model_id, result);
        try self.respondJson(request, .ok, payload, allowMethods(.chat_completions));
    }

    fn handleCompletionsStreaming(
        self: *AnnaHttpServer,
        request: *std.http.Server.Request,
        arena: std.mem.Allocator,
        parsed: CompletionRequest,
    ) !void {
        const response_id = try self.nextResponseId(arena, "cmpl-native");
        const created = self.createdSeconds();
        var send_buffer: [4096]u8 = undefined;
        var response = try self.respondEventStream(request, &send_buffer, allowMethods(.completions));
        var ctx = CompletionStreamContext{
            .allocator = self.allocator,
            .response = &response,
            .response_id = response_id,
            .created = created,
            .model = self.service.model_id,
        };

        self.service_mutex.lockUncancelable(self.io);
        defer self.service_mutex.unlock(self.io);
        const summary = self.service.generateTextStream(parsed.prompt, parsed.config, &ctx, completionStreamOnDelta) catch |err| {
            const engine_error = try generationError(self.allocator, err);
            try self.respondStreamError(&response, engine_error);
            return;
        };

        const final_payload = try openai.completionChunkPayload(self.allocator, response_id, created, self.service.model_id, "", summary.finish_reason);
        defer self.allocator.free(final_payload);
        try writeSsePayload(self.allocator, &response, final_payload);
        try self.finishSse(&response);
    }

    fn handleChatCompletionsStreaming(
        self: *AnnaHttpServer,
        request: *std.http.Server.Request,
        arena: std.mem.Allocator,
        parsed: ChatRequest,
    ) !void {
        const response_id = try self.nextResponseId(arena, "chatcmpl-native");
        const created = self.createdSeconds();
        var send_buffer: [4096]u8 = undefined;
        var response = try self.respondEventStream(request, &send_buffer, allowMethods(.chat_completions));

        const role_payload = try openai.chatChunkPayload(self.allocator, response_id, created, self.service.model_id, "assistant", .{});
        defer self.allocator.free(role_payload);
        try writeSsePayload(self.allocator, &response, role_payload);

        var emitter = ChatStreamEmitter.init(
            self.allocator,
            &self.service.tokenizer,
            &response,
            response_id,
            created,
            self.service.model_id,
            parsed.enable_thinking,
        );
        defer emitter.deinit();

        self.service_mutex.lockUncancelable(self.io);
        defer self.service_mutex.unlock(self.io);
        const summary = self.service.generateChatRawStream(
            parsed.messages,
            parsed.config,
            parsed.enable_thinking,
            &emitter,
            ChatStreamEmitter.onRawDelta,
        ) catch |err| {
            const engine_error = try generationError(self.allocator, err);
            try self.respondStreamError(&response, engine_error);
            return;
        };

        const final = try emitter.finish(self.allocator);
        defer {
            for (final.tool_calls) |tool_call| {
                self.allocator.free(tool_call.name);
                self.allocator.free(tool_call.arguments_json);
            }
            self.allocator.free(final.tool_calls);
        }

        if (final.tool_calls.len > 0) {
            const tool_call_payload = try openai.chatChunkPayload(self.allocator, response_id, created, self.service.model_id, null, .{
                .tool_calls = final.tool_calls,
            });
            defer self.allocator.free(tool_call_payload);
            try writeSsePayload(self.allocator, &response, tool_call_payload);
        }

        const finish_reason = if (std.mem.eql(u8, summary.finish_reason, "length")) "length" else final.finish_reason;
        const final_payload = try openai.chatChunkPayload(self.allocator, response_id, created, self.service.model_id, null, .{
            .finish_reason = finish_reason,
        });
        defer self.allocator.free(final_payload);
        try writeSsePayload(self.allocator, &response, final_payload);
        try self.finishSse(&response);
    }

    fn respondJson(
        self: *AnnaHttpServer,
        request: *std.http.Server.Request,
        status: std.http.Status,
        body: []const u8,
        allow: []const u8,
    ) !void {
        return self.respond(request, status, body, json_content_type, allow, true, null);
    }

    fn respondEventStream(
        self: *AnnaHttpServer,
        request: *std.http.Server.Request,
        send_buffer: []u8,
        allow: []const u8,
    ) !std.http.BodyWriter {
        var headers = [_]std.http.Header{
            .{ .name = "content-type", .value = sse_content_type },
            .{ .name = "cache-control", .value = "no-cache, no-transform" },
            .{ .name = "x-accel-buffering", .value = "no" },
            .{ .name = "access-control-allow-origin", .value = cors_allow_origin },
            .{ .name = "access-control-allow-headers", .value = cors_allow_headers },
            .{ .name = "access-control-allow-methods", .value = allow },
        };
        _ = self;
        return try request.respondStreaming(send_buffer, .{
            .respond_options = .{
                .status = .ok,
                .keep_alive = false,
                .extra_headers = &headers,
            },
        });
    }

    fn respondRequestError(
        self: *AnnaHttpServer,
        request: *std.http.Server.Request,
        status: std.http.Status,
        code: []const u8,
        message: []const u8,
        allow: ?[]const u8,
    ) !void {
        return self.respondAnnaError(request, .{
            .message = message,
            .status_code = @intFromEnum(status),
            .error_type = if (@intFromEnum(status) >= 500) "server_error" else "invalid_request_error",
            .code = code,
        }, allow);
    }

    fn respondAnnaError(
        self: *AnnaHttpServer,
        request: *std.http.Server.Request,
        engine_error: types.AnnaEngineError,
        allow: ?[]const u8,
    ) !void {
        var arena_state = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const payload = try openai.errorPayload(arena, engine_error);
        try self.respond(
            request,
            @enumFromInt(engine_error.status_code),
            payload,
            json_content_type,
            allow orelse allowMethods(.openapi),
            true,
            allow,
        );
    }

    fn respondStreamError(self: *AnnaHttpServer, response: *std.http.BodyWriter, engine_error: types.AnnaEngineError) !void {
        const frame = try openai.sseErrorFrame(self.allocator, engine_error);
        defer self.allocator.free(frame);
        try response.writer.writeAll(frame);
        try response.end();
    }

    fn finishSse(self: *AnnaHttpServer, response: *std.http.BodyWriter) !void {
        const frame = try openai.sseDoneFrame(self.allocator);
        defer self.allocator.free(frame);
        try response.writer.writeAll(frame);
        try response.end();
    }

    fn respond(
        self: *AnnaHttpServer,
        request: *std.http.Server.Request,
        status: std.http.Status,
        body: []const u8,
        content_type: []const u8,
        cors_methods: []const u8,
        include_cors: bool,
        allow_override: ?[]const u8,
    ) !void {
        var headers: [5]std.http.Header = undefined;
        var count: usize = 0;

        headers[count] = .{ .name = "content-type", .value = content_type };
        count += 1;
        if (include_cors) {
            headers[count] = .{ .name = "access-control-allow-origin", .value = cors_allow_origin };
            count += 1;
            headers[count] = .{ .name = "access-control-allow-headers", .value = cors_allow_headers };
            count += 1;
            headers[count] = .{ .name = "access-control-allow-methods", .value = cors_methods };
            count += 1;
        }
        if (allow_override) |allow| {
            headers[count] = .{ .name = "allow", .value = allow };
            count += 1;
        }

        _ = self;
        try request.respond(body, .{
            .status = status,
            .keep_alive = false,
            .extra_headers = headers[0..count],
        });
    }

    fn modelSelectionError(self: *AnnaHttpServer, model: ?[]const u8) ?types.AnnaEngineError {
        const requested = model orelse return null;
        if (std.mem.eql(u8, requested, self.service.model_id)) return null;
        return .{
            .message = "The requested model does not match the model loaded by this Anna native service.",
            .status_code = 404,
            .error_type = "invalid_request_error",
            .code = "model_not_found",
        };
    }

    fn nextResponseId(self: *AnnaHttpServer, allocator: std.mem.Allocator, prefix: []const u8) ![]const u8 {
        const current = self.next_request_id;
        self.next_request_id += 1;
        return try std.fmt.allocPrint(allocator, "{s}-{d}", .{ prefix, current });
    }

    fn createdSeconds(self: *AnnaHttpServer) i64 {
        return std.Io.Timestamp.now(self.io, .real).toSeconds();
    }

    fn freeGenerationResult(self: *AnnaHttpServer, result: types.TextGenerationResult) void {
        self.allocator.free(result.text);
        if (result.reasoning_text) |reasoning| self.allocator.free(reasoning);
        for (result.tool_calls) |tool_call| {
            self.allocator.free(tool_call.name);
            self.allocator.free(tool_call.arguments_json);
        }
        self.allocator.free(result.tool_calls);
    }
};

fn routeFromTarget(target: []const u8) ?Route {
    const path = if (std.mem.indexOfScalar(u8, target, '?')) |idx| target[0..idx] else target;
    if (std.mem.eql(u8, path, "/openapi.json")) return .openapi;
    if (std.mem.eql(u8, path, "/docs")) return .docs;
    if (std.mem.eql(u8, path, "/healthz")) return .healthz;
    if (std.mem.eql(u8, path, "/v1/models")) return .models;
    if (std.mem.eql(u8, path, "/v1/chat/completions")) return .chat_completions;
    if (std.mem.eql(u8, path, "/v1/completions")) return .completions;
    if (std.mem.eql(u8, path, "/v1/audio/speech")) return .audio_speech;
    return null;
}

fn completionStreamOnDelta(ctx: *anyopaque, delta: []const u8) !void {
    const stream_ctx: *CompletionStreamContext = @ptrCast(@alignCast(ctx));
    const payload = try openai.completionChunkPayload(
        stream_ctx.allocator,
        stream_ctx.response_id,
        stream_ctx.created,
        stream_ctx.model,
        delta,
        null,
    );
    defer stream_ctx.allocator.free(payload);
    try writeSsePayload(stream_ctx.allocator, stream_ctx.response, payload);
}

fn writeSsePayload(allocator: std.mem.Allocator, response: *std.http.BodyWriter, payload: []const u8) !void {
    const frame = try openai.sseDataFrame(allocator, payload);
    defer allocator.free(frame);
    try response.writer.writeAll(frame);
    try response.writer.flush();
    try response.flush();
}

fn shiftLeft(buffer: *std.array_list.Managed(u8), count: usize) void {
    if (count == 0) return;
    const remaining = buffer.items.len - count;
    std.mem.copyForwards(u8, buffer.items[0..remaining], buffer.items[count..]);
    buffer.items.len = remaining;
}

fn utf8BoundaryBefore(text: []const u8, proposed: usize) usize {
    var index = @min(proposed, text.len);
    while (index > 0 and index < text.len and (text[index] & 0b1100_0000) == 0b1000_0000) {
        index -= 1;
    }
    return index;
}

fn routeAllowsMethod(route: Route, method: std.http.Method) bool {
    return switch (route) {
        .openapi, .docs, .healthz, .models => method == .GET or method == .HEAD,
        .chat_completions, .completions, .audio_speech => method == .POST,
    };
}

fn allowMethods(route: Route) []const u8 {
    return switch (route) {
        .openapi, .docs, .healthz, .models => "GET, HEAD, OPTIONS",
        .chat_completions, .completions, .audio_speech => "POST, OPTIONS",
    };
}

fn readJsonRequestBody(allocator: std.mem.Allocator, request: *std.http.Server.Request) ![]const u8 {
    if (request.head.content_type) |content_type| {
        var content_type_parts = std.mem.splitScalar(u8, content_type, ';');
        const mime = std.mem.trim(u8, content_type_parts.first(), " \t\r\n");
        if (!std.ascii.eqlIgnoreCase(mime, "application/json")) return error.InvalidContentType;
    }

    var body_buffer: [1024]u8 = undefined;
    const reader = try request.readerExpectContinue(&body_buffer);
    if (request.head.content_length) |content_length| {
        if (content_length > max_request_body_bytes) return error.RequestBodyTooLarge;
        return try reader.readAlloc(allocator, @intCast(content_length));
    }
    return try reader.allocRemaining(allocator, .limited(max_request_body_bytes));
}

fn parseChatRequest(
    allocator: std.mem.Allocator,
    body: []const u8,
    default_enable_thinking: bool,
) !ChatRequest {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.InvalidJson;
    defer parsed.deinit();
    const object = switch (parsed.value) {
        .object => |value| value,
        else => return error.InvalidRootObject,
    };

    const common = try parseCommonRequestOptions(object);

    const messages_value = object.get("messages") orelse return error.MissingMessages;
    const messages = try parseChatMessages(allocator, messages_value);

    var enable_thinking = default_enable_thinking;
    if (object.get("enable_thinking")) |value| enable_thinking = try jsonBool(value);
    if (object.get("thinking")) |value| enable_thinking = try jsonBool(value);

    return .{
        .model = if (object.get("model")) |value| try jsonString(value) else null,
        .messages = messages,
        .config = try parseGenerationConfig(allocator, object),
        .enable_thinking = enable_thinking,
        .stream = common.stream,
    };
}

fn parseCompletionRequest(allocator: std.mem.Allocator, body: []const u8) !CompletionRequest {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.InvalidJson;
    defer parsed.deinit();
    const object = switch (parsed.value) {
        .object => |value| value,
        else => return error.InvalidRootObject,
    };

    const common = try parseCommonRequestOptions(object);

    const prompt_value = object.get("prompt") orelse return error.MissingPrompt;
    return .{
        .model = if (object.get("model")) |value| try jsonString(value) else null,
        .prompt = try parsePrompt(prompt_value),
        .config = try parseGenerationConfig(allocator, object),
        .stream = common.stream,
    };
}

fn parseCommonRequestOptions(object: std.json.ObjectMap) !CommonRequestOptions {
    var options = CommonRequestOptions{};
    if (object.get("stream")) |value| options.stream = try jsonBool(value);
    if (object.get("n")) |value| {
        if (try jsonUsize(value) != 1) return error.InvalidN;
    }
    if (object.get("tools")) |value| switch (value) {
        .null => {},
        .array => |items| if (items.items.len > 0) return error.ToolsUnsupported,
        else => return error.ToolsUnsupported,
    };
    return options;
}

fn parseGenerationConfig(allocator: std.mem.Allocator, object: std.json.ObjectMap) !types.GenerationConfig {
    var config = types.GenerationConfig{};

    if (object.get("max_completion_tokens")) |value| {
        config.max_new_tokens = try parsePositiveTokenCount(value);
    }
    if (object.get("max_tokens")) |value| {
        const parsed = try parsePositiveTokenCount(value);
        if (config.max_new_tokens) |existing| {
            if (existing != parsed) return error.InvalidFieldType;
        }
        config.max_new_tokens = parsed;
    }
    if (object.get("temperature")) |value| config.temperature = try jsonFloat32(value);
    if (object.get("top_p")) |value| config.top_p = try jsonFloat32(value);
    if (object.get("top_k")) |value| config.top_k = try jsonUsize(value);
    if (object.get("repetition_penalty")) |value| config.repetition_penalty = try jsonFloat32(value);
    if (object.get("stop")) |value| config.stop_strings = try parseStopStrings(allocator, value);

    return config;
}

fn parsePositiveTokenCount(value: std.json.Value) !usize {
    const parsed = try jsonUsize(value);
    if (parsed == 0) return error.InvalidFieldType;
    return parsed;
}

fn parsePrompt(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |text| {
            if (text.len == 0) return error.MissingPrompt;
            return text;
        },
        .array => |items| {
            if (items.items.len != 1) return error.InvalidFieldType;
            const inner = items.items[0];
            const text = try jsonString(inner);
            if (text.len == 0) return error.MissingPrompt;
            return text;
        },
        else => error.InvalidFieldType,
    };
}

fn parseChatMessages(allocator: std.mem.Allocator, value: std.json.Value) ![]const tokenizer_mod.ChatMessage {
    const items = switch (value) {
        .array => |array| array.items,
        else => return error.MissingMessages,
    };
    if (items.len == 0) return error.MissingMessages;

    var messages = std.array_list.Managed(tokenizer_mod.ChatMessage).init(allocator);
    defer messages.deinit();

    for (items) |item| {
        const object = switch (item) {
            .object => |entry| entry,
            else => return error.InvalidFieldType,
        };

        const role = try parseRole(try jsonString(object.get("role") orelse return error.MissingRole));
        const content = if (object.get("content")) |content_value|
            try parseMessageContent(allocator, role, content_value)
        else switch (role) {
            .assistant => "",
            else => return error.MissingContent,
        };
        const reasoning_content = if (object.get("reasoning_content")) |reasoning|
            try jsonString(reasoning)
        else
            null;

        try messages.append(.{
            .role = role,
            .content = content,
            .reasoning_content = reasoning_content,
        });
    }

    return try messages.toOwnedSlice();
}

fn parseRole(value: []const u8) !tokenizer_mod.ChatRole {
    if (std.mem.eql(u8, value, "system")) return .system;
    if (std.mem.eql(u8, value, "developer")) return .developer;
    if (std.mem.eql(u8, value, "user")) return .user;
    if (std.mem.eql(u8, value, "assistant")) return .assistant;
    if (std.mem.eql(u8, value, "tool")) return .tool;
    return error.InvalidRole;
}

fn parseMessageContent(
    allocator: std.mem.Allocator,
    role: tokenizer_mod.ChatRole,
    value: std.json.Value,
) ![]const u8 {
    return switch (value) {
        .null => switch (role) {
            .assistant => "",
            else => error.InvalidMessageContent,
        },
        .string => |text| text,
        .array => |parts| {
            var joined = std.array_list.Managed(u8).init(allocator);
            defer joined.deinit();
            for (parts.items) |part| {
                const object = switch (part) {
                    .object => |entry| entry,
                    else => return error.InvalidMessagePart,
                };
                const part_type = try jsonString(object.get("type") orelse return error.InvalidMessagePart);
                if (!std.mem.eql(u8, part_type, "text")) return error.InvalidMessagePart;
                const text = try jsonString(object.get("text") orelse return error.InvalidMessagePart);
                try joined.appendSlice(text);
            }
            return try joined.toOwnedSlice();
        },
        else => error.InvalidMessageContent,
    };
}

fn parseStopStrings(allocator: std.mem.Allocator, value: std.json.Value) ![]const []const u8 {
    return switch (value) {
        .string => |text| {
            const items = try allocator.alloc([]const u8, 1);
            items[0] = text;
            return items;
        },
        .array => |array| {
            var items = std.array_list.Managed([]const u8).init(allocator);
            defer items.deinit();
            for (array.items) |entry| {
                try items.append(try jsonString(entry));
            }
            return try items.toOwnedSlice();
        },
        .null => &.{},
        else => error.InvalidStop,
    };
}

fn jsonString(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |text| text,
        else => error.InvalidFieldType,
    };
}

fn jsonBool(value: std.json.Value) !bool {
    return switch (value) {
        .bool => |flag| flag,
        else => error.InvalidFieldType,
    };
}

fn jsonUsize(value: std.json.Value) !usize {
    return switch (value) {
        .integer => |number| {
            if (number < 0) return error.InvalidFieldType;
            return @intCast(number);
        },
        .float => |number| {
            if (number < 0 or @floor(number) != number) return error.InvalidFieldType;
            return @intFromFloat(number);
        },
        .number_string => |text| std.fmt.parseInt(usize, text, 10) catch return error.InvalidFieldType,
        else => error.InvalidFieldType,
    };
}

fn jsonFloat32(value: std.json.Value) !f32 {
    return switch (value) {
        .integer => |number| @floatFromInt(number),
        .float => |number| @floatCast(number),
        .number_string => |text| std.fmt.parseFloat(f32, text) catch return error.InvalidFieldType,
        else => error.InvalidFieldType,
    };
}

fn requestErrorStatus(err: anyerror) std.http.Status {
    return switch (err) {
        error.RequestBodyTooLarge, error.StreamTooLong => .payload_too_large,
        error.InvalidContentType => .unsupported_media_type,
        error.HttpExpectationFailed => .expectation_failed,
        error.OutOfMemory => .internal_server_error,
        else => .bad_request,
    };
}

fn requestErrorCode(err: anyerror) []const u8 {
    return switch (err) {
        error.InvalidJson => "invalid_json",
        error.InvalidRootObject => "invalid_request",
        error.MissingPrompt => "missing_prompt",
        error.MissingMessages => "missing_messages",
        error.MissingRole => "missing_role",
        error.MissingContent => "missing_content",
        error.InvalidFieldType => "invalid_field_type",
        error.InvalidRole => "invalid_role",
        error.InvalidStop => "invalid_stop",
        error.InvalidN => "unsupported_n",
        error.InvalidModel => "invalid_model",
        error.InvalidContentType => "unsupported_media_type",
        error.InvalidMessageContent => "invalid_message_content",
        error.InvalidMessagePart => "invalid_message_part",
        error.RequestBodyTooLarge, error.StreamTooLong => "request_too_large",
        error.StreamingUnsupported => "stream_unsupported",
        error.ToolsUnsupported => "tools_unsupported",
        error.HttpExpectationFailed => "expectation_failed",
        error.EndOfStream => "truncated_request_body",
        error.ReadFailed => "request_read_failed",
        error.WriteFailed => "request_write_failed",
        error.OutOfMemory => "server_out_of_memory",
        else => "invalid_request",
    };
}

fn requestErrorMessage(err: anyerror) []const u8 {
    return switch (err) {
        error.InvalidJson => "Request body is not valid JSON.",
        error.InvalidRootObject => "Request body must be a JSON object.",
        error.MissingPrompt => "Completion requests must provide a non-empty prompt.",
        error.MissingMessages => "Chat requests must provide a non-empty messages array.",
        error.MissingRole => "Each chat message must include a role field.",
        error.MissingContent => "Each non-assistant chat message must include content.",
        error.InvalidFieldType => "One or more request fields have an invalid type or value.",
        error.InvalidRole => "Chat message role must be one of system, developer, user, assistant, or tool.",
        error.InvalidStop => "The stop field must be a string or an array of strings.",
        error.InvalidN => "Only n=1 is supported by the native Zig service.",
        error.InvalidModel => "The model field must be a string when provided.",
        error.InvalidContentType => "POST request bodies must use application/json.",
        error.InvalidMessageContent => "Chat message content must be a string, null assistant content, or an array of text parts.",
        error.InvalidMessagePart => "Only text content parts are supported in chat messages.",
        error.RequestBodyTooLarge, error.StreamTooLong => "Request body exceeds the 64 MiB service limit.",
        error.StreamingUnsupported => "stream=true is not implemented by the native Zig service.",
        error.ToolsUnsupported => "Tool schema inputs are not implemented by the native Zig service.",
        error.HttpExpectationFailed => "The request Expect header is not supported.",
        error.EndOfStream => "Request body ended before the declared content length was read.",
        error.ReadFailed => "Failed to read the request body.",
        error.WriteFailed => "Failed to write the HTTP response.",
        error.OutOfMemory => "The native service ran out of memory while parsing the request.",
        else => "The request is invalid.",
    };
}

fn generationError(allocator: std.mem.Allocator, err: anyerror) !types.AnnaEngineError {
    return switch (err) {
        error.InvalidChatMessages => .{
            .message = "Chat messages violate the tokenizer chat-template ordering rules.",
            .status_code = 400,
            .error_type = "invalid_request_error",
            .code = "invalid_chat_messages",
        },
        error.MissingChatMessages => .{
            .message = "Chat requests must provide at least one message.",
            .status_code = 400,
            .error_type = "invalid_request_error",
            .code = "missing_messages",
        },
        else => .{
            .message = try std.fmt.allocPrint(allocator, "Native generation failed: {s}", .{@errorName(err)}),
            .status_code = 500,
            .error_type = "server_error",
            .code = "generation_failed",
        },
    };
}

fn healthzPayload(allocator: std.mem.Allocator, model_id: []const u8, backend: types.RuntimeBackend) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("status");
    try jw.write("ok");
    try jw.objectField("model");
    try jw.write(model_id);
    try jw.objectField("backend");
    try jw.write(@tagName(backend));
    try jw.endObject();
    return try out.toOwnedSlice();
}

fn modelsPayload(allocator: std.mem.Allocator, model_id: []const u8, created: i64) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("object");
    try jw.write("list");
    try jw.objectField("data");
    try jw.beginArray();
    try jw.beginObject();
    try jw.objectField("id");
    try jw.write(model_id);
    try jw.objectField("object");
    try jw.write("model");
    try jw.objectField("created");
    try jw.write(created);
    try jw.objectField("owned_by");
    try jw.write("anna-native");
    try jw.endObject();
    try jw.endArray();
    try jw.endObject();
    return try out.toOwnedSlice();
}

fn openapiPayload(allocator: std.mem.Allocator, host: []const u8, port: u16, model_id: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("openapi");
    try jw.write("3.1.0");
    try jw.objectField("info");
    try jw.beginObject();
    try jw.objectField("title");
    try jw.write("Anna Native API");
    try jw.objectField("version");
    try jw.write("0.1.0");
    try jw.endObject();
    try jw.objectField("servers");
    try jw.beginArray();
    try jw.beginObject();
    try jw.objectField("url");
    try jw.write(try std.fmt.allocPrint(allocator, "http://{s}:{d}", .{ host, port }));
    try jw.endObject();
    try jw.endArray();
    try jw.objectField("x-anna-model");
    try jw.write(model_id);
    try jw.objectField("paths");
    try jw.beginObject();
    try writeOpenapiPath(&jw, "/healthz", "get");
    try writeOpenapiPath(&jw, "/v1/models", "get");
    try writeOpenapiPath(&jw, "/v1/completions", "post");
    try writeOpenapiPath(&jw, "/v1/chat/completions", "post");
    try writeOpenapiPath(&jw, "/v1/audio/speech", "post");
    try jw.endObject();
    try jw.endObject();
    return try out.toOwnedSlice();
}

fn writeOpenapiPath(jw: *std.json.Stringify, path: []const u8, method: []const u8) !void {
    try jw.objectField(path);
    try jw.beginObject();
    try jw.objectField(method);
    try jw.beginObject();
    try jw.objectField("responses");
    try jw.beginObject();
    try jw.objectField("200");
    try jw.beginObject();
    try jw.objectField("description");
    try jw.write("OK");
    try jw.endObject();
    try jw.endObject();
    try jw.endObject();
    try jw.endObject();
}

fn docsHtmlPayload(allocator: std.mem.Allocator, host: []const u8, port: u16, model_id: []const u8) ![]const u8 {
    return try std.fmt.allocPrint(
        allocator,
        \\<!doctype html>
        \\<html lang="en">
        \\<head>
        \\  <meta charset="utf-8">
        \\  <title>Anna Native API</title>
        \\  <style>
        \\    body {{ font-family: sans-serif; margin: 2rem auto; max-width: 960px; line-height: 1.6; }}
        \\    code, pre {{ background: #f4f4f4; padding: 0.2rem 0.4rem; border-radius: 4px; }}
        \\    pre {{ padding: 1rem; overflow-x: auto; }}
        \\  </style>
        \\</head>
        \\<body>
        \\  <h1>Anna Native API</h1>
        \\  <p>Base URL: <code>http://{s}:{d}</code></p>
        \\  <p>Loaded model: <code>{s}</code></p>
        \\  <h2>Routes</h2>
        \\  <ul>
        \\    <li><code>GET /healthz</code></li>
        \\    <li><code>GET /v1/models</code></li>
        \\    <li><code>POST /v1/completions</code></li>
        \\    <li><code>POST /v1/chat/completions</code></li>
        \\  </ul>
        \\  <h2>Completion example</h2>
        \\  <pre>curl http://{s}:{d}/v1/completions ^
        \\  -H "Content-Type: application/json" ^
        \\  -d "{{\"model\":\"{s}\",\"prompt\":\"你好，介绍一下 Zig。\",\"max_tokens\":64}}"</pre>
        \\  <p>These routes stream by default. Set <code>"stream": false</code> to request a single JSON response.</p>
        \\  <h2>Chat example</h2>
        \\  <pre>curl http://{s}:{d}/v1/chat/completions ^
        \\  -H "Content-Type: application/json" ^
        \\  -d "{{\"model\":\"{s}\",\"messages\":[{{\"role\":\"user\",\"content\":\"你好，介绍一下 Anna。\"}}],\"max_completion_tokens\":64}}"</pre>
        \\</body>
        \\</html>
    ,
        .{ host, port, model_id, host, port, model_id, host, port, model_id },
    );
}

test "parse completion request accepts stop strings" {
    const raw =
        \\{
        \\  "model": "anna",
        \\  "prompt": "hello",
        \\  "max_tokens": 32,
        \\  "temperature": 0.0,
        \\  "stop": ["</s>", "<|im_end|>"]
        \\}
    ;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const parsed = try parseCompletionRequest(arena_state.allocator(), raw);
    try std.testing.expectEqualStrings("anna", parsed.model.?);
    try std.testing.expectEqualStrings("hello", parsed.prompt);
    try std.testing.expectEqual(@as(?usize, 32), parsed.config.max_new_tokens);
    try std.testing.expectEqual(@as(f32, 0.0), parsed.config.temperature);
    try std.testing.expectEqual(@as(usize, 2), parsed.config.stop_strings.len);
    try std.testing.expect(parsed.stream);
}

test "parse chat request accepts text-part arrays" {
    const raw =
        \\{
        \\  "messages": [
        \\    {"role":"system","content":"Be brief."},
        \\    {"role":"user","content":[{"type":"text","text":"Hello"},{"type":"text","text":" Anna"}]}
        \\  ],
        \\  "enable_thinking": false,
        \\  "max_completion_tokens": 16
        \\}
    ;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const parsed = try parseChatRequest(arena_state.allocator(), raw, true);
    try std.testing.expectEqual(@as(usize, 2), parsed.messages.len);
    try std.testing.expectEqualStrings("Hello Anna", parsed.messages[1].content);
    try std.testing.expectEqual(false, parsed.enable_thinking);
    try std.testing.expectEqual(@as(?usize, 16), parsed.config.max_new_tokens);
    try std.testing.expect(parsed.stream);
}

test "parse chat request accepts explicit stream false" {
    const raw =
        \\{
        \\  "messages": [{"role":"user","content":"hi"}],
        \\  "stream": false
        \\}
    ;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const parsed = try parseChatRequest(arena_state.allocator(), raw, true);
    try std.testing.expect(!parsed.stream);
}
