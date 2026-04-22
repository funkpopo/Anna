const std = @import("std");
const types = @import("../runtime/types.zig");

const TokenPiece = struct {
    text: []const u8,
};

const AddedToken = struct {
    id: u32,
    content: []const u8,
    special: bool = false,
};

const CodepointInfo = struct {
    cp: u21,
    len: usize,
};

pub const ChatRole = enum {
    system,
    developer,
    user,
    assistant,
    tool,
};

pub const ChatMessage = struct {
    role: ChatRole,
    content: []const u8,
    reasoning_content: ?[]const u8 = null,
};

pub const SplitReasoningResult = struct {
    reasoning: ?[]const u8,
    content: []const u8,
};

pub const ToolExtractionResult = struct {
    text: []const u8,
    tool_calls: []const types.ToolCall,
};

pub const QwenTokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(u32),
    id_to_token: [][]const u8,
    merge_ranks: std.StringHashMap(u32),
    added_tokens: []AddedToken,
    special_to_id: std.StringHashMap(u32),
    special_id_set: std.AutoHashMap(u32, void),
    eos_token_ids: []u32,
    image_token: []const u8 = "<|image_pad|>",
    video_token: []const u8 = "<|video_pad|>",
    vision_start_token: []const u8 = "<|vision_start|>",
    vision_end_token: []const u8 = "<|vision_end|>",

    pub fn loadFromModelDir(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !QwenTokenizer {
        var dir = try openModelDir(io, model_dir);
        defer dir.close(io);

        const tokenizer_json = try dir.readFileAlloc(io, "tokenizer.json", allocator, .limited(128 << 20));
        defer allocator.free(tokenizer_json);

        var tokenizer = try parseTokenizerJson(allocator, tokenizer_json);
        errdefer tokenizer.deinit();

        const config_json = dir.readFileAlloc(io, "tokenizer_config.json", allocator, .limited(4 << 20)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        if (config_json) |raw| {
            defer allocator.free(raw);
            try tokenizer.applyTokenizerConfig(raw);
        }

        return tokenizer;
    }

    pub fn deinit(self: *QwenTokenizer) void {
        var vocab_it = self.vocab.iterator();
        while (vocab_it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.vocab.deinit();

        for (self.id_to_token) |token| {
            if (token.len > 0) self.allocator.free(token);
        }
        self.allocator.free(self.id_to_token);

        var merge_it = self.merge_ranks.iterator();
        while (merge_it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.merge_ranks.deinit();

        for (self.added_tokens) |token| self.allocator.free(token.content);
        self.allocator.free(self.added_tokens);

        var special_it = self.special_to_id.iterator();
        while (special_it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.special_to_id.deinit();
        self.special_id_set.deinit();
        self.allocator.free(self.eos_token_ids);
        self.allocator.free(self.image_token);
        self.allocator.free(self.video_token);
        self.allocator.free(self.vision_start_token);
        self.allocator.free(self.vision_end_token);
        self.* = undefined;
    }

    pub fn tokenId(self: *const QwenTokenizer, token: []const u8) ?u32 {
        if (self.special_to_id.get(token)) |id| return id;
        return self.vocab.get(token);
    }

    pub fn isEosToken(self: *const QwenTokenizer, token_id: u32) bool {
        for (self.eos_token_ids) |id| {
            if (id == token_id) return true;
        }
        return false;
    }

    pub fn encodeAlloc(self: *const QwenTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var ids = std.array_list.Managed(u32).init(allocator);
        errdefer ids.deinit();

        var index: usize = 0;
        while (index < text.len) {
            if (self.matchSpecial(text[index..])) |special| {
                try ids.append(special.id);
                index += special.content.len;
                continue;
            }

            const next_special = self.findNextSpecial(text[index..]);
            const end = if (next_special) |relative| index + relative else text.len;
            try self.encodePlainSegment(allocator, &ids, text[index..end]);
            index = end;
        }

        return try ids.toOwnedSlice();
    }

    pub fn decodeAlloc(self: *const QwenTokenizer, allocator: std.mem.Allocator, token_ids: []const u32, skip_special_tokens: bool) ![]u8 {
        var out = std.array_list.Managed(u8).init(allocator);
        errdefer out.deinit();
        const reverse = byteDecoderTable();

        for (token_ids) |token_id| {
            if (token_id >= self.id_to_token.len) return error.TokenOutOfRange;
            const token = self.id_to_token[token_id];
            if (token.len == 0) return error.TokenOutOfRange;
            if (self.special_id_set.contains(token_id)) {
                if (!skip_special_tokens) try out.appendSlice(token);
                continue;
            }

            var view = std.unicode.Utf8View.init(token) catch return error.InvalidTokenizerToken;
            var it = view.iterator();
            while (it.nextCodepoint()) |cp| {
                if (cp >= reverse.len or reverse[@intCast(cp)] < 0) return error.InvalidTokenizerToken;
                try out.append(@intCast(reverse[@intCast(cp)]));
            }
        }

        return try out.toOwnedSlice();
    }

    pub fn decodeTokens(ctx: *const anyopaque, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
        const self: *const QwenTokenizer = @ptrCast(@alignCast(ctx));
        return try self.decodeAlloc(allocator, token_ids, false);
    }

    pub fn renderMessagesAlloc(
        self: *const QwenTokenizer,
        allocator: std.mem.Allocator,
        messages: []const ChatMessage,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) ![]u8 {
        if (messages.len == 0) return error.MissingChatMessages;
        var out = std.array_list.Managed(u8).init(allocator);
        errdefer out.deinit();

        for (messages, 0..) |message, index| {
            const role = if (message.role == .developer) .system else message.role;
            if (role == .system and index != 0) return error.InvalidChatMessages;
            switch (role) {
                .system => {
                    try out.appendSlice("<|im_start|>system\n");
                    try out.appendSlice(std.mem.trim(u8, message.content, " \t\r\n"));
                    try out.appendSlice("<|im_end|>\n");
                },
                .user => {
                    try out.appendSlice("<|im_start|>user\n");
                    try out.appendSlice(std.mem.trim(u8, message.content, " \t\r\n"));
                    try out.appendSlice("<|im_end|>\n");
                },
                .assistant => {
                    try out.appendSlice("<|im_start|>assistant\n");
                    if (message.reasoning_content) |reasoning| {
                        try out.appendSlice("<think>\n");
                        try out.appendSlice(std.mem.trim(u8, reasoning, " \t\r\n"));
                        try out.appendSlice("\n</think>\n\n");
                    }
                    try out.appendSlice(std.mem.trim(u8, message.content, " \t\r\n"));
                    try out.appendSlice("<|im_end|>\n");
                },
                .tool => {
                    try out.appendSlice("<|im_start|>user\n<tool_response>\n");
                    try out.appendSlice(std.mem.trim(u8, message.content, " \t\r\n"));
                    try out.appendSlice("\n</tool_response><|im_end|>\n");
                },
                .developer => unreachable,
            }
        }

        if (add_generation_prompt) {
            if (enable_thinking) {
                try out.appendSlice("<|im_start|>assistant\n<think>\n");
            } else {
                try out.appendSlice("<|im_start|>assistant\n<think>\n\n</think>\n\n");
            }
        }
        _ = self;
        return try out.toOwnedSlice();
    }

    pub fn splitAssistantReasoning(self: *const QwenTokenizer, text: []const u8, enable_thinking: bool) SplitReasoningResult {
        _ = self;
        var normalized = std.mem.trimStart(u8, text, " \t\r\n");
        const explicit_open = std.mem.startsWith(u8, normalized, "<think>");
        if (explicit_open) {
            normalized = normalized["<think>".len..];
            normalized = std.mem.trimStart(u8, normalized, "\r\n");
        }
        if (std.mem.indexOf(u8, normalized, "</think>")) |close_index| {
            const reasoning = std.mem.trim(u8, normalized[0..close_index], " \t\r\n");
            const content = std.mem.trimStart(u8, normalized[close_index + "</think>".len ..], "\r\n");
            return .{ .reasoning = if (reasoning.len == 0) null else reasoning, .content = content };
        }
        if (!explicit_open and !enable_thinking) {
            return .{ .reasoning = null, .content = text };
        }
        const reasoning = std.mem.trim(u8, normalized, " \t\r\n");
        return .{ .reasoning = if (reasoning.len == 0) null else reasoning, .content = "" };
    }

    pub fn extractToolCallsAlloc(self: *const QwenTokenizer, allocator: std.mem.Allocator, text: []const u8) !ToolExtractionResult {
        _ = self;
        var cleaned = std.array_list.Managed(u8).init(allocator);
        errdefer cleaned.deinit();
        var calls = std.array_list.Managed(types.ToolCall).init(allocator);
        errdefer calls.deinit();

        var cursor: usize = 0;
        while (cursor < text.len) {
            const relative_start = std.mem.indexOf(u8, text[cursor..], "<tool_call>") orelse {
                try cleaned.appendSlice(text[cursor..]);
                break;
            };
            const start = cursor + relative_start;
            try cleaned.appendSlice(text[cursor..start]);
            const body_start = start + "<tool_call>".len;
            const relative_end = std.mem.indexOf(u8, text[body_start..], "</tool_call>") orelse {
                try cleaned.appendSlice(text[start..]);
                break;
            };
            const end = body_start + relative_end + "</tool_call>".len;
            if (try parseToolCallBlock(allocator, text[start..end], calls.items.len)) |tool_call| {
                try calls.append(tool_call);
            } else {
                try cleaned.appendSlice(text[start..end]);
            }
            cursor = end;
        }

        const cleaned_text = if (calls.items.len > 0)
            try allocator.dupe(u8, std.mem.trim(u8, cleaned.items, " \t\r\n"))
        else
            try cleaned.toOwnedSlice();
        if (calls.items.len > 0) cleaned.deinit();
        return .{ .text = cleaned_text, .tool_calls = try calls.toOwnedSlice() };
    }

    fn applyTokenizerConfig(self: *QwenTokenizer, raw: []const u8) !void {
        var arena_state = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var parsed = try std.json.parseFromSlice(std.json.Value, arena, raw, .{});
        if (parsed.value != .object) return;

        if (jsonStringOpt(parsed.value.object.get("image_token"))) |value| self.replaceMetadataString(&self.image_token, value) catch return error.OutOfMemory;
        if (jsonStringOpt(parsed.value.object.get("video_token"))) |value| self.replaceMetadataString(&self.video_token, value) catch return error.OutOfMemory;
        if (jsonStringOpt(parsed.value.object.get("vision_bos_token"))) |value| self.replaceMetadataString(&self.vision_start_token, value) catch return error.OutOfMemory;
        if (jsonStringOpt(parsed.value.object.get("vision_eos_token"))) |value| self.replaceMetadataString(&self.vision_end_token, value) catch return error.OutOfMemory;
    }

    fn replaceMetadataString(self: *QwenTokenizer, slot: *[]const u8, value: []const u8) !void {
        self.allocator.free(slot.*);
        slot.* = try self.allocator.dupe(u8, value);
    }

    fn encodePlainSegment(self: *const QwenTokenizer, allocator: std.mem.Allocator, out: *std.array_list.Managed(u32), text: []const u8) !void {
        if (text.len == 0) return;
        var spans = std.array_list.Managed([]const u8).init(allocator);
        defer spans.deinit();
        try pretokenize(allocator, &spans, text);
        for (spans.items) |span| {
            try self.encodePretoken(allocator, out, span);
        }
    }

    fn encodePretoken(self: *const QwenTokenizer, allocator: std.mem.Allocator, out: *std.array_list.Managed(u32), text: []const u8) !void {
        var scratch_state = std.heap.ArenaAllocator.init(allocator);
        defer scratch_state.deinit();
        const scratch = scratch_state.allocator();
        const encoder = byteEncoderTable();

        var mapped = std.array_list.Managed(u8).init(scratch);
        for (text) |byte| {
            const item = encoder[byte];
            try mapped.appendSlice(item.bytes[0..item.len]);
        }

        var pieces = std.array_list.Managed(TokenPiece).init(scratch);
        var view = std.unicode.Utf8View.init(mapped.items) catch return error.InvalidUtf8;
        var it = view.iterator();
        while (it.nextCodepointSlice()) |piece| {
            try pieces.append(.{ .text = piece });
        }
        try self.applyBpe(scratch, &pieces);

        for (pieces.items) |piece| {
            const id = self.vocab.get(piece.text) orelse return error.TokenizerPieceMissing;
            try out.append(id);
        }
    }

    fn applyBpe(self: *const QwenTokenizer, allocator: std.mem.Allocator, pieces: *std.array_list.Managed(TokenPiece)) !void {
        if (pieces.items.len <= 1) return;
        while (pieces.items.len > 1) {
            var best_index: ?usize = null;
            var best_rank: u32 = std.math.maxInt(u32);
            for (0..pieces.items.len - 1) |index| {
                const key = try pairKey(allocator, pieces.items[index].text, pieces.items[index + 1].text);
                if (self.merge_ranks.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_index = index;
                    }
                }
            }
            const merge_index = best_index orelse break;
            pieces.items[merge_index].text = try std.fmt.allocPrint(
                allocator,
                "{s}{s}",
                .{ pieces.items[merge_index].text, pieces.items[merge_index + 1].text },
            );
            _ = pieces.orderedRemove(merge_index + 1);
        }
    }

    fn matchSpecial(self: *const QwenTokenizer, text: []const u8) ?AddedToken {
        var matched: ?AddedToken = null;
        for (self.added_tokens) |token| {
            if (!token.special) continue;
            if (!std.mem.startsWith(u8, text, token.content)) continue;
            if (matched == null or token.content.len > matched.?.content.len) matched = token;
        }
        return matched;
    }

    fn findNextSpecial(self: *const QwenTokenizer, text: []const u8) ?usize {
        var found: ?usize = null;
        for (self.added_tokens) |token| {
            if (!token.special) continue;
            if (std.mem.indexOf(u8, text, token.content)) |idx| {
                found = if (found) |current| @min(current, idx) else idx;
            }
        }
        return found;
    }
};

fn parseTokenizerJson(allocator: std.mem.Allocator, raw: []const u8) !QwenTokenizer {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var parsed = try std.json.parseFromSlice(std.json.Value, arena, raw, .{});
    if (parsed.value != .object) return error.InvalidTokenizerJson;

    const model = parsed.value.object.get("model") orelse return error.InvalidTokenizerJson;
    if (model != .object) return error.InvalidTokenizerJson;
    const vocab_value = model.object.get("vocab") orelse return error.InvalidTokenizerJson;
    if (vocab_value != .object) return error.InvalidTokenizerJson;

    var vocab = std.StringHashMap(u32).init(allocator);
    errdefer freeStringMap(allocator, &vocab);

    var max_id: u32 = 0;
    var vocab_it = vocab_value.object.iterator();
    while (vocab_it.next()) |entry| {
        const id: u32 = @intCast(try jsonInt(entry.value_ptr.*));
        max_id = @max(max_id, id);
        try vocab.put(try allocator.dupe(u8, entry.key_ptr.*), id);
    }

    const added_value = parsed.value.object.get("added_tokens") orelse return error.InvalidTokenizerJson;
    if (added_value != .array) return error.InvalidTokenizerJson;
    var added_tokens = std.array_list.Managed(AddedToken).init(allocator);
    errdefer {
        for (added_tokens.items) |token| allocator.free(token.content);
        added_tokens.deinit();
    }
    for (added_value.array.items) |item| {
        if (item != .object) return error.InvalidTokenizerJson;
        const content = jsonStringOpt(item.object.get("content")) orelse return error.InvalidTokenizerJson;
        const id: u32 = @intCast(try jsonInt(item.object.get("id") orelse return error.InvalidTokenizerJson));
        max_id = @max(max_id, id);
        try added_tokens.append(.{
            .id = id,
            .content = try allocator.dupe(u8, content),
            .special = jsonBoolOpt(item.object.get("special")) orelse false,
        });
    }

    const id_to_token = try allocator.alloc([]const u8, @as(usize, max_id) + 1);
    errdefer allocator.free(id_to_token);
    @memset(id_to_token, "");
    var vocab_it_2 = vocab.iterator();
    while (vocab_it_2.next()) |entry| {
        id_to_token[entry.value_ptr.*] = try allocator.dupe(u8, entry.key_ptr.*);
    }
    for (added_tokens.items) |token| {
        if (id_to_token[token.id].len == 0) {
            id_to_token[token.id] = try allocator.dupe(u8, token.content);
        }
    }

    var merge_ranks = std.StringHashMap(u32).init(allocator);
    errdefer freeStringMap(allocator, &merge_ranks);
    const merges_value = model.object.get("merges") orelse return error.InvalidTokenizerJson;
    if (merges_value != .array) return error.InvalidTokenizerJson;
    for (merges_value.array.items, 0..) |merge, rank| {
        const left, const right = try mergePair(merge);
        try merge_ranks.put(try ownedPairKey(allocator, left, right), @intCast(rank));
    }

    var special_to_id = std.StringHashMap(u32).init(allocator);
    errdefer freeStringMap(allocator, &special_to_id);
    var special_id_set = std.AutoHashMap(u32, void).init(allocator);
    errdefer special_id_set.deinit();
    for (added_tokens.items) |token| {
        if (!token.special) continue;
        try special_to_id.put(try allocator.dupe(u8, token.content), token.id);
        try special_id_set.put(token.id, {});
    }

    var eos_ids = std.array_list.Managed(u32).init(allocator);
    errdefer eos_ids.deinit();
    for ([_][]const u8{ "<|im_start|>", "<|im_end|>", "<|endoftext|>" }) |token| {
        if (special_to_id.get(token) orelse vocab.get(token)) |id| {
            try eos_ids.append(id);
        }
    }

    return .{
        .allocator = allocator,
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .added_tokens = try added_tokens.toOwnedSlice(),
        .special_to_id = special_to_id,
        .special_id_set = special_id_set,
        .eos_token_ids = try eos_ids.toOwnedSlice(),
        .image_token = try allocator.dupe(u8, "<|image_pad|>"),
        .video_token = try allocator.dupe(u8, "<|video_pad|>"),
        .vision_start_token = try allocator.dupe(u8, "<|vision_start|>"),
        .vision_end_token = try allocator.dupe(u8, "<|vision_end|>"),
    };
}

fn pretokenize(allocator: std.mem.Allocator, spans: *std.array_list.Managed([]const u8), text: []const u8) !void {
    _ = allocator;
    var index: usize = 0;
    while (index < text.len) {
        if (matchContraction(text, index)) |end| {
            try spans.append(text[index..end]);
            index = end;
            continue;
        }

        const info = codepointAt(text, index) orelse break;
        if (isHorizontalWhitespace(info.cp)) {
            if (consumeSpacesThenNewlines(text, index)) |end| {
                try spans.append(text[index..end]);
                index = end;
                continue;
            }
            if (info.cp == ' ') {
                if (codepointAt(text, index + info.len)) |next| {
                    if (isPunctuationLike(next.cp)) {
                        const end = consumePunctuationRun(text, index + info.len, true);
                        try spans.append(text[index..end]);
                        index = end;
                        continue;
                    }
                }
            }
            const end = consumeHorizontalWhitespace(text, index);
            try spans.append(text[index..end]);
            index = end;
            continue;
        }
        if (isNewline(info.cp)) {
            const end = consumeNewlines(text, index);
            try spans.append(text[index..end]);
            index = end;
            continue;
        }
        if (isPunctuationLike(info.cp)) {
            if (codepointAt(text, index + info.len)) |next| {
                if (isLetterOrMark(next.cp)) {
                    const end = consumeLetters(text, index + info.len);
                    try spans.append(text[index..end]);
                    index = end;
                    continue;
                }
            }
            const end = consumePunctuationRun(text, index, true);
            try spans.append(text[index..end]);
            index = end;
            continue;
        }
        if (isLetterOrMark(info.cp)) {
            const end = consumeLetters(text, index);
            try spans.append(text[index..end]);
            index = end;
            continue;
        }
        if (isDigit(info.cp)) {
            try spans.append(text[index .. index + info.len]);
            index += info.len;
            continue;
        }
        try spans.append(text[index .. index + info.len]);
        index += info.len;
    }
}

fn consumeSpacesThenNewlines(text: []const u8, start: usize) ?usize {
    var index = start;
    while (codepointAt(text, index)) |info| {
        if (!isHorizontalWhitespace(info.cp)) break;
        index += info.len;
    }
    const newline_start = index;
    while (codepointAt(text, index)) |info| {
        if (!isNewline(info.cp)) break;
        index += info.len;
    }
    return if (index > newline_start) index else null;
}

fn consumeHorizontalWhitespace(text: []const u8, start: usize) usize {
    var index = start;
    while (codepointAt(text, index)) |info| {
        if (!isHorizontalWhitespace(info.cp)) break;
        index += info.len;
    }
    return index;
}

fn consumeNewlines(text: []const u8, start: usize) usize {
    var index = start;
    while (codepointAt(text, index)) |info| {
        if (!isNewline(info.cp)) break;
        index += info.len;
    }
    return index;
}

fn consumeLetters(text: []const u8, start: usize) usize {
    var index = start;
    while (codepointAt(text, index)) |info| {
        if (!isLetterOrMark(info.cp)) break;
        index += info.len;
    }
    return index;
}

fn consumePunctuationRun(text: []const u8, start: usize, include_trailing_newlines: bool) usize {
    var index = start;
    while (codepointAt(text, index)) |info| {
        if (!isPunctuationLike(info.cp)) break;
        index += info.len;
    }
    if (include_trailing_newlines) {
        while (codepointAt(text, index)) |info| {
            if (!isNewline(info.cp)) break;
            index += info.len;
        }
    }
    return index;
}

fn matchContraction(text: []const u8, start: usize) ?usize {
    if (start >= text.len or text[start] != '\'') return null;
    const candidates = [_][]const u8{ "'re", "'ve", "'ll", "'s", "'t", "'m", "'d" };
    for (candidates) |candidate| {
        if (text.len - start < candidate.len) continue;
        if (asciiEqlIgnoreCase(text[start .. start + candidate.len], candidate)) return start + candidate.len;
    }
    return null;
}

fn codepointAt(text: []const u8, index: usize) ?CodepointInfo {
    if (index >= text.len) return null;
    const len: usize = std.unicode.utf8ByteSequenceLength(text[index]) catch return .{ .cp = text[index], .len = 1 };
    if (index + len > text.len) return .{ .cp = text[index], .len = 1 };
    const cp = std.unicode.utf8Decode(text[index .. index + len]) catch return .{ .cp = text[index], .len = 1 };
    return .{ .cp = cp, .len = len };
}

fn isDigit(cp: u21) bool {
    return cp >= '0' and cp <= '9';
}

fn isNewline(cp: u21) bool {
    return cp == '\n' or cp == '\r';
}

fn isHorizontalWhitespace(cp: u21) bool {
    return cp == ' ' or cp == '\t' or cp == 0x0B or cp == 0x0C or cp == 0x00A0 or cp == 0x3000;
}

fn isPunctuationLike(cp: u21) bool {
    return switch (cp) {
        0x21...0x2F, 0x3A...0x40, 0x5B...0x60, 0x7B...0x7E => true,
        0x2000...0x206F, 0x2E00...0x2E7F, 0x3001...0x303F => true,
        0xFF01...0xFF0F, 0xFF1A...0xFF20, 0xFF3B...0xFF40, 0xFF5B...0xFF65 => true,
        else => false,
    };
}

fn isLetterOrMark(cp: u21) bool {
    if (isNewline(cp) or isHorizontalWhitespace(cp) or isDigit(cp) or isPunctuationLike(cp)) return false;
    if (cp < 0x80) return std.ascii.isAlphabetic(@intCast(cp));
    return true;
}

const EncodedByte = struct {
    bytes: [4]u8,
    len: u3,
};

fn byteEncoderTable() [256]EncodedByte {
    var table: [256]EncodedByte = undefined;
    var reverse: [256]bool = [_]bool{false} ** 256;
    var bytes: [256]u16 = undefined;
    var scalars: [256]u21 = undefined;
    var count: usize = 0;

    count = appendRange(&bytes, &scalars, count, &reverse, 33, 126);
    count = appendRange(&bytes, &scalars, count, &reverse, 161, 172);
    count = appendRange(&bytes, &scalars, count, &reverse, 174, 255);
    var next: u21 = 256;
    for (0..256) |byte| {
        if (reverse[byte]) continue;
        bytes[count] = @intCast(byte);
        scalars[count] = next;
        next += 1;
        count += 1;
    }

    for (0..count) |idx| {
        var encoded: [4]u8 = undefined;
        const len = std.unicode.utf8Encode(scalars[idx], &encoded) catch unreachable;
        table[bytes[idx]] = .{ .bytes = encoded, .len = @intCast(len) };
    }
    return table;
}

fn byteDecoderTable() [512]i16 {
    var table: [512]i16 = [_]i16{-1} ** 512;
    const encoder = byteEncoderTable();
    for (encoder, 0..) |item, byte| {
        const cp = std.unicode.utf8Decode(item.bytes[0..item.len]) catch unreachable;
        table[@intCast(cp)] = @intCast(byte);
    }
    return table;
}

fn appendRange(bytes: *[256]u16, scalars: *[256]u21, count_in: usize, used: *[256]bool, start: u16, end: u16) usize {
    var count = count_in;
    var byte = start;
    while (byte <= end) : (byte += 1) {
        bytes[count] = byte;
        scalars[count] = @intCast(byte);
        used[byte] = true;
        count += 1;
    }
    return count;
}

fn pairKey(allocator: std.mem.Allocator, left: []const u8, right: []const u8) ![]const u8 {
    return try std.fmt.allocPrint(allocator, "{s}\x00{s}", .{ left, right });
}

fn ownedPairKey(allocator: std.mem.Allocator, left: []const u8, right: []const u8) ![]const u8 {
    return try pairKey(allocator, left, right);
}

fn mergePair(value: std.json.Value) !struct { []const u8, []const u8 } {
    if (value == .array and value.array.items.len == 2) {
        const left = jsonStringPtr(&value.array.items[0]) orelse return error.InvalidTokenizerJson;
        const right = jsonStringPtr(&value.array.items[1]) orelse return error.InvalidTokenizerJson;
        return .{ left, right };
    }
    if (value == .string) {
        if (std.mem.indexOfScalar(u8, value.string, ' ')) |space| {
            return .{ value.string[0..space], value.string[space + 1 ..] };
        }
    }
    return error.InvalidTokenizerJson;
}

fn jsonStringOpt(value: ?std.json.Value) ?[]const u8 {
    const item = value orelse return null;
    return if (item == .string) item.string else null;
}

fn jsonStringPtr(value: *const std.json.Value) ?[]const u8 {
    return if (value.* == .string) value.string else null;
}

fn jsonBoolOpt(value: ?std.json.Value) ?bool {
    const item = value orelse return null;
    return if (item == .bool) item.bool else null;
}

fn jsonInt(value: std.json.Value) !i64 {
    return switch (value) {
        .integer => |integer| integer,
        .number_string => |text| try std.fmt.parseInt(i64, text, 10),
        else => error.InvalidTokenizerJson,
    };
}

fn freeStringMap(allocator: std.mem.Allocator, map: *std.StringHashMap(u32)) void {
    var it = map.iterator();
    while (it.next()) |entry| allocator.free(entry.key_ptr.*);
    map.deinit();
}

fn asciiEqlIgnoreCase(left: []const u8, right: []const u8) bool {
    if (left.len != right.len) return false;
    for (left, right) |l, r| {
        if (std.ascii.toLower(l) != std.ascii.toLower(r)) return false;
    }
    return true;
}

fn parseToolCallBlock(allocator: std.mem.Allocator, block: []const u8, index: usize) !?types.ToolCall {
    var body = std.mem.trim(u8, block, " \t\r\n");
    if (!std.mem.startsWith(u8, body, "<tool_call>") or !std.mem.endsWith(u8, body, "</tool_call>")) return null;
    body = std.mem.trim(u8, body["<tool_call>".len .. body.len - "</tool_call>".len], " \t\r\n");
    if (!std.mem.startsWith(u8, body, "<function=")) return null;
    const name_start = "<function=".len;
    const name_end = std.mem.indexOfScalar(u8, body[name_start..], '>') orelse return null;
    const function_name = std.mem.trim(u8, body[name_start .. name_start + name_end], " \t\r\n");
    if (function_name.len == 0) return null;
    const close_tag = "</function>";
    if (!std.mem.endsWith(u8, body, close_tag)) return null;
    var params_body = std.mem.trim(u8, body[name_start + name_end + 1 .. body.len - close_tag.len], " \t\r\n");

    var arguments = std.Io.Writer.Allocating.init(allocator);
    errdefer arguments.deinit();
    var jw: std.json.Stringify = .{ .writer = &arguments.writer, .options = .{} };
    try jw.beginObject();
    var first = true;
    while (params_body.len > 0) {
        if (!std.mem.startsWith(u8, params_body, "<parameter=")) return null;
        const param_name_start = "<parameter=".len;
        const param_name_end_rel = std.mem.indexOfScalar(u8, params_body[param_name_start..], '>') orelse return null;
        const param_name_end = param_name_start + param_name_end_rel;
        const param_name = std.mem.trim(u8, params_body[param_name_start..param_name_end], " \t\r\n");
        if (param_name.len == 0) return null;
        const value_start = param_name_end + 1;
        const end_tag = "</parameter>";
        const value_end_rel = std.mem.indexOf(u8, params_body[value_start..], end_tag) orelse return null;
        const raw_value = std.mem.trim(u8, params_body[value_start .. value_start + value_end_rel], "\r\n");
        if (!first) {}
        first = false;
        try jw.objectField(param_name);
        try jw.write(raw_value);
        params_body = std.mem.trim(u8, params_body[value_start + value_end_rel + end_tag.len ..], " \t\r\n");
    }
    try jw.endObject();
    return .{
        .index = index,
        .id = null,
        .name = try allocator.dupe(u8, function_name),
        .arguments_json = try arguments.toOwnedSlice(),
    };
}

fn openModelDir(io: std.Io, path: []const u8) !std.Io.Dir {
    if (std.fs.path.isAbsolute(path)) {
        return try std.Io.Dir.openDirAbsolute(io, path, .{});
    }
    return try std.Io.Dir.cwd().openDir(io, path, .{});
}

test "qwen tokenizer encodes byte-level bpe and special tokens" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tokenizer_json =
        \\{
        \\  "added_tokens": [
        \\    {"id": 10, "content": "<|im_end|>", "special": true},
        \\    {"id": 11, "content": "<|endoftext|>", "special": true}
        \\  ],
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"h":0, "e":1, "l":2, "o":3, "he":4, "ll":5, "Ġ":6, "Ġhe":7, "<|im_end|>":10, "<|endoftext|>":11},
        \\    "merges": [["h","e"], ["l","l"], ["Ġ","he"]]
        \\  }
        \\}
    ;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.json", .data = tokenizer_json });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer_config.json", .data = "{}" });
    const model_dir = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(model_dir);

    var tokenizer = try QwenTokenizer.loadFromModelDir(std.testing.allocator, std.testing.io, model_dir);
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(std.testing.allocator, " hell<|im_end|>");
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 6, 4, 5, 10 }, ids);
    const decoded = try tokenizer.decodeAlloc(std.testing.allocator, ids, false);
    defer std.testing.allocator.free(decoded);
    try std.testing.expectEqualStrings(" hell<|im_end|>", decoded);
    try std.testing.expect(tokenizer.isEosToken(10));
}

test "qwen tokenizer renders chat prompt and extracts reasoning" {
    const empty_tokens = try std.testing.allocator.alloc(AddedToken, 0);
    const empty_eos = try std.testing.allocator.alloc(u32, 0);
    var tokenizer: QwenTokenizer = .{
        .allocator = std.testing.allocator,
        .vocab = std.StringHashMap(u32).init(std.testing.allocator),
        .id_to_token = try std.testing.allocator.alloc([]const u8, 0),
        .merge_ranks = std.StringHashMap(u32).init(std.testing.allocator),
        .added_tokens = empty_tokens,
        .special_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .special_id_set = std.AutoHashMap(u32, void).init(std.testing.allocator),
        .eos_token_ids = empty_eos,
        .image_token = try std.testing.allocator.dupe(u8, "<|image_pad|>"),
        .video_token = try std.testing.allocator.dupe(u8, "<|video_pad|>"),
        .vision_start_token = try std.testing.allocator.dupe(u8, "<|vision_start|>"),
        .vision_end_token = try std.testing.allocator.dupe(u8, "<|vision_end|>"),
    };
    defer tokenizer.deinit();

    const prompt = try tokenizer.renderMessagesAlloc(
        std.testing.allocator,
        &.{
            .{ .role = .system, .content = "Be brief." },
            .{ .role = .user, .content = "Hi" },
        },
        true,
        false,
    );
    defer std.testing.allocator.free(prompt);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>system\nBe brief.<|im_end|>\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));

    const split = tokenizer.splitAssistantReasoning("<think>\nplan\n</think>\n\nanswer", true);
    try std.testing.expectEqualStrings("plan", split.reasoning.?);
    try std.testing.expectEqualStrings("answer", split.content);
}
