const std = @import("std");

pub const DecodeTokensFn = *const fn (ctx: *const anyopaque, token_ids: []const u32, allocator: std.mem.Allocator) anyerror![]u8;

pub const TokenDecoder = struct {
    ctx: *const anyopaque,
    decode_tokens: DecodeTokensFn,
};

pub const FeedResult = struct {
    delta: []const u8,
    stopped: bool,
};

pub fn stripUnstableReplacementSuffix(text: []const u8) []const u8 {
    const replacement_utf8 = "\xEF\xBF\xBD";
    var end = text.len;
    while (end >= replacement_utf8.len and std.mem.eql(u8, text[end - replacement_utf8.len .. end], replacement_utf8)) {
        end -= replacement_utf8.len;
    }
    return text[0..end];
}

pub const IncrementalTextAssembler = struct {
    allocator: std.mem.Allocator,
    decoder: TokenDecoder,
    stop_strings: []const []const u8,
    pending_token_ids: std.array_list.Managed(u32),
    pending_emitted_chars: usize = 0,
    stop_buffer: std.array_list.Managed(u8),
    max_stop_length: usize = 0,
    stopped: bool = false,

    pub fn init(allocator: std.mem.Allocator, decoder: TokenDecoder, stop_strings: []const []const u8) IncrementalTextAssembler {
        var max_stop_length: usize = 0;
        for (stop_strings) |stop| {
            if (stop.len > max_stop_length) max_stop_length = stop.len;
        }
        return .{
            .allocator = allocator,
            .decoder = decoder,
            .stop_strings = stop_strings,
            .pending_token_ids = std.array_list.Managed(u32).init(allocator),
            .stop_buffer = std.array_list.Managed(u8).init(allocator),
            .max_stop_length = max_stop_length,
        };
    }

    pub fn deinit(self: *IncrementalTextAssembler) void {
        self.pending_token_ids.deinit();
        self.stop_buffer.deinit();
    }

    pub fn feedToken(self: *IncrementalTextAssembler, token_id: u32) !FeedResult {
        if (self.stopped) return .{ .delta = "", .stopped = true };

        try self.pending_token_ids.append(token_id);
        const decoded = try self.decoder.decode_tokens(self.decoder.ctx, self.pending_token_ids.items, self.allocator);
        defer self.allocator.free(decoded);
        const stable = stripUnstableReplacementSuffix(decoded);

        var delta: []const u8 = "";
        if (stable.len > self.pending_emitted_chars) {
            delta = stable[self.pending_emitted_chars..];
            self.pending_emitted_chars = stable.len;
        }
        if (stable.len == decoded.len and self.pending_emitted_chars == decoded.len) {
            self.pending_token_ids.clearRetainingCapacity();
            self.pending_emitted_chars = 0;
        }
        return try self.pushStableText(delta);
    }

    pub fn flush(self: *IncrementalTextAssembler) !FeedResult {
        if (self.pending_token_ids.items.len > 0) {
            const decoded = try self.decoder.decode_tokens(self.decoder.ctx, self.pending_token_ids.items, self.allocator);
            defer self.allocator.free(decoded);
            const stable = stripUnstableReplacementSuffix(decoded);
            const delta = stable[self.pending_emitted_chars..];
            self.pending_token_ids.clearRetainingCapacity();
            self.pending_emitted_chars = 0;
            if (delta.len > 0) {
                return try self.flushText(delta);
            }
        }
        return try self.flushText("");
    }

    fn pushStableText(self: *IncrementalTextAssembler, text: []const u8) !FeedResult {
        if (self.stopped) return .{ .delta = "", .stopped = true };
        if (self.stop_strings.len == 0) {
            return .{ .delta = try self.allocator.dupe(u8, text), .stopped = false };
        }

        if (text.len > 0) try self.stop_buffer.appendSlice(text);
        if (findEarliestStop(self.stop_buffer.items, self.stop_strings)) |stop_index| {
            const emitted = try self.allocator.dupe(u8, self.stop_buffer.items[0..stop_index]);
            self.stop_buffer.clearRetainingCapacity();
            self.stopped = true;
            return .{ .delta = emitted, .stopped = true };
        }

        const hold_back = if (self.max_stop_length == 0) 0 else self.max_stop_length - 1;
        if (hold_back == 0) {
            const emitted = try self.allocator.dupe(u8, self.stop_buffer.items);
            self.stop_buffer.clearRetainingCapacity();
            return .{ .delta = emitted, .stopped = false };
        }

        const safe_length = utf8BoundaryBefore(self.stop_buffer.items, self.stop_buffer.items.len -| hold_back);
        const emitted = try self.allocator.dupe(u8, self.stop_buffer.items[0..safe_length]);
        if (safe_length > 0) {
            std.mem.copyForwards(u8, self.stop_buffer.items[0 .. self.stop_buffer.items.len - safe_length], self.stop_buffer.items[safe_length..]);
            self.stop_buffer.items.len -= safe_length;
        }
        return .{ .delta = emitted, .stopped = false };
    }

    fn flushText(self: *IncrementalTextAssembler, text: []const u8) !FeedResult {
        if (self.stopped) return .{ .delta = "", .stopped = true };
        if (text.len > 0) try self.stop_buffer.appendSlice(text);
        if (findEarliestStop(self.stop_buffer.items, self.stop_strings)) |stop_index| {
            const emitted = try self.allocator.dupe(u8, self.stop_buffer.items[0..stop_index]);
            self.stop_buffer.clearRetainingCapacity();
            self.stopped = true;
            return .{ .delta = emitted, .stopped = true };
        }
        const emitted = try self.allocator.dupe(u8, self.stop_buffer.items);
        self.stop_buffer.clearRetainingCapacity();
        return .{ .delta = emitted, .stopped = false };
    }
};

fn findEarliestStop(text: []const u8, stop_strings: []const []const u8) ?usize {
    var result: ?usize = null;
    for (stop_strings) |stop| {
        if (stop.len == 0) continue;
        if (std.mem.indexOf(u8, text, stop)) |index| {
            result = if (result) |current| @min(current, index) else index;
        }
    }
    return result;
}

fn utf8BoundaryBefore(text: []const u8, proposed: usize) usize {
    var index = @min(proposed, text.len);
    while (index > 0 and index < text.len and (text[index] & 0b1100_0000) == 0b1000_0000) {
        index -= 1;
    }
    return index;
}

test "strip unstable replacement suffix removes trailing replacement chars" {
    const text = "夏天。\xEF\xBF\xBD\xEF\xBF\xBD";
    try std.testing.expectEqualStrings("夏天。", stripUnstableReplacementSuffix(text));
}

test "incremental assembler defers stop string tail and flushes correctly" {
    const Harness = struct {
        fn decode(_: *const anyopaque, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
            var buffer = std.array_list.Managed(u8).init(allocator);
            for (token_ids) |token| {
                const piece = switch (token) {
                    1 => "夏",
                    2 => "天",
                    3 => "END",
                    else => "",
                };
                try buffer.appendSlice(piece);
            }
            return buffer.toOwnedSlice();
        }
    };

    var assembler = IncrementalTextAssembler.init(
        std.testing.allocator,
        .{ .ctx = @ptrFromInt(1), .decode_tokens = Harness.decode },
        &.{"END"},
    );
    defer assembler.deinit();

    const first = try assembler.feedToken(1);
    defer std.testing.allocator.free(first.delta);
    try std.testing.expectEqualStrings("", first.delta);
    try std.testing.expect(!first.stopped);

    const second = try assembler.feedToken(2);
    defer std.testing.allocator.free(second.delta);
    try std.testing.expectEqualStrings("夏", second.delta);
    try std.testing.expect(!second.stopped);

    const third = try assembler.feedToken(3);
    defer std.testing.allocator.free(third.delta);
    try std.testing.expectEqualStrings("天", third.delta);
    try std.testing.expect(third.stopped);
}
