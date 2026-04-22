const std = @import("std");

const RankedLogit = struct {
    index: usize,
    value: f32,
    probability: f32 = 0.0,
};

pub fn applyRepetitionPenalty(allocator: std.mem.Allocator, logits: []const f32, generated_ids: []const u32, penalty: f32) ![]f32 {
    const output = try allocator.dupe(f32, logits);
    if (generated_ids.len == 0 or penalty == 1.0) return output;

    var seen = std.AutoHashMap(u32, void).init(allocator);
    defer seen.deinit();
    for (generated_ids) |token_id| {
        if (token_id >= logits.len) continue;
        if ((try seen.getOrPut(token_id)).found_existing) continue;
        output[token_id] = if (output[token_id] < 0.0) output[token_id] * penalty else output[token_id] / penalty;
    }
    return output;
}

pub fn applyTopK(allocator: std.mem.Allocator, logits: []const f32, top_k: usize) ![]f32 {
    const filtered = try allocator.dupe(f32, logits);
    if (top_k == 0 or top_k >= logits.len) return filtered;

    var threshold = -std.math.inf(f32);
    var scratch = try allocator.dupe(f32, logits);
    defer allocator.free(scratch);
    std.mem.copyForwards(f32, scratch, logits);

    var selected: usize = 0;
    while (selected < top_k) : (selected += 1) {
        var best_index: ?usize = null;
        for (scratch, 0..) |value, idx| {
            if (best_index == null or value > scratch[best_index.?]) best_index = idx;
        }
        threshold = scratch[best_index.?];
        scratch[best_index.?] = -std.math.inf(f32);
    }

    for (filtered) |*value| {
        if (value.* < threshold) value.* = -std.math.inf(f32);
    }
    return filtered;
}

pub fn applyTopP(allocator: std.mem.Allocator, logits: []const f32, top_p: f32) ![]f32 {
    const filtered = try allocator.dupe(f32, logits);
    if (top_p >= 1.0) return filtered;

    var items = std.array_list.Managed(RankedLogit).init(allocator);
    defer items.deinit();
    for (logits, 0..) |value, index| {
        try items.append(.{ .index = index, .value = value });
    }
    sortDescending(items.items);

    const max_logit = items.items[0].value;
    var sum: f64 = 0.0;
    for (items.items) |*item| {
        item.probability = @floatCast(std.math.exp(@as(f64, item.value - max_logit)));
        sum += item.probability;
    }

    var cumulative: f64 = 0.0;
    var cutoff_started = false;
    for (items.items, 0..) |item, idx| {
        const probability = @as(f64, item.probability) / sum;
        cumulative += probability;
        if (idx == 0) continue;
        if (!cutoff_started and cumulative > top_p) {
            cutoff_started = true;
        }
        if (cutoff_started) filtered[item.index] = -std.math.inf(f32);
    }
    return filtered;
}

pub fn sampleNextToken(
    allocator: std.mem.Allocator,
    logits: []const f32,
    generated_ids: []const u32,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    rng: *std.Random,
) !u32 {
    const penalized = try applyRepetitionPenalty(allocator, logits, generated_ids, repetition_penalty);
    defer allocator.free(penalized);

    if (temperature <= 0.0) {
        return @intCast(argmax(penalized));
    }

    for (penalized) |*value| {
        value.* /= temperature;
    }

    const top_k_logits = try applyTopK(allocator, penalized, top_k);
    defer allocator.free(top_k_logits);
    const top_p_logits = try applyTopP(allocator, top_k_logits, top_p);
    defer allocator.free(top_p_logits);

    const max_logit = top_p_logits[argmax(top_p_logits)];
    var weights = try allocator.alloc(f64, top_p_logits.len);
    defer allocator.free(weights);
    var total: f64 = 0.0;
    for (top_p_logits, 0..) |value, idx| {
        if (std.math.isInf(value) and value < 0) {
            weights[idx] = 0.0;
        } else {
            weights[idx] = std.math.exp(@as(f64, value - max_logit));
            total += weights[idx];
        }
    }

    const needle = rng.float(f64) * total;
    var running: f64 = 0.0;
    for (weights, 0..) |weight, idx| {
        running += weight;
        if (running >= needle) return @intCast(idx);
    }
    return @intCast(argmax(top_p_logits));
}

fn argmax(values: []const f32) usize {
    var best_index: usize = 0;
    for (values[1..], 1..) |value, idx| {
        if (value > values[best_index]) best_index = idx;
    }
    return best_index;
}

fn sortDescending(items: []RankedLogit) void {
    var i: usize = 0;
    while (i < items.len) : (i += 1) {
        var best = i;
        var j = i + 1;
        while (j < items.len) : (j += 1) {
            if (items[j].value > items[best].value) best = j;
        }
        if (best != i) std.mem.swap(@TypeOf(items[0]), &items[i], &items[best]);
    }
}

test "apply repetition penalty adjusts repeated positive and negative logits" {
    const logits = [_]f32{ 1.0, -2.0, 3.0 };
    const generated = [_]u32{ 0, 1, 1 };
    const output = try applyRepetitionPenalty(std.testing.allocator, &logits, &generated, 2.0);
    defer std.testing.allocator.free(output);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), output[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output[2], 1e-6);
}

test "apply top k masks logits below kth threshold" {
    const logits = [_]f32{ 1.0, 5.0, 4.0, 2.0 };
    const filtered = try applyTopK(std.testing.allocator, &logits, 2);
    defer std.testing.allocator.free(filtered);
    try std.testing.expect(std.math.isInf(filtered[0]) and filtered[0] < 0);
    try std.testing.expectEqual(@as(f32, 5.0), filtered[1]);
    try std.testing.expectEqual(@as(f32, 4.0), filtered[2]);
}

test "temperature zero sampling becomes argmax" {
    var prng = std.Random.DefaultPrng.init(1234);
    const logits = [_]f32{ 0.1, 2.0, 1.2 };
    var random = prng.random();
    const token = try sampleNextToken(std.testing.allocator, &logits, &.{}, 0.0, 1.0, 0, 1.0, &random);
    try std.testing.expectEqual(@as(u32, 1), token);
}
