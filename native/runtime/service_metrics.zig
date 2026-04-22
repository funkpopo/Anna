const std = @import("std");

pub const ServiceMetricsSnapshot = struct {
    timestamp: f64,
    requests_started_total: usize = 0,
    requests_completed_total: usize = 0,
    requests_failed_total: usize = 0,
    prompt_tokens_total: usize = 0,
    generation_tokens_total: usize = 0,
    prompt_cache_queries_total: usize = 0,
    prompt_cache_hits_total: usize = 0,
    running_requests: usize = 0,
    waiting_requests: usize = 0,
    kv_cache_used_pages: usize = 0,
    kv_cache_total_pages: usize = 0,
    prompt_cache_entries: usize = 0,

    pub fn kvCacheUsageRatio(self: ServiceMetricsSnapshot) f64 {
        if (self.kv_cache_total_pages == 0) return 0.0;
        return @as(f64, @floatFromInt(self.kv_cache_used_pages)) / @as(f64, @floatFromInt(self.kv_cache_total_pages));
    }
};

pub const AnnaServiceMetrics = struct {
    mutex: std.atomic.Mutex = .unlocked,
    requests_started_total: usize = 0,
    requests_completed_total: usize = 0,
    requests_failed_total: usize = 0,
    prompt_tokens_total: usize = 0,
    generation_tokens_total: usize = 0,
    prompt_cache_queries_total: usize = 0,
    prompt_cache_hits_total: usize = 0,
    running_requests: usize = 0,
    waiting_requests: usize = 0,
    activity_flag: bool = false,

    pub fn recordRequestSubmitted(self: *AnnaServiceMetrics, waiting: bool) void {
        self.lock();
        defer self.unlock();
        self.requests_started_total += 1;
        if (waiting) self.waiting_requests += 1 else self.running_requests += 1;
        self.activity_flag = true;
    }

    pub fn recordRequestsStartedFromQueue(self: *AnnaServiceMetrics, count: usize) void {
        if (count == 0) return;
        self.lock();
        defer self.unlock();
        self.waiting_requests -|= count;
        self.running_requests += count;
        self.activity_flag = true;
    }

    pub fn recordRequestFinished(self: *AnnaServiceMetrics, success: bool) void {
        self.lock();
        defer self.unlock();
        self.running_requests -|= 1;
        if (success) self.requests_completed_total += 1 else self.requests_failed_total += 1;
        self.activity_flag = true;
    }

    pub fn recordPromptTokens(self: *AnnaServiceMetrics, count: usize) void {
        if (count == 0) return;
        self.lock();
        defer self.unlock();
        self.prompt_tokens_total += count;
        self.activity_flag = true;
    }

    pub fn recordGenerationTokens(self: *AnnaServiceMetrics, count: usize) void {
        if (count == 0) return;
        self.lock();
        defer self.unlock();
        self.generation_tokens_total += count;
        self.activity_flag = true;
    }

    pub fn recordPromptCacheLookup(self: *AnnaServiceMetrics, hit: bool) void {
        self.lock();
        defer self.unlock();
        self.prompt_cache_queries_total += 1;
        if (hit) self.prompt_cache_hits_total += 1;
        self.activity_flag = true;
    }

    pub fn snapshot(self: *AnnaServiceMetrics) ServiceMetricsSnapshot {
        self.lock();
        defer self.unlock();
        return .{
            .timestamp = 0.0,
            .requests_started_total = self.requests_started_total,
            .requests_completed_total = self.requests_completed_total,
            .requests_failed_total = self.requests_failed_total,
            .prompt_tokens_total = self.prompt_tokens_total,
            .generation_tokens_total = self.generation_tokens_total,
            .prompt_cache_queries_total = self.prompt_cache_queries_total,
            .prompt_cache_hits_total = self.prompt_cache_hits_total,
            .running_requests = self.running_requests,
            .waiting_requests = self.waiting_requests,
        };
    }

    pub fn activityIsSet(self: *AnnaServiceMetrics) bool {
        self.lock();
        defer self.unlock();
        return self.activity_flag;
    }

    fn lock(self: *AnnaServiceMetrics) void {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
    }

    fn unlock(self: *AnnaServiceMetrics) void {
        self.mutex.unlock();
    }
};

pub const AnnaServiceMetricsLogger = struct {
    pub fn formatInterval(previous: ServiceMetricsSnapshot, current: ServiceMetricsSnapshot, allocator: std.mem.Allocator) ![]u8 {
        const elapsed = @max(1e-9, current.timestamp - previous.timestamp);
        const prompt_tokens = current.prompt_tokens_total - previous.prompt_tokens_total;
        const generation_tokens = current.generation_tokens_total - previous.generation_tokens_total;
        const cache_queries = current.prompt_cache_queries_total - previous.prompt_cache_queries_total;
        const cache_hits = current.prompt_cache_hits_total - previous.prompt_cache_hits_total;
        const prompt_tokens_per_second = @as(f64, @floatFromInt(prompt_tokens)) / elapsed;
        const generation_tokens_per_second = @as(f64, @floatFromInt(generation_tokens)) / elapsed;
        const hit_rate = if (cache_queries == 0) 0.0 else (@as(f64, @floatFromInt(cache_hits)) / @as(f64, @floatFromInt(cache_queries))) * 100.0;
        const kv_usage = current.kvCacheUsageRatio() * 100.0;

        return try std.fmt.allocPrint(
            allocator,
            "Engine metrics: Avg prompt throughput: {d:.1} tokens/s, Avg generation throughput: {d:.1} tokens/s, Running: {d} reqs, Waiting: {d} reqs, GPU KV cache usage: {d:.1}% ({d}/{d} pages), Prompt cache hit rate: {d:.1}%",
            .{
                prompt_tokens_per_second,
                generation_tokens_per_second,
                current.running_requests,
                current.waiting_requests,
                kv_usage,
                current.kv_cache_used_pages,
                current.kv_cache_total_pages,
                hit_rate,
            },
        );
    }

    pub fn shouldLogInterval(previous: ServiceMetricsSnapshot, current: ServiceMetricsSnapshot) bool {
        if (current.running_requests > 0 or current.waiting_requests > 0) return true;
        return current.requests_started_total != previous.requests_started_total or
            current.requests_completed_total != previous.requests_completed_total or
            current.requests_failed_total != previous.requests_failed_total or
            current.prompt_tokens_total != previous.prompt_tokens_total or
            current.generation_tokens_total != previous.generation_tokens_total or
            current.prompt_cache_queries_total != previous.prompt_cache_queries_total or
            current.prompt_cache_hits_total != previous.prompt_cache_hits_total or
            current.kv_cache_used_pages != previous.kv_cache_used_pages or
            current.kv_cache_total_pages != previous.kv_cache_total_pages or
            current.prompt_cache_entries != previous.prompt_cache_entries;
    }
};

test "service metrics tracks request queueing and counters" {
    var metrics = AnnaServiceMetrics{};
    try std.testing.expect(!metrics.activityIsSet());

    metrics.recordRequestSubmitted(true);
    metrics.recordRequestSubmitted(false);
    metrics.recordRequestsStartedFromQueue(1);
    metrics.recordPromptTokens(12);
    metrics.recordGenerationTokens(5);
    metrics.recordPromptCacheLookup(true);
    metrics.recordPromptCacheLookup(false);
    metrics.recordRequestFinished(true);
    metrics.recordRequestFinished(false);

    const snapshot = metrics.snapshot();
    try std.testing.expectEqual(@as(usize, 2), snapshot.requests_started_total);
    try std.testing.expectEqual(@as(usize, 1), snapshot.requests_completed_total);
    try std.testing.expectEqual(@as(usize, 1), snapshot.requests_failed_total);
    try std.testing.expectEqual(@as(usize, 12), snapshot.prompt_tokens_total);
    try std.testing.expectEqual(@as(usize, 5), snapshot.generation_tokens_total);
    try std.testing.expectEqual(@as(usize, 2), snapshot.prompt_cache_queries_total);
    try std.testing.expectEqual(@as(usize, 1), snapshot.prompt_cache_hits_total);
    try std.testing.expectEqual(@as(usize, 0), snapshot.running_requests);
    try std.testing.expectEqual(@as(usize, 0), snapshot.waiting_requests);
    try std.testing.expect(metrics.activityIsSet());
}

test "service metrics logger formats interval rates" {
    const previous: ServiceMetricsSnapshot = .{
        .timestamp = 10.0,
        .prompt_tokens_total = 8,
        .generation_tokens_total = 4,
        .prompt_cache_queries_total = 1,
        .prompt_cache_hits_total = 0,
    };
    const current: ServiceMetricsSnapshot = .{
        .timestamp = 12.0,
        .prompt_tokens_total = 24,
        .generation_tokens_total = 14,
        .prompt_cache_queries_total = 5,
        .prompt_cache_hits_total = 3,
        .running_requests = 2,
        .waiting_requests = 1,
        .kv_cache_used_pages = 6,
        .kv_cache_total_pages = 12,
    };

    const line = try AnnaServiceMetricsLogger.formatInterval(previous, current, std.testing.allocator);
    defer std.testing.allocator.free(line);

    try std.testing.expect(std.mem.indexOf(u8, line, "Avg prompt throughput: 8.0 tokens/s") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "Avg generation throughput: 5.0 tokens/s") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "Running: 2 reqs") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "Waiting: 1 reqs") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "GPU KV cache usage: 50.0% (6/12 pages)") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "Prompt cache hit rate: 75.0%") != null);
}

test "service metrics logger skips idle intervals without changes" {
    const previous: ServiceMetricsSnapshot = .{
        .timestamp = 10.0,
        .kv_cache_total_pages = 128,
    };
    const current: ServiceMetricsSnapshot = .{
        .timestamp = 20.0,
        .kv_cache_total_pages = 128,
    };

    try std.testing.expect(!AnnaServiceMetricsLogger.shouldLogInterval(previous, current));
}

test "service metrics logger logs idle interval after completed work" {
    const previous: ServiceMetricsSnapshot = .{ .timestamp = 10.0 };
    const current: ServiceMetricsSnapshot = .{
        .timestamp = 20.0,
        .requests_started_total = 1,
        .requests_completed_total = 1,
        .prompt_tokens_total = 32,
        .generation_tokens_total = 8,
        .kv_cache_total_pages = 128,
    };
    try std.testing.expect(AnnaServiceMetricsLogger.shouldLogInterval(previous, current));
}
