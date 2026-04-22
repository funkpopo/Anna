const std = @import("std");

pub const RouteSpec = struct {
    path: []const u8,
    methods: []const u8,
};

pub fn defaultRoutes() []const RouteSpec {
    return &.{
        .{ .path = "/openapi.json", .methods = "HEAD, GET" },
        .{ .path = "/docs", .methods = "HEAD, GET" },
        .{ .path = "/healthz", .methods = "GET" },
        .{ .path = "/v1/models", .methods = "GET" },
        .{ .path = "/v1/chat/completions", .methods = "POST" },
        .{ .path = "/v1/completions", .methods = "POST" },
        .{ .path = "/v1/audio/speech", .methods = "POST" },
    };
}

pub fn writeRouteAnnouncement(writer: anytype, routes: []const RouteSpec, host: []const u8, port: u16) !void {
    try writer.print("Starting Anna server on http://{s}:{d}\n", .{ host, port });
    try writer.writeAll("Available routes are:\n");
    for (routes) |route| {
        try writer.print("Route: {s}, Methods: {s}\n", .{ route.path, route.methods });
    }
}

test "default routes include OpenAI surface" {
    const routes = defaultRoutes();
    try std.testing.expect(routes.len >= 6);

    var saw_openapi = false;
    var saw_chat = false;
    var saw_audio = false;
    for (routes) |route| {
        saw_openapi = saw_openapi or (std.mem.eql(u8, route.path, "/openapi.json") and std.mem.eql(u8, route.methods, "HEAD, GET"));
        saw_chat = saw_chat or (std.mem.eql(u8, route.path, "/v1/chat/completions") and std.mem.eql(u8, route.methods, "POST"));
        saw_audio = saw_audio or (std.mem.eql(u8, route.path, "/v1/audio/speech") and std.mem.eql(u8, route.methods, "POST"));
    }
    try std.testing.expect(saw_openapi);
    try std.testing.expect(saw_chat);
    try std.testing.expect(saw_audio);
}

test "route announcement mirrors python cli log" {
    var buf: [1024]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buf);
    try writeRouteAnnouncement(&writer, defaultRoutes(), "127.0.0.1", 8000);
    const output = writer.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "Starting Anna server on http://127.0.0.1:8000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Route: /healthz, Methods: GET") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Route: /v1/chat/completions, Methods: POST") != null);
}
