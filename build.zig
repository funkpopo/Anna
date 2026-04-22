const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const anna_native = b.addModule("anna_native", .{
        .root_source_file = b.path("native/root.zig"),
        .target = target,
    });

    const exe = b.addExecutable(.{
        .name = "anna-native",
        .root_module = b.createModule(.{
            .root_source_file = b.path("native/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "anna_native", .module = anna_native },
            },
        }),
    });
    b.installArtifact(exe);

    const sycl_backend_cmd = b.addSystemCommand(&.{
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        b.pathFromRoot("native/xpu/build_xpu_backend.ps1"),
        "-Source",
        b.pathFromRoot("native/xpu/sycl_backend.cpp"),
        "-OutDir",
        b.pathFromRoot("zig-out/bin"),
    });
    const xpu_backend_step = b.step("xpu-backend", "Build the native XPU backend DLL");
    xpu_backend_step.dependOn(&sycl_backend_cmd.step);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run anna-native");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("native/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run anna-native tests");
    test_step.dependOn(&run_unit_tests.step);
}
