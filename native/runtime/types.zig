const std = @import("std");

pub const SupportedModelFamily = enum {
    qwen3_5_text,
    qwen3_tts,
    gemma4,
};

pub const RuntimeBackend = enum {
    cpu,
    xpu_opencl,
};

pub const RuntimeSafetyPolicy = struct {
    min_free_bytes: u64 = 1 << 30,
    reserve_margin_bytes: u64 = 512 << 20,
    max_estimated_usage_ratio: f64 = 0.9,
    generation_memory_safety_factor: f64 = 2.0,
};

pub const GenerationConfig = struct {
    max_new_tokens: ?usize = null,
    temperature: f32 = 0.7,
    top_p: f32 = 0.95,
    top_k: usize = 50,
    repetition_penalty: f32 = 1.0,
    stop_strings: []const []const u8 = &.{},
};

pub const GenerationPerfStats = struct {
    total_seconds: f64,
    prefill_seconds: f64,
    ttft_seconds: f64,
    decode_seconds: f64,
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_tokens_per_second: f64,
    decode_tokens: usize,
    decode_tokens_per_second: f64,
    total_tokens_per_second: f64,
};

pub const ToolCall = struct {
    index: usize = 0,
    id: ?[]const u8 = null,
    name: []const u8,
    arguments_json: []const u8,
};

pub const StreamEvent = struct {
    text: []const u8 = "",
    reasoning_text: ?[]const u8 = null,
    tool_calls: []const ToolCall = &.{},
    finish_reason: ?[]const u8 = null,
};

pub const TextGenerationResult = struct {
    text: []const u8,
    finish_reason: []const u8,
    prompt_tokens: usize,
    completion_tokens: usize,
    reasoning_text: ?[]const u8 = null,
    tool_calls: []const ToolCall = &.{},
    perf: ?GenerationPerfStats = null,
};

pub const AnnaEngineError = struct {
    message: []const u8,
    status_code: u16 = 400,
    error_type: []const u8 = "invalid_request_error",
    code: ?[]const u8 = null,
};

pub const SpeechSynthesisConfig = struct {
    max_new_tokens: ?usize = null,
    do_sample: bool = true,
    temperature: f32 = 0.9,
    top_p: f32 = 1.0,
    top_k: usize = 50,
    repetition_penalty: f32 = 1.05,
    subtalker_do_sample: bool = true,
    subtalker_temperature: f32 = 0.9,
    subtalker_top_p: f32 = 1.0,
    subtalker_top_k: usize = 50,
    non_streaming_mode: bool = true,
};

pub const SpeechSynthesisResult = struct {
    audio: []const f32,
    sample_rate: u32,
    duration_seconds: f64,
    total_seconds: f64,
};

pub fn defaultStopStrings() []const []const u8 {
    return &.{};
}
