from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class MessageContentPart(BaseModel):
    type: Literal["text", "image_url", "video_url", "audio_url"]
    text: str | None = None
    image_url: Any | None = None
    video_url: Any | None = None
    audio_url: Any | None = None


class ChatToolCallFunction(BaseModel):
    name: str
    arguments: str | dict[str, Any] | None = None


class ChatToolCall(BaseModel):
    id: str | None = None
    type: Literal["function"] = "function"
    function: ChatToolCallFunction


class ChatToolFunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None
    response: dict[str, Any] | None = None


class ChatCompletionTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    function: ChatToolFunctionDefinition


class NamedToolChoiceFunction(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: NamedToolChoiceFunction


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[MessageContentPart] | None = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ChatToolCall] | None = None


class ChatTemplateKwargs(BaseModel):
    enable_thinking: bool | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    tools: list[ChatCompletionTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    enable_thinking: bool | None = None
    chat_template_kwargs: ChatTemplateKwargs | None = None
    reasoning_format: Literal["none", "deepseek"] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=0.1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stream_include_usage: bool | None = None
    stop: str | list[str] | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=0.1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stream_include_usage: bool | None = None
    stop: str | list[str] | None = None


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str = Field(min_length=1)
    voice: str | None = None
    speaker: str | None = None
    language: str | None = None
    instruct: str | None = None
    ref_audio: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ref_audio", "reference_audio"),
    )
    ref_text: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ref_text", "reference_text"),
    )
    x_vector_only_mode: bool = False
    response_format: Literal["wav", "flac", "pcm"] = "wav"
    max_new_tokens: int | None = Field(default=None, ge=1)
    do_sample: bool = True
    temperature: float = Field(default=0.9, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.05, ge=0.1)
    subtalker_do_sample: bool = True
    subtalker_temperature: float = Field(default=0.9, ge=0.0)
    subtalker_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    subtalker_top_k: int = Field(default=50, ge=0)
    non_streaming_mode: bool = True
