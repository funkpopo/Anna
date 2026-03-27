from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageContentPart(BaseModel):
    type: Literal["text", "image_url", "video_url"]
    text: str | None = None
    image_url: Any | None = None
    video_url: Any | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[MessageContentPart] | None = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None


class ChatTemplateKwargs(BaseModel):
    enable_thinking: bool | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
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
    stop: str | list[str] | None = None
