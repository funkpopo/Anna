from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

ToolChoiceMode = Literal["none", "auto", "required", "named"]


@dataclass(slots=True)
class ResolvedToolChoice:
    mode: ToolChoiceMode
    function_name: str | None = None


@dataclass(slots=True)
class ParsedToolCall:
    name: str
    arguments: str
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex}")
    type: str = "function"

    def to_openai_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(slots=True)
class ToolCallDelta:
    index: int
    id: str
    name: str | None = None
    arguments: str | None = None
    type: str = "function"

    def to_openai_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "index": self.index,
        }
        if self.id:
            payload["id"] = self.id
        if self.type:
            payload["type"] = self.type
        function: dict[str, str] = {}
        if self.name is not None:
            function["name"] = self.name
        if self.arguments is not None:
            function["arguments"] = self.arguments
        if function:
            payload["function"] = function
        return payload


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("Expected a mapping-like value.")


def normalize_tool_choice(
    tool_choice: Any,
    tools: list[Any] | None,
) -> ResolvedToolChoice:
    has_tools = bool(tools)
    if tool_choice is None:
        return ResolvedToolChoice("auto" if has_tools else "none")

    if isinstance(tool_choice, str):
        normalized = tool_choice.strip().lower()
        if normalized not in {"none", "auto", "required"}:
            raise ValueError("tool_choice must be one of: none, auto, required, or a named function choice.")
        if normalized != "none" and not has_tools:
            raise ValueError("tool_choice requires tools to be provided.")
        return ResolvedToolChoice(normalized)

    choice_dict = model_to_dict(tool_choice)
    choice_type = str(choice_dict.get("type") or "").strip().lower()
    if choice_type != "function":
        raise ValueError("Named tool_choice objects must use type='function'.")
    function = choice_dict.get("function")
    function_dict = model_to_dict(function)
    function_name = str(function_dict.get("name") or "").strip()
    if not function_name:
        raise ValueError("Named tool_choice objects must include function.name.")
    if not has_tools:
        raise ValueError("Named tool_choice requires tools to be provided.")
    return ResolvedToolChoice("named", function_name=function_name)


def select_tools_for_choice(
    tools: list[Any] | None,
    tool_choice: ResolvedToolChoice,
) -> list[dict[str, Any]]:
    tool_dicts = [model_to_dict(tool) for tool in tools or []]
    if tool_choice.mode == "none":
        return []
    if tool_choice.mode != "named":
        return tool_dicts

    selected = []
    for tool in tool_dicts:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        if str(function.get("name") or "").strip() == tool_choice.function_name:
            selected.append(tool)
    if not selected:
        raise ValueError(f"tool_choice referenced unknown function: {tool_choice.function_name}")
    return selected


def coerce_arguments_mapping(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if hasattr(arguments, "model_dump"):
        dumped = arguments.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool call arguments strings must decode to a JSON object.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Tool call arguments must decode to a JSON object.")
        return parsed
    raise ValueError("Tool call arguments must be provided as an object or a JSON object string.")


def normalize_arguments_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return "{}"
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("Tool call arguments must decode to a JSON object.")
        return compact_json(parsed)
    return compact_json(coerce_arguments_mapping(arguments))


def parse_tool_response_content(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def extract_delimited_tool_calls(
    text: str,
    *,
    start_tag: str,
    end_tag: str,
    block_parser: Callable[[str], ParsedToolCall | None],
) -> tuple[str, list[ParsedToolCall]]:
    if not text:
        return text, []

    cursor = 0
    parts: list[str] = []
    calls: list[ParsedToolCall] = []

    while cursor < len(text):
        start = text.find(start_tag, cursor)
        if start == -1:
            parts.append(text[cursor:])
            break
        parts.append(text[cursor:start])
        end = text.find(end_tag, start + len(start_tag))
        if end == -1:
            parts.append(text[start:])
            break
        end += len(end_tag)
        block = text[start:end]
        parsed = block_parser(block)
        if parsed is None:
            parts.append(block)
        else:
            calls.append(parsed)
        cursor = end

    cleaned = "".join(parts)
    if calls:
        cleaned = cleaned.strip()
    return cleaned, calls


class ThinkingStreamParser:
    CLOSE_TAG = "</think>"

    def __init__(self, *, enable_thinking: bool) -> None:
        self.enable_thinking = enable_thinking
        self.state = "reasoning" if enable_thinking else "content"
        self.buffer = ""

    @staticmethod
    def _strip_think_open_tag(text: str) -> str:
        normalized = text
        if normalized.startswith("<think>"):
            normalized = normalized[len("<think>") :]
            normalized = normalized.lstrip("\r\n")
        return normalized

    def _emit_reasoning_prefix(self) -> list[tuple[str, str]]:
        if self.state != "reasoning":
            return []
        self.buffer = self._strip_think_open_tag(self.buffer)
        hold_back = 0
        max_suffix = min(len(self.buffer), len(self.CLOSE_TAG) - 1)
        for suffix_length in range(max_suffix, 0, -1):
            if self.buffer.endswith(self.CLOSE_TAG[:suffix_length]):
                hold_back = suffix_length
                break
        safe_length = max(0, len(self.buffer) - hold_back)
        if safe_length <= 0:
            return []
        reasoning = self.buffer[:safe_length]
        self.buffer = self.buffer[safe_length:]
        return [("reasoning", reasoning)]

    def feed(self, text: str) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        self.buffer += text

        while True:
            if self.state == "content":
                stripped = self.buffer.lstrip()
                if stripped.startswith("<think>"):
                    self.buffer = stripped
                    self.state = "reasoning"
                    continue

            if self.state == "reasoning":
                self.buffer = self._strip_think_open_tag(self.buffer)
                close_index = self.buffer.find(self.CLOSE_TAG)
                if close_index == -1:
                    outputs.extend(self._emit_reasoning_prefix())
                    break
                reasoning = self.buffer[:close_index].rstrip("\r\n")
                if reasoning:
                    outputs.append(("reasoning", reasoning))
                self.buffer = self.buffer[close_index + len(self.CLOSE_TAG) :].lstrip("\r\n")
                self.state = "content"
                continue

            if self.buffer:
                outputs.append(("content", self.buffer))
                self.buffer = ""
            break

        return outputs

    def flush(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        if self.state == "reasoning":
            self.buffer = self._strip_think_open_tag(self.buffer)
            reasoning = self.buffer.rstrip("\r\n")
            if reasoning:
                outputs.append(("reasoning", reasoning))
        elif self.buffer:
            outputs.append(("content", self.buffer))
        self.buffer = ""
        return outputs


class GemmaThinkingStreamParser:
    OPEN_TAG = "<|channel>thought\n"
    CLOSE_TAG = "<channel|>"

    def __init__(self, *, enable_thinking: bool) -> None:
        self.enable_thinking = enable_thinking
        self.state = "reasoning" if enable_thinking else "content"
        self.buffer = ""

    def _emit_reasoning_prefix(self) -> list[tuple[str, str]]:
        if self.state != "reasoning":
            return []
        hold_back = 0
        max_suffix = min(len(self.buffer), len(self.CLOSE_TAG) - 1)
        for suffix_length in range(max_suffix, 0, -1):
            if self.buffer.endswith(self.CLOSE_TAG[:suffix_length]):
                hold_back = suffix_length
                break
        safe_length = max(0, len(self.buffer) - hold_back)
        if safe_length <= 0:
            return []
        reasoning = self.buffer[:safe_length]
        self.buffer = self.buffer[safe_length:]
        return [("reasoning", reasoning)]

    def feed(self, text: str) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        self.buffer += text

        while True:
            if self.state == "content":
                start_index = self.buffer.find(self.OPEN_TAG)
                if start_index == -1:
                    hold_back = 0
                    max_suffix = min(len(self.buffer), len(self.OPEN_TAG) - 1)
                    for suffix_length in range(max_suffix, 0, -1):
                        if self.OPEN_TAG.startswith(self.buffer[-suffix_length:]):
                            hold_back = suffix_length
                            break
                    safe_length = max(0, len(self.buffer) - hold_back)
                    if safe_length <= 0:
                        break
                    outputs.append(("content", self.buffer[:safe_length]))
                    self.buffer = self.buffer[safe_length:]
                    break
                if start_index > 0:
                    outputs.append(("content", self.buffer[:start_index]))
                    self.buffer = self.buffer[start_index:]
                if self.buffer.startswith(self.OPEN_TAG):
                    self.buffer = self.buffer[len(self.OPEN_TAG) :]
                    self.state = "reasoning"
                    continue

            if self.state == "reasoning":
                close_index = self.buffer.find(self.CLOSE_TAG)
                if close_index == -1:
                    outputs.extend(self._emit_reasoning_prefix())
                    break
                reasoning = self.buffer[:close_index].rstrip("\r\n")
                if reasoning:
                    outputs.append(("reasoning", reasoning))
                self.buffer = self.buffer[close_index + len(self.CLOSE_TAG) :].lstrip("\r\n")
                self.state = "content"
                continue

            break

        return outputs

    def flush(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        if self.state == "reasoning":
            reasoning = self.buffer.rstrip("\r\n")
            if reasoning:
                outputs.append(("reasoning", reasoning))
        elif self.buffer:
            outputs.append(("content", self.buffer))
        self.buffer = ""
        return outputs


class DelimitedToolCallStreamParser:
    def __init__(
        self,
        *,
        start_tag: str,
        end_tag: str,
        block_parser: Callable[[str], ParsedToolCall | None],
    ) -> None:
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.block_parser = block_parser
        self.buffer = ""
        self.next_index = 0
        self.saw_tool_calls = False

    def _parse_buffer(self, *, final: bool) -> list[tuple[str, str | ToolCallDelta]]:
        outputs: list[tuple[str, str | ToolCallDelta]] = []
        while self.buffer:
            start_index = self.buffer.find(self.start_tag)
            if start_index == -1:
                if final:
                    outputs.append(("content", self.buffer))
                    self.buffer = ""
                else:
                    hold_back = 0
                    max_suffix = min(len(self.buffer), len(self.start_tag) - 1)
                    for suffix_length in range(max_suffix, 0, -1):
                        if self.start_tag.startswith(self.buffer[-suffix_length:]):
                            hold_back = suffix_length
                            break
                    safe_length = max(0, len(self.buffer) - hold_back)
                    if safe_length <= 0:
                        break
                    outputs.append(("content", self.buffer[:safe_length]))
                    self.buffer = self.buffer[safe_length:]
                break

            if start_index > 0:
                outputs.append(("content", self.buffer[:start_index]))
                self.buffer = self.buffer[start_index:]

            end_index = self.buffer.find(self.end_tag, len(self.start_tag))
            if end_index == -1:
                if final:
                    outputs.append(("content", self.buffer))
                    self.buffer = ""
                break

            end_index += len(self.end_tag)
            block = self.buffer[:end_index]
            self.buffer = self.buffer[end_index:]
            parsed = self.block_parser(block)
            if parsed is None:
                outputs.append(("content", block))
                continue
            self.saw_tool_calls = True
            outputs.append(
                (
                    "tool_call",
                    ToolCallDelta(
                        index=self.next_index,
                        id=parsed.id,
                        name=parsed.name,
                        arguments=parsed.arguments,
                    ),
                )
            )
            self.next_index += 1
        return outputs

    def feed(self, text: str) -> list[tuple[str, str | ToolCallDelta]]:
        self.buffer += text
        return self._parse_buffer(final=False)

    def flush(self) -> list[tuple[str, str | ToolCallDelta]]:
        return self._parse_buffer(final=True)
