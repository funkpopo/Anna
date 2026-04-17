from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from anna.core.function_calling import (
    DelimitedToolCallStreamParser,
    ParsedToolCall,
    ThinkingStreamParser,
    coerce_arguments_mapping,
    compact_json,
    extract_delimited_tool_calls,
    model_to_dict,
    normalize_tool_choice,
    select_tools_for_choice,
)
from tokenizers import Tokenizer


class Qwen3_5TextTokenizer:
    _QWEN_TOOL_CALL_RE = re.compile(
        r"^\s*<tool_call>\s*<function=([^\n>]+)>\s*(.*?)</function>\s*</tool_call>\s*$",
        re.DOTALL,
    )
    _QWEN_PARAMETER_RE = re.compile(r"<parameter=([^\n>]+)>\s*(.*?)</parameter>", re.DOTALL)

    def __init__(self, backend: Tokenizer, metadata: dict[str, Any] | None = None):
        self.backend = backend
        self.metadata = metadata or {}
        extra = self.metadata.get("extra_special_tokens", {})
        self.image_token = extra.get("image_token", "<|image_pad|>")
        self.video_token = extra.get("video_token", "<|video_pad|>")
        self.vision_start_token = extra.get("vision_bos_token", "<|vision_start|>")
        self.vision_end_token = extra.get("vision_eos_token", "<|vision_end|>")

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen3_5TextTokenizer":
        model_path = Path(model_dir)
        tokenizer_path = model_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Missing tokenizer.json in {model_path}")

        metadata = None
        config_path = model_path / "tokenizer_config.json"
        if config_path.exists():
            metadata = json.loads(config_path.read_text(encoding="utf-8"))

        return cls(Tokenizer.from_file(str(tokenizer_path)), metadata=metadata)

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text).ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        if not token_ids:
            return ""
        return self.backend.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_id(self, token: str) -> int | None:
        token_id = self.backend.token_to_id(token)
        return None if token_id is None else int(token_id)

    @property
    def image_token_id(self) -> int | None:
        return self.token_id(self.image_token)

    @property
    def video_token_id(self) -> int | None:
        return self.token_id(self.video_token)

    @property
    def vision_start_token_id(self) -> int | None:
        return self.token_id(self.vision_start_token)

    @property
    def vision_end_token_id(self) -> int | None:
        return self.token_id(self.vision_end_token)

    @property
    def eos_token_ids(self) -> set[int]:
        # Qwen chat templates should never surface a fresh role boundary in assistant content.
        tokens = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
        return {token_id for token in tokens if (token_id := self.token_id(token)) is not None}

    @staticmethod
    def _message_value(message: Any, key: str, default: Any = None) -> Any:
        value = getattr(message, key, None)
        if value is not None:
            return value
        if isinstance(message, dict):
            return message.get(key, default)
        return default

    def split_assistant_reasoning(self, text: str, *, enable_thinking: bool) -> tuple[str | None, str]:
        normalized = text.lstrip()
        explicit_open = normalized.startswith("<think>")
        if explicit_open:
            normalized = normalized.removeprefix("<think>").lstrip("\r\n")
        closing_tag = "</think>"
        closing_index = normalized.find(closing_tag)
        if closing_index != -1:
            reasoning = normalized[:closing_index].strip()
            content = normalized[closing_index + len(closing_tag) :].lstrip("\r\n")
            return reasoning or None, content
        if not explicit_open and not enable_thinking:
            return None, text
        reasoning = normalized.strip()
        content = ""
        return reasoning or None, content

    def _flatten_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if hasattr(item, "type"):
                    item_type = getattr(item, "type")
                    text = getattr(item, "text", None)
                else:
                    item_type = item.get("type")
                    text = item.get("text")

                if item_type == "text":
                    chunks.append(text or "")
                elif item_type in {"image", "image_url"}:
                    chunks.append(f"{self.vision_start_token}{self.image_token}{self.vision_end_token}")
                elif item_type in {"video", "video_url"}:
                    chunks.append(f"{self.vision_start_token}{self.video_token}{self.vision_end_token}")
                else:
                    raise ValueError(f"Unsupported content part type: {item_type}")
            return "".join(chunks)
        raise ValueError("Unsupported chat content format.")

    def _normalize_role(self, role: str, *, index: int) -> str:
        normalized = role.strip().lower()
        if normalized == "developer":
            if index != 0:
                raise ValueError("Developer message must be the first message.")
            return "system"
        return normalized

    def _assistant_history_text(self, message: Any, *, include_reasoning: bool) -> str:
        text = self._flatten_content(self._message_value(message, "content")).strip()
        reasoning_content = self._message_value(message, "reasoning_content")
        parsed_reasoning, parsed_content = self.split_assistant_reasoning(text, enable_thinking=False)
        if reasoning_content is None:
            reasoning_content = parsed_reasoning
            text = parsed_content.strip()
        elif parsed_reasoning is not None:
            text = parsed_content.strip()
        elif text.strip() == str(reasoning_content).strip():
            text = ""

        if include_reasoning:
            reasoning_text = "" if reasoning_content is None else str(reasoning_content).strip()
            return f"<think>\n{reasoning_text}\n</think>\n\n{text}"
        return text

    def _render_qwen_tool_call_history(self, tool_calls: list[Any]) -> str:
        blocks: list[str] = []
        for raw_tool_call in tool_calls:
            tool_call = model_to_dict(raw_tool_call)
            function = tool_call.get("function")
            function_dict = model_to_dict(function)
            function_name = str(function_dict.get("name") or "").strip()
            if not function_name:
                raise ValueError("Assistant tool_calls entries must include function.name.")
            arguments = coerce_arguments_mapping(function_dict.get("arguments"))
            block_lines = ["<tool_call>", f"<function={function_name}>"]
            for parameter_name, parameter_value in arguments.items():
                if isinstance(parameter_value, (dict, list)):
                    serialized = compact_json(parameter_value)
                else:
                    serialized = str(parameter_value)
                block_lines.append(f"<parameter={parameter_name}>")
                block_lines.append(serialized)
                block_lines.append("</parameter>")
            block_lines.append("</function>")
            block_lines.append("</tool_call>")
            blocks.append("\n".join(block_lines))
        return "\n".join(blocks)

    def _render_tools_preamble(
        self,
        *,
        tools: list[Any] | None,
        tool_choice: Any,
        parallel_tool_calls: bool | None,
        first_system_content: str,
    ) -> str:
        resolved_tool_choice = normalize_tool_choice(tool_choice, tools)
        selected_tools = select_tools_for_choice(tools, resolved_tool_choice)
        if not selected_tools:
            if first_system_content:
                return f"<|im_start|>system\n{first_system_content}<|im_end|>\n"
            return ""

        parts = [
            "<|im_start|>system\n",
            "# Tools\n\nYou have access to the following functions:\n\n<tools>",
        ]
        for tool in selected_tools:
            parts.append("\n")
            parts.append(json.dumps(tool, ensure_ascii=False))
        parts.append("\n</tools>")
        parts.append(
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls"
        )
        if resolved_tool_choice.mode == "required":
            parts.append("\n- You MUST call one or more available functions in your next reply")
        elif resolved_tool_choice.mode == "named" and resolved_tool_choice.function_name is not None:
            parts.append(f"\n- You MUST call the function {resolved_tool_choice.function_name} in your next reply")
        if parallel_tool_calls is False:
            parts.append("\n- You MUST NOT call more than one function in your next reply")
        parts.append("\n</IMPORTANT>")
        if first_system_content:
            parts.append(f"\n\n{first_system_content}")
        parts.append("<|im_end|>\n")
        return "".join(parts)

    def _parse_parameter_value(self, value: str) -> Any:
        normalized = value.strip("\r\n")
        if not normalized:
            return ""
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            if normalized == "True":
                return True
            if normalized == "False":
                return False
            if normalized in {"None", "null"}:
                return None
            try:
                return int(normalized)
            except ValueError:
                try:
                    return float(normalized)
                except ValueError:
                    return normalized

    def _parse_qwen_tool_call_block(self, block: str) -> ParsedToolCall | None:
        match = self._QWEN_TOOL_CALL_RE.match(block)
        if match is None:
            return None
        function_name = match.group(1).strip()
        if not function_name:
            return None

        body = match.group(2)
        arguments: dict[str, Any] = {}
        cursor = 0
        for parameter_match in self._QWEN_PARAMETER_RE.finditer(body):
            if body[cursor : parameter_match.start()].strip():
                return None
            parameter_name = parameter_match.group(1).strip()
            if not parameter_name:
                return None
            arguments[parameter_name] = self._parse_parameter_value(parameter_match.group(2))
            cursor = parameter_match.end()
        if body[cursor:].strip():
            return None

        return ParsedToolCall(
            name=function_name,
            arguments=compact_json(arguments),
        )

    def extract_tool_calls(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        return extract_delimited_tool_calls(
            text,
            start_tag="<tool_call>",
            end_tag="</tool_call>",
            block_parser=self._parse_qwen_tool_call_block,
        )

    def create_reasoning_parser(self, *, enable_thinking: bool) -> ThinkingStreamParser:
        return ThinkingStreamParser(enable_thinking=enable_thinking)

    def create_tool_call_stream_parser(self) -> DelimitedToolCallStreamParser:
        return DelimitedToolCallStreamParser(
            start_tag="<tool_call>",
            end_tag="</tool_call>",
            block_parser=self._parse_qwen_tool_call_block,
        )

    def render_messages(
        self,
        messages: list[Any],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
    ) -> str:
        parts: list[str] = []
        if not messages:
            raise ValueError("At least one message is required.")

        first_role = self._normalize_role(str(self._message_value(messages[0], "role", "")), index=0)
        first_system_content = ""
        if first_role == "system":
            first_system_content = self._flatten_content(self._message_value(messages[0], "content")).strip()
        parts.append(
            self._render_tools_preamble(
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                first_system_content=first_system_content,
            )
        )

        last_query_index = None
        for idx in range(len(messages) - 1, -1, -1):
            role = self._normalize_role(str(self._message_value(messages[idx], "role", "")), index=idx)
            if role == "user":
                last_query_index = idx
                break
        if last_query_index is None:
            raise ValueError("No user query found in messages.")

        tool_group_open = False
        for idx, message in enumerate(messages):
            role = self._normalize_role(str(self._message_value(message, "role", "")), index=idx)
            if role == "system":
                if idx != 0:
                    raise ValueError("System message must be the first message.")
                continue

            if role == "user":
                text = self._flatten_content(self._message_value(message, "content")).strip()
                parts.append(f"<|im_start|>user\n{text}<|im_end|>\n")
                continue

            if role == "assistant":
                text = self._assistant_history_text(
                    message,
                    include_reasoning=idx > last_query_index,
                )
                tool_calls = self._message_value(message, "tool_calls") or []
                if tool_calls:
                    rendered_tool_calls = self._render_qwen_tool_call_history(tool_calls)
                    if text.strip():
                        text = f"{text}\n\n{rendered_tool_calls}"
                    else:
                        text = rendered_tool_calls
                parts.append(f"<|im_start|>assistant\n{text}<|im_end|>\n")
                continue

            if role == "tool":
                text = self._flatten_content(self._message_value(message, "content")).strip()
                if not tool_group_open:
                    parts.append("<|im_start|>user")
                    tool_group_open = True
                parts.append(f"\n<tool_response>\n{text}\n</tool_response>")
                next_role = None
                if idx + 1 < len(messages):
                    next_role = self._normalize_role(str(self._message_value(messages[idx + 1], "role", "")), index=idx + 1)
                if next_role != "tool":
                    parts.append("<|im_end|>\n")
                    tool_group_open = False
                continue

            raise ValueError(f"Unsupported chat role: {role}")

        if add_generation_prompt:
            if enable_thinking:
                parts.append("<|im_start|>assistant\n<think>\n")
            else:
                # Match the Qwen3 chat template behavior for non-thinking mode:
                # pre-close an empty think block so generation starts directly in answer mode.
                parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)
