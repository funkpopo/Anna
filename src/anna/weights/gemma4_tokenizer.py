from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from anna.core.function_calling import (
    DelimitedToolCallStreamParser,
    GemmaThinkingStreamParser,
    ParsedToolCall,
    coerce_arguments_mapping,
    compact_json,
    extract_delimited_tool_calls,
    model_to_dict,
    normalize_tool_choice,
    parse_tool_response_content,
    select_tools_for_choice,
)
from tokenizers import Tokenizer


class _GemmaStructuredValueParser:
    _NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def parse(self) -> Any:
        value = self._parse_value()
        self._skip_ws()
        if self.pos != len(self.text):
            raise ValueError("Unexpected trailing content in Gemma structured value.")
        return value

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _consume(self, literal: str) -> bool:
        if self.text.startswith(literal, self.pos):
            self.pos += len(literal)
            return True
        return False

    def _parse_value(self) -> Any:
        self._skip_ws()
        if self._consume('<|"|>'):
            return self._parse_string_body()
        if self._consume("{"):
            return self._parse_object_body()
        if self._consume("["):
            return self._parse_array_body()
        if self._consume("true"):
            return True
        if self._consume("false"):
            return False
        if self._consume("null"):
            return None

        number_match = self._NUMBER_RE.match(self.text, self.pos)
        if number_match is not None:
            token = number_match.group(0)
            self.pos = number_match.end()
            return float(token) if any(char in token for char in ".eE") else int(token)

        return self._parse_identifier()

    def _parse_string_body(self) -> str:
        end = self.text.find('<|"|>', self.pos)
        if end == -1:
            raise ValueError("Unterminated Gemma escaped string.")
        value = self.text[self.pos:end]
        self.pos = end + len('<|"|>')
        return value

    def _parse_identifier(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ",:{}[]":
            if self.text[self.pos].isspace():
                break
            self.pos += 1
        if self.pos == start:
            raise ValueError("Expected an identifier in Gemma structured value.")
        return self.text[start:self.pos]

    def _parse_key(self) -> str:
        self._skip_ws()
        if self._consume('<|"|>'):
            return self._parse_string_body()
        return self._parse_identifier()

    def _parse_object_body(self) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        self._skip_ws()
        if self._consume("}"):
            return obj
        while True:
            key = self._parse_key()
            self._skip_ws()
            if not self._consume(":"):
                raise ValueError("Expected ':' in Gemma object value.")
            obj[key] = self._parse_value()
            self._skip_ws()
            if self._consume("}"):
                break
            if not self._consume(","):
                raise ValueError("Expected ',' or '}' in Gemma object value.")
        return obj

    def _parse_array_body(self) -> list[Any]:
        values: list[Any] = []
        self._skip_ws()
        if self._consume("]"):
            return values
        while True:
            values.append(self._parse_value())
            self._skip_ws()
            if self._consume("]"):
                break
            if not self._consume(","):
                raise ValueError("Expected ',' or ']' in Gemma array value.")
        return values


class Gemma4Tokenizer:
    def __init__(self, backend: Tokenizer, metadata: dict[str, Any] | None = None):
        self.backend = backend
        self.metadata = metadata or {}
        self.bos_token = str(self.metadata.get("bos_token", "<bos>"))
        self.eos_token = str(self.metadata.get("eos_token", "<eos>"))
        self.sot_token = str(self.metadata.get("sot_token", "<|turn>"))
        self.eot_token = str(self.metadata.get("eot_token", "<turn|>"))
        self.soc_token = str(self.metadata.get("soc_token", "<|channel>"))
        self.eoc_token = str(self.metadata.get("eoc_token", "<channel|>"))
        self.think_token = str(self.metadata.get("think_token", "<|think|>"))
        self.boi_token = str(self.metadata.get("boi_token", "<|image>"))
        self.eoi_token = str(self.metadata.get("eoi_token", "<image|>"))
        self.image_token = str(self.metadata.get("image_token", "<|image|>"))
        self.boa_token = str(self.metadata.get("boa_token", "<|audio>"))
        self.eoa_token = str(self.metadata.get("eoa_token", "<audio|>"))
        self.audio_token = str(self.metadata.get("audio_token", "<|audio|>"))
        self.video_token = "<|video|>"
        self.tool_call_start_token = str(self.metadata.get("stc_token", "<|tool_call>"))
        self.tool_call_end_token = str(self.metadata.get("etc_token", "<tool_call|>"))
        self.tool_decl_start_token = str(self.metadata.get("std_token", "<|tool>"))
        self.tool_decl_end_token = str(self.metadata.get("etd_token", "<tool|>"))
        self.tool_response_start_token = str(self.metadata.get("str_token", "<|tool_response>"))
        self.tool_response_end_token = str(self.metadata.get("etr_token", "<tool_response|>"))

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Gemma4Tokenizer":
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
    def audio_token_id(self) -> int | None:
        return self.token_id(self.audio_token)

    @property
    def eos_token_ids(self) -> set[int]:
        eos_ids: set[int] = set()
        for token in (self.eos_token, self.eot_token):
            token_id = self.token_id(token)
            if token_id is not None:
                eos_ids.add(token_id)
        return eos_ids

    @staticmethod
    def _message_value(message: Any, key: str, default: Any = None) -> Any:
        value = getattr(message, key, None)
        if value is not None:
            return value
        if isinstance(message, dict):
            return message.get(key, default)
        return default

    @staticmethod
    def _strip_thinking(text: str) -> str:
        result = text
        while "<|channel>" in result and "<channel|>" in result:
            start = result.find("<|channel>")
            end = result.find("<channel|>", start)
            if end == -1:
                break
            prefix = result[:start]
            suffix = result[end + len("<channel|>") :]
            result = (prefix + suffix).strip()
        return result.strip()

    def _flatten_content(self, content: Any, *, role: str) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return self._strip_thinking(content) if role == "assistant" else content.strip()
        if not isinstance(content, list):
            raise ValueError("Unsupported chat content format.")

        chunks: list[str] = []
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")
            if item_type == "text":
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    chunks.append(self._strip_thinking(text) if role == "assistant" else str(text).strip())
            elif item_type in {"image", "image_url"}:
                chunks.append(self.image_token)
            elif item_type in {"video", "video_url"}:
                chunks.append(self.video_token)
            elif item_type in {"audio", "audio_url"}:
                chunks.append(self.audio_token)
            else:
                raise ValueError(f"Unsupported content part type: {item_type}")
        return "".join(chunks)

    def split_assistant_reasoning(self, text: str, *, enable_thinking: bool) -> tuple[str | None, str]:
        del enable_thinking
        normalized = text.lstrip()
        open_tag = f"{self.soc_token}thought\n"
        if not normalized.startswith(open_tag):
            return None, text
        normalized = normalized[len(open_tag) :]
        close_index = normalized.find(self.eoc_token)
        if close_index == -1:
            reasoning = normalized.strip()
            return reasoning or None, ""
        reasoning = normalized[:close_index].strip()
        content = normalized[close_index + len(self.eoc_token) :].lstrip("\r\n")
        return reasoning or None, content

    def _format_argument(self, argument: Any, *, escape_keys: bool = True) -> str:
        if isinstance(argument, str):
            return f'<|"|>{argument}<|"|>'
        if isinstance(argument, bool):
            return "true" if argument else "false"
        if argument is None:
            return "null"
        if isinstance(argument, dict):
            parts = []
            for key in sorted(argument):
                rendered_key = f'<|"|>{key}<|"|>' if escape_keys else str(key)
                parts.append(f"{rendered_key}:{self._format_argument(argument[key], escape_keys=escape_keys)}")
            return "{" + ",".join(parts) + "}"
        if isinstance(argument, (list, tuple)):
            return "[" + ",".join(self._format_argument(item, escape_keys=escape_keys) for item in argument) + "]"
        return str(argument)

    def _format_required_items(self, required: list[Any] | tuple[Any, ...]) -> str:
        return "[" + ",".join(self._format_argument(str(item)) for item in required) + "]"

    def _format_parameters(self, properties: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in sorted(properties):
            value = properties[key]
            if not isinstance(value, dict):
                continue
            item_fields: list[str] = []
            description = value.get("description")
            if description is not None:
                item_fields.append(f"description:{self._format_argument(str(description))}")
            if value.get("nullable"):
                item_fields.append("nullable:true")

            item_type = value.get("type", "string")
            if isinstance(item_type, list):
                item_type_render = self._format_argument([str(item).upper() for item in item_type])
                item_fields.append(f"type:{item_type_render}")
            else:
                upper_type = str(item_type).upper()
                if upper_type == "STRING" and value.get("enum") is not None:
                    item_fields.append(f"enum:{self._format_argument(value.get('enum'))}")
                elif upper_type == "OBJECT":
                    nested_properties = value.get("properties") or {}
                    item_fields.append(f"properties:{{{self._format_parameters(nested_properties)}}}")
                    nested_required = value.get("required") or []
                    if nested_required:
                        item_fields.append(f"required:{self._format_required_items(nested_required)}")
                elif upper_type == "ARRAY":
                    items = value.get("items")
                    if isinstance(items, dict) and items:
                        item_parts: list[str] = []
                        item_properties = items.get("properties")
                        if isinstance(item_properties, dict) and item_properties:
                            item_parts.append(f"properties:{{{self._format_parameters(item_properties)}}}")
                        item_required = items.get("required") or []
                        if item_required:
                            item_parts.append(f"required:{self._format_required_items(item_required)}")
                        item_type_value = items.get("type")
                        if item_type_value is not None:
                            if isinstance(item_type_value, list):
                                item_type_render = self._format_argument([str(item).upper() for item in item_type_value])
                            else:
                                item_type_render = self._format_argument(str(item_type_value).upper())
                            item_parts.append(f"type:{item_type_render}")
                        for item_key in sorted(items):
                            if item_key in {"properties", "required", "type"}:
                                continue
                            item_value = items[item_key]
                            if item_value is not None:
                                item_parts.append(f"{item_key}:{self._format_argument(item_value)}")
                        item_fields.append(f"items:{{{','.join(item_parts)}}}")
                item_fields.append(f"type:{self._format_argument(upper_type)}")

            parts.append(f"{key}:{{{','.join(item_fields)}}}")
        return ",".join(parts)

    def _format_function_declaration(self, tool: Any) -> str:
        tool_dict = model_to_dict(tool)
        function_dict = model_to_dict(tool_dict.get("function"))
        function_name = str(function_dict.get("name") or "").strip()
        if not function_name:
            raise ValueError("Tool definitions must include function.name.")

        fields: list[str] = []
        description = function_dict.get("description")
        if description is not None:
            fields.append(f"description:{self._format_argument(str(description))}")

        parameters = function_dict.get("parameters")
        if isinstance(parameters, dict) and parameters:
            parameter_fields: list[str] = []
            properties = parameters.get("properties")
            if isinstance(properties, dict) and properties:
                parameter_fields.append(f"properties:{{{self._format_parameters(properties)}}}")
            required = parameters.get("required") or []
            if required:
                parameter_fields.append(f"required:{self._format_required_items(required)}")
            parameter_type = parameters.get("type")
            if parameter_type is not None:
                if isinstance(parameter_type, list):
                    type_render = self._format_argument([str(item).upper() for item in parameter_type])
                else:
                    type_render = self._format_argument(str(parameter_type).upper())
                parameter_fields.append(f"type:{type_render}")
            fields.append(f"parameters:{{{','.join(parameter_fields)}}}")

        response = function_dict.get("response")
        if isinstance(response, dict) and response:
            response_fields: list[str] = []
            response_description = response.get("description")
            if response_description is not None:
                response_fields.append(f"description:{self._format_argument(str(response_description))}")
            response_type = response.get("type")
            if response_type is not None:
                response_fields.append(f"type:{self._format_argument(str(response_type).upper())}")
            fields.append(f"response:{{{','.join(response_fields)}}}")

        return f"declaration:{function_name}{{{','.join(fields)}}}"

    def _render_tool_selection_instructions(
        self,
        *,
        tool_choice: Any,
        tools: list[Any] | None,
        parallel_tool_calls: bool | None,
    ) -> str:
        resolved_tool_choice = normalize_tool_choice(tool_choice, tools)
        lines: list[str] = []
        if resolved_tool_choice.mode == "required":
            lines.append("You must call at least one tool in your next reply.")
        elif resolved_tool_choice.mode == "named" and resolved_tool_choice.function_name is not None:
            lines.append(f"You must call the tool {resolved_tool_choice.function_name} in your next reply.")
        if parallel_tool_calls is False:
            lines.append("Do not call more than one tool in your next reply.")
        if not lines:
            return ""
        return "\n" + "\n".join(lines)

    def _build_tool_name_lookup(self, messages: list[Any]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for message in messages:
            for raw_tool_call in self._message_value(message, "tool_calls") or []:
                tool_call = model_to_dict(raw_tool_call)
                tool_call_id = str(tool_call.get("id") or "").strip()
                function = tool_call.get("function")
                function_dict = model_to_dict(function)
                function_name = str(function_dict.get("name") or "").strip()
                if tool_call_id and function_name:
                    mapping[tool_call_id] = function_name
        return mapping

    def _render_tool_call_history(self, tool_calls: list[Any]) -> str:
        parts: list[str] = []
        for raw_tool_call in tool_calls:
            tool_call = model_to_dict(raw_tool_call)
            function = tool_call.get("function")
            function_dict = model_to_dict(function)
            function_name = str(function_dict.get("name") or "").strip()
            if not function_name:
                raise ValueError("Assistant tool_calls entries must include function.name.")
            arguments = coerce_arguments_mapping(function_dict.get("arguments"))
            serialized_arguments = ",".join(
                f"{key}:{self._format_argument(value, escape_keys=False)}"
                for key, value in sorted(arguments.items())
            )
            parts.append(
                f"{self.tool_call_start_token}call:{function_name}{{{serialized_arguments}}}{self.tool_call_end_token}"
            )
        return "".join(parts)

    def _render_tool_response_message(self, message: Any, tool_name_by_id: dict[str, str]) -> str:
        tool_name = str(self._message_value(message, "name") or "").strip()
        if not tool_name:
            tool_call_id = str(self._message_value(message, "tool_call_id") or "").strip()
            tool_name = tool_name_by_id.get(tool_call_id, "")
        if not tool_name:
            raise ValueError("Gemma tool messages require name or a tool_call_id that matches a prior assistant tool call.")

        response_text = self._flatten_content(self._message_value(message, "content"), role="tool")
        response_value = parse_tool_response_content(response_text)
        if isinstance(response_value, dict):
            body = ",".join(
                f"{key}:{self._format_argument(value, escape_keys=False)}"
                for key, value in sorted(response_value.items())
            )
            payload = f"response:{tool_name}{{{body}}}"
        else:
            payload = (
                f"response:{tool_name}{{value:{self._format_argument(response_value, escape_keys=False)}}}"
            )
        return f"{self.tool_response_start_token}{payload}{self.tool_response_end_token}"

    def _parse_gemma_tool_call_block(self, block: str) -> ParsedToolCall | None:
        if not block.startswith(self.tool_call_start_token) or not block.endswith(self.tool_call_end_token):
            return None
        inner = block[len(self.tool_call_start_token) : -len(self.tool_call_end_token)].strip()
        if not inner.startswith("call:"):
            return None
        brace_index = inner.find("{")
        if brace_index == -1 or not inner.endswith("}"):
            return None
        function_name = inner[len("call:") : brace_index].strip()
        if not function_name:
            return None
        try:
            arguments = _GemmaStructuredValueParser(inner[brace_index:]).parse()
        except ValueError:
            return None
        if not isinstance(arguments, dict):
            return None
        return ParsedToolCall(
            name=function_name,
            arguments=compact_json(arguments),
        )

    def extract_tool_calls(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        return extract_delimited_tool_calls(
            text,
            start_tag=self.tool_call_start_token,
            end_tag=self.tool_call_end_token,
            block_parser=self._parse_gemma_tool_call_block,
        )

    def create_reasoning_parser(self, *, enable_thinking: bool) -> GemmaThinkingStreamParser:
        return GemmaThinkingStreamParser(enable_thinking=enable_thinking)

    def create_tool_call_stream_parser(self) -> DelimitedToolCallStreamParser:
        return DelimitedToolCallStreamParser(
            start_tag=self.tool_call_start_token,
            end_tag=self.tool_call_end_token,
            block_parser=self._parse_gemma_tool_call_block,
        )

    def render_messages(
        self,
        messages: list[Any],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
    ) -> str:
        if not messages:
            raise ValueError("At least one message is required.")

        parts: list[str] = [self.bos_token]
        loop_messages = messages
        first_role = str(self._message_value(messages[0], "role") or "")
        resolved_tool_choice = normalize_tool_choice(tool_choice, tools)
        selected_tools = select_tools_for_choice(tools, resolved_tool_choice)
        tool_name_by_id = self._build_tool_name_lookup(messages)
        needs_system_block = enable_thinking or bool(selected_tools) or first_role in {"system", "developer"}
        if needs_system_block:
            parts.append(f"{self.sot_token}system\n")
            if enable_thinking:
                parts.append(self.think_token)
            if first_role in {"system", "developer"}:
                first_content = self._message_value(messages[0], "content")
                parts.append(self._flatten_content(first_content, role="system"))
                loop_messages = messages[1:]
            if selected_tools:
                for tool in selected_tools:
                    parts.append(self.tool_decl_start_token)
                    parts.append(self._format_function_declaration(tool))
                    parts.append(self.tool_decl_end_token)
                instructions = self._render_tool_selection_instructions(
                    tool_choice=tool_choice,
                    tools=tools,
                    parallel_tool_calls=parallel_tool_calls,
                )
                if instructions:
                    parts.append(instructions)
            parts.append(f"{self.eot_token}\n")

        prev_message_type = None
        for message in loop_messages:
            prev_message_type = None
            role = str(self._message_value(message, "role") or "")
            content = self._message_value(message, "content")
            normalized_role = "model" if role == "assistant" else role
            if normalized_role not in {"system", "user", "model", "tool"}:
                raise ValueError(f"Unsupported chat role: {role}")
            parts.append(f"{self.sot_token}{normalized_role}\n")

            tool_calls = self._message_value(message, "tool_calls") or []
            if tool_calls:
                parts.append(self._render_tool_call_history(tool_calls))
                prev_message_type = "tool_call"

            if role == "tool":
                parts.append(self._render_tool_response_message(message, tool_name_by_id))
                prev_message_type = "tool_response"

            rendered_content = "" if role == "tool" else self._flatten_content(content, role=role)
            parts.append(rendered_content)
            if not (role == "tool" and not rendered_content):
                parts.append(f"{self.eot_token}\n")

        if add_generation_prompt:
            if prev_message_type != "tool_response":
                parts.append(f"{self.sot_token}model\n")
        return "".join(parts)
