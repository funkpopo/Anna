from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


class Gemma4Tokenizer:
    def __init__(self, backend: Tokenizer, metadata: dict[str, Any] | None = None):
        self.backend = backend
        self.metadata = metadata or {}
        self.bos_token = str(self.metadata.get("bos_token", "<bos>"))
        self.eos_token = str(self.metadata.get("eos_token", "<eos>"))
        self.sot_token = str(self.metadata.get("sot_token", "<|turn>"))
        self.eot_token = str(self.metadata.get("eot_token", "<turn|>"))
        self.soc_token = str(self.metadata.get("soc_token", "<|channel>"))
        self.think_token = str(self.metadata.get("think_token", "<|think|>"))
        self.boi_token = str(self.metadata.get("boi_token", "<|image>"))
        self.eoi_token = str(self.metadata.get("eoi_token", "<image|>"))
        self.image_token = str(self.metadata.get("image_token", "<|image|>"))
        self.boa_token = str(self.metadata.get("boa_token", "<|audio>"))
        self.eoa_token = str(self.metadata.get("eoa_token", "<audio|>"))
        self.audio_token = str(self.metadata.get("audio_token", "<|audio|>"))
        self.video_token = "<|video|>"

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
    def eos_token_ids(self) -> set[int]:
        eos_ids: set[int] = set()
        for token in (self.eos_token, self.eot_token):
            token_id = self.token_id(token)
            if token_id is not None:
                eos_ids.add(token_id)
        return eos_ids

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
            if item_type != "text":
                raise ValueError(
                    "Gemma4 text runtime currently loads only the language tower. "
                    "Image, video, and audio chat inputs are not supported in this runtime path."
                )
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text:
                chunks.append(self._strip_thinking(text) if role == "assistant" else str(text).strip())
        return "".join(chunks)

    def render_messages(
        self,
        messages: list[Any],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        if not messages:
            raise ValueError("At least one message is required.")

        parts: list[str] = [self.bos_token]
        loop_messages = messages
        first_role = getattr(messages[0], "role", None) or messages[0]["role"]
        needs_system_block = enable_thinking or first_role in {"system", "developer"}
        if needs_system_block:
            parts.append(f"{self.sot_token}system\n")
            if enable_thinking:
                parts.append(self.think_token)
            if first_role in {"system", "developer"}:
                first_content = getattr(messages[0], "content", None)
                if first_content is None and isinstance(messages[0], dict):
                    first_content = messages[0].get("content")
                parts.append(self._flatten_content(first_content, role="system"))
                loop_messages = messages[1:]
            parts.append(f"{self.eot_token}\n")

        for message in loop_messages:
            role = getattr(message, "role", None) or message["role"]
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            normalized_role = "model" if role == "assistant" else role
            if normalized_role not in {"system", "user", "model"}:
                raise ValueError(f"Unsupported chat role: {role}")
            parts.append(f"{self.sot_token}{normalized_role}\n")
            parts.append(self._flatten_content(content, role=role))
            parts.append(f"{self.eot_token}\n")

        if add_generation_prompt:
            parts.append(f"{self.sot_token}model\n")
        return "".join(parts)
