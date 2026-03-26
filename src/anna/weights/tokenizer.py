from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


class QwenTokenizer:
    def __init__(self, backend: Tokenizer, metadata: dict[str, Any] | None = None):
        self.backend = backend
        self.metadata = metadata or {}
        extra = self.metadata.get("extra_special_tokens", {})
        self.image_token = extra.get("image_token", "<|image_pad|>")
        self.video_token = extra.get("video_token", "<|video_pad|>")
        self.vision_start_token = extra.get("vision_bos_token", "<|vision_start|>")
        self.vision_end_token = extra.get("vision_eos_token", "<|vision_end|>")

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "QwenTokenizer":
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
        tokens = {"<|im_end|>", "<|endoftext|>"}
        return {token_id for token in tokens if (token_id := self.token_id(token)) is not None}

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
                elif item_type == "image_url":
                    chunks.append(f"{self.vision_start_token}{self.image_token}{self.vision_end_token}")
                elif item_type == "video_url":
                    chunks.append(f"{self.vision_start_token}{self.video_token}{self.vision_end_token}")
                else:
                    raise ValueError(f"Unsupported content part type: {item_type}")
            return "".join(chunks)
        raise ValueError("Unsupported chat content format.")

    def render_messages(self, messages: list[Any], *, add_generation_prompt: bool = True) -> str:
        parts: list[str] = []
        for idx, message in enumerate(messages):
            role = getattr(message, "role", None) or message["role"]
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            text = self._flatten_content(content).strip()

            if role == "system" and idx != 0:
                raise ValueError("System message must be the first message.")
            if role == "tool":
                text = f"<tool_response>\n{text}\n</tool_response>"
                role = "user"
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported chat role: {role}")
            parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)
