from __future__ import annotations

from typing import Any

import torch

from anna.mm.qwen3_5_text_processor import PreparedInputs
from anna.weights.gemma4_tokenizer import Gemma4Tokenizer


class Gemma4TextProcessor:
    def __init__(self, tokenizer: Gemma4Tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _resolve_tensor_device(tensor_device: torch.device | str | None) -> torch.device | None:
        if tensor_device is None:
            return None
        return tensor_device if isinstance(tensor_device, torch.device) else torch.device(tensor_device)

    def _build_prepared_inputs(
        self,
        *,
        prompt: str,
        tensor_device: torch.device | str | None = None,
    ) -> PreparedInputs:
        resolved_device = self._resolve_tensor_device(tensor_device)
        tensor_kwargs = {} if resolved_device is None else {"device": resolved_device}
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, **tensor_kwargs)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        return PreparedInputs(
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )

    def encode_text(
        self,
        prompt: str,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> PreparedInputs:
        return self._build_prepared_inputs(prompt=prompt, tensor_device=tensor_device)

    def prepare_messages(
        self,
        messages: list[Any],
        *,
        enable_thinking: bool = False,
        tensor_device: torch.device | str | None = None,
        tensor_dtype: torch.dtype | None = None,
    ) -> PreparedInputs:
        del tensor_dtype
        prompt = self.tokenizer.render_messages(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return self._build_prepared_inputs(prompt=prompt, tensor_device=tensor_device)
