from __future__ import annotations

from dataclasses import dataclass, field


def strip_unstable_replacement_suffix(text: str) -> str:
    return text.rstrip("\ufffd")


@dataclass(slots=True)
class IncrementalTextAssembler:
    tokenizer: object
    stop_strings: list[str]
    pending_token_ids: list[int] = field(default_factory=list, init=False)
    pending_emitted_chars: int = field(default=0, init=False)
    stop_buffer: str = field(default="", init=False)
    max_stop_length: int = field(default=0, init=False)
    stopped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.max_stop_length = max((len(stop) for stop in self.stop_strings if stop), default=0)

    def feed_token(self, token_id: int) -> tuple[str, bool]:
        if self.stopped:
            return "", True

        self.pending_token_ids.append(token_id)
        decoded = self.tokenizer.decode(self.pending_token_ids, skip_special_tokens=False)
        stable_text = strip_unstable_replacement_suffix(decoded)
        delta = ""
        if len(stable_text) > self.pending_emitted_chars:
            delta = stable_text[self.pending_emitted_chars :]
            self.pending_emitted_chars = len(stable_text)

        if stable_text == decoded and self.pending_emitted_chars == len(decoded):
            self.pending_token_ids.clear()
            self.pending_emitted_chars = 0

        return self._push_stable_text(delta)

    def flush(self) -> tuple[str, bool]:
        if self.pending_token_ids:
            decoded = self.tokenizer.decode(self.pending_token_ids, skip_special_tokens=False)
            stable_text = strip_unstable_replacement_suffix(decoded)
            delta = stable_text[self.pending_emitted_chars :]
            self.pending_token_ids.clear()
            self.pending_emitted_chars = 0
            if delta:
                return self._flush_text(delta)
        return self._flush_text("")

    def _push_stable_text(self, text: str) -> tuple[str, bool]:
        if self.stopped:
            return "", True
        if not self.stop_strings:
            return text, False

        if text:
            self.stop_buffer += text
        stop_index = self._find_earliest_stop(self.stop_buffer)
        if stop_index is not None:
            emitted = self.stop_buffer[:stop_index]
            self.stop_buffer = ""
            self.stopped = True
            return emitted, True

        hold_back = max(0, self.max_stop_length - 1)
        if hold_back == 0:
            emitted = self.stop_buffer
            self.stop_buffer = ""
            return emitted, False

        safe_length = max(0, len(self.stop_buffer) - hold_back)
        emitted = self.stop_buffer[:safe_length]
        self.stop_buffer = self.stop_buffer[safe_length:]
        return emitted, False

    def _flush_text(self, text: str) -> tuple[str, bool]:
        if self.stopped:
            return "", True
        if text:
            self.stop_buffer += text

        stop_index = self._find_earliest_stop(self.stop_buffer)
        if stop_index is not None:
            emitted = self.stop_buffer[:stop_index]
            self.stop_buffer = ""
            self.stopped = True
            return emitted, True

        emitted = self.stop_buffer
        self.stop_buffer = ""
        return emitted, False

    def _find_earliest_stop(self, text: str) -> int | None:
        indexes = [text.find(stop) for stop in self.stop_strings if stop and stop in text]
        if not indexes:
            return None
        return min(indexes)
