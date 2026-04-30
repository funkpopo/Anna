"""Shared multimodal message media loading (chat image/video URLs and local paths)."""

from __future__ import annotations

import base64
import io
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


def collect_message_media_refs(messages: list[Any], part_type: str) -> list[Any]:
    refs: list[Any] = []
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if hasattr(item, "type"):
                item_type = getattr(item, "type")
                value = getattr(item, part_type, None)
            else:
                item_type = item.get("type")
                value = item.get(part_type)
            if item_type == part_type:
                refs.append(value)
    return refs


def resolve_media_url(ref: Any) -> str:
    if isinstance(ref, str):
        return ref
    if isinstance(ref, dict):
        url = ref.get("url")
        if not url:
            raise ValueError("Media URL object is missing the 'url' field.")
        return url
    raise ValueError("Unsupported media reference format.")


def read_media_bytes(media_ref: Any) -> bytes:
    url = resolve_media_url(media_ref)
    if url.startswith("data:"):
        _, payload = url.split(",", 1)
        return base64.b64decode(payload)
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url) as response:
            return response.read()
    return Path(url).read_bytes()


def load_image_pil(media_ref: Any) -> Image.Image:
    return Image.open(io.BytesIO(read_media_bytes(media_ref))).convert("RGB")


def load_video_frames(media_ref: Any) -> tuple[list[Image.Image], float]:
    try:
        import imageio
    except Exception as exc:  # pragma: no cover - dependency availability is environment-specific
        raise RuntimeError("Video input requires imageio with ffmpeg support installed.") from exc

    url = resolve_media_url(media_ref)
    temp_path: str | None = None
    if url.startswith(("http://", "https://", "data:")):
        suffix = Path(urllib.parse.urlparse(url).path).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(read_media_bytes(media_ref))
            temp_path = handle.name
        video_path = temp_path
    else:
        video_path = url

    reader = None
    try:
        reader = imageio.get_reader(video_path, format="FFMPEG")
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps") or 24.0)
        frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
    finally:
        if reader is not None:
            reader.close()
        if temp_path is not None:
            os.unlink(temp_path)
    if not frames:
        raise ValueError("Decoded video contained zero frames.")
    return frames, fps
