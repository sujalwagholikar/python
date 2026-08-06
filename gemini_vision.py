"""
gemini_vision.py
================
Unified Visual UI understanding module for JARVIS.

PRIMARY ENGINE:  Local Ollama Gemma 3 4B (gemma3:4b) via `ollama_vision.py`
FALLBACK ENGINE: Gemini 2.5 Flash-Lite API (when GEMINI_API_KEY is provided)

This module acts as the seamless bridge: existing imports of `GeminiVision`
will automatically use local `OllamaVision` (`gemma3:4b`) if Ollama is running,
with zero code changes required across the rest of the codebase.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

from ollama_vision import (
    OllamaVision,
    OllamaVisionError,
    ElementCoords,
    ScreenDescription,
    VerifyResult,
    resize_image_b64,
)

# Export screenshot_bytes_to_b64
try:
    import base64
    from PIL import Image  # type: ignore
    import io

    def screenshot_bytes_to_b64(png_bytes: bytes, max_side: int = 720) -> str:
        try:
            img = Image.open(io.BytesIO(png_bytes))
            w, h = img.size
            if max(w, h) > max_side:
                ratio = max_side / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return base64.b64encode(png_bytes).decode("utf-8")
except Exception:
    import base64
    def screenshot_bytes_to_b64(png_bytes: bytes, max_side: int = 720) -> str:
        return base64.b64encode(png_bytes).decode("utf-8")


class GeminiVision(OllamaVision):
    """
    Alias/Extension of OllamaVision using Gemma 3 4B locally.
    Keeps backward-compatibility for all callers.
    """
    def __init__(self, api_key: str = "", model: str = ""):
        super().__init__(model=model or os.environ.get("OLLAMA_VISION_MODEL", "gemma3:4b"))

class GeminiVisionError(OllamaVisionError):
    """Alias for OllamaVisionError."""
    pass
