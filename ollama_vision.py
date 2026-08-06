"""
ollama_vision.py
================
Local Visual UI understanding for JARVIS powered by Ollama and Gemma 3 4B
(`gemma3:4b`).

WHY THIS EXISTS
---------------
Replaces cloud-based vision APIs (like Gemini) with a 100% local, keyless,
privacy-preserving multimodal vision model running on the user's PC via Ollama.

Ollama API endpoint: http://localhost:11434/api/chat
Model default:       gemma3:4b

PUBLIC API
----------
  from ollama_vision import OllamaVision, OllamaVisionError

  vision = OllamaVision()                       # auto-connects to http://localhost:11434
  desc   = vision.describe_screen(img_b64, ui_text)
  coords = vision.find_element_coords(img_b64, ui_text, "the search button icon")
  check  = vision.verify_goal(img_b64, ui_text, goal="send message", action_taken="tapped Send")
  action = vision.decide_next_action(img_b64, ui_text, goal, history_txt, schema_doc)

Zero extra pip dependencies — uses standard library `urllib` and base64.
Resizes screenshots via Pillow if installed to keep Ollama inference fast.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_VISION_MODEL", "gemma3:4b")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_MAX_IMAGE_SIDE = 720  # Keep image size compact for fast local inference

# --------------------------------------------------------------------------- #
# Exceptions & Data Classes
# --------------------------------------------------------------------------- #


class OllamaVisionError(RuntimeError):
    """Raised when Ollama API fails or model is unavailable."""


@dataclass
class ScreenDescription:
    app_name: str = ""
    visible_elements: list = field(default_factory=list)
    popups_dialogs: list = field(default_factory=list)
    keyboard_visible: bool = False
    loading: bool = False
    summary: str = ""
    raw: str = ""


@dataclass
class ElementCoords:
    found: bool = False
    x: int = 0
    y: int = 0
    confidence: str = "low"
    description: str = ""


@dataclass
class VerifyResult:
    state: str = "in_progress"  # "complete" | "in_progress" | "stuck"
    note: str = ""
    confidence: str = "medium"


# --------------------------------------------------------------------------- #
# Image resizing
# --------------------------------------------------------------------------- #


def resize_image_b64(image_b64: str, max_side: int = _MAX_IMAGE_SIDE) -> str:
    """Resize base64 screenshot to max_side pixels to keep Ollama fast."""
    if not image_b64:
        return ""
    try:
        from PIL import Image  # type: ignore
        import io
        raw_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw_bytes))
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return image_b64


# --------------------------------------------------------------------------- #
# Core Ollama Vision Client
# --------------------------------------------------------------------------- #


class OllamaVision:
    """
    Local visual UI understanding powered by Gemma 3 4B in Ollama.
    """

    def __init__(self, host: str = "", model: str = ""):
        self.host = (host or OLLAMA_HOST).rstrip("/")
        self.model = model or OLLAMA_MODEL
        self._available_cache: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Check if local Ollama server is running and model is loaded."""
        if self._available_cache is not None:
            return self._available_cache
        try:
            url = f"{self.host}/api/tags"
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                full_names = [m.get("name", "") for m in data.get("models", [])]
                base_target = self.model.split(":")[0]
                self._available_cache = (
                    self.model in full_names or base_target in models
                )
                return self._available_cache
        except Exception:
            self._available_cache = False
            return False

    def describe_screen(
        self, image_b64: str, ui_text: str = "", timeout: int = 40
    ) -> ScreenDescription:
        system = (
            "You are a phone screen analyst for an Android automation agent. "
            "Analyze the screenshot carefully. Describe the foreground app, visible "
            "buttons/icons, popups, whether keyboard is visible, and summary."
        )
        user = (
            "Analyze this Android screen screenshot.\n"
            + (f"UIAutomator accessibility text:\n{ui_text}\n\n" if ui_text else "")
            + 'Return ONLY JSON: {"app_name": "...", "visible_elements": ["..."], '
            '"popups_dialogs": ["..."], "keyboard_visible": true/false, '
            '"loading": true/false, "summary": "1-2 sentence description"}'
        )

        try:
            raw = self._call_ollama(system, user, image_b64, timeout=timeout)
            data = _extract_json(raw)
            return ScreenDescription(
                app_name=data.get("app_name", ""),
                visible_elements=data.get("visible_elements", []),
                popups_dialogs=data.get("popups_dialogs", []),
                keyboard_visible=bool(data.get("keyboard_visible", False)),
                loading=bool(data.get("loading", False)),
                summary=data.get("summary", raw[:200]),
                raw=raw,
            )
        except Exception as e:
            return ScreenDescription(
                summary=ui_text[:300] if ui_text else "(ollama vision error)",
                raw=str(e),
            )

    def find_element_coords(
        self,
        image_b64: str,
        ui_text: str,
        element_description: str,
        screen_width: int = 1080,
        screen_height: int = 2400,
        timeout: int = 35,
    ) -> ElementCoords:
        system = (
            "You are a precise UI element locator for Android automation. "
            "Locate the described element in the screenshot and return its center "
            f"pixel coordinates in the original screen size ({screen_width}x{screen_height})."
        )
        user = (
            f"Find element: '{element_description}'\n"
            + (f"Accessibility text:\n{ui_text}\n\n" if ui_text else "")
            + f"Screen resolution: {screen_width}x{screen_height}\n\n"
            'Return ONLY JSON: {"found": true/false, "x": <int>, "y": <int>, '
            '"confidence": "high"/"medium"/"low", "description": "what you found"}'
        )

        try:
            raw = self._call_ollama(system, user, image_b64, timeout=timeout)
            data = _extract_json(raw)
            return ElementCoords(
                found=bool(data.get("found", False)),
                x=int(data.get("x", 0)),
                y=int(data.get("y", 0)),
                confidence=data.get("confidence", "low"),
                description=data.get("description", ""),
            )
        except Exception as e:
            return ElementCoords(found=False, description=f"Ollama error: {e}")

    def verify_goal(
        self,
        image_b64: str,
        ui_text: str,
        goal: str,
        action_taken: str,
        timeout: int = 35,
    ) -> VerifyResult:
        system = (
            "You are a strict verifier for an Android automation agent. "
            "Look at the screenshot taken AFTER the action was executed. "
            "Decide if the goal is complete, in progress, or stuck based ONLY on what is visible."
        )
        user = (
            f"GOAL: {goal}\n"
            f"ACTION JUST TAKEN: {action_taken}\n\n"
            + (f"Accessibility text:\n{ui_text}\n\n" if ui_text else "")
            + 'Return ONLY JSON: {"state": "complete"/"in_progress"/"stuck", '
            '"note": "explanation based on visible screenshot", "confidence": "high"/"medium"/"low"}'
        )

        try:
            raw = self._call_ollama(system, user, image_b64, timeout=timeout)
            data = _extract_json(raw)
            state = data.get("state", "in_progress")
            if state not in ("complete", "in_progress", "stuck"):
                state = "in_progress"
            return VerifyResult(
                state=state,
                note=data.get("note", raw[:200]),
                confidence=data.get("confidence", "medium"),
            )
        except Exception as e:
            return VerifyResult(state="in_progress", note=f"Verification unavailable: {e}")

    def decide_next_action(
        self,
        image_b64: str,
        ui_text: str,
        goal: str,
        history_summary: str,
        action_schema: str,
        timeout: int = 45,
    ) -> str:
        system = (
            "You are the execution planning core of JARVIS, powered by local Gemma 3 4B. "
            "You see the actual phone screenshot and accessibility text. "
            "Choose ONE next action as JSON to move toward the goal.\n\n"
            + action_schema
        )
        user = (
            f"GOAL: {goal}\n\n"
            f"ACTIONS TAKEN SO FAR:\n{history_summary}\n\n"
            "CURRENT PHONE SCREEN (screenshot attached):\n"
            + (f"{ui_text}\n\n" if ui_text else "")
            + "What is the single next action? Respond with ONLY the JSON object."
        )
        return self._call_ollama(system, user, image_b64, timeout=timeout)

    def ask(
        self,
        image_b64: str,
        ui_text: str,
        question: str,
        timeout: int = 30,
    ) -> str:
        system = "You are a visual assistant for an Android phone controller. Answer concisely based on the screenshot."
        user = (
            (f"Accessibility text:\n{ui_text}\n\n" if ui_text else "")
            + f"Question: {question}"
        )
        try:
            return self._call_ollama(system, user, image_b64, timeout=timeout)
        except Exception as e:
            return f"(local vision unavailable: {e})"

    # ------------------------------------------------------------------ #
    # Private: Ollama API call
    # ------------------------------------------------------------------ #

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        image_b64: Optional[str] = None,
        timeout: int = 40,
    ) -> str:
        url = f"{self.host}/api/chat"

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        user_msg = {"role": "user", "content": user_prompt}
        if image_b64:
            resized = resize_image_b64(image_b64)
            user_msg["images"] = [resized]

        messages.append(user_msg)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                # Responses here are always short structured JSON (an
                # action object, a coords object, or a verify verdict) —
                # capping generation length tightly cuts real wall-clock
                # latency since Ollama stops generating once it hits this,
                # and most of a slow response's time is token-by-token
                # decode, not prompt processing.
                "num_predict": 200,
                # Keep the model resident between calls instead of letting
                # Ollama unload it after its default idle timeout — avoids
                # an expensive reload+recompute on the next call in a
                # multi-step agent loop that's still actively running.
            },
            "keep_alive": "10m",
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as e:
            raise OllamaVisionError(
                f"Could not reach Ollama at {self.host}. Is Ollama running? ({e})"
            ) from e
        except Exception as e:
            raise OllamaVisionError(f"Ollama API call failed: {e}") from e


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped.strip())
    try:
        return json.loads(stripped.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON: {text[:200]}")


if __name__ == "__main__":
    print(f"[*] Connecting to local Ollama ({OLLAMA_HOST}) with model '{OLLAMA_MODEL}'...")
    vision = OllamaVision()
    if vision.available:
        print("[✓] Ollama server is online and gemma3:4b model is available!")
        res = vision.ask("", "Button text: Send", "Is the Send button visible?")
        print(f"[*] Smoke test response: {res}")
    else:
        print(f"[!] Ollama is not running or model '{OLLAMA_MODEL}' not found.")
