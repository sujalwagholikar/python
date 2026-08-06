"""
jarvis_brain.py
================
The "personality" layer of JARVIS — powered by Groq's
llama-3.3-70b-versatile.

This is deliberately kept SEPARATE from llm_parser.py (which still uses
llama-3.1-8b-instant purely for turning messy sentences into structured
device-control JSON). Splitting the two models like this means:

  - llama-3.1-8b-instant  -> fast, cheap, JSON-only, "the hands"
                              (command_parser.py / llm_parser.py)
  - llama-3.3-70b-versatile -> smarter, conversational, "the voice/mind"
                              (this file)

Responsibilities of this module
--------------------------------
1. classify_utterance(text) -> "command" | "chat"
   Fast heuristic + LLM-assisted routing: decide whether what the user
   said is a phone-control command (should go to the existing regex/8B
   pipeline) or general conversation/question (should be answered by
   JARVIS directly).

2. jarvis_reply(text, history, device_context) -> str
   Generates an in-character JARVIS response for anything that is NOT a
   device command — general questions, chit-chat, reasoning, clarifying
   questions, or a spoken acknowledgement of a command result phrased the
   way JARVIS would say it back to Tony.

3. narrate_result(raw_command, result_text) -> str
   Takes the raw mechanical result string produced by executing a phone
   command (e.g. "Opened whatsapp") and asks the 70B model to restate it
   briefly, in JARVIS's voice, for text-to-speech playback. Falls back to
   the raw string if the API is unavailable, so the assistant never goes
   silent just because narration failed.

Design notes
------------
- Every network call degrades gracefully: if GROQ_API_KEY_JARVIS is
  missing or the API call fails, callers get a clear exception
  (JarvisBrainError) or, for narration, a safe fallback string — never a
  crash that would kill the GUI event loop.
- No third-party HTTP client required — uses urllib, same approach as
  llm_parser.py, so requirements.txt doesn't need to grow for this.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY_JARVIS = os.environ.get("GROQ_API_KEY_JARVIS") or os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL_JARVIS = os.environ.get("GROQ_MODEL_JARVIS", "llama-3.3-70b-versatile")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

JARVIS_SYSTEM_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System),
a witty, composed, and extremely capable AI assistant in the style of Tony
Stark's assistant from Iron Man. You are speaking with your principal user
through a real-time voice interface, and you ALSO have direct control over
their Android phone via a separate command-execution system.

PERSONALITY
- Calm, precise, dryly witty, unfailingly polite but never obsequious.
- Address the user as "sir" or "boss" occasionally (not every sentence).
- Confident and efficient. You don't ramble.
- You may make a brief, tasteful quip, but you always deliver the actual
  information or answer first.

RESPONSE RULES (read carefully — this output is converted to SPEECH)
- Keep replies SHORT: 1–3 sentences for most things, unless the user
  explicitly asks for detail, a list, code, or a long explanation.
- NEVER use markdown formatting (no *, #, bullet points, code fences) —
  this text is spoken aloud by a text-to-speech engine, so it must be
  plain, natural spoken English.
- Do not narrate your own reasoning process ("Let me think...", "As an
  AI..."). Just answer.
- If you don't know something or it requires live/real-time data you
  don't have, say so briefly and, if relevant, suggest the user ask you
  to search or check the phone.
- If the user's message was actually a device command that got routed to
  you by mistake, you may gently note you can also just do it directly,
  but still be helpful.
"""

CLASSIFY_SYSTEM_PROMPT = """You are a strict binary router for a phone-control
voice assistant. Given one user utterance, decide whether it is:

  "command" — an instruction to control the Android phone or perform a
              concrete device action: opening apps, sending messages,
              calling someone, taking a screenshot, toggling settings
              (wifi/bluetooth/flashlight/volume/etc.), navigating
              (home/back/scroll), tapping/typing on screen, setting
              alarms/timers, reading the screen, checking battery/device
              info, or any other physical/system action on the phone.

  "chat"    — anything else: general questions, conversation, requests
              for information/explanations/opinions/jokes, small talk,
              or requests for JARVIS to reason/answer rather than act on
              the phone.

Respond with ONLY a single JSON object: {"route": "command"} or
{"route": "chat"}. No other text.
"""


class JarvisBrainError(RuntimeError):
    """Raised when the JARVIS (70B) model call fails outright."""


@dataclass
class ChatTurn:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class JarvisConversation:
    """Rolling short-term chat history fed to the 70B model for continuity."""
    turns: list[ChatTurn] = field(default_factory=list)
    max_turns: int = 12

    def add(self, role: str, content: str) -> None:
        self.turns.append(ChatTurn(role=role, content=content))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def as_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def clear(self) -> None:
        self.turns.clear()


# --------------------------------------------------------------------------- #
# Low level Groq call (shared)
# --------------------------------------------------------------------------- #

def _groq_chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.6,
    max_tokens: int = 400,
    json_mode: bool = False,
    timeout: int = 20,
) -> str:
    if not GROQ_API_KEY_JARVIS or GROQ_API_KEY_JARVIS == "paste_your_groq_key_here":
        raise JarvisBrainError(
            "No Groq API key configured for JARVIS. Get a free key at "
            "https://console.groq.com/keys and put it in .env as:\n"
            "  GROQ_API_KEY_JARVIS=your-key-here"
        )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        GROQ_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY_JARVIS}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        if e.code == 401:
            raise JarvisBrainError(
                "Groq rejected the JARVIS API key (401). Check "
                "GROQ_API_KEY_JARVIS in your .env file."
            ) from e
        raise JarvisBrainError(f"Groq API error {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise JarvisBrainError(f"Could not reach Groq API: {e}") from e

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise JarvisBrainError(f"Unexpected Groq response shape: {body}") from e


# --------------------------------------------------------------------------- #
# Fast local heuristics (avoid a network round-trip for obvious cases)
# --------------------------------------------------------------------------- #

_COMMAND_HINT_WORDS = re.compile(
    r"\b(open|close|launch|send|call|dial|text|whatsapp|telegram|sms|"
    r"screenshot|record|camera|photo|flashlight|torch|lock|unlock|wake|"
    r"sleep|search|google|volume|mute|wifi|bluetooth|airplane|scroll|"
    r"tap|swipe|type|press|home|back|recent|battery|vibrate|reboot|"
    r"brightness|clipboard|paste|alarm|timer|settings|uninstall|"
    r"clear\s+app|split\s*screen|dnd|do\s+not\s+disturb|silent|vibrate\s+mode|"
    r"rotate|hotspot|dark\s+mode|notification|keyboard|ping|status)\b",
    re.I,
)

_CHAT_HINT_WORDS = re.compile(
    r"^\s*(what|who|why|how|when|where|which|tell me|explain|do you|"
    r"can you (?:tell|explain)|are you|is it|what's|who's|joke|thanks|"
    r"thank you|hello|hi\b|hey\b|good morning|good night|how are you)\b",
    re.I,
)

# Real-time internet lookup — deliberately distinct phrasing from the
# on-phone "search X" / "google search X" commands (which open the
# phone's own browser via command_parser.py). These trigger a live
# web_search.py call from the desktop instead.
_WEB_SEARCH_HINT_WORDS = re.compile(
    r"\b(look\s*up|latest\s+news|current\s+news|what'?s\s+happening|"
    r"news\s+(?:on|about)|search\s+the\s+web|web\s+search|"
    r"browse\s+the\s+internet|check\s+the\s+internet|"
    r"today'?s\s+(?:news|headlines)|what'?s\s+the\s+(?:latest|current)|"
    r"find\s+(?:me\s+)?(?:information|info|articles?)\s+(?:on|about))\b",
    re.I,
)


def is_web_search_request(text: str) -> bool:
    """True if the utterance is asking JARVIS to actually go look
    something up on the live internet (routed to web_search.py), as
    opposed to an on-phone 'search X' browser command."""
    return bool(_WEB_SEARCH_HINT_WORDS.search(text.strip()))


def classify_utterance(text: str, use_llm_fallback: bool = True) -> str:
    """
    Return "command", "chat", or "search".

    "search" takes priority over the other two: it means the user wants
    JARVIS to actually go look something up on the live internet right
    now (routed to web_search.py), as distinct from an on-phone
    "search X" browser command (which stays "command" and is handled by
    command_parser.py / adb_controller.py as before).

    Fast path: obvious lexical cues resolve instantly without a network
    call. Ambiguous utterances fall back to a cheap classification call
    on the 70B model itself (kept to a 1-token-ish JSON answer, so it's
    fast) — unless use_llm_fallback is False, in which case ambiguous
    text defaults to "command" (so the existing regex/8B pipeline gets
    first refusal, which already fails safely with a clear message).
    """
    stripped = text.strip()
    if not stripped:
        return "chat"

    if is_web_search_request(stripped):
        return "search"

    has_command_hint = bool(_COMMAND_HINT_WORDS.search(stripped))
    has_chat_hint = bool(_CHAT_HINT_WORDS.match(stripped))

    if has_command_hint and not has_chat_hint:
        return "command"
    if has_chat_hint and not has_command_hint:
        return "chat"
    if not has_command_hint and not has_chat_hint and len(stripped.split()) <= 3:
        # short ambiguous fragment ("yes", "stop", "cancel") -> treat as chat
        return "chat"

    if not use_llm_fallback:
        return "command"

    try:
        raw = _groq_chat(
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": stripped},
            ],
            model=GROQ_MODEL_JARVIS,
            temperature=0,
            max_tokens=20,
            json_mode=True,
            timeout=8,
        )
        data = json.loads(raw)
        route = data.get("route")
        if route in ("command", "chat"):
            return route
    except Exception:
        pass  # fall through to default below

    return "command" if has_command_hint else "chat"


# --------------------------------------------------------------------------- #
# Public: conversational reply
# --------------------------------------------------------------------------- #

def jarvis_reply(
    text: str,
    conversation: Optional[JarvisConversation] = None,
    device_context: str = "",
) -> str:
    """
    Generate JARVIS's spoken reply to a general (non-command) utterance.
    Raises JarvisBrainError if the API call fails — callers should catch
    this and speak/display a graceful fallback line.
    """
    messages = [{"role": "system", "content": JARVIS_SYSTEM_PROMPT}]
    if device_context:
        messages.append({
            "role": "system",
            "content": f"Current device/session context:\n{device_context}",
        })
    if conversation:
        messages.extend(conversation.as_messages())
    messages.append({"role": "user", "content": text})

    reply = _groq_chat(
        messages=messages,
        model=GROQ_MODEL_JARVIS,
        temperature=0.65,
        max_tokens=300,
    )
    reply = reply.strip()

    if conversation is not None:
        conversation.add("user", text)
        conversation.add("assistant", reply)

    return reply


def narrate_result(raw_command: str, result_text: str) -> str:
    """
    Turn a mechanical execution result (e.g. "Opened whatsapp") into a
    short, natural, in-character JARVIS confirmation line for TTS.
    Falls back to the raw result string if the model call fails, so a
    narration failure never silences the assistant.
    """
    try:
        prompt = (
            f"The user asked: {raw_command!r}\n"
            f"The system just executed it with this raw result: {result_text!r}\n"
            f"Say ONE short natural sentence confirming this back to the user, "
            f"in character as JARVIS. Do not repeat the raw result verbatim; "
            f"paraphrase it naturally. If the result looks like an error, "
            f"acknowledge it briefly and calmly."
        )
        reply = _groq_chat(
            messages=[
                {"role": "system", "content": JARVIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL_JARVIS,
            temperature=0.5,
            max_tokens=80,
            timeout=10,
        )
        return reply.strip()
    except JarvisBrainError:
        return result_text


def jarvis_greeting() -> str:
    """A quick static greeting used at startup (no network call needed)."""
    return "Systems online. Good to see you, sir. JARVIS at your service."
