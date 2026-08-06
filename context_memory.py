"""
context_memory.py
=================
Conversation context memory for the Android controller.

Tracks what happened in previous turns so that follow-up commands like:
  > open whatsapp
  > type sujal on search bar
  > click it
  > type hi
  > press send

...can be understood as a coherent multi-turn sequence without the user
re-stating what app they're in or who they're talking to.

What it tracks
--------------
  last_app       – the app most recently opened ("whatsapp", "youtube", …)
  last_contact   – the person most recently mentioned ("sujal", "mom", …)
  last_query     – the last search query or typed text
  last_message   – the last message body sent / typed
  last_intent    – the Intent name that just executed
  last_result    – the human-readable result string from execute_intent()
  turn_history   – ordered list of (raw_input, intent_name, params) tuples,
                   most-recent last, capped at HISTORY_LIMIT turns

Resolution helpers
------------------
  resolve_app(text)     → fills "whatsapp" when user says "it" / "that app"
  resolve_contact(text) → fills "sujal" when user says "him" / "her" / "them"
  resolve_query(text)   → fills last query when user says "same thing" / "that"
  enrich_intent(intent) → auto-fills missing params from context before execution

The LLM receives a compact context block prepended to every prompt so it
can reason across turns without the user repeating themselves.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Optional

if TYPE_CHECKING:
    from command_parser import Intent  # avoid circular import at runtime

HISTORY_LIMIT = 20  # how many past turns to keep in memory

# Pronouns that refer back to the last contact
_CONTACT_PRONOUNS = {
    "him", "her", "them", "he", "she", "they",
    "this person", "that person", "the same person",
}

# Pronouns / phrases that refer back to the last app
_APP_PRONOUNS = {
    "it", "that", "that app", "this app", "the app",
    "same app", "there", "the same app",
}

# Phrases meaning "same query as before"
_QUERY_PRONOUNS = {
    "same thing", "that", "same", "same query", "same search",
    "it", "the same",
}


@dataclass
class _Turn:
    """A single completed turn."""
    raw: str
    intent_name: str
    params: dict
    result: str = ""


class ConversationMemory:
    """
    Holds conversation state across REPL turns.

    Usage (in the REPL loop):
        memory = ConversationMemory()
        ...
        # before parsing/executing:
        raw = memory.resolve_references(raw_user_input)
        intent = parse(raw)
        intent = memory.enrich_intent(intent)
        result = execute_intent(controller, intent)
        memory.update(raw_user_input, intent, result)
    """

    def __init__(self) -> None:
        self.last_app: Optional[str] = None
        self.last_contact: Optional[str] = None
        self.last_query: Optional[str] = None
        self.last_message: Optional[str] = None
        self.last_intent: Optional[str] = None
        self.last_result: Optional[str] = None
        self._history: Deque[_Turn] = deque(maxlen=HISTORY_LIMIT)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, raw: str, intent: "Intent", result: str = "") -> None:
        """Call this after every successful execution to record what happened."""
        p = intent.params or {}
        name = intent.name

        # Update last_app
        if name in ("open_app", "open_app_and_type", "open_app_and_search",
                     "open_app_and_do", "open_camera", "open_video_camera",
                     "whatsapp_send", "telegram_send"):
            app = (p.get("app") or "").strip().lower()
            if app:
                self.last_app = app
            elif name == "whatsapp_send":
                self.last_app = "whatsapp"
            elif name == "telegram_send":
                self.last_app = "telegram"
            elif name in ("open_camera", "open_video_camera"):
                self.last_app = "camera"

        # Update last_contact
        contact = (p.get("contact") or "").strip().lower()
        if contact:
            self.last_contact = contact

        # Update last_query / last_message
        query = (p.get("query") or "").strip()
        if query:
            self.last_query = query

        msg = (p.get("message") or p.get("text") or "").strip()
        if msg:
            self.last_message = msg
            self.last_query = msg  # typed text is also a valid "last thing"

        self.last_intent = name
        self.last_result = result

        self._history.append(_Turn(
            raw=raw,
            intent_name=name,
            params=dict(p),
            result=result,
        ))

    def resolve_references(self, text: str) -> str:
        """
        Replace conversational pronouns and shorthand with their referents
        before the text is sent to the parser.

        Examples:
          "message him" → "message sujal"  (if last_contact = sujal)
          "search in it"  → "search in youtube"  (if last_app = youtube)
        """
        lowered = text.strip().lower()

        # --- contact pronoun resolution ---
        if self.last_contact:
            for pronoun in sorted(_CONTACT_PRONOUNS, key=len, reverse=True):
                # whole-word match only
                pattern = rf"\b{re.escape(pronoun)}\b"
                if re.search(pattern, lowered, re.I):
                    text = re.sub(pattern, self.last_contact, text, flags=re.I)
                    lowered = text.lower()
                    break  # only replace the first matching pronoun per turn

        # --- app pronoun resolution ---
        if self.last_app:
            for pronoun in sorted(_APP_PRONOUNS, key=len, reverse=True):
                pattern = rf"\b{re.escape(pronoun)}\b"
                if re.search(pattern, lowered, re.I):
                    text = re.sub(pattern, self.last_app, text, flags=re.I)
                    lowered = text.lower()
                    break

        # --- query pronoun resolution (only for search-like commands) ---
        if self.last_query and re.search(r"\b(search|find|look up|type)\b", lowered, re.I):
            for pronoun in sorted(_QUERY_PRONOUNS, key=len, reverse=True):
                pattern = rf"\b{re.escape(pronoun)}\b"
                # Don't replace "it" if it already got replaced by last_app above
                if re.search(pattern, lowered, re.I):
                    text = re.sub(pattern, self.last_query, text, flags=re.I)
                    lowered = text.lower()
                    break

        return text

    def enrich_intent(self, intent: "Intent") -> "Intent":
        """
        Fill in missing intent params from context memory.

        For example:
          - whatsapp_send with no contact → fill from last_contact
          - open_app_and_search with no app → fill from last_app
          - type_and_send with no text → fill from last_message
        """
        p = dict(intent.params or {})
        changed = False

        # Fill missing contact from context
        if intent.name in ("whatsapp_send", "sms_send", "call", "dial",
                            "telegram_send") and self.last_contact:
            if not p.get("contact") and not p.get("number"):
                p["contact"] = self.last_contact
                changed = True

        # Fill missing app from context
        if intent.name in ("open_app_and_type", "open_app_and_search",
                            "open_app_and_do", "close_app",
                            "clear_app_data", "open_app_settings") and self.last_app:
            if not (p.get("app") or "").strip():
                p["app"] = self.last_app
                changed = True

        # Fill missing message from context (e.g. "send it" after typing)
        if intent.name in ("type_and_send",) and self.last_message:
            if not (p.get("message") or p.get("text") or "").strip():
                p["text"] = self.last_message
                changed = True

        if changed:
            from command_parser import Intent as _Intent
            return _Intent(name=intent.name, params=p, raw_text=intent.raw_text)

        return intent

    def context_block_for_llm(self) -> str:
        """
        Returns a compact natural-language summary of the current conversation
        context to prepend to the LLM prompt, so the LLM can resolve references
        and understand what the user is continuing from.

        Returns an empty string when there's no context yet (first turn).
        """
        if not self._history:
            return ""

        lines: list[str] = ["=== CONVERSATION CONTEXT (most recent first) ==="]

        # Show last N turns compactly
        for turn in reversed(list(self._history)[-6:]):
            line = f"  [{turn.intent_name}]"
            p = turn.params
            details: list[str] = []
            if p.get("app"):
                details.append(f"app={p['app']}")
            if p.get("contact"):
                details.append(f"contact={p['contact']}")
            if p.get("message"):
                details.append(f"message={p['message']!r}")
            if p.get("query"):
                details.append(f"query={p['query']!r}")
            if p.get("text"):
                details.append(f"text={p['text']!r}")
            if details:
                line += " " + ", ".join(details)
            if turn.raw:
                line += f"  ← user said: {turn.raw!r}"
            lines.append(line)

        lines.append("")

        # Add active context pointers
        active: list[str] = []
        if self.last_app:
            active.append(f"current_app={self.last_app!r}")
        if self.last_contact:
            active.append(f"last_contact={self.last_contact!r}")
        if self.last_query:
            active.append(f"last_typed={self.last_query!r}")
        if self.last_message:
            active.append(f"last_message={self.last_message!r}")
        if active:
            lines.append("Active context: " + ", ".join(active))

        lines.append(
            "\nIMPORTANT: The user may use pronouns like 'him', 'her', 'it', 'that app', "
            "'same thing', 'there' — resolve them using the context above before "
            "choosing an intent. If the user says 'type sujal' after 'open whatsapp', "
            "they mean type sujal inside whatsapp. If they say 'message him', 'him' "
            "refers to the last_contact above."
        )
        lines.append("=== END CONTEXT ===\n")

        return "\n".join(lines)

    def formatted_history(self) -> str:
        """Human-readable history for the 'memory' REPL command."""
        if not self._history:
            return "(no history yet)"
        lines = []
        for i, turn in enumerate(self._history, 1):
            lines.append(f"  {i:2d}. [{turn.intent_name}] ← {turn.raw!r}")
            if turn.result:
                lines.append(f"       → {turn.result}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Wipe all memory (user command: 'forget' / 'clear memory')."""
        self.last_app = None
        self.last_contact = None
        self.last_query = None
        self.last_message = None
        self.last_intent = None
        self.last_result = None
        self._history.clear()
