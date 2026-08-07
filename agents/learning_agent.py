"""
agents/learning_agent.py
==========================
LearningAgent -- observes JARVIS usage over time and persists lightweight
patterns to disk (learning_memory.json, next to contacts.json), so future
sessions can be smarter:

  - Most used apps       -- counted every time any agent/command opens an app
  - Favorite contacts     -- counted every time a contact name is referenced
  - Typing style           -- aggregate stats over typed messages (length,
                               punctuation/emoji habits, sentence-casing) --
                               NOT a stored log of message content itself
  - Frequent locations     -- inferred from WiFi SSID (adb_controller
                               already exposes get_wifi_ssid() -- this is
                               the one location-shaped signal actually
                               available without root/location permissions;
                               see the honesty note below)

HONESTY NOTE ON "LOCATIONS"
-----------------------------
Plain `adb shell` (no root) cannot read the phone's GPS/network location
-- that requires the ACCESS_FINE_LOCATION/ACCESS_COARSE_LOCATION runtime
permission to be granted to a caller, which shell isn't by default, and
faking it would silently produce wrong data. What IS reliably readable
via ADB is the current WiFi SSID (already used elsewhere in this project
via `AndroidController.get_wifi_ssid()`), which is a genuine, honest proxy
for location clusters ("Home-WiFi" vs "Office-5G" vs unknown/mobile-data)
-- most people's WiFi networks map 1:1 to a small number of real places.
This module tracks THAT, not raw GPS coordinates, and the field is named
`frequent_networks` internally to keep this honest, with a `location_hint`
alias exposed for anything that wants to treat it as "places".

WHAT THIS DOES NOT DO
------------------------
- It never stores raw message text long-term, only aggregate stats
  (counts, average lengths, punctuation ratios) computed from messages as
  they pass through -- the same "durable pattern, not the transcript"
  principle used for the assistant's own memory elsewhere.
- It never uses any of this to auto-act without being asked; it's a
  read-only signal store other agents/prompts can consult
  ("open <most-used app>" style shortcuts, contact-name disambiguation,
  etc.) -- wiring that consumption in is a separate step.

PUBLIC API
----------
    learner = LearningAgent()
    learner.record_app_open("whatsapp")
    learner.record_contact_mention("rahul")
    learner.record_typed_message("hey are you free later?")
    learner.record_network_seen("Home-WiFi-5G")

    learner.top_apps(n=3)          -> [("whatsapp", 42), ("gmail", 17), ...]
    learner.top_contacts(n=3)      -> [("rahul", 9), ("mom", 6), ...]
    learner.typing_style_summary() -> "short, casual, low punctuation, no emoji"
    learner.top_networks(n=3)      -> [("Home-WiFi-5G", 30), ...]
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

LEARNING_FILE = Path(__file__).parent.parent / "learning_memory.json"

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)


def _now() -> float:
    return time.time()


class LearningAgent:
    """Persists lightweight usage-pattern counters to a local JSON file.
    Loads existing data on construction; every record_* call saves
    immediately (same "small file, write-through" approach contacts.py
    already uses -- no separate save() step to forget to call)."""

    def __init__(self, path: Path = LEARNING_FILE):
        self.path = path
        self._data = self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> dict:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return self._empty()
        # Backfill any keys an older version of this file might be missing.
        empty = self._empty()
        for key, val in empty.items():
            data.setdefault(key, val)
        return data

    @staticmethod
    def _empty() -> dict:
        return {
            "app_opens": {},        # {package_or_name: count}
            "contact_mentions": {}, # {contact_name: count}
            "networks_seen": {},    # {ssid: count}
            "typing_stats": {
                "message_count": 0,
                "total_chars": 0,
                "total_words": 0,
                "exclamation_count": 0,
                "question_count": 0,
                "emoji_count": 0,
                "lowercase_start_count": 0,
            },
            "last_updated": None,
        }

    def _save(self) -> None:
        self._data["last_updated"] = _now()
        try:
            self.path.write_text(json.dumps(self._data, indent=2))
        except OSError:
            pass  # learning is best-effort; never crash the assistant over it

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_app_open(self, app_name: str) -> None:
        name = (app_name or "").strip().lower()
        if not name:
            return
        counts = self._data["app_opens"]
        counts[name] = counts.get(name, 0) + 1
        self._save()

    def record_contact_mention(self, contact_name: str) -> None:
        name = (contact_name or "").strip().lower()
        if not name:
            return
        counts = self._data["contact_mentions"]
        counts[name] = counts.get(name, 0) + 1
        self._save()

    def record_network_seen(self, ssid: str) -> None:
        name = (ssid or "").strip()
        if not name or name == "(not connected)":
            return
        counts = self._data["networks_seen"]
        counts[name] = counts.get(name, 0) + 1
        self._save()

    def record_typed_message(self, text: str) -> None:
        """Update aggregate typing-style stats from one message. Does NOT
        store the message text itself anywhere."""
        stripped = (text or "").strip()
        if not stripped:
            return
        stats = self._data["typing_stats"]
        stats["message_count"] += 1
        stats["total_chars"] += len(stripped)
        stats["total_words"] += len(stripped.split())
        stats["exclamation_count"] += stripped.count("!")
        stats["question_count"] += stripped.count("?")
        stats["emoji_count"] += len(_EMOJI_RE.findall(stripped))
        if stripped[0:1].islower():
            stats["lowercase_start_count"] += 1
        self._save()

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def top_apps(self, n: int = 5) -> List[Tuple[str, int]]:
        return self._top(self._data["app_opens"], n)

    def top_contacts(self, n: int = 5) -> List[Tuple[str, int]]:
        return self._top(self._data["contact_mentions"], n)

    def top_networks(self, n: int = 5) -> List[Tuple[str, int]]:
        return self._top(self._data["networks_seen"], n)

    # Alias made explicit per the honesty note above -- same data,
    # named the way a caller thinking in terms of "places" would look
    # for it, without implying GPS-level precision.
    def location_hints(self, n: int = 5) -> List[Tuple[str, int]]:
        return self.top_networks(n)

    def typing_style_summary(self) -> str:
        s = self._data["typing_stats"]
        count = s["message_count"]
        if count < 5:
            return "Not enough typed messages yet to characterise typing style."

        avg_words = s["total_words"] / count
        avg_chars = s["total_chars"] / count
        excl_rate = s["exclamation_count"] / count
        q_rate = s["question_count"] / count
        emoji_rate = s["emoji_count"] / count
        lowercase_rate = s["lowercase_start_count"] / count

        length_desc = (
            "short" if avg_words < 6 else "medium-length" if avg_words < 15 else "long"
        )
        tone_bits = []
        if excl_rate > 0.3:
            tone_bits.append("enthusiastic (frequent !)")
        if q_rate > 0.3:
            tone_bits.append("asks a lot of questions")
        if emoji_rate > 0.2:
            tone_bits.append("uses emoji often")
        if lowercase_rate > 0.6:
            tone_bits.append("casual lowercase style")
        tone_desc = ", ".join(tone_bits) if tone_bits else "plain, neutral tone"

        return (
            f"{length_desc} messages (avg {avg_words:.0f} words / "
            f"{avg_chars:.0f} chars), {tone_desc}. Based on {count} messages."
        )

    def summary(self) -> str:
        """One human-readable block covering everything learned so far --
        used for a spoken/status report, e.g. 'what have you learned about me'."""
        lines = []
        apps = self.top_apps(3)
        if apps:
            lines.append("Most used apps: " + ", ".join(f"{a} ({c})" for a, c in apps))
        contacts = self.top_contacts(3)
        if contacts:
            lines.append("Favorite contacts: " + ", ".join(f"{c} ({n})" for c, n in contacts))
        nets = self.top_networks(3)
        if nets:
            lines.append("Frequent locations (by WiFi network): " + ", ".join(f"{s} ({c})" for s, c in nets))
        lines.append(self.typing_style_summary())
        return "\n".join(lines) if lines else "I haven't learned any usage patterns yet."

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _top(counts: Dict[str, int], n: int) -> List[Tuple[str, int]]:
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]