"""
command_parser.py
==================
A regex/rule-based natural-language command parser.

Feed it lines like:
    "open whatsapp and send hi how are you to 919876543210"
    "take a screenshot"
    "open camera and click photo"
    "google search best pizza near me"
    "turn on flashlight"
    "unlock my phone"
    "open youtube"
    "call mom"                      (needs a contacts.py lookup, see below)
    "increase volume by 3"
    "lock my phone"

It resolves each command to a single normalized Intent, then dispatches
it to the corresponding AndroidController method. No ML model is used —
this is a fast, fully offline, deterministic pattern matcher, which is
more reliable for device-control commands than a generic LLM call and
has zero extra dependencies or latency.

Extend the PATTERNS list to teach it new phrasings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from adb_controller import AndroidController
from contacts import resolve_contact


@dataclass
class Intent:
    name: str
    params: dict = field(default_factory=dict)
    raw_text: str = ""


class ParseError(ValueError):
    """Raised when no pattern matches the input command."""


# --------------------------------------------------------------------------- #
# Helper regex fragments
# --------------------------------------------------------------------------- #

_PHONE = r"(?P<number>\+?\d[\d\s\-]{6,})"
_CONTACT = r"(?P<contact>[A-Za-z][A-Za-z .]{0,40}?)"
_APP = r"(?P<app>[A-Za-z0-9][A-Za-z0-9 ]{0,30}?)"
_MSG = r"(?P<message>.+?)"
_QUERY = r"(?P<query>.+)"
_NUM = r"(?P<n>\d+)"


def _target(match: re.Match) -> str:
    """Return contact if present else number, for send/call commands."""
    gd = match.groupdict()
    return gd.get("contact") or gd.get("number") or ""


# --------------------------------------------------------------------------- #
# Pattern table: (regex, intent_name)
# Ordered most-specific first — first match wins.
# All patterns are matched case-insensitively against the whole command.
# --------------------------------------------------------------------------- #

PATTERNS: list[tuple[re.Pattern, str]] = [
    # --- WhatsApp: "open whatsapp and send <message> to <contact/number>" ---
    (re.compile(
        rf"^open\s+whatsapp\s+and\s+send\s+{_MSG}\s+to\s+(?:{_PHONE}|{_CONTACT})$",
        re.I), "whatsapp_send"),
    (re.compile(
        rf"^send\s+{_MSG}\s+(?:on|via)\s+whatsapp\s+to\s+(?:{_PHONE}|{_CONTACT})$",
        re.I), "whatsapp_send"),
    (re.compile(
        rf"^whatsapp\s+{_MSG}\s+to\s+(?:{_PHONE}|{_CONTACT})$",
        re.I), "whatsapp_send"),

    # --- SMS ---
    (re.compile(
        rf"^send\s+(?:an?\s+)?sms\s+{_MSG}\s+to\s+(?:{_PHONE}|{_CONTACT})$",
        re.I), "sms_send"),
    (re.compile(
        rf"^text\s+(?:{_PHONE}|{_CONTACT})\s+{_MSG}$",
        re.I), "sms_send"),

    # --- Calls ---
    (re.compile(rf"^call\s+(?:{_PHONE}|{_CONTACT})$", re.I), "call"),
    (re.compile(rf"^dial\s+(?:{_PHONE}|{_CONTACT})$", re.I), "dial"),

    # --- Screenshot / screen recording ---
    (re.compile(r"^(?:take\s+a?\s*)?screenshot$", re.I), "screenshot"),
    (re.compile(rf"^(?:record|screen\s*record)(?:\s+for\s+{_NUM}\s*(?:s|sec|seconds)?)?$", re.I), "screen_record"),

    # --- Camera ---
    (re.compile(r"^open\s+camera\s+and\s+(?:click|take)\s+(?:a\s+)?(?:photo|picture|pic)$", re.I), "camera_photo"),
    (re.compile(r"^(?:click|take)\s+(?:a\s+)?(?:photo|picture|pic)$", re.I), "camera_photo"),
    (re.compile(r"^open\s+camera$", re.I), "open_camera"),
    (re.compile(r"^(?:record|open)\s+video(?:\s+camera)?$", re.I), "open_video_camera"),

    # --- Flashlight ---
    (re.compile(r"^(?:turn\s+on|start|enable)\s+(?:the\s+)?flash(?:light)?$", re.I), "flashlight_on"),
    (re.compile(r"^(?:turn\s+off|stop|disable)\s+(?:the\s+)?flash(?:light)?$", re.I), "flashlight_off"),

    # --- Lock / unlock ---
    (re.compile(r"^unlock(?:\s+(?:my\s+)?phone)?(?:\s+with\s+pin\s+(?P<pin>\d+))?$", re.I), "unlock"),
    (re.compile(r"^lock(?:\s+(?:my\s+)?phone)?$", re.I), "lock"),
    (re.compile(r"^wake\s*(?:up)?(?:\s+(?:my\s+)?(?:phone|screen))?$", re.I), "wake"),
    (re.compile(r"^(?:sleep|turn\s+off\s+screen)$", re.I), "sleep"),

    # --- Browser / search ---
    (re.compile(rf"^(?:google\s+)?search\s+(?:for\s+)?{_QUERY}$", re.I), "google_search"),
    (re.compile(rf"^open\s+google\s+and\s+search\s+(?:for\s+)?{_QUERY}$", re.I), "google_search"),
    (re.compile(rf"^open\s+(?:url|website|link)\s+{_QUERY}$", re.I), "open_url"),
    (re.compile(rf"^go\s+to\s+{_QUERY}$", re.I), "open_url"),

    # --- App management (must come before the generic open/close patterns below) ---
    (re.compile(rf"^clear\s+(?:app\s+)?data\s+(?:for\s+|of\s+)?{_APP}$", re.I), "clear_app_data"),
    (re.compile(rf"^uninstall\s+{_APP}$", re.I), "uninstall_app"),
    (re.compile(rf"^open\s+(?:app\s+)?settings\s+for\s+{_APP}$", re.I), "open_app_settings"),
    (re.compile(r"^close\s+all\s+(?:recent\s+)?apps$", re.I), "close_all_apps"),
    (re.compile(rf"^open\s+{_QUERY}\s+settings$", re.I), "open_settings_page"),

    # --- Generic "open app and do X" (X becomes typed text, best-effort) ---
    # With optional "and press enter/send/send button/type enter" suffix
    (re.compile(
        rf"^open\s+{_APP}\s+and\s+type\s+{_MSG}"
        r"(?:\s+and\s+(?:press\s+)?(?:enter|send(?:\s+button)?|send\s+it|submit|go))?$",
        re.I), "open_app_and_type"),
    (re.compile(
        rf"^open\s+{_APP}\s+and\s+search\s+(?:for\s+)?{_QUERY}"
        r"(?:\s+and\s+(?:press\s+)?(?:enter|send(?:\s+button)?|search|go))?$",
        re.I), "open_app_and_search"),
    (re.compile(rf"^open\s+{_APP}$", re.I), "open_app"),
    (re.compile(rf"^close\s+{_APP}$", re.I), "close_app"),
    (re.compile(rf"^launch\s+{_APP}$", re.I), "open_app"),

    # --- Navigation ---
    (re.compile(r"^(?:go\s+)?home$", re.I), "home"),
    (re.compile(r"^(?:go\s+)?back$", re.I), "back"),
    (re.compile(r"^(?:show\s+)?recent\s*apps?$", re.I), "recents"),

    # --- Volume ---
    (re.compile(rf"^(?:increase|raise|turn\s+up)\s+volume(?:\s+by\s+{_NUM})?$", re.I), "volume_up"),
    (re.compile(rf"^(?:decrease|lower|turn\s+down)\s+volume(?:\s+by\s+{_NUM})?$", re.I), "volume_down"),
    (re.compile(r"^mute(?:\s+(?:the\s+)?(?:phone|volume|sound))?$", re.I), "mute"),

    # --- WiFi / Bluetooth / Airplane mode ---
    (re.compile(r"^(?:turn\s+on|enable)\s+wi[\-\s]?fi$", re.I), "wifi_on"),
    (re.compile(r"^(?:turn\s+off|disable)\s+wi[\-\s]?fi$", re.I), "wifi_off"),
    (re.compile(r"^(?:turn\s+on|enable)\s+bluetooth$", re.I), "bluetooth_on"),
    (re.compile(r"^(?:turn\s+off|disable)\s+bluetooth$", re.I), "bluetooth_off"),
    (re.compile(r"^(?:turn\s+on|enable)\s+airplane\s*mode$", re.I), "airplane_on"),
    (re.compile(r"^(?:turn\s+off|disable)\s+airplane\s*mode$", re.I), "airplane_off"),

    # --- Media playback ---
    (re.compile(r"^(?:play|pause|play\s*/\s*pause)(?:\s+music)?$", re.I), "play_pause"),
    (re.compile(r"^next\s*(?:track|song)?$", re.I), "next_track"),
    (re.compile(r"^(?:previous|prev)\s*(?:track|song)?$", re.I), "prev_track"),

    # --- Battery / status / misc ---
    (re.compile(r"^battery(?:\s+status)?$", re.I), "battery"),
    (re.compile(r"^(?:what.?s\s+)?(?:the\s+)?current\s+app$", re.I), "current_app"),
    (re.compile(r"^vibrate$", re.I), "vibrate"),
    (re.compile(r"^reboot(?:\s+(?:my\s+)?phone)?$", re.I), "reboot"),

    # --- Do Not Disturb / ringer ---
    (re.compile(r"^(?:turn\s+on|enable)\s+(?:do\s+not\s+disturb|dnd)$", re.I), "dnd_on"),
    (re.compile(r"^(?:turn\s+off|disable)\s+(?:do\s+not\s+disturb|dnd)$", re.I), "dnd_off"),
    (re.compile(r"^(?:set\s+(?:phone\s+)?to\s+)?silent(?:\s+mode)?$", re.I), "ringer_silent"),
    (re.compile(r"^(?:set\s+(?:phone\s+)?to\s+)?vibrate(?:\s+mode)?$", re.I), "ringer_vibrate"),
    (re.compile(r"^(?:set\s+(?:phone\s+)?to\s+)?normal(?:\s+mode)?$", re.I), "ringer_normal"),

    # --- Rotation / display ---
    (re.compile(r"^(?:turn\s+on|enable)\s+auto\s*rotate$", re.I), "auto_rotate_on"),
    (re.compile(r"^(?:turn\s+off|disable)\s+auto\s*rotate$", re.I), "auto_rotate_off"),
    (re.compile(rf"^(?:rotate|set\s+rotation\s+to)\s+{_NUM}(?:\s*degrees?)?$", re.I), "set_rotation"),
    (re.compile(rf"^set\s+screen\s+timeout\s+(?:to\s+)?{_NUM}\s*(?:s|sec|seconds)?$", re.I), "screen_timeout"),

    # --- Scrolling ---
    (re.compile(r"^scroll\s+up(?:\s+(?P<amount>small|medium|large))?$", re.I), "scroll_up"),
    (re.compile(r"^scroll\s+down(?:\s+(?P<amount>small|medium|large))?$", re.I), "scroll_down"),
    (re.compile(r"^scroll\s+left(?:\s+(?P<amount>small|medium|large))?$", re.I), "scroll_left"),
    (re.compile(r"^scroll\s+right(?:\s+(?P<amount>small|medium|large))?$", re.I), "scroll_right"),

    # --- Press enter / send explicitly (MUST come before generic tap_text below) ---
    (re.compile(r"^(?:press|hit)\s+enter$", re.I), "press_enter"),
    (re.compile(r"^(?:press|tap|hit)\s+(?:the\s+)?send\s+button$", re.I), "press_enter"),
    (re.compile(r"^(?:press|tap|hit)\s+(?:the\s+)?(?:go|submit|done)\s+button$", re.I), "press_enter"),
    (re.compile(r"^send\s+it$", re.I), "smart_send"),
    (re.compile(r"^(?:submit|done)$", re.I), "smart_send"),

    # --- Find/tap text visible on screen ---
    (re.compile(rf"^tap\s+(?:on\s+)?{_QUERY}$", re.I), "tap_text"),
    (re.compile(rf"^(?:press|click)\s+(?:the\s+)?{_QUERY}\s+button$", re.I), "tap_text"),

    # --- Text field editing ---
    (re.compile(r"^select\s+all(?:\s+text)?$", re.I), "select_all_text"),
    (re.compile(r"^clear\s+(?:the\s+)?(?:text\s+)?field$", re.I), "clear_text_field"),
    (re.compile(r"^paste(?:\s+clipboard)?$", re.I), "paste"),
    (re.compile(r"^(?:what.?s\s+(?:in\s+)?(?:the\s+)?clipboard|get\s+clipboard)$", re.I), "get_clipboard"),

    # --- Alarms / timers ---
    (re.compile(rf"^set\s+(?:an?\s+)?alarm\s+(?:for\s+)?{_NUM}(?::(?P<minute>\d{{2}}))?\s*(?P<ampm>am|pm)?(?:\s+(?:called|labeled)\s+{_MSG})?$", re.I), "set_alarm"),
    (re.compile(rf"^set\s+(?:a\s+)?timer\s+(?:for\s+)?{_NUM}\s*(?P<unit>s|sec|secs|seconds|m|min|mins|minutes)?(?:\s+(?:called|labeled)\s+{_MSG})?$", re.I), "set_timer"),

    # --- Telegram ---
    (re.compile(
        rf"^(?:open\s+)?telegram\s+(?:and\s+)?send\s+{_MSG}\s+to\s+(?:{_PHONE}|{_CONTACT})$",
        re.I), "telegram_send"),

    # --- Split screen ---
    (re.compile(r"^split\s*screen$", re.I), "split_screen"),

    # --- Read screen ---
    (re.compile(r"^(?:read|what(?:'s|\s+is)\s+on)\s+(?:the\s+)?screen$", re.I), "read_screen"),
    (re.compile(r"^(?:what(?:'s|\s+is)\s+(?:showing|visible|displayed)(?:\s+on\s+(?:the\s+)?screen)?)$", re.I), "read_screen"),

    # --- Device info ---
    (re.compile(r"^(?:device|phone)\s+info(?:rmation)?$", re.I), "device_info"),
    (re.compile(r"^(?:what(?:'s|\s+is)\s+(?:my\s+)?(?:device|phone)\s+(?:model|info|name))$", re.I), "device_info"),
    (re.compile(r"^screen\s+(?:size|resolution)$", re.I), "device_info"),

    # --- Dark mode ---
    (re.compile(r"^(?:turn\s+on|enable)\s+dark\s+mode$", re.I), "dark_mode_on"),
    (re.compile(r"^(?:turn\s+off|disable)\s+dark\s+mode$", re.I), "dark_mode_off"),

    # --- Hotspot ---
    (re.compile(r"^(?:turn\s+on|enable|start)\s+(?:wifi\s+)?hotspot$", re.I), "hotspot_on"),
    (re.compile(r"^(?:turn\s+off|disable|stop)\s+(?:wifi\s+)?hotspot$", re.I), "hotspot_off"),

    # --- Notifications shade ---
    (re.compile(r"^(?:expand|pull\s+down|open|show)\s+notifications?(?:\s+shade)?$", re.I), "expand_notifications"),
    (re.compile(r"^(?:collapse|close|hide)\s+notifications?(?:\s+shade)?$", re.I), "collapse_notifications"),
    (re.compile(r"^(?:expand|pull\s+down|open|show)\s+quick\s+settings?$", re.I), "expand_quick_settings"),

    # --- Keyboard ---
    (re.compile(r"^(?:hide|dismiss|close)\s+(?:the\s+)?keyboard$", re.I), "hide_keyboard"),

    # --- Set volume absolute ---
    (re.compile(rf"^set\s+(?:(?P<stream>music|ring|ringer|alarm|notification|call|media)\s+)?volume\s+(?:to\s+)?{_NUM}$", re.I), "set_volume_level"),

    # --- Font size ---
    (re.compile(r"^set\s+font\s+(?:size\s+)?(?:to\s+)?(?P<size>small|normal|large|larger|largest)$", re.I), "set_font_size"),

    # --- WiFi SSID / network info ---
    (re.compile(r"^(?:what(?:'s|\s+is)\s+(?:my\s+)?wi[- ]?fi|connected\s+wi[- ]?fi|wifi\s+(?:info|name|status)|wi[- ]?fi\s+info|ssid)$", re.I), "wifi_info"),

    # --- Ping / internet check ---
    (re.compile(r"^(?:check\s+)?(?:internet|network|ping)(?:\s+connection)?$", re.I), "ping"),

    # --- List running apps ---
    (re.compile(r"^(?:list|show|what)\s+(?:are\s+(?:the\s+)?)?running\s+apps?$", re.I), "list_running_apps"),

    # --- App version ---
    (re.compile(rf"^(?:version|app\s+version)\s+(?:of\s+)?{_APP}$", re.I), "app_version"),

    # --- type_and_send shortcut (type something and immediately send) ---
    (re.compile(rf"^type\s+{_MSG}\s+and\s+(?:press\s+)?(?:enter|send|submit|go)$", re.I), "type_and_send"),
    (re.compile(rf"^(?:send|submit)\s+{_MSG}$", re.I), "type_and_send"),

    # --- Full app switcher swipe-away-all handled by close_all_apps above ---
]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_command(text: str) -> Intent:
    """
    Parse a single natural-language command string into an Intent.
    Raises ParseError if nothing matches.
    """
    cleaned = text.strip().rstrip(".")
    for pattern, name in PATTERNS:
        match = pattern.match(cleaned)
        if match:
            return Intent(name=name, params=match.groupdict(), raw_text=text)
    raise ParseError(
        f"Could not understand command: {text!r}\n"
        f"Try phrasings like: 'open whatsapp and send hello to 91XXXXXXXXXX', "
        f"'take screenshot', 'open camera and click photo', 'search cheap flights', "
        f"'turn on flashlight', 'call mom', 'open youtube'."
    )


def parse_multi_command(text: str) -> list[Intent]:
    """
    Split on ' and then ' / ' then ' / ';' to support chained commands like:
    "open camera and click photo then take screenshot"
    Each segment is parsed independently. Falls back to a single Intent
    if only one segment matches cleanly (protects phrases like
    "open whatsapp and send X" which contain 'and' but are one intent).
    """
    # Try parsing the whole string as one intent first (handles the common
    # single-action-with-'and' cases like whatsapp_send, camera_photo, etc.)
    try:
        return [parse_command(text)]
    except ParseError:
        pass

    segments = re.split(r"\s*(?:;|\bthen\b)\s*", text.strip(), flags=re.I)
    intents = []
    for seg in segments:
        seg = seg.strip()
        if seg:
            intents.append(parse_command(seg))
    if not intents:
        raise ParseError(f"Could not understand command: {text!r}")
    return intents


# --------------------------------------------------------------------------- #
# Dispatch: Intent -> AndroidController method call
# --------------------------------------------------------------------------- #

def execute_intent(controller: AndroidController, intent: Intent) -> str:
    """
    Run the given Intent against a connected AndroidController.
    Returns a short human-readable status string.
    """
    p = intent.params
    name = intent.name

    if name == "whatsapp_send":
        target = _target_from_params(p)
        number = resolve_contact(target)
        controller.send_whatsapp_message(number, p["message"].strip())
        return f"Sent WhatsApp message to {target}: {p['message'].strip()!r}"

    if name == "sms_send":
        target = _target_from_params(p)
        number = resolve_contact(target)
        controller.send_sms(number, p["message"].strip())
        return f"Opened SMS to {target} with message {p['message'].strip()!r} (review & tap send)"

    if name == "call":
        target = _target_from_params(p)
        number = resolve_contact(target)
        controller.call_number(number)
        return f"Calling {target} ({number})"

    if name == "dial":
        target = _target_from_params(p)
        number = resolve_contact(target)
        controller.dial_number(number)
        return f"Dialer opened with {number}"

    if name == "screenshot":
        path = controller.screenshot()
        return f"Screenshot saved: {path}"

    if name == "screen_record":
        seconds = int(p.get("n") or 10)
        path = controller.screen_record(seconds=seconds)
        return f"Recorded {seconds}s to {path}"

    if name == "camera_photo":
        controller.open_camera(take_photo=True)
        return "Opened camera and captured a photo"

    if name == "open_camera":
        controller.open_camera()
        return "Camera opened"

    if name == "open_video_camera":
        controller.open_video_camera()
        return "Video camera opened"

    if name == "flashlight_on":
        controller.flashlight(True)
        return "Flashlight ON"

    if name == "flashlight_off":
        controller.flashlight(False)
        return "Flashlight OFF"

    if name == "unlock":
        controller.unlock(pin=p.get("pin"))
        return "Unlock attempted"

    if name == "lock":
        controller.lock()
        return "Phone locked"

    if name == "wake":
        controller.wake_screen()
        return "Screen woken"

    if name == "sleep":
        controller.sleep_screen()
        return "Screen turned off"

    if name == "google_search":
        controller.google_search(p["query"].strip())
        return f"Searched Google for {p['query'].strip()!r}"

    if name == "open_url":
        controller.open_url(p["query"].strip())
        return f"Opened URL {p['query'].strip()!r}"

    if name == "open_app_and_type":
        import time as _t
        app = p["app"].strip()
        msg = p["message"].strip()
        controller.open_app(app)
        _t.sleep(2.2)
        controller.type_text(msg)
        return f"Opened {app} and typed {msg!r}"

    if name == "open_app_and_search":
        import time as _t
        app = p["app"].strip()
        query = p["query"].strip()
        controller.open_app(app)
        _t.sleep(2.2)
        controller.type_text(query)
        _t.sleep(0.3)
        # Always press enter/search after typing a search query
        controller.smart_send()
        return f"Opened {app}, searched {query!r} and submitted"

    if name == "open_app":
        controller.open_app(p["app"].strip())
        return f"Opened {p['app'].strip()}"

    if name == "close_app":
        controller.close_app(p["app"].strip())
        return f"Closed {p['app'].strip()}"

    if name == "home":
        controller.go_home()
        return "Went home"

    if name == "back":
        controller.go_back()
        return "Went back"

    if name == "recents":
        controller.recent_apps()
        return "Showing recent apps"

    if name == "volume_up":
        n = int(p.get("n") or 1)
        controller.volume_up(n)
        return f"Volume up x{n}"

    if name == "volume_down":
        n = int(p.get("n") or 1)
        controller.volume_down(n)
        return f"Volume down x{n}"

    if name == "mute":
        controller.mute()
        return "Muted"

    if name == "wifi_on":
        controller.set_wifi(True)
        return "WiFi ON"

    if name == "wifi_off":
        controller.set_wifi(False)
        return "WiFi OFF"

    if name == "bluetooth_on":
        controller.set_bluetooth(True)
        return "Bluetooth ON"

    if name == "bluetooth_off":
        controller.set_bluetooth(False)
        return "Bluetooth OFF"

    if name == "airplane_on":
        controller.set_airplane_mode(True)
        return "Airplane mode ON"

    if name == "airplane_off":
        controller.set_airplane_mode(False)
        return "Airplane mode OFF"

    if name == "play_pause":
        controller.play_pause_media()
        return "Play/Pause toggled"

    if name == "next_track":
        controller.next_track()
        return "Next track"

    if name == "prev_track":
        controller.prev_track()
        return "Previous track"

    if name == "battery":
        info = controller.battery_status()
        level = info.get("level", "?")
        status = info.get("status", "?")
        return f"Battery: {level}% (status code {status})"

    if name == "current_app":
        return f"Current app: {controller.get_current_app()}"

    if name == "vibrate":
        controller.vibrate()
        return "Vibrated"

    if name == "reboot":
        controller.reboot()
        return "Rebooting phone..."

    # ---- Newly added: raw input control (tap/swipe/type/keys) ----
    if name == "tap":
        x, y = int(p["x"]), int(p["y"])
        controller.tap(x, y)
        return f"Tapped ({x}, {y})"

    if name == "swipe":
        x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
        duration = int(p.get("duration_ms") or 300)
        controller.swipe(x1, y1, x2, y2, duration_ms=duration)
        return f"Swiped ({x1},{y1}) -> ({x2},{y2})"

    if name == "long_press":
        x, y = int(p["x"]), int(p["y"])
        duration = int(p.get("duration_ms") or 800)
        controller.long_press(x, y, duration_ms=duration)
        return f"Long-pressed ({x}, {y})"

    if name == "type_text_raw":
        controller.type_text(p["text"])
        return f"Typed {p['text']!r} into the focused field"

    if name == "press_key":
        controller.press_key(p["key"].strip())
        return f"Pressed key '{p['key'].strip()}'"

    if name == "set_brightness":
        level = int(p["level"])
        controller.set_brightness(level)
        return f"Brightness set to {level}/255"

    if name == "share_text":
        controller.share_text(p["text"])
        return f"Opened share sheet with {p['text']!r}"

    if name == "get_notifications":
        dump = controller.get_notifications()
        preview = dump[:800] + ("..." if len(dump) > 800 else "")
        return f"Notification dump (raw, truncated):\n{preview}"

    if name == "clear_notifications":
        controller.clear_notifications()
        return "Notifications cleared"

    if name == "set_clipboard":
        controller.set_clipboard(p["text"])
        return f"Clipboard set to {p['text']!r}"

    # ---- Newly added: generic chained "open app then do steps" ----
    if name == "open_app_and_do":
        app = p["app"].strip()
        controller.open_app(app)
        import time as _t
        steps = p.get("steps") or []
        _t.sleep(2)  # let the app launch before the first step
        done = [f"opened {app}"]
        for step in steps:
            action = (step.get("action") or "").lower()
            if action == "wait":
                secs = float(step.get("seconds") or 1)
                _t.sleep(secs)
                done.append(f"waited {secs}s")
            elif action == "tap":
                controller.tap(int(step["x"]), int(step["y"]))
                done.append(f"tapped ({step['x']},{step['y']})")
            elif action == "swipe":
                controller.swipe(int(step["x1"]), int(step["y1"]),
                                  int(step["x2"]), int(step["y2"]),
                                  duration_ms=int(step.get("duration_ms") or 300))
                done.append("swiped")
            elif action == "type":
                controller.type_text(str(step.get("text", "")))
                done.append(f"typed {step.get('text', '')!r}")
            elif action == "key":
                controller.press_key(str(step.get("key", "")).lower())
                done.append(f"pressed {step.get('key', '')}")
            elif action == "back":
                controller.go_back()
                done.append("pressed back")
            elif action == "home":
                controller.go_home()
                done.append("pressed home")
            elif action == "tap_text":
                label = str(step.get("text") or step.get("query") or "")
                found = controller.tap_text(label)
                done.append(f"tapped text {label!r}" if found else f"(text {label!r} not found)")
            elif action == "enter":
                controller.press_enter()
                done.append("pressed enter")
            elif action == "send":
                controller.smart_send()
                done.append("pressed send/enter")
            elif action == "scroll_up":
                controller.scroll_up(str(step.get("amount") or "medium"))
                done.append("scrolled up")
            elif action == "scroll_down":
                controller.scroll_down(str(step.get("amount") or "medium"))
                done.append("scrolled down")
            elif action == "screenshot":
                controller.screenshot()
                done.append("screenshot taken")
            # unknown step actions are silently skipped rather than
            # raising, so one bad step in a long chain doesn't abort
            # everything already completed
        return f"Ran chained sequence: {'; '.join(done)}"

    # ---- Do Not Disturb / ringer ----
    if name == "dnd_on":
        controller.set_dnd(True)
        return "Do Not Disturb ON"

    if name == "dnd_off":
        controller.set_dnd(False)
        return "Do Not Disturb OFF"

    if name == "ringer_silent":
        controller.set_ringer_mode("silent")
        return "Ringer set to silent"

    if name == "ringer_vibrate":
        controller.set_ringer_mode("vibrate")
        return "Ringer set to vibrate"

    if name == "ringer_normal":
        controller.set_ringer_mode("normal")
        return "Ringer set to normal"

    # ---- Rotation / display ----
    if name == "auto_rotate_on":
        controller.set_auto_rotate(True)
        return "Auto-rotate ON"

    if name == "auto_rotate_off":
        controller.set_auto_rotate(False)
        return "Auto-rotate OFF"

    if name == "set_rotation":
        degrees = int(p["n"])
        controller.set_rotation(degrees)
        return f"Rotation set to {degrees} degrees"

    if name == "screen_timeout":
        seconds = int(p["n"])
        controller.set_screen_timeout(seconds)
        return f"Screen timeout set to {seconds}s"

    # ---- App management ----
    if name == "clear_app_data":
        app = p["app"].strip()
        controller.clear_app_data(app)
        return f"Cleared app data for {app}"

    if name == "uninstall_app":
        app = p["app"].strip()
        controller.uninstall_app(app)
        return f"Uninstalled {app}"

    if name == "open_app_settings":
        app = p["app"].strip()
        controller.open_app_settings(app)
        return f"Opened settings for {app}"

    if name == "close_all_apps":
        controller.force_stop_all_recent()
        return "Force-stopped all backgrounded third-party apps"

    # ---- Scrolling ----
    if name == "scroll_up":
        controller.scroll_up(p.get("amount") or "medium")
        return "Scrolled up"

    if name == "scroll_down":
        controller.scroll_down(p.get("amount") or "medium")
        return "Scrolled down"

    if name == "scroll_left":
        controller.scroll_left(p.get("amount") or "medium")
        return "Scrolled left"

    if name == "scroll_right":
        controller.scroll_right(p.get("amount") or "medium")
        return "Scrolled right"

    # ---- Find/tap text on screen ----
    if name == "tap_text":
        target_text = p["query"].strip()
        found = controller.tap_text(target_text)
        if found:
            return f"Tapped element matching {target_text!r}"
        raise ParseError(
            f"Couldn't find anything matching {target_text!r} on the current "
            f"screen. Make sure it's actually visible right now."
        )

    # ---- Text field editing ----
    if name == "select_all_text":
        controller.select_all_text()
        return "Selected all text in focused field"

    if name == "clear_text_field":
        controller.clear_text_field()
        return "Cleared focused text field"

    if name == "paste":
        controller.paste_clipboard()
        return "Pasted clipboard into focused field"

    if name == "get_clipboard":
        text = controller.get_clipboard()
        return f"Clipboard: {text!r}"

    # ---- Alarms / timers ----
    # (accepts params from either the regex parser, which captures "n" for
    # the hour, or the LLM parser, which sends "hour" directly)
    if name == "set_alarm":
        hour = int(p.get("hour") if p.get("hour") is not None else p["n"])
        minute = int(p.get("minute") or 0)
        ampm = (p.get("ampm") or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        label = (p.get("label") if p.get("label") is not None else p.get("message")) or ""
        label = label.strip()
        controller.set_alarm(hour, minute, label)
        return f"Alarm set for {hour:02d}:{minute:02d}" + (f" ({label})" if label else "")

    if name == "set_timer":
        if p.get("seconds") is not None:
            seconds = int(p["seconds"])
        else:
            n = int(p["n"])
            unit = (p.get("unit") or "s").lower()
            seconds = n * 60 if unit.startswith("m") else n
        label = (p.get("label") if p.get("label") is not None else p.get("message")) or ""
        label = label.strip()
        controller.set_timer(seconds, label)
        return f"Timer set for {seconds}s" + (f" ({label})" if label else "")

    # ---- Settings pages ----
    if name == "open_settings_page":
        page_query = (p.get("page") or p.get("query") or "").strip().lower().replace(" ", "_")
        controller.open_settings_page(page_query)
        return f"Opened {page_query.replace('_', ' ')} settings"

    # ---- Telegram ----
    if name == "telegram_send":
        target = _target_from_params(p)
        number = resolve_contact(target)
        controller.send_telegram_message(number, p["message"].strip())
        return f"Sent Telegram message to {target}: {p['message'].strip()!r}"

    # ---- Split screen ----
    if name == "split_screen":
        controller.split_screen_current()
        return "Attempted split-screen (behavior varies by device/launcher)"

    # ---- Press Enter / smart send ----
    if name == "press_enter":
        controller.press_enter()
        return "Pressed Enter / Send"

    if name == "smart_send":
        found = controller.smart_send()
        return "Tapped Send button" if found else "Pressed Enter (send button not found)"

    if name == "type_and_send":
        text = (p.get("message") or p.get("text") or "").strip()
        controller.type_and_send(text)
        return f"Typed {text!r} and submitted"

    # ---- Read screen ----
    if name == "read_screen":
        summary = controller.read_screen_summary()
        return f"Screen shows: {summary}"

    # ---- Device info ----
    if name == "device_info":
        info = controller.get_device_info()
        lines = [f"{k}: {v}" for k, v in info.items()]
        return "\n".join(lines)

    # ---- Dark mode ----
    if name == "dark_mode_on":
        controller.set_dark_mode(True)
        return "Dark mode ON"

    if name == "dark_mode_off":
        controller.set_dark_mode(False)
        return "Dark mode OFF"

    # ---- Hotspot ----
    if name == "hotspot_on":
        controller.set_wifi_hotspot(True)
        return "WiFi hotspot ON"

    if name == "hotspot_off":
        controller.set_wifi_hotspot(False)
        return "WiFi hotspot OFF"

    # ---- Notifications shade ----
    if name == "expand_notifications":
        controller.expand_notifications()
        return "Notification shade expanded"

    if name == "collapse_notifications":
        controller.collapse_notifications()
        return "Notification shade collapsed"

    if name == "expand_quick_settings":
        controller.expand_quick_settings()
        return "Quick settings expanded"

    # ---- Keyboard ----
    if name == "hide_keyboard":
        controller.hide_keyboard()
        return "Keyboard dismissed"

    # ---- Set volume absolute ----
    if name == "set_volume_level":
        level = int(p.get("n") or 5)
        stream = (p.get("stream") or "music").strip()
        controller.set_volume_level(stream, level)
        return f"{stream.capitalize()} volume set to {level}"

    # ---- Font size ----
    if name == "set_font_size":
        size_map = {"small": 0.85, "normal": 1.0, "large": 1.15, "larger": 1.3, "largest": 1.45}
        size_name = (p.get("size") or "normal").lower()
        scale = size_map.get(size_name, 1.0)
        controller.set_font_size(scale)
        return f"Font size set to {size_name} ({scale})"

    # ---- WiFi info ----
    if name == "wifi_info":
        ssid = controller.get_wifi_ssid()
        ip = controller.get_device_ip() or "unknown"
        return f"WiFi: {ssid} | IP: {ip}"

    # ---- Ping ----
    if name == "ping":
        result = controller.ping()
        return f"Ping result:\n{result}"

    # ---- List running apps ----
    if name == "list_running_apps":
        apps = controller.list_running_apps()
        return "Running apps: " + (", ".join(apps) if apps else "(none found)")

    # ---- App version ----
    if name == "app_version":
        app = p["app"].strip()
        version = controller.get_app_version(app)
        return f"{app} version: {version}"

    raise ParseError(f"No handler wired up for intent '{name}'")


def _target_from_params(p: dict) -> str:
    contact = (p.get("contact") or "").strip()
    number = (p.get("number") or "").strip()
    return contact or number
