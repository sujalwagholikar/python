"""
llm_parser.py
=============
Natural-language intent understanding powered by Groq's hosted
Llama 3.1 8B (model id: "llama-3.1-8b-instant").

Why this exists
----------------
The regex parser in command_parser.py is fast and 100% offline, but it
only understands phrasings it has a pattern for. Real speech is messier:
    "hey can you message rohan on whatsapp and tell him im running late"
    "yaar whatsapp pe mom ko bol do ki mai aa raha hu"
    "pull up youtube for me"
None of those match a fixed regex. This module sends the raw sentence to
Llama 3.1 8B on Groq (extremely fast inference, generous free tier) and
asks it to return ONLY a strict JSON object describing the action —
never asks the model to execute anything itself. Python still does 100%
of the actual device control; the LLM's only job is turning fuzzy English
into a structured Intent your code already knows how to execute.

Flow
----
1. Try the offline regex parser first (command_parser.py) — it's instant
   and free, so use it whenever it confidently matches.
2. If it doesn't match, call Groq/Llama 3.1 8B to classify the sentence
   into one of the SAME intent names command_parser.py already knows how
   to execute, with the same parameter names.
3. Validate the model's JSON strictly before ever touching the phone —
   unknown intent names or malformed JSON are rejected, never guessed at
   or run blindly.

Setup
-----
1. Get a free API key at https://console.groq.com/keys
2. Copy .env.example to .env (project root) and set:
       GROQ_API_KEY_INTENT=your-key-here
   (a plain GROQ_API_KEY is also accepted as a fallback name).
3. No extra pip installs needed — this uses only Python's built-in
   `urllib`, so you don't need the `groq` or `openai` packages.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

from command_parser import Intent, ParseError

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# API key — set GROQ_API_KEY_INTENT (preferred) or GROQ_API_KEY in your .env
# file / environment variables. NEVER hardcode a real key in source code.
# Get a free key at: https://console.groq.com/keys
#
# This key powers the FAST, cheap intent-parsing model (llama-3.1-8b-instant)
# used as a fallback when the offline regex parser can't match a command.
# The separate JARVIS conversational brain (llama-3.3-70b-versatile) uses
# its own key — see jarvis_brain.py / GROQ_API_KEY_JARVIS in .env.
# --------------------------------------------------------------------------- #
GROQ_API_KEY = os.environ.get("GROQ_API_KEY_INTENT") or os.environ.get("GROQ_API_KEY", "")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL_INTENT", "llama-3.1-8b-instant")

# The exact set of intents command_parser.execute_intent() knows how to run,
# and the parameter names each one expects. Keeping the LLM constrained to
# this exact list is what makes this safe — it can't invent a new action
# that has no handler.
INTENT_SCHEMA = """
Valid "intent" values and their required "params" keys:

  whatsapp_send   { "contact": str or null, "number": str or null, "message": str }
  sms_send        { "contact": str or null, "number": str or null, "message": str }
  call            { "contact": str or null, "number": str or null }
  dial            { "contact": str or null, "number": str or null }
  screenshot      {}
  screen_record   { "n": str or null }              // seconds, as a string, e.g. "10"
  camera_photo    {}
  open_camera     {}
  open_video_camera {}
  flashlight_on   {}
  flashlight_off  {}
  unlock          { "pin": str or null }
  lock            {}
  wake            {}
  sleep           {}
  google_search   { "query": str }
  open_url        { "query": str }                   // the URL/website
  open_app_and_type { "app": str, "message": str }   // opens app, types message — auto-submits if user says "and send/enter"
  open_app_and_search { "app": str, "query": str }   // opens app, types query and presses enter/search
  open_app        { "app": str }
  close_app       { "app": str }
  home            {}
  back            {}
  recents         {}
  volume_up       { "n": str or null }
  volume_down     { "n": str or null }
  mute            {}
  wifi_on         {}
  wifi_off        {}
  bluetooth_on    {}
  bluetooth_off   {}
  airplane_on     {}
  airplane_off    {}
  play_pause      {}
  next_track      {}
  prev_track      {}
  battery         {}
  current_app     {}
  vibrate         {}
  reboot          {}

  // Raw input control — use when the user describes a physical gesture
  // or names on-screen coordinates directly.
  tap             { "x": str, "y": str }
  swipe           { "x1": str, "y1": str, "x2": str, "y2": str, "duration_ms": str or null }
  long_press      { "x": str, "y": str, "duration_ms": str or null }
  type_text_raw   { "text": str }          // types into whatever field is currently focused
  press_key       { "key": str }           // one of: home back call endcall volume_up
                                            // volume_down power camera menu enter delete
                                            // app_switch notification search play_pause
                                            // next_track prev_track screenshot sleep wakeup

  // Device extras
  set_brightness  { "level": str }         // 0-255
  share_text      { "text": str }          // opens Android share sheet with this text
  get_notifications {}
  clear_notifications {}
  set_clipboard   { "text": str }
  get_clipboard   {}
  paste           {}                       // paste clipboard into focused field (Android 12+)
  select_all_text {}
  clear_text_field {}

  // Do Not Disturb / ringer
  dnd_on          {}
  dnd_off         {}
  ringer_silent   {}
  ringer_vibrate  {}
  ringer_normal   {}

  // Rotation / display
  auto_rotate_on  {}
  auto_rotate_off {}
  set_rotation    { "n": str }             // one of 0, 90, 180, 270
  screen_timeout  { "n": str }             // seconds

  // App management
  clear_app_data  { "app": str }
  uninstall_app   { "app": str }
  open_app_settings { "app": str }
  close_all_apps  {}
  split_screen    {}

  // Scrolling
  scroll_up       { "amount": str or null }    // small/medium/large, default medium
  scroll_down     { "amount": str or null }
  scroll_left     { "amount": str or null }
  scroll_right    { "amount": str or null }

  // Find & tap something currently VISIBLE on screen by its label/text
  // (uses a live UI scan, not guessed coordinates) — use this whenever
  // the user names a button/label/icon rather than giving coordinates,
  // e.g. "tap send", "press the send button", "tap on settings".
  tap_text        { "query": str }

  // Alarms / timers (opens the clock app's official set-alarm/set-timer
  // action; a confirmation screen may briefly appear, which is normal
  // Android behavior, not something this tool can or should suppress)
  set_alarm       { "hour": str, "minute": str or null, "ampm": str or null, "label": str or null }
  set_timer       { "seconds": str, "label": str or null }

  // System settings pages (opens the named settings screen directly)
  open_settings_page { "page": str }   // one of: wifi bluetooth display sound apps battery
                                        // storage location security accounts date_time
                                        // language accessibility developer notifications airplane

  // Additional messaging platforms
  telegram_send   { "contact": str or null, "number": str or null, "message": str }

  // GENERIC MULTI-STEP ACTION — use this whenever the user describes
  // opening an app and then doing a SEQUENCE of things inside it that
  // no other single intent above covers (tapping a specific button,
  // typing into a field, waiting, going back, etc). This is your
  // general-purpose fallback for "open X and do this and that"-style
  // requests that don't map to whatsapp_send/open_app_and_type/etc.
  //
  // NOTE: the "steps" you list below are only a rough initial hint of
  // intent — at execution time this is handed to a closed-loop agent
  // (task_agent.py) that re-plans ONE action at a time against the
  // REAL current screen and keeps going until the goal is verified
  // complete (or honestly reports it couldn't be done), rather than
  // blindly running your guessed steps in order. Still fill in a
  // reasonable "steps" list — it helps signal what you think the
  // sequence roughly looks like — but don't worry about getting every
  // exact tap right; the agent will adapt to whatever's actually on
  // screen.
  open_app_and_do {
    "app": str,
    "steps": [
      // each step is one of:
      { "action": "wait",      "seconds": str },
      { "action": "tap",       "x": str, "y": str },
      { "action": "tap_text",  "text": str },        // tap a visible label/button by name (PREFERRED over tap with coords)
      { "action": "swipe",     "x1": str, "y1": str, "x2": str, "y2": str, "duration_ms": str or null },
      { "action": "type",      "text": str },
      { "action": "key",       "key": str },          // same key names as press_key
      { "action": "enter" },                          // press Enter/Done key
      { "action": "send" },                           // tap Send button OR press Enter
      { "action": "scroll_up",   "amount": str or null },
      { "action": "scroll_down", "amount": str or null },
      { "action": "back" },
      { "action": "home" },
      { "action": "screenshot" }
    ]
  }

  // NEW intents:
  press_enter     {}                            // press the Enter / Done / Go key
  smart_send      {}                            // tap Send button if visible, else press Enter
  type_and_send   { "text": str }              // type text into focused field then press send/enter
  read_screen     {}                            // read and return all visible text on screen
  device_info     {}                            // return model, Android version, resolution, IP
  dark_mode_on    {}
  dark_mode_off   {}
  hotspot_on      {}                            // enable WiFi hotspot / tethering
  hotspot_off     {}
  expand_notifications  {}                      // pull down notification shade
  collapse_notifications {}
  expand_quick_settings {}                      // pull down quick settings panel
  hide_keyboard   {}                            // dismiss soft keyboard
  set_volume_level { "stream": str or null, "n": str }  // stream: music/ring/alarm/notification/call; n: 0-15
  set_font_size   { "size": str }              // small / normal / large / larger / largest
  wifi_info       {}                            // return WiFi SSID and device IP
  ping            {}                            // test internet connection from device
  list_running_apps {}                          // list currently running app packages
  app_version     { "app": str }               // get installed version of an app

Rules:
- For whatsapp_send/sms_send/call/dial: if the target looks like a phone
  number (mostly digits, 7+ digits), put it in "number" and set "contact"
  to null. Otherwise put the name in "contact" (lowercase) and set
  "number" to null.
- "message" must be the exact words the user wants sent, nothing added.
- CRITICAL — entering/submitting text: when the user says "open X and type Y
  and press enter", "open X and type Y and send", "open X and type Y and
  send it", "open X and search Y" — ALWAYS add a final step
  { "action": "send" } to open_app_and_do, OR use open_app_and_search which
  auto-submits. Never leave the user to press Enter manually unless they
  explicitly said not to send.
- Prefer the specific named intents (whatsapp_send, open_app, google_search,
  etc.) whenever the request matches one of them — they're more reliable.
  Only use open_app_and_do for requests that genuinely need a custom
  sequence of taps/types/waits inside an app that no specific intent covers.
- CRITICAL — tapping buttons: You do NOT know exact on-screen pixel coordinates.
  NEVER invent tap x/y coordinates unless the user explicitly provided them.
  When the user says "tap Send", "press the follow button", "tap on Settings",
  "click search" — use { "action": "tap_text", "text": "Send" } which finds
  the real element on screen. Only use { "action": "key", "key": "enter" }
  as a last resort when no visible label is available.
- If the sentence requests something with NO matching intent above
  (e.g. "read my messages", "hack into", "unlock without pin/password",
  anything unsafe or unsupported), respond with:
      {"intent": null, "reason": "<short reason>"}
- Output ONLY the JSON object. No markdown, no code fences, no commentary.
"""

SYSTEM_PROMPT = f"""You are a strict intent-classifier for an Android phone
automation tool. Given one user sentence, output a single JSON object with
exactly two top-level keys: "intent" (one of the allowed names below, or
null) and "params" (an object with that intent's required keys).

{INTENT_SCHEMA}

Respond with raw JSON only. Do not explain. Do not use markdown formatting.
"""


class LLMParseError(ParseError):
    """Raised when the LLM can't map the sentence to a supported action,
    or when the API call itself fails."""


def _call_groq(user_text: str, timeout: int = 20) -> str:
    if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_GROQ_API_KEY_HERE":
        raise LLMParseError(
            "No Groq API key configured for intent parsing. Get a free key "
            "at https://console.groq.com/keys and put it in the .env file "
            "next to main.py as:\n"
            "  GROQ_API_KEY_INTENT=your-key-here\n"
            "(or GROQ_API_KEY as a fallback name)."
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    request = urllib.request.Request(
        GROQ_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            # Groq's Cloudflare protection blocks Python's default
            # "Python-urllib/x.x" User-Agent (returns 403 / Cloudflare
            # error 1010). Sending a normal browser-like UA avoids that.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36",
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
            raise LLMParseError(
                "Groq API key was rejected (401 Unauthorized). "
                "Double-check the key in llm_parser.py / GROQ_API_KEY."
            ) from e
        if e.code == 403 and ("1010" in error_body or "cloudflare" in error_body.lower()):
            raise LLMParseError(
                "Groq blocked the request at the network level (Cloudflare "
                "error 1010) rather than rejecting your API key. This is "
                "usually caused by antivirus/VPN/proxy software intercepting "
                "HTTPS traffic, or a corporate network. Try: (1) temporarily "
                "disabling VPN/proxy, (2) checking antivirus 'HTTPS "
                "scanning' or 'web shield' settings, (3) a different network. "
                "Your API key itself is fine."
            ) from e
        raise LLMParseError(f"Groq API error {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise LLMParseError(
            f"Could not reach Groq API — check your internet connection. ({e})"
        ) from e

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMParseError(f"Unexpected Groq response shape: {body}") from e


def _extract_json(raw: str) -> dict:
    """Groq with response_format=json_object should return clean JSON, but
    strip markdown fences defensively in case the model adds them anyway."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMParseError(f"Model did not return valid JSON: {raw!r}") from e


_ALLOWED_INTENTS = {
    "whatsapp_send", "sms_send", "call", "dial", "screenshot", "screen_record",
    "camera_photo", "open_camera", "open_video_camera", "flashlight_on",
    "flashlight_off", "unlock", "lock", "wake", "sleep", "google_search",
    "open_url", "open_app_and_type", "open_app_and_search", "open_app",
    "close_app", "home", "back", "recents", "volume_up", "volume_down",
    "mute", "wifi_on", "wifi_off", "bluetooth_on", "bluetooth_off",
    "airplane_on", "airplane_off", "play_pause", "next_track", "prev_track",
    "battery", "current_app", "vibrate", "reboot",
    # raw input control
    "tap", "swipe", "long_press", "type_text_raw", "press_key",
    "set_brightness", "share_text", "get_notifications",
    "clear_notifications", "set_clipboard", "open_app_and_do",
    # clipboard / text editing
    "get_clipboard", "paste", "select_all_text", "clear_text_field",
    # DND / ringer
    "dnd_on", "dnd_off", "ringer_silent", "ringer_vibrate", "ringer_normal",
    # rotation / display
    "auto_rotate_on", "auto_rotate_off", "set_rotation", "screen_timeout",
    # app management
    "clear_app_data", "uninstall_app", "open_app_settings", "close_all_apps",
    "split_screen",
    # scrolling
    "scroll_up", "scroll_down", "scroll_left", "scroll_right",
    # visible-text tap
    "tap_text",
    # alarms/timers
    "set_alarm", "set_timer",
    # settings pages
    "open_settings_page",
    # messaging
    "telegram_send",
    # NEW: enter/send/submit
    "press_enter", "smart_send", "type_and_send",
    # NEW: screen reading
    "read_screen",
    # NEW: device info
    "device_info",
    # NEW: dark mode
    "dark_mode_on", "dark_mode_off",
    # NEW: hotspot
    "hotspot_on", "hotspot_off",
    # NEW: notifications shade
    "expand_notifications", "collapse_notifications", "expand_quick_settings",
    # NEW: keyboard
    "hide_keyboard",
    # NEW: volume absolute
    "set_volume_level",
    # NEW: font size
    "set_font_size",
    # NEW: wifi info / ping
    "wifi_info", "ping",
    # NEW: running apps / app version
    "list_running_apps", "app_version",
}


def parse_with_llm(text: str, context_block: str = "") -> Intent:
    """
    Send `text` to Llama 3.1 8B on Groq and return a validated Intent.
    Raises LLMParseError if the model can't/won't map it to a supported
    action, or if the API call fails.

    context_block — optional conversation history/state string from
    ConversationMemory.context_block_for_llm(), prepended to the user
    message so the LLM can resolve pronouns and continue across turns.
    """
    prompt = f"{context_block}{text}" if context_block else text
    raw = _call_groq(prompt)
    data = _extract_json(raw)

    intent_name = data.get("intent")
    if intent_name is None:
        reason = data.get("reason", "not a supported phone action")
        raise LLMParseError(f"Couldn't map that to a supported action: {reason}")

    if intent_name not in _ALLOWED_INTENTS:
        raise LLMParseError(
            f"Model returned an unrecognized intent '{intent_name}' — refusing "
            f"to execute it for safety."
        )

    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise LLMParseError("Model returned malformed params — refusing to execute.")

    return Intent(name=intent_name, params=params, raw_text=text)


def parse_command_smart(text: str, context_block: str = "") -> Intent:
    """
    Best-of-both parser: try the fast offline regex parser first; only
    fall back to the Groq/Llama LLM if the regex parser can't understand
    the sentence. This keeps common commands instant and free, while
    handling messy/conversational phrasing through the LLM.

    context_block is forwarded to the LLM when it is needed.
    """
    from command_parser import parse_command  # local import avoids cycle at module load

    try:
        return parse_command(text)
    except ParseError:
        pass  # fall through to the LLM
    return parse_with_llm(text, context_block=context_block)
