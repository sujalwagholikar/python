# J.A.R.V.I.S. — Voice-Controlled Android Assistant (Python + ADB + Groq)

A real-time, voice-driven, Iron-Man-style assistant that controls your
Android phone over **USB or WiFi** (no root required) and holds normal
conversation — all from one graphical window, **no terminal interaction
required** once it's running.

```
🎙  "Jarvis, open whatsapp and tell mom I'm running late"
🔊  "On it, sir — message sent to mom."

🎙  "What's a good comeback for a bad pun?"
🔊  "A groan, sir. Delivered with feeling."
```

## What's new in this version

| Feature | Details |
|---|---|
| 🖥️ **Full GUI, zero terminal** | `jarvis_launcher.pyw` opens a dark, arc-reactor-themed window. Double-click it — no console ever appears (Windows `pythonw.exe`). |
| 🎙️ **Real voice input** | `SpeechRecognition` listens through your microphone; click the mic once, or enable always-on wake-word listening ("jarvis, …"). |
| 🔊 **Real voice output** | `gTTS` + `pygame` speak every reply back to you, in character. |
| 🧠 **Two-model architecture** | **`llama-3.3-70b-versatile`** is the JARVIS "brain" — conversation, reasoning, and narrating results in character. The original **`llama-3.1-8b-instant`** stays exactly where it was — fast, cheap structured-JSON command parsing — completely unchanged in role. |
| 💬 **Real conversation, not just commands** | Ask JARVIS a question, chat, or just say hi — it routes automatically between "have a conversation" and "control the phone" per utterance. |
| 🔑 **`.env`-based API keys** | No more hardcoded keys in source. Copy `.env.example` → `.env` and paste your own two Groq keys (one per model — see below). |
| 🧵 **Fully threaded** | Mic listening, TTS playback, ADB calls, and Groq calls all run off the UI thread — the window never freezes. |
| 📡 **Live system telemetry HUD** | Top-left strip shows real-time CPU %, GPU % (NVIDIA via `nvidia-smi`, "N/A" gracefully elsewhere), RAM %, and network throughput in KB/s or MB/s — sampled every second via `psutil`, off the UI thread. |
| 🌀 **Dynamic multi-ring reactor HUD** | The animated core now has tick-mark rims, dual counter-rotating segmented rings, a satellite gauge trio (CPU/GPU/RAM as mini dials), a live network sparkline, and corner brackets. Color, rotation speed, and pattern visibly change per state — idle, listening, thinking, **searching the web**, executing a phone command, speaking, or error. |
| 🌐 **Real-time web search** | Ask JARVIS to "look up…" or "what's happening with…" and it queries DuckDuckGo + Google News RSS (both keyless) plus Tavily (optional, if `TAVILY_API_KEY` is set) in parallel, merges/deduplicates the results, and has the 70B brain answer grounded only in what was actually retrieved. |

### How routing works

```
你说话/type
    │
    ▼
jarvis_brain.classify_utterance()   (instant local heuristics, LLM tiebreak only if ambiguous)
    │
    ├── "chat"   ──────────► jarvis_brain.jarvis_reply()        [llama-3.3-70b-versatile]
    │                              → spoken back immediately
    │
    ├── "search" ──────────► web_search.web_search()
    │                              (DuckDuckGo + Google News RSS + optional Tavily)
    │                              → jarvis_brain.jarvis_reply() paraphrases the
    │                                results into a short grounded spoken answer
    │
    └── "command" ─► context_memory (resolve "him"/"it"/"that app")
                         │
                         ▼
                    command_parser (offline regex, instant)
                         │  (no match)
                         ▼
                    llm_parser.parse_with_llm()                 [llama-3.1-8b-instant]
                         │
                         ▼
                    adb_controller executes on the phone
                         │
                         ▼
                    jarvis_brain.narrate_result()                [llama-3.3-70b-versatile]
                         → spoken back in character
```

The 8B model's job never changed — it still only ever emits a strict,
validated JSON intent from a fixed allow-list, same as before. The 70B
model is purely additive: it decides *whether* something is a command at
all, and it's the "voice" for everything else.

## Quick start

1. **Install Python 3.10+** and the **Android platform-tools** (`adb`) —
   see the original setup steps further down this README (unchanged).
2. **Install the new dependencies:**
   ```
   pip install -r requirements.txt
   ```
   (`tkinter` ships with the standard Windows/macOS Python installer; on
   minimal Linux distros install your package manager's `python3-tk`.)
3. **Set up your API keys:**
   ```
   copy .env.example .env      (Windows)
   cp .env.example .env        (macOS/Linux)
   ```
   Open `.env` and paste in your own Groq keys — get free ones at
   https://console.groq.com/keys:
   ```
   GROQ_API_KEY_JARVIS=gsk_...your_key_for_llama-3.3-70b-versatile...
   GROQ_API_KEY_INTENT=gsk_...your_key_for_llama-3.1-8b-instant...
   ```
   You can reuse the same key for both if you only have one — two
   separate keys just let you track usage per model on Groq's dashboard.

   Optionally, also paste a Tavily key (free at https://app.tavily.com)
   for a third, higher-quality search source:
   ```
   TAVILY_API_KEY=tvly-...your_key...
   ```
   This is fully optional — JARVIS still searches the web fine with just
   DuckDuckGo + Google News RSS (both work with no key at all) if you
   skip this.
4. **Add your contacts** to `contacts.json` (unchanged from before — see
   below).
5. **Launch JARVIS:**
   - **Windows:** double-click `jarvis_launcher.pyw` — no console window
     will appear at all.
   - **macOS/Linux:** `python3 jarvis_gui.py`
6. Plug your phone in via USB (or set up WiFi ADB — see below), click
   **🎙 LISTEN**, and talk.

The old terminal-only interface still exists unchanged at `main.py` if
you ever want it (`python main.py`) — the GUI is simply the new default,
recommended way to use the project.

## Voice tips

- **Single command:** click **🎙 LISTEN**, speak, wait for the reply.
- **Hands-free mode:** tick **"Always listening"** and say the wake word
  ("jarvis" by default — change it via `JARVIS_WAKE_WORD` in `.env`)
  followed by your request, e.g. *"Jarvis, take a screenshot."*
- **Voice replies off:** untick **"Voice replies"** to keep everything
  text-only in the transcript (useful in quiet environments).
- Both speech-to-text and text-to-speech need an internet connection
  (they use Google's free web APIs) — same as the Groq calls already did.

## Live telemetry HUD & web search

- The top-left strip in the window shows live **CPU %**, **GPU %**,
  **RAM %**, and **network throughput** (KB/s or MB/s), refreshed every
  second. GPU % needs an NVIDIA card + driver (uses `nvidia-smi`); on
  anything else it just shows `N/A` rather than a fake number.
- The reactor ring's color, spin speed, and pattern change depending on
  what JARVIS is doing right now — cyan/slow when idle, gold when
  listening, cyan/fast when thinking or executing a phone command,
  violet when searching the web, glowing cyan when speaking, red when
  something went wrong.
- To trigger a **live internet search** (as opposed to opening a search
  in the phone's browser), phrase it as a lookup: *"look up the latest
  news on X"*, *"what's happening with Y"*, *"find me information about
  Z"*. JARVIS queries DuckDuckGo + Google News RSS (and Tavily, if
  configured), then answers grounded only in what was actually found —
  it won't make something up if the search comes back empty.

---

# Android Phone Controller (Python + ADB)

Control your Android phone from Python using plain-English commands, over
**USB or WiFi**. No root required.

```
> open whatsapp and send hi how are you to mom
> take screenshot
> open camera and click photo
> google search best pizza near me
> turn on flashlight
> call dad
> open youtube
```

## How it works

This is built on **ADB (Android Debug Bridge)** — the same official tool
Android Studio uses. There is no other legitimate, reliable way to control
a real Android phone from a desktop Python script without root; anything
claiming otherwise (fake "hacking" scripts) either doesn't work or requires
root/an accessibility-service app installed on the phone. ADB is the real
answer and it's fully supported by Google.

Five files:
- **`adb_controller.py`** — low-level engine. Every phone action (tap, type,
  open app, screenshot, flashlight, WiFi/Bluetooth toggle, send WhatsApp
  message, etc.) is a method on `AndroidController`.
- **`command_parser.py`** — turns an English sentence into one of those
  method calls using a fast, deterministic, fully-offline regex parser.
  Tried first for every command, since it's instant and free.
- **`llm_parser.py`** — AI fallback. When the regex parser can't match a
  sentence, this sends it to **Llama 3.1 8B via Groq** (very fast, free
  tier available) and asks it to return a strict JSON intent from a fixed
  allow-list — so it can only ever trigger actions this project actually
  implements, never something invented. Handles messy/conversational
  phrasing the regex patterns don't cover.
- **`contacts.py`** — resolves names ("mom") to phone numbers using a local
  `contacts.json` file you fill in.
- **`main.py`** — the interactive command-line program you actually run.

## 1. Install prerequisites (one time)

1. Install **Python 3.9+**.
2. Install **Android Platform Tools** (contains `adb`):
   - Download: https://developer.android.com/tools/releases/platform-tools
   - Unzip it, add the folder to your **PATH** (Windows: System Properties >
     Environment Variables > Path > add the `platform-tools` folder).
   - Verify: open a new terminal and run `adb version` — it should print a
     version number, not an error.
3. On your **phone**: Settings > About phone > tap "Build number" 7 times
   to unlock Developer Options. Then Settings > Developer options > enable
   **USB debugging**.

No extra Python packages are required — everything uses the standard
library (`subprocess`, `socket`, `re`, etc).

## 2. First connection (USB)

1. Plug the phone into your PC with a USB cable.
2. On the phone, a popup appears: **"Allow USB debugging?"** — tap **Allow**,
   and check "Always allow from this computer" so you don't get asked again.
3. Run:
   ```
   cd phone_control
   python main.py
   ```
   You should see `[connected] <your phone model> ... via USB`.

If you get "No authorized device found", run `adb devices` directly in a
terminal to see the raw status (`unauthorized` means you need to tap Allow
on the phone; empty list means the cable/drivers aren't working).

## 3. Switch to WiFi (optional, no cable needed after this)

With the phone still plugged in via USB and on the same WiFi network as
your PC:

```
python main.py --wifi-setup
```

This prints the phone's WiFi IP address and switches it into wireless ADB
mode. You can now **unplug the USB cable**. From then on, connect with:

```
python main.py --wifi 192.168.1.42
```

(Replace with whatever IP was printed. This wireless pairing resets if the
phone reboots or leaves the WiFi network — just redo `--wifi-setup` once
more while plugged in.)

Running `python main.py` with no flags tries USB first automatically and
uses whichever device adb finds.

## 4. AI understanding — Groq / Llama models (set your own keys in `.env`)

The offline regex parser only catches phrasings it has a pattern for.
For anything messier — "hey can you tell mom on whatsapp im running late",
"pull up youtube for me" — this project sends the sentence to
**`llama-3.1-8b-instant` on Groq** (extremely fast, free tier) and asks
it to return a structured intent from a fixed allow-list, so it can
never trigger something this project doesn't actually implement.

**Set your own key** — no key is baked into the source code (an earlier
version of this project did that; if you're reusing a copy with a key
hardcoded in it, treat that key as compromised and revoke it at
https://console.groq.com/keys immediately). Instead:

1. Copy `.env.example` to `.env` in this folder.
2. Get a free key at https://console.groq.com/keys.
3. Paste it in as `GROQ_API_KEY_INTENT=your-key-here`.

`main.py` (terminal mode) and `jarvis_gui.py` (GUI mode) both load `.env`
automatically. AI understanding is on by default in the terminal mode;
run with `--no-ai` to use only the offline parser. No extra `pip
install` needed for this part — uses Python's built-in `urllib`.

The JARVIS GUI additionally uses a second, more capable model —
**`llama-3.3-70b-versatile`** — as its conversational "brain" for
everything that isn't a direct device command, and to narrate command
results back to you in character. Set that key as
`GROQ_API_KEY_JARVIS=your-key-here` in the same `.env` file. See the
"What's new in this version" section at the top of this README for
details.

### What the AI can do now (expanded intent set)

Beyond the named actions (WhatsApp, calls, camera, etc.), the AI layer
has access to a much larger command surface — around 80 distinct
intents in total, all still routed through the same fixed allow-list so
the model can never trigger anything this project doesn't implement:

- **Raw input**: `tap x,y`, `swipe`, `long press`, typing into whatever
  field is focused, and named key presses (back/home/enter/etc.)
- **Find & tap by label** (`tap_text`): "tap send", "press the follow
  button", "tap on settings" — this reads the actual current screen
  (via Android's official `uiautomator dump`) and taps whatever element
  matches that text/description right now, instead of guessing a fixed
  pixel position. This is the preferred way to describe UI interactions
  and is far more reliable across different phones/screen sizes than
  coordinates.
- **Text & clipboard**: select all, clear the current field, paste,
  read back what's currently on the clipboard
- **Do Not Disturb / ringer**: "turn on do not disturb", "set phone to
  silent/vibrate/normal"
- **Rotation & display**: auto-rotate on/off, "rotate 90 degrees", set
  screen timeout
- **App management**: clear an app's data, uninstall an app, open an
  app's settings page, force-stop all backgrounded apps, split screen
- **Scrolling**: up/down/left/right, with small/medium/large distances
- **Alarms & timers**: "set an alarm for 7:30am called wake up", "set a
  timer for 10 minutes" — uses Android's official alarm/timer intents
- **System settings shortcuts**: jump straight to WiFi, Bluetooth,
  display, sound, storage, security, accessibility, developer options,
  and more, by name
- **More messaging platforms**: Telegram sends (via its official `t.me`
  deep link), alongside the existing WhatsApp/SMS support
- **Brightness**: "set brightness to 200"
- **Share sheet**: "share this text: ..."
- **Notifications**: "show my notifications", "clear notifications"
- **Generic multi-step chains** (`open_app_and_do`): for requests like
  *"open Instagram, wait 2 seconds, go back, then type hello world"* —
  the model breaks it into an ordered sequence of wait/tap/swipe/type/key
  steps and Python runs them one after another. This is the general
  fallback for "open X and do this and that" phrasing that doesn't match
  a specific named intent.

Note: the AI is intentionally told **not to invent exact pixel
coordinates** it can't actually know for your phone's screen layout.
For a step that needs to hit a specific on-screen element, it's told to
prefer `tap_text` with the element's visible label (looked up live on
the real screen) over guessing coordinates; only true coordinates you
give it yourself, or a named key/back/home step, are used otherwise. It
also refuses anything unsafe or unsupported (bypassing a lock without a
PIN, reading private message content, etc.) rather than guessing.

## 5. Add your contacts

Edit the auto-generated `contacts.json` in this folder:

```json
{
  "mom": "919876543210",
  "dad": "919812345678",
  "john": "14155552671"
}
```

Numbers: country code + number, digits only, no `+`/spaces/dashes.
Now `"call mom"` or `"send hi to mom"` will work. You can also just use a
raw phone number directly in any command instead of a saved contact.

## 6. Usage

Interactive mode:
```
python main.py
> open whatsapp and send hello to mom
> take screenshot
> help          (full command list)
> quit
```

One-off command (useful for scripting / scheduled tasks):
```
python main.py -c "turn on flashlight"
```

Chain multiple actions with "then":
```
> open camera and click photo then take screenshot
```

Type `help` inside the program for the full list of supported phrasings —
covers apps, WhatsApp/Telegram/SMS/calls, camera, screenshots/recording,
flashlight, lock/unlock, WiFi/Bluetooth/airplane mode, Do Not Disturb and
ringer mode, screen rotation, volume, navigation (home/back/scroll),
tapping on-screen buttons by name, text-field/clipboard actions, app
management (clear data/uninstall/settings/close-all/split-screen),
alarms and timers, direct links to system settings pages, media
playback, battery status, vibrate, and reboot.

Anything that doesn't match those phrasings is automatically sent to the
AI fallback (Groq/Llama 3.1 8B) if you've set up a key — try natural
phrasing like `"tell mom on whatsapp im running late"`,
`"pull up youtube for me"`, or `"open settings and turn on developer
options"`.

## Adding more apps

`APP_PACKAGES` near the top of `adb_controller.py` maps friendly names to
Android package IDs. Add any app you use:

```python
APP_PACKAGES["snapchat"] = "com.snapchat.android"
```

Find a package name for an installed app with:
```
python -c "from adb_controller import AndroidController; c=AndroidController(); c.connect(); print([p for p in c.list_installed_packages() if 'snap' in p.lower()])"
```

## Honest limitations (things ADB genuinely cannot do without root)

- **Cannot bypass a PIN/pattern/biometric lock.** `unlock` only works if
  there's no lock, or if you supply the PIN you already know
  (`unlock my phone with pin 1234`) — this *types* the PIN, it doesn't crack
  anything. Pattern and fingerprint/face locks can't be automated this way.
- **Cannot read message content back** from WhatsApp/SMS/etc — sending is
  supported (via official deep links / SEND intents), reading someone's
  replies back into Python isn't wired up here (would need root or a
  companion Accessibility-Service Android app).
- **WhatsApp auto-send tap coordinates** (`WHATSAPP_SEND_BUTTON_COORDS` in
  `adb_controller.py`) assume a common screen layout. If the send button is
  in a different spot on your specific phone, take a screenshot
  (`take screenshot`) after opening a chat once, note the pixel coordinates
  of the send button, and update that constant.
- **Clipboard set/get** requires Android 10+; **paste/select-all via
  key-combination** requires Android 12+ (older versions can still be
  cleared via `clear the field`, and text can always be typed directly
  instead of pasted).
- **Do Not Disturb** (`cmd notification set_dnd`) needs Notification
  Policy Access granted to the shell. Most devices grant this
  automatically for adb-shell callers; if `dnd on`/`dnd off` silently
  has no effect, grant it once manually: Settings > Apps > Special app
  access > Do Not Disturb access > (allow for the relevant caller), or
  just toggle DND from the notification shade / Settings > Sound
  directly.
- **`tap <label>`** only works on elements that are actually visible on
  the current screen right now (it reads a live `uiautomator dump`) —
  it can't tap something that requires scrolling into view first or
  that's inside a WebView/game canvas that doesn't expose accessibility
  labels.
- **Split screen** behavior varies significantly by OEM/launcher and
  isn't guaranteed to work identically (or at all) on every device —
  it's a best-effort gesture, not an official ADB API.
- Some actions (installing apps, changing certain system settings) may
  show an on-screen confirmation the first time regardless of ADB, by
  Android design.

## Safety note

This gives Python full input-injection control over your own phone via a
tool (ADB) that only works with your explicit on-device authorization. Only
use it on your own device, and be mindful that anyone with USB/WiFi ADB
access to an authorized PC has the same level of control you do.
