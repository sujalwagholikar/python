"""
main.py
=======
Entry point. Run this file to get an interactive command prompt for
controlling your Android phone in plain English.

USAGE
-----
    python main.py                  # USB first, fallback to WiFi if configured
    python main.py --wifi-setup     # one-time: switch a USB-plugged phone to WiFi ADB
    python main.py --wifi 192.168.1.42   # connect directly over WiFi
    python main.py --usb            # force USB only

Then at the prompt, type things like:
    > open whatsapp and send hi how are you to mom
    > take screenshot
    > open camera and click photo
    > google search best biryani near me
    > turn on flashlight
    > call mom
    > open youtube
    > unlock my phone
    > status
    > help
    > quit
"""

from __future__ import annotations

import argparse
import os
import sys

from adb_controller import ADBBridge, ADBError, AndroidController, DeviceNotConnected
from command_parser import ParseError, execute_intent, parse_multi_command
from context_memory import ConversationMemory
from llm_parser import LLMParseError, parse_with_llm
from task_agent import TaskAgent, VisualTaskAgent, TaskAgentError, StepRecord

try:
    from autonomous_planner import AutonomousPlanner, TaskPlan  # noqa: F401
    _HAS_PLANNER = True
except ImportError:
    _HAS_PLANNER = False

try:
    from ollama_vision import OllamaVision
    _ollama = OllamaVision()
    _HAS_VISION = _ollama.available or bool(os.environ.get("GEMINI_API_KEY", ""))
    _VISION_MODEL_NAME = f"Ollama {_ollama.model} (Local)" if _ollama.available else "Gemini 2.5 Flash-Lite"
except ImportError:
    _HAS_VISION = bool(os.environ.get("GEMINI_API_KEY", ""))
    _VISION_MODEL_NAME = "Gemini 2.5 Flash-Lite"


BANNER = r"""
========================================================
   Android Phone Controller  (ADB + natural language)
========================================================
Type a command in plain English, e.g.:
  open whatsapp and send hello to mom
  take screenshot
  open camera and click photo
  google search cheapest flights to goa
  turn on flashlight
  call dad
  open youtube
  lock / unlock my phone

Type 'help' for the full command guide, 'quit' to exit.
--------------------------------------------------------
"""

def _build_ai_banner() -> str:
    lines = [
        "[AI understanding: ON] Commands that don't match a known phrasing are",
        "sent to Llama 3.1 8B (Groq) to figure out what you meant.",
    ]
    if _HAS_VISION:
        lines.append(
            f"[Vision: ON] Multi-step tasks use {_VISION_MODEL_NAME} for visual"
        )
        lines.append(
            "UI understanding + autonomous planning (sees real screenshots)."
        )
    else:
        lines.append(
            "[Vision: OFF] Make sure Ollama (gemma3:4b) is running locally or set GEMINI_API_KEY in .env."
        )
    lines.append("Run with --no-ai to use only the offline parser.")
    lines.append("--------------------------------------------------------")
    return "\n".join(lines) + "\n"

AI_BANNER_NOTE = _build_ai_banner()

HELP_TEXT = """
COMMAND CHEAT SHEET
====================
Apps & browsing
  open <app>                                   e.g. "open youtube"
  close <app>
  open <app> and search <query>
  open <app> and type <text>
  google search <query>  /  search <query>
  open url <website>  /  go to <website>

Messaging & calls  (add names to contacts.json first)
  open whatsapp and send <message> to <contact/number>
  send sms <message> to <contact/number>
  call <contact/number>
  dial <contact/number>

Camera & screen
  take screenshot
  record [for N seconds]
  open camera and click photo
  open camera
  open video / record video

Device controls
  turn on / turn off flashlight
  lock / unlock (my) phone [with pin 1234]
  wake / sleep
  turn on / off wifi
  turn on / off bluetooth
  turn on / off airplane mode
  increase / decrease volume [by N]
  mute
  turn on / off do not disturb (dnd)
  set phone to silent / vibrate / normal
  turn on / off auto rotate
  rotate 90 / 180 / 270 degrees
  set screen timeout to N seconds

Navigation
  home / back / recent apps
  scroll up / down / left / right [small/medium/large]
  tap <label>                 e.g. "tap send", "press the follow button"
                               (finds the real button on screen right now)

Text & clipboard (acts on whatever field is currently focused)
  select all
  clear the field
  paste
  what's in the clipboard

App management
  clear app data for <app>
  uninstall <app>
  open app settings for <app>
  close all apps
  split screen

Alarms & timers
  set an alarm for 7:30 am [called <label>]
  set a timer for N minutes/seconds [called <label>]

System settings shortcuts
  open <wifi/bluetooth/display/sound/apps/battery/storage/location/
        security/accounts/date_time/language/accessibility/developer/
        notifications/airplane> settings

Messaging & calls  (add names to contacts.json first)
  open whatsapp and send <message> to <contact/number>
  telegram send <message> to <contact/number>
  send sms <message> to <contact/number>
  call <contact/number>
  dial <contact/number>

Camera & screen
  take screenshot
  record [for N seconds]
  open camera and click photo
  open camera
  open video / record video

Media
  play / pause
  next track / previous track

Info
  battery status
  current app
  status              -> shows adb device connection info

Enter / Send
  press enter                          send whatever is typed in focused field
  press send (button)                  tap the visible Send button on screen
  send it / submit / done              smart-send: tap Send button OR press Enter
  type <text> and send                 type text then immediately submit it

Screen / UI
  read the screen                      list all visible text on screen
  hide keyboard                        dismiss soft keyboard
  expand / collapse notifications
  expand quick settings (shade)

Device
  device info                          model, Android version, resolution, WiFi IP
  wifi info / ssid                     show connected WiFi name and IP address
  check internet / ping                test device internet connection
  turn on / off dark mode
  turn on / off hotspot
  set font size to large               small / normal / large / larger / largest
  set volume to N                      0–15 for music; prefix with stream name
  screen resolution / screen size

App info
  version of <app>                     show installed version
  list running apps                    show backgrounded app packages

Other
  vibrate
  reboot phone

Chain multiple commands with "then", e.g.:
  open camera and click photo then take screenshot

AI understanding
  Anything not matching the phrasings above is sent to Llama 3.1 8B
  (via Groq) to figure out what you meant, e.g.:
    "hey can you message rohan on whatsapp and tell him im running late"
    "pull up youtube for me"
    "open instagram, wait 2 seconds, tap the search icon, then type cats"
  The AI also has access to every command above plus raw tap/swipe/type/
  key control and "open X and do this and that" multi-step chains for
  requests that don't match a specific named command.
  Disable with --no-ai to force offline-only parsing.
"""


def build_controller(args: argparse.Namespace) -> AndroidController:
    preferred = "usb" if args.usb else ("wifi" if args.wifi else "auto")
    bridge = ADBBridge(preferred=preferred)
    controller = AndroidController(bridge)

    if args.wifi_setup:
        print("[setup] Make sure your phone is currently connected via USB...")
        controller.connect_over_wifi()  # auto-detects IP, switches to tcpip
        return controller

    if args.wifi:
        controller.bridge.connect_wifi(args.wifi)
        info = controller.bridge.device
        print(f"[connected] {info.model} via WiFi ({info.serial})")
        return controller

    controller.connect()
    return controller


def run_single_command(
    controller: AndroidController,
    raw: str,
    use_ai: bool,
    memory: ConversationMemory | None = None,
) -> None:
    """
    Parse + execute one command line, with optional conversation memory.

    Flow:
      1. Memory resolves pronouns ("him" → "sujal", "it" → "whatsapp")
      2. Regex multi-command parser tries to match (handles "then" chains)
      3. Memory enriches any missing params from context
      4. Execute each intent and record the result in memory
      5. If regex fails entirely and AI is on, hand to LLM with full context
    """
    if memory is None:
        memory = ConversationMemory()

    # --- Step 1: pronoun / reference resolution ---
    resolved = memory.resolve_references(raw)
    if resolved != raw:
        print(f"  [memory] resolved: {resolved!r}")

    # --- Step 2: regex parser ---
    try:
        intents = parse_multi_command(resolved)
        for intent in intents:
            # Step 3: fill missing params from context
            intent = memory.enrich_intent(intent)
            result = _execute_intent_tracked(controller, intent, raw)
            print(f"  -> {result}")
            # Step 4: record in memory
            memory.update(raw, intent, result)
        return
    except ParseError:
        if not use_ai:
            print(
                f"[!] Could not understand: {raw!r}\n"
                f"    Type 'help' for supported phrasings, or run with AI "
                f"understanding enabled (see --no-ai flag)."
            )
            return

    # --- Step 5: LLM fallback with full context block ---
    try:
        print("  [thinking with AI...]")
        context_block = memory.context_block_for_llm()
        intent = parse_with_llm(resolved, context_block=context_block)
        intent = memory.enrich_intent(intent)
        result = _execute_intent_tracked(controller, intent, raw)
        print(f"  -> (AI: {intent.name}) {result}")
        memory.update(raw, intent, result)
    except LLMParseError as e:
        print(f"[!] {e}")
    except ValueError as e:
        print(f"[!] {e}")


def _execute_intent_tracked(controller: AndroidController, intent, raw_text: str) -> str:
    """
    Execute one Intent, routing multi-step/ambiguous goals (open_app_and_do)
    through VisualTaskAgent (Gemini vision + autonomous planning) when
    GEMINI_API_KEY is available, otherwise falling back to the legacy
    text-only TaskAgent (Groq).

    VisualTaskAgent:
      - Decomposes complex goals into verifiable sub-tasks (AutonomousPlanner)
      - Takes real phone screenshots and sends them to Gemini 2.5 Flash-Lite
      - Visually confirms each step and the final goal (not just text matching)
      - Can tap icon-only buttons using visual description

    Behavior is identical to the GUI's _run_task_agent(), just with
    plain print() progress lines instead of transcript widgets.
    """
    if intent.name != "open_app_and_do":
        return execute_intent(controller, intent)

    goal_text = intent.raw_text or raw_text

    # Determine which agent to use
    use_visual = _HAS_VISION
    agent_label = "VisualAgent" if use_visual else "Agent"
    print(f"  [{agent_label}] planning goal: {goal_text!r}")

    def _on_step(rec: StepRecord) -> None:
        icon = {"complete": "✓", "in_progress": "…", "stuck": "⚠"}.get(
            rec.verify_state, "…"
        )
        note = f" — {rec.verify_note}" if rec.verify_note else ""
        vis = " 👁" if rec.screenshot_b64 else ""
        print(f"    [{icon}] step {rec.step_num}: "
              f"{rec.action.get('action')}{vis}{note}")

    def _on_plan(plan) -> None:
        print(f"  [plan] {plan.estimated_complexity} complexity "
              f"— {len(plan.subtasks)} sub-task(s):")
        for i, st in enumerate(plan.subtasks, 1):
            print(f"     {i}. {st.description}")

    try:
        if use_visual:
            agent = VisualTaskAgent(controller, on_step=_on_step,
                                    on_plan=_on_plan)
        else:
            agent = TaskAgent(controller, on_step=_on_step)
        run = agent.run(goal_text)
    except TaskAgentError as e:
        return f"Task automation couldn't start: {e}"

    if run.success:
        return f"Task completed and verified: {run.final_summary}"
    steps_taken = len(run.steps)
    return (
        f"Task NOT completed after {steps_taken} step(s) — "
        f"{run.aborted_reason or 'goal was not confirmed done'}."
    )


def run_repl(controller: AndroidController, use_ai: bool = True) -> None:
    print(BANNER)
    if use_ai:
        print(AI_BANNER_NOTE)

    memory = ConversationMemory()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if raw.lower() in ("help", "?"):
            print(HELP_TEXT)
            continue
        if raw.lower() == "status":
            print(controller.status())
            continue

        # Memory management commands
        if raw.lower() in ("memory", "history", "context"):
            print("\n[Conversation Memory]")
            print(f"  last app     : {memory.last_app or '(none)'}")
            print(f"  last contact : {memory.last_contact or '(none)'}")
            print(f"  last typed   : {memory.last_query or '(none)'}")
            print(f"  last message : {memory.last_message or '(none)'}")
            print(f"  last intent  : {memory.last_intent or '(none)'}")
            print("\n[History]")
            print(memory.formatted_history())
            continue
        if raw.lower() in ("forget", "clear memory", "reset memory", "clear context"):
            memory.clear()
            print("  [memory cleared]")
            continue

        try:
            run_single_command(controller, raw, use_ai, memory=memory)
        except ADBError as e:
            print(f"[adb error] {e}")
        except Exception as e:  # noqa: BLE001 - top level CLI safety net
            print(f"[unexpected error] {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural-language Android phone controller")
    parser.add_argument("--usb", action="store_true", help="Force USB-only connection")
    parser.add_argument("--wifi", metavar="IP", nargs="?", const="", default=None,
                         help="Connect over WiFi. Pass an IP, or omit to auto-pick a WiFi device.")
    parser.add_argument("--wifi-setup", action="store_true",
                         help="One-time: switch a USB-connected phone into WiFi ADB mode")
    parser.add_argument("-c", "--command", metavar="CMD", default=None,
                         help="Run a single command non-interactively and exit")
    parser.add_argument("--no-ai", action="store_true",
                         help="Disable Groq/Llama AI fallback; only use the offline regex parser")
    args = parser.parse_args()
    use_ai = not args.no_ai

    try:
        controller = build_controller(args)
    except DeviceNotConnected as e:
        print(f"[!] {e}")
        sys.exit(1)
    except ADBError as e:
        print(f"[!] {e}")
        sys.exit(1)

    if args.command:
        try:
            run_single_command(controller, args.command, use_ai, memory=ConversationMemory())
        except ADBError as e:
            print(f"[!] {e}")
            sys.exit(1)
        return

    run_repl(controller, use_ai=use_ai)


if __name__ == "__main__":
    main()
