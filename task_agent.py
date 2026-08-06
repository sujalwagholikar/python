"""
task_agent.py
=============
Closed-loop, screen-aware, VISUALLY-GROUNDED task execution for JARVIS.

WHAT'S NEW IN THIS VERSION
---------------------------
Before this upgrade, JARVIS was "blind" — it could only read UIAutomator
XML text from the screen, meaning:
  - Buttons with only icons (no text) were invisible to the agent.
  - The LLM had to guess screen layout from raw accessibility text.
  - Verification was string-matching on text, not visual confirmation.
  - Complex goals went straight to execution with no structured planning.

This version fixes all of that with three integrated systems:

1. VISUAL PERCEPTION (GeminiVision)
   ─────────────────────────────────
   Every perceive step takes a real ADB screenshot AND reads the
   UIAutomator dump, then sends BOTH to Gemini 2.5 Flash-Lite. The model
   can now literally SEE the screen layout, icon states, button positions,
   loading spinners, popups with no text, and anything a human could see.

2. AUTONOMOUS PLANNING (AutonomousPlanner)
   ─────────────────────────────────────────
   Before touching the phone, Gemini reasons about the goal and produces
   a structured TaskPlan: an ordered list of SubTask objects, each with
   a clear description AND a visual success criterion. Complex goals get
   proper decomposition; simple goals go through without overhead.

3. VISUAL VERIFICATION
   ─────────────────────
   After every action, Gemini looks at the real screenshot and decides
   whether the step is complete, still in progress, or stuck — based on
   what it SEES, not what text happens to appear.

PUBLIC API
----------
  from task_agent import VisualTaskAgent, TaskAgent, TaskRun, StepRecord

  # New visual agent (recommended):
  agent = VisualTaskAgent(controller, on_step=callback)
  run = agent.run("send a message to mom on whatsapp")

  # Legacy text-only agent (kept for backward-compatibility):
  agent = TaskAgent(controller, on_step=callback)
  run = agent.run("open settings")

BACKWARD COMPATIBILITY
----------------------
  TaskAgent (text-only, Groq) is preserved unchanged for callers that
  don't want to use Gemini or for simple commands that don't need vision.
  VisualTaskAgent is the new primary agent class.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

from adb_controller import ADBError, AndroidController

# Optional vision & planner imports — Ollama Gemma 3 4B primary, Gemini fallback
try:
    from ollama_vision import OllamaVision, OllamaVisionError, resize_image_b64
    from gemini_vision import GeminiVision, screenshot_bytes_to_b64
    _HAS_VISION = True
except ImportError:
    try:
        from gemini_vision import GeminiVision, GeminiVisionError, screenshot_bytes_to_b64
        OllamaVision = GeminiVision  # type: ignore
        _HAS_VISION = True
    except ImportError:
        _HAS_VISION = False
        OllamaVision = None  # type: ignore
        GeminiVision = None  # type: ignore

try:
    from autonomous_planner import AutonomousPlanner, TaskPlan, SubTask
    _HAS_PLANNER = True
except ImportError:
    _HAS_PLANNER = False
    AutonomousPlanner = None  # type: ignore

# Fast, deterministic, zero-LLM-latency decision/verify layer. Always
# available (stdlib only) — used as the first attempt on every step before
# falling back to vision/LLM calls.
from fast_engine import FastDecider, FastVerifier, looks_atomic

# --------------------------------------------------------------------------- #
# Legacy Groq config (for backward-compatible TaskAgent)
# --------------------------------------------------------------------------- #

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = (
    os.environ.get("GROQ_API_KEY_JARVIS")
    or os.environ.get("GROQ_API_KEY_INTENT")
    or os.environ.get("GROQ_API_KEY", "")
)
GROQ_MODEL = os.environ.get("GROQ_MODEL_JARVIS", "llama-3.3-70b-versatile")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MAX_STEPS = 20           # hard cap so a confused loop can't run forever
MAX_STUCK_REPEATS = 3    # same screenshot hash seen N times → abort as stuck
STEP_SETTLE_SECONDS = 1.2  # pause after each action before re-reading screen
SCREENSHOT_SETTLE_SECONDS = 0.5  # extra wait before screenshot (let animations finish)

# --------------------------------------------------------------------------- #
# Action schema (shared between legacy and visual agents)
# --------------------------------------------------------------------------- #

_ACTION_SCHEMA_DOC = """
Each turn you choose exactly ONE next action as a JSON object:

  {"action": "open_app", "app": "whatsapp"}
  {"action": "tap_text", "text": "Send"}              // tap visible element by text/label
  {"action": "tap_xy", "x": 500, "y": 800}             // use ONLY when find_element returns coords
  {"action": "tap_visual", "description": "the blue send button icon"}  // NEW: tap by visual description
  {"action": "type", "text": "hello there"}            // types into the currently focused field
  {"action": "key", "key": "back"}                     // back | home | enter | recent
  {"action": "swipe", "direction": "up"}               // up | down | left | right
  {"action": "wait", "seconds": 2}                     // let something load/animate
  {"action": "smart_send"}                             // tap Send/Go/Submit or press Enter
  {"action": "screenshot"}                             // capture current screen
  {"action": "done", "success": true, "summary": "..."}   // YOU confirm the goal is complete
  {"action": "give_up", "reason": "..."}                  // YOU decide goal cannot be completed

Rules:
- Prefer "tap_text" for elements with visible text — most reliable.
- Use "tap_visual" for icon-only buttons, image buttons, avatar circles —
  describe what you SEE (color, shape, position) not what you guess it does.
- Only use "tap_xy" when tap_visual returns specific coordinates you trust.
- Call "done" ONLY when the CURRENT SCREEN visually confirms the goal was
  achieved — never assume something worked without seeing it.
- Call "give_up" if you've tried reasonable alternatives and the screen
  shows the goal cannot be completed — explain the visual evidence in "reason".
- Respond with ONLY the JSON object, no other text.
""".strip()

_VISUAL_AGENT_SYSTEM = f"""You are the vision-guided execution planning core of JARVIS,
an Android phone automation assistant. You are given a user's goal, the ACTUAL
current phone screen as a SCREENSHOT (you can see it visually) AND as
accessibility text. You choose ONE next action to move toward completing the goal.

You can literally SEE the screen — use that to make better decisions than any
text-only agent could. Identify icons, button positions, loading states, dialogs,
and visual cues that text can't capture.

{_ACTION_SCHEMA_DOC}

Never invent screen contents. Base your decision on what you actually see in the
screenshot. If the same screen isn't changing after your actions, try a
fundamentally different approach — don't repeat the same failed action."""

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class TaskAgentError(RuntimeError):
    """Raised when the agent can't start (no API key, no device, etc.)."""


# --------------------------------------------------------------------------- #
# Shared data classes
# --------------------------------------------------------------------------- #


@dataclass
class StepRecord:
    step_num: int
    action: dict
    screen_before: str        # text description
    screen_after: str         # text description
    verify_state: str         # "complete" | "in_progress" | "stuck"
    verify_note: str
    screenshot_b64: str = ""  # base64 screenshot captured at this step
    subtask_index: int = 0    # which subtask this step belongs to


@dataclass
class TaskRun:
    goal: str
    steps: list = field(default_factory=list)   # list[StepRecord]
    success: Optional[bool] = None
    final_summary: str = ""
    aborted_reason: str = ""
    plan: Optional[object] = None               # TaskPlan if planner was used

    def log_lines(self) -> list:
        out = []
        for s in self.steps:
            out.append(
                f"step {s.step_num}: {_describe_action(s.action)} "
                f"→ {s.verify_state}"
                + (f" ({s.verify_note})" if s.verify_note else "")
            )
        return out


# --------------------------------------------------------------------------- #
# Action description helper (shared)
# --------------------------------------------------------------------------- #


def _describe_action(action: dict) -> str:
    name = action.get("action", "?")
    if name == "tap_text":
        return f"tap '{action.get('text', '')}'"
    if name == "tap_xy":
        return f"tap ({action.get('x')},{action.get('y')})"
    if name == "tap_visual":
        return f"tap visual '{action.get('description', '')}'"
    if name == "type":
        return f"type {action.get('text', '')!r}"
    if name == "key":
        return f"press {action.get('key', '')}"
    if name == "swipe":
        return f"swipe {action.get('direction', '')}"
    if name == "wait":
        return f"wait {action.get('seconds', 1)}s"
    if name == "open_app":
        return f"open {action.get('app', '')}"
    if name == "done":
        return "✓ done"
    if name == "give_up":
        return "✗ give_up"
    return name


def _screenshot_hash(img_b64: str) -> str:
    """Cheap hash of screenshot bytes for stuck-loop detection."""
    if not img_b64:
        return ""
    # Hash the first 8KB of the decoded bytes (fast, sensitive enough)
    try:
        sample = base64.b64decode(img_b64[:10000])[:8192]
        return hashlib.md5(sample).hexdigest()
    except Exception:
        return img_b64[:100]


def _screen_signature(screen_text: str) -> str:
    """Cheap text signature for stuck-loop detection."""
    return re.sub(r"\s+", " ", screen_text).strip()[:400]


# --------------------------------------------------------------------------- #
#  VisualTaskAgent — the new primary agent with Gemini vision + planning
# --------------------------------------------------------------------------- #


class VisualTaskAgent:
    """
    Vision-guided, autonomously-planned, closed-loop Android automation agent.

    Execution flow:
      1. PLAN  — AutonomousPlanner decomposes the goal into SubTasks.
      2. For each SubTask:
           a. PERCEIVE  — Take screenshot + UIAutomator text dump.
           b. DECIDE    — Gemini Vision sees the real screen and picks ONE action.
           c. ACT       — Execute the action via AndroidController.
           d. VERIFY    — Gemini Vision looks at the new screenshot and
                          confirms success/in-progress/stuck.
           e. REPEAT    — Until sub-task criterion met, step cap, or stuck.
      3. REPLAN — If a sub-task fails, ask Gemini to generate a recovery plan.
      4. Honest final report (success/failure/partial with visual evidence).

    Falls back gracefully to text-only mode if Gemini is unavailable.
    """

    def __init__(
        self,
        controller: AndroidController,
        on_step: Optional[Callable[[StepRecord], None]] = None,
        on_plan: Optional[Callable[[object], None]] = None,  # called with TaskPlan
    ):
        self.controller = controller
        self.on_step = on_step
        self.on_plan = on_plan

        # Initialize Ollama Gemma 3 4B vision engine (falls back to Gemini API if key set)
        if _HAS_VISION and OllamaVision:
            self.vision = OllamaVision()
        elif _HAS_VISION and GeminiVision:
            self.vision = GeminiVision()
        else:
            self.vision = None

        if _HAS_PLANNER:
            self.planner = AutonomousPlanner()
        else:
            self.planner = None

        # Fast, zero-latency deterministic engine — tried before any LLM call.
        self.fast_decider = FastDecider()
        self.fast_verifier = FastVerifier()
        self._last_ui_xml = ""
        self._last_current_app = ""

        # Resolve screen resolution once on connect
        try:
            self._screen_w, self._screen_h = controller.get_screen_resolution()
        except Exception:
            self._screen_w, self._screen_h = 1080, 2400

    # ------------------------------------------------------------------ #
    # Main entry: run a goal to completion
    # ------------------------------------------------------------------ #

    def run(self, goal: str) -> TaskRun:
        """
        Run a complete goal-directed session.
        Returns a TaskRun with full step logs and success/failure state.
        """
        run = TaskRun(goal=goal)

        # Step 1: Autonomous planning
        plan = self._make_plan(goal)
        run.plan = plan

        if self.on_plan and plan is not None:
            self.on_plan(plan)

        # Step 2: Execute each subtask
        global_step = 0
        completed_subtask_descriptions: List[str] = []

        subtasks = plan.subtasks if plan else [
            _make_simple_subtask(goal)
        ]

        for st_idx, subtask in enumerate(subtasks):
            subtask_run, last_screen, last_img = self._run_subtask(
                subtask, run, global_step, st_idx
            )
            global_step += len([s for s in run.steps if s.subtask_index == st_idx])

            if subtask_run == "complete":
                completed_subtask_descriptions.append(subtask.description)
            elif subtask_run == "failed":
                if subtask.optional:
                    # Optional subtask failure: continue with next subtask
                    continue

                # Try adaptive replanning if planner available
                if self.planner and self.planner.available and plan:
                    failure_reason = (
                        run.steps[-1].verify_note
                        if run.steps else "Unknown failure"
                    )
                    recovery = self.planner.replan(
                        original_goal=goal,
                        failed_subtask=subtask,
                        failure_reason=failure_reason,
                        current_screen=last_screen,
                        completed_descriptions=completed_subtask_descriptions,
                    )
                    if recovery.subtasks:
                        # Inject recovery subtasks and continue
                        remaining = subtasks[st_idx + 1:]
                        subtasks = (
                            subtasks[: st_idx + 1]
                            + recovery.subtasks
                            + remaining
                        )
                        continue

                # No recovery: abort
                run.success = False
                run.aborted_reason = (
                    f"Sub-task {st_idx + 1} failed: {subtask.description[:80]}. "
                    + (run.steps[-1].verify_note if run.steps else "")
                )
                return run

            # Global step cap
            if global_step >= (plan.total_max_steps if plan else MAX_STEPS):
                run.success = False
                run.aborted_reason = f"Reached global step cap ({global_step}) without completing all sub-tasks."
                return run

        # All subtasks complete
        run.success = True
        # Summarise from last successful step
        final_note = ""
        for step in reversed(run.steps):
            if step.verify_note:
                final_note = step.verify_note
                break
        run.final_summary = final_note or "All sub-tasks completed and visually verified."
        return run

    # ------------------------------------------------------------------ #
    # Sub-task execution loop
    # ------------------------------------------------------------------ #

    def _run_subtask(
        self,
        subtask: "SubTask",
        run: TaskRun,
        global_step_offset: int,
        subtask_index: int,
    ) -> Tuple[str, str, str]:
        """
        Execute one SubTask.
        Returns ("complete" | "failed" | "stuck", last_screen_text, last_img_b64).
        """
        history_actions: List[dict] = []
        recent_hashes: List[str] = []
        last_screen = ""
        last_img = ""
        already_typed = False

        step_cap = subtask.max_steps if hasattr(subtask, "max_steps") else 8

        for step_num in range(1, step_cap + 1):
            global_step = global_step_offset + step_num

            # PERCEIVE — text/UI-tree only first; screenshot is deferred
            # until we know we actually need vision (fast path resolves
            # most steps without ever taking or encoding a screenshot).
            screen_text, _ = self._perceive(need_screenshot=False)
            last_screen = screen_text
            current_app_before = self._last_current_app

            # DECIDE — try the deterministic fast path first (near-zero
            # latency: pure regex + UI-tree matching, no LLM round trip).
            fast_action = self.fast_decider.decide(
                subtask.description, history_actions, self._last_ui_xml,
                already_typed=already_typed,
            )

            if fast_action is not None:
                action = fast_action
                img_b64 = ""  # not needed — fast path never used vision
            else:
                # Escalate to the LLM/vision agent for just this one step.
                # Now (and only now) do we pay for a screenshot.
                screen_text, img_b64 = self._perceive(need_screenshot=True)
                last_screen = screen_text
                last_img = img_b64
                try:
                    action = self._decide(subtask.description, history_actions,
                                          screen_text, img_b64)
                except TaskAgentError as e:
                    break

            # Terminal actions
            if action.get("action") == "give_up":
                rec = StepRecord(
                    step_num=global_step,
                    action=action,
                    screen_before=screen_text,
                    screen_after=screen_text,
                    verify_state="stuck",
                    verify_note=action.get("reason", "Agent gave up"),
                    screenshot_b64=img_b64,
                    subtask_index=subtask_index,
                )
                run.steps.append(rec)
                if self.on_step:
                    self.on_step(rec)
                return "failed", last_screen, last_img

            if action.get("action") == "done":
                rec = StepRecord(
                    step_num=global_step,
                    action=action,
                    screen_before=screen_text,
                    screen_after=screen_text,
                    verify_state="complete",
                    verify_note=action.get("summary", ""),
                    screenshot_b64=img_b64,
                    subtask_index=subtask_index,
                )
                run.steps.append(rec)
                if self.on_step:
                    self.on_step(rec)
                return "complete", last_screen, last_img

            # ACT
            act_error = None
            try:
                self._act(action, img_b64)
            except (ADBError, ValueError, KeyError, TypeError) as e:
                act_error = str(e)

            if action.get("action") == "type":
                already_typed = True

            time.sleep(STEP_SETTLE_SECONDS)

            # PERCEIVE after action — text/UI-tree only for now; fast
            # verifier works off text, so we still avoid a screenshot here
            # unless fast verification can't decide.
            screen_after, _ = self._perceive(need_screenshot=False)
            current_app_after = self._last_current_app
            last_screen = screen_after

            if act_error:
                rec = StepRecord(
                    step_num=global_step,
                    action=action,
                    screen_before=screen_text,
                    screen_after=screen_after,
                    verify_state="stuck",
                    verify_note=f"Action error: {act_error}",
                    screenshot_b64="",
                    subtask_index=subtask_index,
                )
                run.steps.append(rec)
                history_actions.append(action)
                if self.on_step:
                    self.on_step(rec)
                continue

            # VERIFY — try the deterministic fast verifier first.
            fast_result = self.fast_verifier.verify(
                subtask.description,
                subtask.success_criterion if hasattr(subtask, "success_criterion") else "",
                action,
                current_app_before,
                current_app_after,
                screen_text,
                screen_after,
            )

            img_after = ""
            if fast_result is not None:
                verify_state, verify_note = fast_result
            else:
                # Fast verifier couldn't confidently decide — escalate to
                # vision/LLM verification, now paying for a screenshot.
                screen_after, img_after = self._perceive(need_screenshot=True)
                last_screen = screen_after
                last_img = img_after
                verify_state, verify_note = self._verify(
                    subtask.description, action, screen_after, img_after
                )

            rec = StepRecord(
                step_num=global_step,
                action=action,
                screen_before=screen_text,
                screen_after=screen_after,
                verify_state=verify_state,
                verify_note=verify_note,
                screenshot_b64=img_after,
                subtask_index=subtask_index,
            )
            run.steps.append(rec)
            history_actions.append(action)
            if self.on_step:
                self.on_step(rec)

            if verify_state == "complete":
                return "complete", last_screen, last_img

            # Stuck-loop detection. Prefer the cheap text signature (always
            # available); fold in the screenshot hash too when we happened
            # to take one this step, for stronger duplicate detection.
            text_sig = _screen_signature(screen_after)
            action_sig = json.dumps(action, sort_keys=True)
            sig = (action_sig, text_sig, _screenshot_hash(img_after) if img_after else "")
            recent_hashes.append(sig)
            if recent_hashes.count(sig) >= MAX_STUCK_REPEATS:
                return "failed", last_screen, last_img

        # Step cap hit
        return "failed", last_screen, last_img

    # ------------------------------------------------------------------ #
    # Perceive — take screenshot + read UIAutomator dump
    # ------------------------------------------------------------------ #

    def _perceive(self, need_screenshot: bool = True) -> Tuple[str, str]:
        """
        Returns (screen_text, image_b64).
        screen_text: human-readable UIAutomator summary
        image_b64: base64-encoded screenshot (JPEG), or "" if unavailable

        Also caches the raw UI XML dump (self._last_ui_xml) and current app
        package (self._last_current_app) for the fast deterministic engine
        to use — avoiding a second uiautomator dump just for that.

        need_screenshot=False skips the (relatively slow) screencap+encode
        step entirely — used when the fast path resolves the whole step
        and no vision call will be made, cutting real wall-clock time.
        """
        # Get screen text
        try:
            current_app = self.controller.get_current_app()
        except Exception:
            current_app = "(unknown)"
        self._last_current_app = current_app

        try:
            ui_xml = self.controller.dump_ui()
        except Exception:
            ui_xml = ""
        self._last_ui_xml = ui_xml

        if ui_xml:
            texts = [m.group(1).strip() for m in re.finditer(r'text="([^"]+)"', ui_xml) if m.group(1).strip()]
            seen: set = set()
            unique = [t for t in texts if not (t in seen or seen.add(t))]
            summary = " | ".join(unique[:30]) if unique else "(nothing readable on screen)"
        else:
            try:
                summary = self.controller.read_screen_summary()
            except Exception:
                summary = "(could not read screen)"

        screen_text = f"Current app: {current_app}\nVisible on screen: {summary}"

        # Take screenshot (only if actually needed — this is the slowest
        # part of perceive, so the fast path skips it entirely).
        img_b64 = ""
        if need_screenshot:
            try:
                time.sleep(SCREENSHOT_SETTLE_SECONDS)
                png_bytes = self.controller.bridge.exec_out("screencap -p")
                if png_bytes:
                    if _HAS_VISION:
                        # screenshot_bytes_to_b64 resizes/compresses before
                        # encoding — used regardless of which vision backend
                        # (Ollama or Gemini) is active, since both benefit
                        # from a smaller payload and OllamaVision.resize_
                        # image_b64() only re-resizes what's already sane.
                        img_b64 = screenshot_bytes_to_b64(png_bytes)
                    else:
                        img_b64 = base64.b64encode(png_bytes).decode("utf-8")
            except Exception:
                pass  # screenshot failure is non-fatal

        return screen_text, img_b64

    # ------------------------------------------------------------------ #
    # Decide — ask Gemini Vision for the next action
    # ------------------------------------------------------------------ #

    def _decide(
        self,
        goal: str,
        history_actions: list,
        screen_text: str,
        img_b64: str,
    ) -> dict:
        history_txt = "\n".join(
            f"  {i + 1}. {_describe_action(a)}" for i, a in enumerate(history_actions)
        ) or "  (no actions taken yet)"

        # Use the active vision backend (Ollama local, or Gemini fallback)
        # if available and a screenshot was captured this step.
        if self.vision and self.vision.available and img_b64:
            try:
                raw = self.vision.decide_next_action(
                    image_b64=img_b64,
                    ui_text=screen_text,
                    goal=goal,
                    history_summary=history_txt,
                    action_schema=_ACTION_SCHEMA_DOC,
                )
                # Parse the JSON response
                data = _extract_json_safe(raw)
                if "action" not in data:
                    raise TaskAgentError("Vision model response missing 'action' field")
                return data
            except Exception:
                # Vision call failed or returned something unusable — fall
                # through to the text-only fallback below rather than
                # crashing the whole agent loop.
                pass

        # Fallback: text-only Groq call (legacy behavior)
        if GROQ_API_KEY:
            return self._decide_groq(goal, history_txt, screen_text)

        # No API available
        raise TaskAgentError(
            "No AI backend available. Set GEMINI_API_KEY (recommended) "
            "or GROQ_API_KEY_JARVIS in your .env file."
        )

    def _decide_groq(self, goal: str, history_txt: str, screen_text: str) -> dict:
        """Fallback text-only decision using Groq/Llama."""
        user_msg = (
            f"GOAL: {goal}\n\n"
            f"ACTIONS TAKEN SO FAR:\n{history_txt}\n\n"
            f"CURRENT REAL SCREEN:\n{screen_text}\n\n"
            f"What is the single next action?"
        )
        raw = _groq_json_call(_VISUAL_AGENT_SYSTEM, user_msg)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise TaskAgentError(f"Invalid JSON from Groq: {e}") from e
        if "action" not in data:
            raise TaskAgentError("Groq response missing 'action' field")
        return data

    # ------------------------------------------------------------------ #
    # Act — execute one validated action
    # ------------------------------------------------------------------ #

    def _act(self, action: dict, img_b64: str = "") -> None:
        c = self.controller
        name = action.get("action")

        if name == "open_app":
            c.open_app(str(action.get("app", "")).strip())
        elif name == "tap_text":
            c.tap_text(str(action.get("text", "")))
        elif name == "tap_xy":
            c.tap(int(action["x"]), int(action["y"]))
        elif name == "tap_visual":
            # NEW: Use Gemini Vision to find the element's coords, then tap
            description = str(action.get("description", ""))
            self._act_tap_visual(description, img_b64)
        elif name == "type":
            c.type_text(str(action.get("text", "")))
        elif name == "key":
            key = str(action.get("key", "")).lower()
            if key == "back":
                c.go_back()
            elif key == "home":
                c.go_home()
            elif key == "enter":
                c.press_enter()
            elif key == "recent":
                c.recent_apps()
            else:
                c.press_key(key)
        elif name == "swipe":
            direction = str(action.get("direction", "up")).lower()
            {
                "up": c.scroll_up,
                "down": c.scroll_down,
                "left": c.scroll_left,
                "right": c.scroll_right,
            }.get(direction, c.scroll_up)()
        elif name == "wait":
            time.sleep(float(action.get("seconds", 1)))
        elif name == "smart_send":
            c.smart_send()
        elif name == "screenshot":
            c.screenshot()
        else:
            raise TaskAgentError(f"Unknown action '{name}' — refusing to execute")

    def _act_tap_visual(self, description: str, img_b64: str) -> None:
        """
        Tap an element described visually (e.g. "the blue send icon button").
        Uses Gemini Vision to find the element's coordinates, then taps.
        Falls back to tap_text if vision unavailable.
        """
        if self.vision and self.vision.available and img_b64:
            try:
                # First try text-based tap as it's cheaper
                if self.controller.tap_text(description):
                    return

                # If text tap failed, ask Gemini for coordinates
                result = self.vision.find_element_coords(
                    image_b64=img_b64,
                    ui_text="",  # coords-only call, no need for full dump
                    element_description=description,
                    screen_width=self._screen_w,
                    screen_height=self._screen_h,
                )
                if result.found and result.confidence in ("high", "medium"):
                    self.controller.tap(result.x, result.y)
                    return
                # Low confidence → try text-based tap as last resort
                self.controller.tap_text(description)
                return
            except Exception:
                pass

        # Fallback: treat description as text to tap
        if not self.controller.tap_text(description):
            raise ADBError(
                f"Could not find element '{description}' on screen "
                f"(vision unavailable and text not found)"
            )

    # ------------------------------------------------------------------ #
    # Verify — visual confirmation of goal completion
    # ------------------------------------------------------------------ #

    def _verify(
        self,
        goal: str,
        action: dict,
        screen_after: str,
        img_after: str,
    ) -> Tuple[str, str]:
        """
        Visually verify whether the goal/subtask is complete.
        Returns (state, note) where state ∈ {"complete", "in_progress", "stuck"}.
        """
        action_desc = _describe_action(action)

        # Gemini visual verification
        if self.vision and self.vision.available and img_after:
            try:
                result = self.vision.verify_goal(
                    image_b64=img_after,
                    ui_text=screen_after,
                    goal=goal,
                    action_taken=action_desc,
                )
                return result.state, result.note
            except Exception:
                pass

        # Text-only fallback (Groq)
        if GROQ_API_KEY:
            return self._verify_groq(goal, action_desc, screen_after)

        # No backend: treat as in_progress
        return "in_progress", ""

    def _verify_groq(self, goal: str, action_desc: str, screen_after: str) -> Tuple[str, str]:
        """Text-only Groq verification (fallback)."""
        _VERIFY_PROMPT = (
            "You verify Android automation. Respond ONLY with JSON: "
            "{\"state\": \"complete\"|\"in_progress\"|\"stuck\", \"note\": \"...\"}"
        )
        user_msg = (
            f"GOAL: {goal}\n"
            f"ACTION JUST EXECUTED: {action_desc}\n\n"
            f"REAL SCREEN RIGHT NOW:\n{screen_after}\n\n"
            f"What is the current state?"
        )
        try:
            raw = _groq_json_call(_VERIFY_PROMPT, user_msg)
            data = json.loads(raw)
            state = data.get("state", "in_progress")
            note = data.get("summary") or data.get("note") or ""
            if state not in ("complete", "in_progress", "stuck"):
                state = "in_progress"
            return state, note
        except Exception:
            return "in_progress", ""

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #

    def _make_plan(self, goal: str) -> Optional[object]:
        """
        Generate a TaskPlan for the goal.
        Returns None if planner is unavailable (falls back to single subtask).

        Obviously-atomic goals ("open X", "call mom", one clear action with
        no "and then"/multi-step language) skip the planner LLM entirely —
        both the classify_complexity AND plan calls — and go straight to a
        single-subtask plan built locally in Python. This is the single
        biggest latency win: planning used to cost 1-2 LLM round trips
        before the agent even touched the phone, on every single command.
        """
        if looks_atomic(goal):
            if self.planner:
                return self.planner._fallback_plan(goal)
            return None

        if not self.planner or not self.planner.available:
            return None

        try:
            # Quick complexity check — skip full planning for simple goals
            complexity = self.planner.classify_complexity(goal)
            if complexity == "simple":
                # Simple: use minimal fallback plan (1 subtask), no 2nd call
                return self.planner._fallback_plan(goal)
            return self.planner.plan(goal)
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# Legacy TaskAgent — text-only, backward-compatible (uses Groq)
# --------------------------------------------------------------------------- #

_LEGACY_AGENT_SYSTEM = f"""You are the execution-planning core of JARVIS, an
Android-control assistant. You are given a user's goal and the ACTUAL
current phone screen contents (read live via Android's accessibility tree).
You choose ONE next action each turn.

{_ACTION_SCHEMA_DOC}

Never invent screen contents. If the same screen isn't changing after your
action, try a different approach rather than repeating the same action."""

_LEGACY_VERIFY_SYSTEM = (
    "You verify Android automation. Respond ONLY with JSON: "
    "{\"state\": \"complete\"|\"in_progress\"|\"stuck\", "
    "\"summary\": \"one sentence (for complete)\", \"note\": \"brief note\"}"
)


class TaskAgent:
    """
    Legacy text-only closed-loop agent (uses Groq, no vision).
    Preserved for backward compatibility with existing callers.

    For new code, use VisualTaskAgent instead.
    """

    def __init__(
        self,
        controller: AndroidController,
        on_step: Optional[Callable[[StepRecord], None]] = None,
    ):
        self.controller = controller
        self.on_step = on_step

    def run(self, goal: str) -> TaskRun:
        if not GROQ_API_KEY or GROQ_API_KEY == "paste_your_groq_key_here":
            raise TaskAgentError(
                "No Groq API key configured (GROQ_API_KEY_JARVIS). "
                "Set it in .env, or use VisualTaskAgent with GEMINI_API_KEY instead."
            )

        run = TaskRun(goal=goal)
        history_actions: list = []
        recent_signatures: list = []

        for step_num in range(1, MAX_STEPS + 1):
            screen_before = self._perceive()

            try:
                action = self._decide(goal, history_actions, screen_before)
            except TaskAgentError as e:
                run.success = False
                run.aborted_reason = f"planning failed: {e}"
                break

            if action.get("action") == "give_up":
                run.success = False
                run.aborted_reason = action.get("reason", "agent judged the goal unreachable")
                break

            if action.get("action") == "done":
                verify_state, verify_note = "complete", action.get("summary", "")
                rec = StepRecord(step_num, action, screen_before, screen_before,
                                 verify_state, verify_note)
                run.steps.append(rec)
                if self.on_step:
                    self.on_step(rec)
                run.success = True
                run.final_summary = verify_note or "Goal completed."
                break

            try:
                self._act(action)
            except (ADBError, ValueError, KeyError, TypeError) as e:
                screen_after = self._perceive()
                rec = StepRecord(step_num, action, screen_before, screen_after,
                                 "stuck", f"action raised an error: {e}")
                run.steps.append(rec)
                history_actions.append(action)
                if self.on_step:
                    self.on_step(rec)
                continue

            time.sleep(STEP_SETTLE_SECONDS)
            screen_after = self._perceive()

            verify_state, verify_note = self._verify(goal, action, screen_after)
            rec = StepRecord(step_num, action, screen_before, screen_after,
                             verify_state, verify_note)
            run.steps.append(rec)
            history_actions.append(action)
            if self.on_step:
                self.on_step(rec)

            if verify_state == "complete":
                run.success = True
                run.final_summary = verify_note or "Goal completed."
                break

            signature = (json.dumps(action, sort_keys=True), _screen_signature(screen_after))
            recent_signatures.append(signature)
            if recent_signatures.count(signature) >= MAX_STUCK_REPEATS:
                run.success = False
                run.aborted_reason = (
                    f"Same action produced no screen change after "
                    f"{MAX_STUCK_REPEATS} attempts — aborting."
                )
                break
        else:
            run.success = False
            run.aborted_reason = f"Reached {MAX_STEPS}-step safety limit without confirming completion."

        if run.success is None:
            run.success = False
            run.aborted_reason = run.aborted_reason or "Task did not reach a confirmed state."
        return run

    def _perceive(self) -> str:
        try:
            current_app = self.controller.get_current_app()
        except Exception:
            current_app = "(unknown)"
        try:
            summary = self.controller.read_screen_summary()
        except Exception:
            summary = "(could not read screen)"
        return f"Current app: {current_app}\nVisible on screen: {summary}"

    def _decide(self, goal: str, history_actions: list, screen_state: str) -> dict:
        history_txt = "\n".join(
            f"  {i + 1}. {_describe_action(a)}" for i, a in enumerate(history_actions)
        ) or "  (no actions taken yet)"
        user_msg = (
            f"GOAL: {goal}\n\n"
            f"ACTIONS TAKEN SO FAR:\n{history_txt}\n\n"
            f"CURRENT REAL SCREEN:\n{screen_state}\n\n"
            f"What is the single next action?"
        )
        raw = _groq_json_call(_LEGACY_AGENT_SYSTEM, user_msg)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise TaskAgentError(f"agent returned invalid JSON: {e}") from e
        if "action" not in data:
            raise TaskAgentError("agent response missing 'action' field")
        return data

    def _act(self, action: dict) -> None:
        c = self.controller
        name = action.get("action")
        if name == "open_app":
            c.open_app(str(action.get("app", "")).strip())
        elif name == "tap_text":
            c.tap_text(str(action.get("text", "")))
        elif name == "tap_xy":
            c.tap(int(action["x"]), int(action["y"]))
        elif name == "type":
            c.type_text(str(action.get("text", "")))
        elif name == "key":
            key = str(action.get("key", "")).lower()
            if key == "back":
                c.go_back()
            elif key == "home":
                c.go_home()
            elif key == "enter":
                c.press_enter()
            elif key == "recent":
                c.recent_apps()
            else:
                c.press_key(key)
        elif name == "swipe":
            direction = str(action.get("direction", "up")).lower()
            {"up": c.scroll_up, "down": c.scroll_down,
             "left": c.scroll_left, "right": c.scroll_right}.get(
                direction, c.scroll_up)()
        elif name == "wait":
            time.sleep(float(action.get("seconds", 1)))
        elif name == "smart_send":
            c.smart_send()
        elif name == "screenshot":
            c.screenshot()
        else:
            raise TaskAgentError(f"unknown action '{name}'")

    def _verify(self, goal: str, action: dict, screen_after: str) -> tuple:
        user_msg = (
            f"GOAL: {goal}\n"
            f"ACTION JUST EXECUTED: {_describe_action(action)}\n\n"
            f"REAL SCREEN RIGHT NOW:\n{screen_after}\n\n"
            f"What is the current state?"
        )
        try:
            raw = _groq_json_call(_LEGACY_VERIFY_SYSTEM, user_msg)
            data = json.loads(raw)
            state = data.get("state", "in_progress")
            note = data.get("summary") or data.get("note") or ""
            if state not in ("complete", "in_progress", "stuck"):
                state = "in_progress"
            return state, note
        except Exception:
            return "in_progress", ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_simple_subtask(goal: str):
    """Create a minimal single-subtask fallback (no AutonomousPlanner)."""
    if _HAS_PLANNER:
        return SubTask(
            description=goal,
            success_criterion="Goal appears complete on screen",
            max_steps=MAX_STEPS,
        )
    # Create a minimal duck-typed object if planner module isn't available
    class _FallbackSubTask:
        def __init__(self, g):
            self.description = g
            self.success_criterion = "Goal appears complete on screen"
            self.max_steps = MAX_STEPS
            self.fallback = ""
            self.depends_on = -1
            self.optional = False
    return _FallbackSubTask(goal)


def _extract_json_safe(text: str) -> dict:
    """Extract JSON dict from model response text, with fallbacks."""
    if not text:
        return {}
    stripped = text.strip()
    # Remove markdown fences
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped.strip())
    # Direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Find first JSON object
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _groq_json_call(system_prompt: str, user_msg: str, timeout: int = 20) -> str:
    """Legacy Groq JSON call (used by TaskAgent fallback)."""
    if not GROQ_API_KEY:
        raise TaskAgentError("No Groq API key configured.")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
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
        raise TaskAgentError(f"Groq API error {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise TaskAgentError(f"Network error reaching Groq: {e}") from e

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise TaskAgentError(f"Unexpected Groq response: {body}") from e
