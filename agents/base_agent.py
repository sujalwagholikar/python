"""
agents/base_agent.py
=====================
Shared base class for every SPECIALIST agent (WhatsApp, Gmail, Files, ...).

WHY SPECIALIST AGENTS EXIST
----------------------------
The existing VisualTaskAgent (task_agent.py) is a general-purpose,
app-agnostic perceive/decide/act/verify loop — it can be pointed at *any*
goal and it'll try its best. That's powerful, but it also means:

  - It has no memory of "how WhatsApp specifically behaves" (e.g. that the
    chat search icon is top-right, that a freshly-opened chat auto-focuses
    the text field, that the send button is bottom-right near the mic).
  - It has no fixed vocabulary of allowed actions for a domain, so it's
    equally happy to (incorrectly) wander into Settings while trying to
    "find Rahul".
  - There's no hard boundary stopping the Gmail flow from accidentally
    opening WhatsApp mid-task if the model gets confused.

A SpecialistAgent fixes this by:
  1. Declaring an APP_SCOPE (the ONLY app package(s) it's allowed to
     operate inside once a task starts).
  2. Declaring a fixed COMMANDS vocabulary (e.g. WhatsApp's "find_contact",
     "read_latest", "reply", "send_pdf", "find_image", "delete_chat",
     "archive_chat") that maps free-text goals to one concrete op.
  3. Reusing VisualTaskAgent under the hood for the actual perceive/decide/
     act/verify execution of each op, but with an op-specific goal string
     and a guard that aborts if execution ever leaves APP_SCOPE.

This keeps "no other agent touches WhatsApp" a structural guarantee
(enforced by ToolSelectorAgent routing + the scope guard below), not just
a prompt-level suggestion.

PUBLIC API
----------
    class MyAgent(SpecialistAgent):
        AGENT_ID = "myapp"
        APP_SCOPE = ("com.example.myapp",)
        COMMANDS = {"do_thing": DoThingOp}

    agent = MyAgent(controller, on_step=cb, on_status=status_cb)
    result: AgentResult = agent.handle("do the thing")
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Pattern, Tuple

from adb_controller import ADBError, AndroidController
from task_agent import VisualTaskAgent, TaskAgent, TaskAgentError, StepRecord, TaskRun

try:
    from ollama_vision import OllamaVision
    _HAS_VISION = True
except ImportError:
    try:
        from gemini_vision import GeminiVision
        _HAS_VISION = True
    except ImportError:
        _HAS_VISION = False


# --------------------------------------------------------------------------- #
# Result / step types
# --------------------------------------------------------------------------- #


@dataclass
class AgentResult:
    """Uniform result shape every specialist agent returns, regardless of
    which underlying op ran."""
    agent_id: str
    op_name: str
    goal_text: str
    success: bool
    summary: str
    steps_taken: int = 0
    aborted_reason: str = ""
    out_of_scope: bool = False   # True if execution tried to leave APP_SCOPE
    run: Optional[TaskRun] = None


@dataclass
class AgentOp:
    """One recognised operation within an agent's vocabulary.

    `patterns` are matched against the free-text goal (case-insensitive,
    first match wins) to decide which op the user meant, without needing
    an LLM round-trip for the obvious cases — mirrors the fast-path
    heuristics already used by jarvis_brain.classify_utterance and
    fast_engine.FastDecider elsewhere in this project.

    `goal_template` is the concrete instruction handed to VisualTaskAgent,
    with `{arg}` substituted from whatever free text followed the trigger
    phrase (e.g. a contact name, a search term).
    """
    name: str
    patterns: Tuple[Pattern, ...]
    goal_template: str
    needs_arg: bool = True
    max_steps: int = 12


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class SpecialistAgentError(RuntimeError):
    """Raised when a specialist agent can't even start (no controller,
    unrecognised command, etc.)."""


class SpecialistAgent:
    """
    Base class for a single-app specialist agent.

    Subclasses set:
      AGENT_ID    -- short slug, e.g. "whatsapp"
      DISPLAY_NAME-- human label, e.g. "WhatsApp Agent"
      APP_SCOPE   -- tuple of Android package names this agent is allowed
                     to operate inside (from adb_controller.APP_PACKAGES)
      APP_NAME    -- friendly app name passed to controller.open_app()
      COMMANDS    -- dict[str, AgentOp] vocabulary of recognised ops
      RING_STATE  -- GUI status-key this agent should drive the reactor
                     ring into while active (see jarvis_gui.STATE_STYLE)
    """

    AGENT_ID: str = "base"
    DISPLAY_NAME: str = "Base Agent"
    APP_SCOPE: Tuple[str, ...] = ()
    APP_NAME: str = ""
    COMMANDS: Dict[str, AgentOp] = {}
    RING_STATE: str = "AGENT"

    def __init__(
        self,
        controller: AndroidController,
        on_step: Optional[Callable[[StepRecord], None]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        # on_status(agent_id, ring_state) -- lets the GUI swap ring style
        # the instant this agent takes over, before the first step runs.
        learner=None,
        # Optional LearningAgent instance. When provided, every run_op()
        # call records an app-open (and subclasses can record contact
        # mentions / typed messages on top of that). Left as None by
        # default so specialist agents remain fully usable standalone,
        # without importing learning_agent.
    ):
        if controller is None:
            raise SpecialistAgentError(f"{self.DISPLAY_NAME}: no phone connected.")
        self.controller = controller
        self.on_step = on_step
        self.on_status = on_status
        self.learner = learner
        self._scope_breach = False

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def handle(self, free_text: str) -> AgentResult:
        """Match free_text against this agent's COMMANDS vocabulary, then
        execute the matched op end-to-end inside this agent's app scope."""
        if self.on_status:
            self.on_status(self.AGENT_ID, self.RING_STATE)

        op, arg = self.match_command(free_text)
        if op is None:
            return AgentResult(
                agent_id=self.AGENT_ID,
                op_name="unknown",
                goal_text=free_text,
                success=False,
                summary=(
                    f"{self.DISPLAY_NAME} didn't recognise that as a "
                    f"{self.DISPLAY_NAME.lower()} action."
                ),
            )
        return self.run_op(op, arg, free_text)

    @staticmethod
    def _format_goal(goal_template: str, arg: str) -> str:
        """
        Safely fill a goal_template's {arg} placeholder.

        Bug this fixes: when arg is empty/falsy AND needs_arg=False (e.g.
        WhatsApp's read_latest/send_pdf/delete_chat/archive_chat, Gmail's
        compose), the old code skipped .format() entirely whenever `arg`
        was falsy -- leaving the literal text "{arg}" inside the goal
        string handed to the vision model (e.g. "open the chat with
        {arg}"), which confused or broke every one of those ops when
        called without an explicit contact/recipient name. Always format,
        substituting a neutral, readable phrase in place of a missing arg
        instead of leaving the placeholder un-filled.
        """
        if not goal_template or "{arg}" not in goal_template:
            return goal_template
        filler = arg.strip() if arg else "whatever is currently relevant on screen"
        return goal_template.format(arg=filler)

    # Filler words people naturally add that don't change the intent --
    # stripped before pattern matching so "can you please find rahul" and
    # "find rahul for me" match the same op as "find rahul". Conservative
    # list: only stripped from the very start/end of the utterance, never
    # from the middle, so it can't accidentally eat words that are part of
    # a contact name, message, or file path.
    _LEADING_FILLER = re.compile(
        r"^(?:can\s+you\s+|could\s+you\s+|please\s+|hey\s+|jarvis\s+|"
        r"i\s+want\s+you\s+to\s+|i\s+need\s+you\s+to\s+)+",
        re.IGNORECASE,
    )
    _TRAILING_FILLER = re.compile(
        r"(?:\s+please|\s+for\s+me|\s+now|\s+thanks|\s+thank\s+you)+$",
        re.IGNORECASE,
    )

    def match_command(self, free_text: str) -> Tuple[Optional[AgentOp], str]:
        """Return (matched AgentOp, captured argument text) or (None, "")."""
        text = free_text.strip()
        text = self._LEADING_FILLER.sub("", text).strip()
        text = self._TRAILING_FILLER.sub("", text).strip()
        for op in self.COMMANDS.values():
            for pat in op.patterns:
                m = pat.search(text)
                if m:
                    arg = ""
                    if m.groupdict().get("arg"):
                        arg = m.group("arg").strip()
                    elif m.groups():
                        arg = (m.group(1) or "").strip()
                    return op, arg
        return None, ""

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def run_op(self, op: AgentOp, arg: str, raw_text: str) -> AgentResult:
        goal = self._format_goal(op.goal_template, arg)

        # Ensure we start inside this agent's own app -- every op assumes
        # the app is already foregrounded, same convention the generic
        # VisualTaskAgent uses (its first planned subtask is normally
        # "open <app>").
        try:
            if self.APP_NAME:
                self.controller.open_app(self.APP_NAME)
                time.sleep(1.2)
                if self.learner:
                    self.learner.record_app_open(self.APP_NAME)
        except ADBError as e:
            return AgentResult(
                agent_id=self.AGENT_ID, op_name=op.name, goal_text=goal,
                success=False,
                summary=f"{self.DISPLAY_NAME} couldn't open {self.APP_NAME}: {e}",
            )

        def _scoped_on_step(rec: StepRecord) -> None:
            self._check_scope()
            if self.on_step:
                self.on_step(rec)

        try:
            if _HAS_VISION:
                agent = VisualTaskAgent(self.controller, on_step=_scoped_on_step)
            else:
                agent = TaskAgent(self.controller, on_step=_scoped_on_step)
            run = agent.run(goal)
        except TaskAgentError as e:
            return AgentResult(
                agent_id=self.AGENT_ID, op_name=op.name, goal_text=goal,
                success=False, summary=f"{self.DISPLAY_NAME} automation error: {e}",
            )

        if self._scope_breach:
            return AgentResult(
                agent_id=self.AGENT_ID, op_name=op.name, goal_text=goal,
                success=False, out_of_scope=True,
                summary=(
                    f"{self.DISPLAY_NAME} aborted — execution left its "
                    f"allowed app scope ({', '.join(self.APP_SCOPE)})."
                ),
                steps_taken=len(run.steps), run=run,
            )

        return AgentResult(
            agent_id=self.AGENT_ID, op_name=op.name, goal_text=goal,
            success=bool(run.success),
            summary=run.final_summary if run.success else (
                run.aborted_reason or "Task did not complete."
            ),
            steps_taken=len(run.steps),
            aborted_reason=run.aborted_reason,
            run=run,
        )

    # ------------------------------------------------------------------ #
    # Scope guard -- "no other agent touches WhatsApp" is enforced by
    # ToolSelectorAgent routing, but this catches the runtime case where
    # the underlying VisualTaskAgent (e.g. via a stray "open_app" action)
    # tries to jump to a different app mid-task.
    # ------------------------------------------------------------------ #

    def _check_scope(self) -> None:
        if not self.APP_SCOPE:
            return
        try:
            current = self.controller.get_current_app() or ""
        except Exception:
            return
        if current and not any(pkg in current for pkg in self.APP_SCOPE):
            self._scope_breach = True
