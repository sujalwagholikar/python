"""
agents/recovery_agent.py
==========================
RecoveryAgent -- detects when a specialist agent's target app has crashed
or dropped out of the foreground mid-task, restarts it, and resumes the
same goal. This isn't a fourth thing users talk to directly; it's a
supervisor any SpecialistAgent.handle() call can be wrapped in.

    WhatsApp crashes -> Restart -> Continue

DETECTING A "CRASH" WITHOUT ROOT
----------------------------------
ADB without root can't hook into Android's crash reporter (that needs
either root or reading logcat crash buffers, which requires READ_LOGS --
not reliably grantable via plain adb shell on stock ROMs). So "crash" here
is detected behaviourally, the same honest way the rest of this project
already treats "stuck" states (see task_agent.py's MAX_STUCK_REPEATS):

  1. The target app UNEXPECTEDLY LEFT THE FOREGROUND during a run (i.e.
     controller.get_current_app() stops matching the agent's APP_SCOPE)
     without the agent itself having asked to navigate away -- this is
     exactly what SpecialistAgent._check_scope() already flags as
     `out_of_scope`.
  2. AND the app is no longer even in the recent-apps list
     (controller.list_running_apps()) shortly after -- i.e. it didn't
     just go to background, the process actually died.

If only (1) is true (app still running, just navigated somewhere odd),
that's a normal task-agent "stuck" failure, not a crash -- RecoveryAgent
leaves it alone and lets the existing failure path report it normally.
If both (1) and (2) are true, RecoveryAgent treats it as a crash: it
reopens the app and re-runs the ORIGINAL goal from scratch (bounded
retries), since a freshly relaunched app has no in-progress state to
resume mid-way through -- restarting the goal is the only honest way to
"continue".

PUBLIC API
----------
    recovery = RecoveryAgent(controller, on_status=cb, on_detail=cb)
    result = recovery.run(specialist_agent, free_text)
        # -> AgentResult, same shape as SpecialistAgent.handle() returns,
        #    with extra fields (via .run recorded in .summary) noting any
        #    restart(s) that happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from adb_controller import ADBError, AndroidController
from agents.base_agent import AgentResult, SpecialistAgent

MAX_RESTART_ATTEMPTS = 2          # bounded -- never loop forever on a truly broken app
POST_RESTART_SETTLE_SECONDS = 2.5  # give the app time to finish (re)launching
CRASH_CONFIRM_DELAY_SECONDS = 1.0  # brief pause before checking if the process really died


@dataclass
class RecoveryLog:
    """Record of what RecoveryAgent had to do, attached to the final
    AgentResult's summary so the user hears about it, not just a silent
    retry."""
    crash_detected: bool = False
    restart_attempts: int = 0
    restart_succeeded: bool = False
    notes: List[str] = field(default_factory=list)


class RecoveryAgent:
    """Wraps a single SpecialistAgent.handle() call with crash detection
    and restart-and-continue behaviour."""

    def __init__(
        self,
        controller: AndroidController,
        on_status: Optional[Callable[[str, str], None]] = None,
        on_detail: Optional[Callable[[str], None]] = None,
    ):
        self.controller = controller
        self.on_status = on_status
        self.on_detail = on_detail

    def run(self, agent: SpecialistAgent, free_text: str) -> AgentResult:
        log = RecoveryLog()
        result = agent.handle(free_text)

        attempt = 0
        while (
            result.out_of_scope
            and self._looks_like_crash(agent)
            and attempt < MAX_RESTART_ATTEMPTS
        ):
            attempt += 1
            log.crash_detected = True
            log.restart_attempts = attempt
            self._detail(
                f"⚠ {agent.DISPLAY_NAME} target app appears to have crashed "
                f"(attempt {attempt}/{MAX_RESTART_ATTEMPTS}) — restarting…"
            )
            if self.on_status:
                self.on_status(agent.AGENT_ID, "AGENT_RECOVERY")

            restarted = self._restart_app(agent)
            if not restarted:
                log.notes.append(f"Restart attempt {attempt} failed to relaunch the app.")
                break

            log.notes.append(f"Restarted {agent.DISPLAY_NAME.lower()} after a crash.")
            self._detail(f"✓ {agent.APP_NAME or agent.DISPLAY_NAME} restarted — resuming task…")

            # A freshly relaunched app has no partial progress to resume
            # from -- the only honest "continue" is re-running the same
            # goal from the top, same as a person reopening a crashed app
            # and doing the thing again.
            agent._scope_breach = False  # reset the guard for the retry
            result = agent.handle(free_text)

        if log.crash_detected:
            log.restart_succeeded = result.success
            prefix = (
                f"[Recovered after {log.restart_attempts} restart"
                f"{'s' if log.restart_attempts != 1 else ''}] "
            )
            result.summary = prefix + result.summary

        return result

    # ------------------------------------------------------------------ #
    # Crash detection
    # ------------------------------------------------------------------ #

    def _looks_like_crash(self, agent: SpecialistAgent) -> bool:
        """True only if the app both left scope AND is no longer running
        at all -- distinguishes an actual crash from the agent merely
        having wandered to a different screen/app."""
        if not agent.APP_SCOPE:
            return False
        time.sleep(CRASH_CONFIRM_DELAY_SECONDS)
        try:
            running = self.controller.list_running_apps()
        except ADBError:
            return False  # can't tell -- don't guess a crash
        return not any(
            any(pkg in r for pkg in agent.APP_SCOPE) for r in running
        )

    # ------------------------------------------------------------------ #
    # Restart
    # ------------------------------------------------------------------ #

    def _restart_app(self, agent: SpecialistAgent) -> bool:
        if not agent.APP_NAME:
            return False
        try:
            self.controller.open_app(agent.APP_NAME)
            time.sleep(POST_RESTART_SETTLE_SECONDS)
            current = self.controller.get_current_app() or ""
            return any(pkg in current for pkg in agent.APP_SCOPE)
        except ADBError as e:
            self._detail(f"Restart failed: {e}")
            return False

    def _detail(self, msg: str) -> None:
        if self.on_detail:
            self.on_detail(msg)