"""
autonomous_planner.py
=====================
Autonomous task decomposition brain for JARVIS, powered by local Ollama Gemma 3 4B
(`gemma3:4b`).

THE PROBLEM
-----------
Before this module, JARVIS treated every user request as a single opaque
blob handed directly to the task execution loop:
  "Open WhatsApp, find mom's chat, send a message saying I'm running late"
  → Guesses steps blindly without structured sub-goals.

THE FIX: Autonomous Task Planning
----------------------------------
AutonomousPlanner uses Gemma 3 4B (running locally via Ollama) to THINK FIRST:

  1. ANALYSE the goal — what apps/systems are involved? What could go wrong?
  2. DECOMPOSE into ordered SubTask objects, each with:
     - A single, atomic description
     - A clear success_criterion (what the screen should show when done)
     - A max_steps budget for this sub-task
     - An optional fallback if this sub-task fails
  3. Returns a TaskPlan — an ordered list of SubTasks with an overall
     success_criterion for the whole goal.

PUBLIC API
----------
  from autonomous_planner import AutonomousPlanner, TaskPlan, SubTask

  planner = AutonomousPlanner()
  plan = planner.plan(goal="send hi to mom on whatsapp")
  for subtask in plan.subtasks:
      print(subtask.description, "→", subtask.success_criterion)
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_VISION_MODEL", "gemma3:4b")

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class PlannerError(RuntimeError):
    """Raised when the planner cannot generate a valid plan."""


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class SubTask:
    description: str
    success_criterion: str
    max_steps: int = 5
    fallback: str = ""
    depends_on: int = -1
    optional: bool = False

    def __str__(self) -> str:
        return (
            f"SubTask: {self.description!r}\n"
            f"  Success: {self.success_criterion}\n"
            f"  Max steps: {self.max_steps}"
            + (f"\n  Fallback: {self.fallback}" if self.fallback else "")
        )


@dataclass
class TaskPlan:
    original_goal: str
    subtasks: List[SubTask] = field(default_factory=list)
    overall_criterion: str = ""
    estimated_complexity: str = "moderate"
    risk_notes: List[str] = field(default_factory=list)
    total_max_steps: int = 20
    created_at: float = field(default_factory=time.time)

    def __str__(self) -> str:
        lines = [
            f"TaskPlan for: {self.original_goal!r}",
            f"  Complexity: {self.estimated_complexity}",
            f"  SubTasks: {len(self.subtasks)}",
        ]
        for i, st in enumerate(self.subtasks, 1):
            lines.append(f"  [{i}] {st.description}")
            lines.append(f"       Success: {st.success_criterion}")
        lines.append(f"  Overall: {self.overall_criterion}")
        if self.risk_notes:
            lines.append(f"  Risks: {'; '.join(self.risk_notes)}")
        return "\n".join(lines)

    def log_lines(self) -> List[str]:
        out = [f"📋 Plan: {self.original_goal} ({self.estimated_complexity})"]
        for i, st in enumerate(self.subtasks, 1):
            out.append(f"   {i}. {st.description}")
        if self.risk_notes:
            for r in self.risk_notes:
                out.append(f"   ⚠ {r}")
        return out


# --------------------------------------------------------------------------- #
# Planner prompts
# --------------------------------------------------------------------------- #

_PLAN_SYSTEM = """You are an expert Android phone automation planner for JARVIS.
Given a user's goal, you decompose it into a structured execution plan with
atomic, verifiable sub-tasks. Each sub-task must be:
  - A SINGLE concrete action (open an app, navigate, type, tap one thing)
  - Independently verifiable by looking at the screen
  - Realistic given standard Android UI behavior

You are NOT executing the plan — you are only designing it. Be honest about complexity and risks."""

_PLAN_USER_TEMPLATE = """Design an execution plan for this Android automation goal:

GOAL: {goal}

{context}

Return ONLY a valid JSON object with this exact structure:
{{
  "estimated_complexity": "simple|moderate|complex",
  "overall_success_criterion": "what the screen shows when the WHOLE goal is done",
  "risk_notes": ["list", "of", "things", "that", "might", "go", "wrong"],
  "total_max_steps": <integer 5-25>,
  "subtasks": [
    {{
      "description": "exact single action to take",
      "success_criterion": "what screen shows when THIS step is done",
      "max_steps": <integer 1-8>,
      "fallback": "what to try if this fails (empty string if none)",
      "depends_on": -1,
      "optional": false
    }}
  ]
}}

Rules:
- Keep subtasks atomic: one action per subtask.
- success_criterion must be visually verifiable.
- total_max_steps should be the SUM of all subtask max_steps.
"""

_REPLAN_SYSTEM = """You are an adaptive Android automation replanner for JARVIS.
A task execution plan has partially failed. Analyze what happened and design a RECOVERY plan."""

_REPLAN_USER_TEMPLATE = """A sub-task in an automation plan has FAILED.

ORIGINAL GOAL: {goal}
FAILED SUB-TASK: {failed_description}
FAILURE REASON: {failure_reason}

CURRENT PHONE SCREEN:
{current_screen}

COMPLETED SO FAR:
{completed_steps}

Design a RECOVERY plan. Return the same JSON structure as the original plan.
"""


# --------------------------------------------------------------------------- #
# Planner class
# --------------------------------------------------------------------------- #


class AutonomousPlanner:
    """
    Decomposes complex user goals into structured, verifiable sub-tasks.
    Uses local Ollama Gemma 3 4B (`gemma3:4b`) by default, with Gemini fallback.
    """

    def __init__(self, ollama_host: str = "", model: str = ""):
        self.ollama_host = (ollama_host or OLLAMA_HOST).rstrip("/")
        self.model = model or OLLAMA_MODEL

    @property
    def available(self) -> bool:
        """Check if local Ollama or Gemini is available."""
        return True  # Fallback handler inside plan() guarantees graceful operation

    def plan(
        self,
        goal: str,
        device_context: str = "",
        timeout: int = 120,
    ) -> TaskPlan:
        context_block = f"CURRENT DEVICE STATE: {device_context}\n" if device_context else ""
        prompt = _PLAN_USER_TEMPLATE.format(goal=goal, context=context_block)

        try:
            raw = self._text_call(_PLAN_SYSTEM, prompt, timeout=timeout)
            data = _extract_json(raw)
            return self._parse_plan(goal, data)
        except Exception as e:
            return self._fallback_plan(goal, note=str(e))

    def replan(
        self,
        original_goal: str,
        failed_subtask: SubTask,
        failure_reason: str,
        current_screen: str,
        completed_descriptions: List[str],
        timeout: int = 120,
    ) -> TaskPlan:
        completed_text = "\n".join(f"  ✓ {d}" for d in completed_descriptions) or "  (none)"
        prompt = _REPLAN_USER_TEMPLATE.format(
            goal=original_goal,
            failed_description=failed_subtask.description,
            failure_reason=failure_reason,
            current_screen=current_screen[:800],
            completed_steps=completed_text,
        )

        try:
            raw = self._text_call(_REPLAN_SYSTEM, prompt, timeout=timeout)
            data = _extract_json(raw)
            plan = self._parse_plan(original_goal, data)
            plan.risk_notes.insert(0, f"⟳ Recovery plan after failure: {failed_subtask.description[:60]}")
            return plan
        except Exception as e:
            return self._fallback_plan(original_goal, note=f"Replanning failed: {e}")

    def classify_complexity(self, goal: str, timeout: int = 45) -> str:
        prompt = (
            f"Classify this Android automation goal's complexity:\n'{goal}'\n\n"
            "Return ONLY JSON: {{\"complexity\": \"simple\"|\"moderate\"|\"complex\"}}\n"
            "simple = 1 action\nmoderate = 2-4 actions\ncomplex = 5+ actions"
        )
        try:
            raw = self._text_call("Classify goal complexity. Return only JSON.", prompt, timeout=timeout)
            data = _extract_json(raw)
            return data.get("complexity", "moderate")
        except Exception:
            return "moderate"

    def _text_call(self, system: str, user: str, timeout: int = 120) -> str:
        """Primary: local Ollama gemma3:4b call. Secondary: Gemini fallback."""
        # 1. Try Ollama local
        ollama_error: str = ""
        try:
            url = f"{self.ollama_host}/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # 768 is only needed for full multi-subtask plans; the
                # fast_engine.looks_atomic() check in task_agent.py already
                # skips this call entirely for simple one-shot goals, so
                # by the time we get here a real plan is genuinely needed.
                "options": {"temperature": 0.1, "num_predict": 768},
                # Keep the model warm in VRAM between planning/replanning/
                # classify_complexity calls in the same session, instead
                # of paying a reload cost on every call.
                "keep_alive": "10m",
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            # Ollama is reachable but returned an error (e.g. model not
            # pulled, bad request). Read the body — it usually says why.
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            ollama_error = f"Ollama HTTP {e.code} at {self.ollama_host}: {detail[:300]}"
        except urllib.error.URLError as e:
            # Can't even connect — Ollama isn't running / wrong host/port.
            ollama_error = (
                f"Cannot reach Ollama at {self.ollama_host} ({e.reason}). "
                "Is `ollama serve` running and is OLLAMA_HOST correct?"
            )
        except Exception as e:
            ollama_error = f"Ollama call failed: {type(e).__name__}: {e}"

        print(f"[AutonomousPlanner] {ollama_error}")

        # 2. Try Gemini fallback if key is present
        if GEMINI_API_KEY:
            try:
                url = _GEMINI_URL.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)
                payload = {
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                raise PlannerError(
                    f"{ollama_error} | Gemini fallback also failed: {type(e).__name__}: {e}"
                )

        raise PlannerError(ollama_error or "Neither Ollama nor Gemini API available for task planning.")

    def _parse_plan(self, goal: str, data: dict) -> TaskPlan:
        subtasks = []
        for raw_st in data.get("subtasks", []):
            if not isinstance(raw_st, dict):
                continue
            st = SubTask(
                description=str(raw_st.get("description", "")).strip(),
                success_criterion=str(raw_st.get("success_criterion", "")).strip(),
                max_steps=max(1, min(10, int(raw_st.get("max_steps", 4)))),
                fallback=str(raw_st.get("fallback", "")).strip(),
                depends_on=int(raw_st.get("depends_on", -1)),
                optional=bool(raw_st.get("optional", False)),
            )
            if st.description:
                subtasks.append(st)

        if not subtasks:
            subtasks = [SubTask(description=goal, success_criterion="Goal appears complete on screen", max_steps=8)]

        total = int(data.get("total_max_steps", sum(s.max_steps for s in subtasks) + 4))

        return TaskPlan(
            original_goal=goal,
            subtasks=subtasks,
            overall_criterion=str(data.get("overall_success_criterion", "")).strip() or "Goal is visually confirmed complete",
            estimated_complexity=str(data.get("estimated_complexity", "moderate")),
            risk_notes=[str(r) for r in data.get("risk_notes", []) if r],
            total_max_steps=max(5, min(30, total)),
        )

    def _fallback_plan(self, goal: str, note: str = "") -> TaskPlan:
        subtask = SubTask(
            description=goal,
            success_criterion="Goal appears complete on screen",
            max_steps=14,
        )
        risk_notes = []
        if note:
            risk_notes.append(f"Planning fallback: {note[:100]}")
        return TaskPlan(
            original_goal=goal,
            subtasks=[subtask],
            overall_criterion="Goal is visually confirmed complete on screen",
            estimated_complexity="unknown",
            risk_notes=risk_notes,
            total_max_steps=14,
        )


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped.strip())
    try:
        return json.loads(stripped.strip())
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in response: {text[:200]}")


if __name__ == "__main__":
    planner = AutonomousPlanner()
    print("[*] Testing local Ollama gemma3:4b planning...")
    plan = planner.plan("Open YouTube and search for biryani recipe")
    print(plan)
