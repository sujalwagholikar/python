"""
agents/files_agent.py
=======================
Specialist agent for on-device file management.

UNLIKE WhatsApp/Gmail, most Files operations don't need the vision/tap
loop at all -- ADB shell already gives deterministic, scriptable file
access (find/rm/mv/zip/unzip), which is faster and more reliable than
tapping through a Files app UI. So this agent runs raw `adb shell`
commands directly via AndroidController.shell(), and only falls back to
the visual agent for genuinely UI-only actions (e.g. "open the Files app
and show me screenshots" where the user wants to *see* it).

Handles: Find Resume, Delete ZIPs, Compress, Extract, Move Files.

Safety: delete/compress/extract/move all operate under /sdcard/ only --
this agent refuses any path that resolves outside the shared storage
root, so a garbled voice command can't be turned into "rm -rf /".
"""

from __future__ import annotations

import posixpath
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from adb_controller import ADBError, AndroidController
from agents.base_agent import AgentOp, AgentResult, SpecialistAgent, SpecialistAgentError

_SDCARD_ROOT = "/sdcard/"


def _p(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def _safe_join(user_path: str) -> str:
    """Resolve a user-given fragment to an absolute path under /sdcard/,
    refusing anything that tries to climb outside it (../, absolute paths
    elsewhere, etc.)."""
    cleaned = (user_path or "").strip().strip("'\"")
    if not cleaned:
        return _SDCARD_ROOT
    if cleaned.startswith("/"):
        candidate = cleaned
    else:
        candidate = posixpath.join(_SDCARD_ROOT, cleaned)
    normalized = posixpath.normpath(candidate)
    if not (normalized == _SDCARD_ROOT.rstrip("/") or normalized.startswith(_SDCARD_ROOT)):
        raise ValueError(f"Refusing to operate outside {_SDCARD_ROOT}: {user_path!r}")
    return normalized


class FilesAgent(SpecialistAgent):
    AGENT_ID = "files"
    DISPLAY_NAME = "Files Agent"
    APP_SCOPE = ()  # shell-driven, not app-UI-scoped
    APP_NAME = ""   # don't force-open a Files app for shell ops
    RING_STATE = "AGENT_FILES"

    COMMANDS = {
        "find_resume": AgentOp(
            name="find_resume",
            patterns=(
                _p(r"^find\s+(?:my\s+)?resume$"),
                _p(r"^find\s+(?:my\s+)?cv$"),
            ),
            goal_template="resume", needs_arg=False,
        ),
        "delete_zips": AgentOp(
            name="delete_zips",
            patterns=(
                _p(r"^delete\s+zips?$"),
                _p(r"^delete\s+all\s+zip\s+files?$"),
            ),
            goal_template="", needs_arg=False,
        ),
        "compress": AgentOp(
            name="compress",
            patterns=(
                _p(r"^compress\s+(?P<arg>.+)$"),
            ),
            goal_template="",
        ),
        "extract": AgentOp(
            name="extract",
            patterns=(
                _p(r"^extract\s+(?P<arg>.+)$"),
                _p(r"^unzip\s+(?P<arg>.+)$"),
            ),
            goal_template="",
        ),
        "move_files": AgentOp(
            name="move_files",
            patterns=(
                _p(r"^move\s+(?P<arg>.+?)\s+to\s+(?P<dest>.+)$"),
            ),
            goal_template="",
        ),
    }

    # ------------------------------------------------------------------ #
    # Override: entirely shell-driven, deterministic. No VisualTaskAgent.
    # ------------------------------------------------------------------ #

    def run_op(self, op: AgentOp, arg: str, raw_text: str) -> AgentResult:
        if self.on_status:
            self.on_status(self.AGENT_ID, self.RING_STATE)

        dispatch = {
            "find_resume": self._find_resume,
            "delete_zips": self._delete_zips,
            "compress": self._compress,
            "extract": self._extract,
            "move_files": self._move_files,
        }
        fn = dispatch.get(op.name)
        if fn is None:
            return AgentResult(
                agent_id=self.AGENT_ID, op_name=op.name, goal_text=raw_text,
                success=False, summary=f"Files Agent has no handler for '{op.name}'.",
            )
        try:
            return fn(arg, raw_text)
        except ValueError as e:
            return AgentResult(
                agent_id=self.AGENT_ID, op_name=op.name, goal_text=raw_text,
                success=False, summary=str(e),
            )
        except ADBError as e:
            return AgentResult(
                agent_id=self.AGENT_ID, op_name=op.name, goal_text=raw_text,
                success=False, summary=f"Files Agent hit a device error: {e}",
            )

    # ------------------------------------------------------------------ #
    # Ops
    # ------------------------------------------------------------------ #

    def _find_resume(self, arg: str, raw_text: str) -> AgentResult:
        out = self.controller.bridge.shell(
            f"find {_SDCARD_ROOT} -iname '*resume*' -o -iname '*cv*.pdf' "
            f"-o -iname '*cv*.docx' 2>/dev/null | head -20",
            timeout=25,
        )
        matches = [f.strip() for f in out.splitlines() if f.strip()]
        if matches:
            summary = "Found: " + "; ".join(matches[:5])
            if len(matches) > 5:
                summary += f" (+{len(matches) - 5} more)"
        else:
            summary = "No resume/CV file found under device storage."
        return AgentResult(
            agent_id=self.AGENT_ID, op_name="find_resume", goal_text=raw_text,
            success=bool(matches), summary=summary, steps_taken=1,
        )

    def _delete_zips(self, arg: str, raw_text: str) -> AgentResult:
        out = self.controller.bridge.shell(
            f"find {_SDCARD_ROOT} -iname '*.zip' 2>/dev/null", timeout=25,
        )
        zips = [f.strip() for f in out.splitlines() if f.strip()]
        if not zips:
            return AgentResult(
                agent_id=self.AGENT_ID, op_name="delete_zips", goal_text=raw_text,
                success=True, summary="No .zip files found -- nothing to delete.",
                steps_taken=1,
            )
        deleted = 0
        for path in zips:
            safe = _safe_join(path)
            self.controller.delete_file(safe)
            deleted += 1
        return AgentResult(
            agent_id=self.AGENT_ID, op_name="delete_zips", goal_text=raw_text,
            success=True, summary=f"Deleted {deleted} .zip file(s).",
            steps_taken=deleted,
        )

    def _compress(self, arg: str, raw_text: str) -> AgentResult:
        target = _safe_join(arg)
        archive = target.rstrip("/") + ".zip"
        # Android's toolbox doesn't ship `zip` on every ROM -- try, and
        # report clearly if it's unavailable rather than silently failing.
        check = self.controller.bridge.shell("which zip", timeout=10)
        if not check.strip():
            return AgentResult(
                agent_id=self.AGENT_ID, op_name="compress", goal_text=raw_text,
                success=False,
                summary=(
                    "This device's shell has no 'zip' binary available, so "
                    "Files Agent can't compress on-device. Try a Files-app "
                    "with built-in compress support instead."
                ),
            )
        parent = posixpath.dirname(target)
        base = posixpath.basename(target)
        self.controller.bridge.shell(
            f"cd {parent} && zip -r {posixpath.basename(archive)} {base}",
            timeout=60,
        )
        return AgentResult(
            agent_id=self.AGENT_ID, op_name="compress", goal_text=raw_text,
            success=True, summary=f"Compressed {target} -> {archive}",
            steps_taken=1,
        )

    def _extract(self, arg: str, raw_text: str) -> AgentResult:
        target = _safe_join(arg)
        if not target.lower().endswith(".zip"):
            return AgentResult(
                agent_id=self.AGENT_ID, op_name="extract", goal_text=raw_text,
                success=False, summary="Extract currently supports .zip files only.",
            )
        check = self.controller.bridge.shell("which unzip", timeout=10)
        if not check.strip():
            return AgentResult(
                agent_id=self.AGENT_ID, op_name="extract", goal_text=raw_text,
                success=False,
                summary="This device's shell has no 'unzip' binary available.",
            )
        dest_dir = posixpath.dirname(target) or _SDCARD_ROOT
        self.controller.bridge.shell(f"unzip -o {target} -d {dest_dir}", timeout=60)
        return AgentResult(
            agent_id=self.AGENT_ID, op_name="extract", goal_text=raw_text,
            success=True, summary=f"Extracted {target} into {dest_dir}",
            steps_taken=1,
        )

    def _move_files(self, arg: str, raw_text: str) -> AgentResult:
        # arg here is only the source half; dest was captured separately
        # by the "dest" named group -- re-match to pull both out cleanly.
        m = re.search(r"^move\s+(?P<src>.+?)\s+to\s+(?P<dst>.+)$", raw_text.strip(), re.I)
        if not m:
            return AgentResult(
                agent_id=self.AGENT_ID, op_name="move_files", goal_text=raw_text,
                success=False,
                summary="Couldn't parse a source and destination -- say \"move X to Y\".",
            )
        src = _safe_join(m.group("src"))
        dst = _safe_join(m.group("dst"))
        self.controller.bridge.shell(f"mkdir -p {dst}", timeout=10)
        self.controller.bridge.shell(f"mv {src} {dst}", timeout=30)
        return AgentResult(
            agent_id=self.AGENT_ID, op_name="move_files", goal_text=raw_text,
            success=True, summary=f"Moved {src} -> {dst}",
            steps_taken=1,
        )
