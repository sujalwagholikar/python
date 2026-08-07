"""
agents/tool_selector.py
=========================
ToolSelectorAgent -- decides WHICH specialist agent (if any) should handle
a given utterance, and enforces exclusivity: WhatsApp phrasing can only
ever route to WhatsAppAgent, Gmail phrasing only to GmailAgent, Files
phrasing only to FilesAgent. If nothing matches, the caller falls through
to the existing general-purpose command pipeline (command_parser /
llm_parser / VisualTaskAgent) unchanged.

DESIGN
------
Mirrors jarvis_brain.classify_utterance()'s two-tier approach already used
elsewhere in this project: cheap local regex first (near-zero latency,
handles the overwhelming majority of phrasing), LLM tiebreak only for
genuinely ambiguous cases, and a safe default (None -> fall through to
the general pipeline) if nothing is confident.

Selection is a TWO-STEP decision:
  1. Which AGENT (app) does this belong to? -- app-name keywords are the
     strongest, cheapest signal ("whatsapp", "gmail", "on gmail", a saved
     contact name only really makes sense for WhatsApp, etc.)
  2. Does that agent actually recognise the phrasing as one of ITS
     commands? -- delegated to SpecialistAgent.match_command(). If step 1
     picks an agent but step 2 finds no matching op, we do NOT silently
     fall through to a different agent (that would violate the "no other
     agent touches WhatsApp" rule) -- we report "recognised as a <agent>
     request but not a supported action" instead.

PUBLIC API
----------
    selector = ToolSelectorAgent(controller, on_step=cb, on_status=cb)
    routed = selector.route(text)        # -> Optional[AgentResult]
    if routed is None:
        ... fall through to existing command_parser/llm_parser pipeline ...
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Type

from adb_controller import AndroidController
from agents.base_agent import AgentResult, SpecialistAgent
from agents.whatsapp_agent import WhatsAppAgent
from agents.gmail_agent import GmailAgent
from agents.files_agent import FilesAgent
from agents.recovery_agent import RecoveryAgent
from agents.learning_agent import LearningAgent


def _p(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Agent-level (app) selection signals -- cheap, local, checked before any
# per-command matching happens. Ordered by specificity; WhatsApp/Gmail app
# names are checked before the more generic Files vocabulary so "find my
# resume" doesn't get mistaken for WhatsApp's "find <contact>" just
# because both start with "find".
# --------------------------------------------------------------------------- #

_AGENT_HINTS: Tuple[Tuple[str, Type[SpecialistAgent], re.Pattern], ...] = (
    ("whatsapp", WhatsAppAgent, _p(r"\bwhatsapp\b")),
    ("gmail", GmailAgent, _p(r"\bgmail\b|\be-?mail\b|\binbox\b")),
    ("files", FilesAgent, _p(
        r"\b(resume|cv|zip|compress|extract|unzip|move\s+files?|"
        r"delete\s+(?:the\s+)?zips?)\b"
    )),
)

# Command-shape signals that imply an agent even without the app name
# being said explicitly, e.g. "find Rahul" (WhatsApp-style contact find)
# vs "find my resume" (Files) -- these run only if _AGENT_HINTS found
# nothing, and only as a secondary, lower-confidence pass.
_SECONDARY_HINTS: Tuple[Tuple[str, Type[SpecialistAgent], re.Pattern], ...] = (
    ("files", FilesAgent, _p(r"^find\s+(?:my\s+)?(resume|cv)\b")),
    ("files", FilesAgent, _p(r"^move\s+.+\s+to\s+.+$")),
    ("whatsapp", WhatsAppAgent, _p(
        r"^(?:find|reply|archive\s+chat|delete\s+chat|send\s+pdf|find\s+image|"
        r"read\s+(?:the\s+)?latest\s+messages?)\b"
    )),
)


class ToolSelectorAgent:
    """Routes an utterance to the single correct specialist agent, or
    returns None so the caller can fall through to the general pipeline."""

    def __init__(
        self,
        controller: Optional[AndroidController],
        on_step=None,
        on_status=None,
        on_detail=None,
        # on_detail(msg) -- lets RecoveryAgent surface "restarting after
        # crash" notices to the transcript, same channel append_detail()
        # already uses elsewhere in the GUI.
        learner: Optional[LearningAgent] = None,
        # Shared LearningAgent instance -- when provided, gets threaded
        # into every specialist agent so app-opens/contact-mentions/typed
        # messages get recorded automatically as a side effect of normal
        # routing, with zero extra calls needed at call sites.
    ):
        self.controller = controller
        self.on_step = on_step
        self.on_status = on_status
        self.on_detail = on_detail
        self.learner = learner
        self._agent_classes: Dict[str, Type[SpecialistAgent]] = {
            "whatsapp": WhatsAppAgent,
            "gmail": GmailAgent,
            "files": FilesAgent,
        }

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #

    def select_agent_id(self, text: str) -> Optional[str]:
        """Return which agent's territory this utterance belongs to, or
        None if it isn't clearly any specialist agent's job."""
        stripped = text.strip()
        for agent_id, _cls, pattern in _AGENT_HINTS:
            if pattern.search(stripped):
                return agent_id
        for agent_id, _cls, pattern in _SECONDARY_HINTS:
            if pattern.search(stripped):
                return agent_id
        return None

    def route(self, text: str) -> Optional[AgentResult]:
        """
        Full dispatch: pick the agent, strip the app-name if the user said
        it explicitly (so "on whatsapp find rahul" -> "find rahul" for the
        agent's own command matcher), run it, and return the AgentResult.

        Returns None (not an AgentResult) if no specialist agent's
        territory matches at all -- this is the signal for the caller to
        use the existing general-purpose pipeline instead.
        """
        agent_id = self.select_agent_id(text)
        if agent_id is None:
            return None

        agent_cls = self._agent_classes[agent_id]
        cleaned = self._strip_app_name(text, agent_id)

        if self.controller is None:
            return AgentResult(
                agent_id=agent_id, op_name="unknown", goal_text=text,
                success=False,
                summary=f"{agent_cls.DISPLAY_NAME} needs a connected phone first.",
            )

        agent = agent_cls(
            self.controller, on_step=self.on_step, on_status=self.on_status,
            learner=self.learner,
        )

        # Route the actual op through RecoveryAgent rather than calling
        # agent.handle() directly, so a crashed target app gets restarted
        # and the goal re-run automatically, instead of the failure just
        # being reported. RecoveryAgent only intervenes when it detects a
        # genuine crash (app left scope AND is no longer even running) --
        # anything else passes through as agent.handle() would return
        # unmodified.
        recovery = RecoveryAgent(self.controller, on_status=self.on_status, on_detail=self.on_detail)
        result = recovery.run(agent, cleaned)

        # If the matched agent's fixed vocabulary didn't recognise the
        # exact phrasing, do NOT hard-fail and do NOT hand it to a
        # different specialist agent (that would break the "no other
        # agent touches WhatsApp" guarantee) -- but DO let it fall
        # through to the general-purpose pipeline (command_parser /
        # llm_parser / VisualTaskAgent), which already knows how to open
        # this exact app and can still complete the request generically.
        # This keeps specialist agents strictly additive: they can only
        # make a request succeed faster/more reliably, never regress a
        # phrasing that used to work before they existed.
        if result.op_name == "unknown":
            if self.on_detail:
                self.on_detail(
                    f"'{cleaned}' didn't match a known {agent_cls.DISPLAY_NAME} "
                    f"action ({', '.join(sorted(agent_cls.COMMANDS.keys()))}) "
                    f"-- falling back to general command handling."
                )
            return None
        return result

    @staticmethod
    def _strip_app_name(text: str, agent_id: str) -> str:
        # Swallow "open <app> and" as a unit first -- very common phrasing
        # ("open whatsapp and message rahul") that would otherwise leave
        # an orphaned "open and" behind once just the app name is
        # stripped, breaking every downstream pattern match.
        text = re.sub(
            r"^open\s+(?:whatsapp|gmail)\s+and\s+", "", text, flags=re.IGNORECASE
        ).strip()

        app_words = {
            "whatsapp": r"\b(?:on\s+)?whatsapp\b",
            "gmail": r"\b(?:on\s+|in\s+)?gmail\b",
            "files": r"",
        }
        pattern = app_words.get(agent_id, "")
        if not pattern:
            return text.strip()
        cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned or text.strip()