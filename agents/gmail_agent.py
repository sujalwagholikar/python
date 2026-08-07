"""
agents/gmail_agent.py
=======================
Specialist agent for the Gmail Android app. Same ADB/phone-UI automation
approach as every other agent here (no Gmail API / OAuth) -- it drives the
real Gmail app the same way a person's thumb would.

Handles: inbox, search, compose, reply, summarize, archive.
"""

from __future__ import annotations

import re

from agents.base_agent import AgentOp, SpecialistAgent


def _p(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


class GmailAgent(SpecialistAgent):
    AGENT_ID = "gmail"
    DISPLAY_NAME = "Gmail Agent"
    APP_SCOPE = ("com.google.android.gm",)
    APP_NAME = "gmail"
    RING_STATE = "AGENT_GMAIL"

    COMMANDS = {
        "inbox": AgentOp(
            name="inbox",
            patterns=(
                _p(r"^(?:open|check|show)\s+(?:my\s+)?inbox$"),
                _p(r"^(?:check|any)\s+(?:new\s+)?(?:mail|email)s?$"),
            ),
            goal_template="In Gmail, go to the main inbox and report the subjects of the newest 3-5 emails visible.",
            needs_arg=False,
        ),
        "search": AgentOp(
            name="search",
            patterns=(
                _p(r"^search\s+(?:for\s+)?(?:mail|email)?\s*(?:from|about)?\s*(?P<arg>.+)$"),
                _p(r"^find\s+(?:the\s+)?email\s+(?:from|about)\s+(?P<arg>.+)$"),
            ),
            goal_template="In Gmail, tap the search bar, type \"{arg}\", submit the search, and report the top results.",
        ),
        "compose": AgentOp(
            name="compose",
            patterns=(
                _p(r"^compose(?:\s+(?:an?\s+)?email)?(?:\s+to\s+(?P<arg>.+))?$"),
                _p(r"^(?:write|send)\s+(?:an?\s+)?email(?:\s+to\s+(?P<arg>.+))?$"),
            ),
            goal_template=(
                "In Gmail, tap Compose{arg_clause}, and leave the draft open "
                "for the user to add a subject and body before sending -- do "
                "not send automatically."
            ),
            needs_arg=False,
        ),
        "reply": AgentOp(
            name="reply",
            patterns=(
                _p(r"^reply(?:\s+to\s+(?P<who>.+?))?\s*(?:saying|:)\s*(?P<arg>.+)$"),
                _p(r"^reply\s*:?\s*(?P<arg>.+)$"),
            ),
            goal_template=(
                "In Gmail, in the currently open email thread, tap Reply, "
                "type \"{arg}\" into the body, and send it."
            ),
        ),
        "summarize": AgentOp(
            name="summarize",
            patterns=(
                _p(r"^summari[sz]e\s+(?:this\s+)?(?:email|inbox|thread)?(?:\s+(?:from|about)\s+(?P<arg>.+))?$"),
            ),
            goal_template=(
                "In Gmail, open the email {arg_clause} and report a short "
                "summary of its content -- do not take any other action."
            ),
            needs_arg=False,
        ),
        "archive": AgentOp(
            name="archive",
            patterns=(
                _p(r"^archive(?:\s+(?:the\s+)?email(?:\s+(?:from|about)\s+(?P<arg>.+))?)?$"),
            ),
            goal_template=(
                "In Gmail, locate the email {arg_clause} in the inbox and "
                "swipe or tap it to archive it."
            ),
            needs_arg=False,
        ),
    }

    def run_op(self, op: AgentOp, arg: str, raw_text: str):
        if op.name == "compose":
            clause = f", fill in the recipient {arg}" if arg else ""
            op = AgentOp(
                name=op.name, patterns=op.patterns,
                goal_template=op.goal_template.replace("{arg_clause}", clause),
                needs_arg=False,
            )
            return super().run_op(op, "", raw_text)
        if op.name in ("summarize", "archive"):
            clause = f"from {arg}" if arg else "currently open or topmost in the inbox"
            op = AgentOp(
                name=op.name, patterns=op.patterns,
                goal_template=op.goal_template.replace("{arg_clause}", clause),
                needs_arg=False,
            )
            return super().run_op(op, "", raw_text)
        return super().run_op(op, arg, raw_text)
