"""
agents/whatsapp_agent.py
=========================
The ONLY agent permitted to touch WhatsApp. ToolSelectorAgent enforces this
at the routing level (no other agent's command vocabulary matches WhatsApp
phrasing), and SpecialistAgent's scope guard enforces it at runtime (if
execution ever drifts outside com.whatsapp, the run is aborted rather than
silently continuing in some other app).

Supported commands (matches the examples in the request):
    Find Rahul          -> open his chat
    Read latest message -> open chat, report back what's visible
    Reply                -> reply in the currently-open (or named) chat
    Send PDF              -> attach & send a PDF from Files/Downloads
    Find Image            -> locate an image in a chat or in the gallery share sheet
    Delete Chat           -> delete a named chat
    Archive Chat          -> archive a named chat

Contacts: for "find X" / "reply to X" / etc., if X matches a saved name in
contacts.json, we resolve straight to a phone number and open the chat
directly via adb_controller.open_whatsapp_chat() (fast, deterministic --
no vision call needed at all). If it's not a saved contact, we fall back
to the visual agent searching WhatsApp's own contact/chat search by name.
"""

from __future__ import annotations

import re

from agents.base_agent import AgentOp, AgentResult, SpecialistAgent

try:
    from contacts import resolve_contact
except ImportError:
    def resolve_contact(name: str) -> str:
        raise ValueError("contacts module unavailable")


def _p(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


class WhatsAppAgent(SpecialistAgent):
    AGENT_ID = "whatsapp"
    DISPLAY_NAME = "WhatsApp Agent"
    APP_SCOPE = ("com.whatsapp", "com.whatsapp.w4b")
    APP_NAME = "whatsapp"
    RING_STATE = "AGENT_WHATSAPP"

    COMMANDS = {
        # NOTE: dict iteration order = match priority (base_agent.match_command
        # tries each op in this order, first match wins). More specific
        # patterns like find_image must be listed BEFORE find_contact's
        # broad "^find <anything>$" catch-all, or "find image" would
        # always be swallowed by find_contact first.
        "find_image": AgentOp(
            name="find_image",
            patterns=(
                _p(r"^find\s+image(?:\s+(?:of|in|from)\s+(?P<arg>.+))?$"),
                _p(r"^find\s+(?:the\s+)?photo(?:\s+(?:of|in|from)\s+(?P<arg>.+))?$"),
            ),
            goal_template=(
                "In WhatsApp, locate an image {arg_clause} — check the "
                "current chat's media/gallery view and report what you find, "
                "without sending anything."
            ),
            needs_arg=False,
        ),
        "find_contact": AgentOp(
            name="find_contact",
            patterns=(
                _p(r"^find\s+(?P<arg>.+)$"),
                _p(r"^(?:open|go to)\s+(?P<arg>.+?)(?:'s)?\s+chat$"),
                _p(r"^open\s+chat\s+with\s+(?P<arg>.+)$"),
            ),
            goal_template="In WhatsApp, open the chat with {arg}.",
        ),
        "send_message": AgentOp(
            name="send_message",
            patterns=(
                # "message/text <contact> saying/that/: <message>" -- tried
                # first because the "saying"/"that"/":" delimiter is
                # unambiguous (unlike bare "to", which can also appear
                # inside the message body itself, e.g. "coming to the
                # party"), so this ordering avoids misparsing those cases.
                _p(r"^(?:message|text)\s+(?P<arg>[a-z0-9 .'\-]{2,40}?)\s+"
                   r"(?:saying|that|and\s+say)\s*:?\s+(?P<msg>.+)$"),
                # "send/text (a message) to <contact>" with NO body at all
                # (just the bare verb + optional filler words + "to X") --
                # matched explicitly BEFORE the body-capturing pattern below
                # so filler words like "message"/"a message" never get
                # mistaken for the message content itself.
                _p(r"^(?:send|text)\s+(?:an?\s+)?message\s+to\s+"
                   r"(?P<arg>[a-z0-9 .'\-]{2,40})$"),
                # "send/text <message body> to <contact>"
                _p(r"^(?:send|text)\s+(?:saying\s+)?"
                   r"(?P<msg>.+)\s+to\s+(?P<arg>[a-z0-9 .'\-]{2,40})$"),
                # "message/text <contact>" with no message body -- opens
                # the chat with an empty compose box (nothing to send yet)
                _p(r"^(?:message|text)\s+(?P<arg>[a-z0-9 .'\-]{2,40})$"),
            ),
            goal_template=(
                "In WhatsApp, open the chat with {arg}, type the message "
                "\"{msg}\" into the message field, and send it."
            ),
            needs_arg=False,
        ),
        "read_latest": AgentOp(
            name="read_latest",
            patterns=(
                _p(r"^read\s+(?:the\s+)?latest\s+message(?:s)?(?:\s+from\s+(?P<arg>.+))?$"),
                _p(r"^what\s+did\s+(?P<arg>.+?)\s+say$"),
            ),
            goal_template=(
                "In WhatsApp, open the chat with {arg} and report the text "
                "of the most recent message visible on screen."
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
                "In WhatsApp, in the currently open chat (or the chat "
                "already implied by context), type the reply \"{arg}\" "
                "into the message field and send it."
            ),
        ),
        "send_pdf": AgentOp(
            name="send_pdf",
            patterns=(
                _p(r"^send\s+(?:a\s+)?pdf(?:\s+(?:to|called|named)\s+(?P<arg>.+))?$"),
            ),
            goal_template=(
                "In WhatsApp, in the currently open chat (or the chat for "
                "{arg} if that names a contact), use the attachment/paperclip "
                "button, choose Document, pick the most relevant PDF file, "
                "and send it."
            ),
            needs_arg=False,
        ),
        "delete_chat": AgentOp(
            name="delete_chat",
            patterns=(
                _p(r"^delete\s+chat(?:\s+(?:with|for)\s+(?P<arg>.+))?$"),
            ),
            goal_template=(
                "In WhatsApp, long-press the chat with {arg} in the chat "
                "list to select it, then tap the delete/trash icon and "
                "confirm deletion."
            ),
            needs_arg=False,
        ),
        "archive_chat": AgentOp(
            name="archive_chat",
            patterns=(
                _p(r"^archive\s+chat(?:\s+(?:with|for)\s+(?P<arg>.+))?$"),
            ),
            goal_template=(
                "In WhatsApp, long-press the chat with {arg} in the chat "
                "list to select it, then tap the archive icon."
            ),
            needs_arg=False,
        ),
    }

    # ------------------------------------------------------------------ #
    # Override: try the deterministic contacts.json fast path before
    # falling through to the general vision agent -- this is the same
    # "fast path first" philosophy fast_engine.py already uses elsewhere.
    # ------------------------------------------------------------------ #

    def run_op(self, op: AgentOp, arg: str, raw_text: str):
        if op.name == "find_contact" and arg:
            try:
                number = resolve_contact(arg.lower().strip())
                self.controller.open_whatsapp_chat(number, message="")
                if self.learner:
                    self.learner.record_contact_mention(arg.lower().strip())
                return AgentResult(
                    agent_id=self.AGENT_ID, op_name=op.name,
                    goal_text=f"open chat with {arg}", success=True,
                    summary=f"Opened WhatsApp chat with {arg}.",
                    steps_taken=1,
                )
            except ValueError:
                pass  # not a saved contact -- fall through to visual agent search
            except Exception:
                pass  # ADB hiccup -- fall through to visual agent search

        if op.name == "send_message":
            return self._run_send_message(op, raw_text)

        # find_image needs a slightly different template shape (arg_clause)
        if op.name == "find_image":
            clause = f"related to {arg}" if arg else "in the currently open chat"
            op = AgentOp(
                name=op.name, patterns=op.patterns,
                goal_template=op.goal_template.replace("{arg_clause}", clause),
                needs_arg=False,
            )
            return super().run_op(op, "", raw_text)

        if op.name == "reply" and arg and self.learner:
            self.learner.record_typed_message(arg)

        return super().run_op(op, arg, raw_text)

    def _run_send_message(self, op: AgentOp, raw_text: str) -> AgentResult:
        """
        "send/message/text <message> to <contact>" -- re-matches raw_text
        against send_message's own patterns to pull out BOTH the contact
        (`arg`) and message body (`msg`) groups, since the shared
        match_command()/run_op() contract only carries a single `arg`
        string through. If the contact resolves via contacts.json, this
        goes straight through the deterministic wa.me deep-link + tap-send
        path (adb_controller.send_whatsapp_message) -- fast, no vision
        call needed at all, same philosophy as find_contact's fast path
        above. Only falls back to the visual agent if the name isn't a
        saved contact.
        """
        cleaned = self._LEADING_FILLER.sub("", raw_text.strip()).strip()
        cleaned = self._TRAILING_FILLER.sub("", cleaned).strip()

        contact = ""
        message = ""
        for pat in op.patterns:
            m = pat.search(cleaned)
            if m:
                gd = m.groupdict()
                contact = (gd.get("arg") or "").strip()
                message = (gd.get("msg") or "").strip()
                break

        if contact and message:
            try:
                number = resolve_contact(contact.lower())
                self.controller.send_whatsapp_message(number, message)
                if self.learner:
                    self.learner.record_contact_mention(contact.lower())
                    self.learner.record_typed_message(message)
                return AgentResult(
                    agent_id=self.AGENT_ID, op_name=op.name,
                    goal_text=f"send \"{message}\" to {contact}", success=True,
                    summary=f"Sent \"{message}\" to {contact} on WhatsApp.",
                    steps_taken=1,
                )
            except ValueError:
                pass  # not a saved contact -- fall through to visual agent
            except Exception:
                pass  # ADB hiccup -- fall through to visual agent

        # No saved-contact fast path available (unknown name, or no
        # message body given yet) -- let the visual agent open the chat
        # and/or type+send via WhatsApp's own UI search.
        if contact and message:
            goal = f"In WhatsApp, open the chat with {contact}, type the message \"{message}\", and send it."
        elif contact:
            goal = f"In WhatsApp, open the chat with {contact}."
        else:
            goal = "In WhatsApp, open the chat implied by context and send the intended message."
        fallback_op = AgentOp(name=op.name, patterns=op.patterns, goal_template=goal, needs_arg=False)
        return super().run_op(fallback_op, "", raw_text)