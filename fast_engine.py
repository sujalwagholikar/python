"""
fast_engine.py
==============
A deterministic, pure-Python "fast path" for JARVIS's autonomous agent loop.

WHY THIS EXISTS
----------------
Every LLM call (planning, deciding the next tap, verifying a screen) costs
anywhere from 1-30+ seconds on local Ollama/Gemma hardware. The previous
architecture called the LLM on almost EVERY step: once to plan, then once
to decide + once to verify PER STEP, and sometimes a THIRD time to find
tap coordinates. For a 5-step task that's 10-15+ LLM round-trips.

Most of those calls don't need an LLM at all. Android already exposes a
full accessibility tree (via `adb shell uiautomator dump`) listing every
visible button/label and its exact bounding box. If the goal or subtask
names a concrete UI action ("open X", "tap Send", "type this text", "go
back", "search for Y", "scroll down"), Python can resolve it directly and
instantly against that tree — same reliability as a human matching text
on screen, with zero model latency.

This module implements:
  1. FastDecider   — regex/rule-based next-action chooser, given a subtask
                      description + the UI tree. Returns None if the intent
                      is genuinely ambiguous (icon-only UI, vague goal, etc.)
                      so the caller can fall back to the LLM.
  2. FastVerifier  — rule-based "did this succeed?" check using screen-text
                      deltas and keyword matching against the subtask's
                      success criterion. Returns None (defer to LLM/vision)
                      when it can't confidently decide.
  3. FastPlanner   — splits an already-simple goal into a single subtask
                      instantly, skipping the classify+plan LLM round trip
                      for goals that are clearly one atomic action.

DESIGN PRINCIPLE
-----------------
Fast path is CONSERVATIVE: it only commits to an action/verdict when it is
reasonably confident. Any ambiguity returns None, which callers treat as
"escalate to the LLM/vision agent for this one step only." This keeps
correctness on hard/icon-only screens while making the 80% of steps that
are simple (open app, tap visible button, type text, press back/enter,
search, scroll) run in milliseconds instead of seconds.

ZERO extra dependencies — standard library only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Public result types
# --------------------------------------------------------------------------- #


@dataclass
class UIElement:
    text: str
    desc: str
    resource_id: str
    clickable: bool
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def label(self) -> str:
        return self.text or self.desc or self.resource_id


# --------------------------------------------------------------------------- #
# UI tree parsing (shared, cheap, pure regex — no XML lib needed)
# --------------------------------------------------------------------------- #

_NODE_RE = re.compile(r"<node\b([^>]*?)/?\s*>", re.DOTALL)
_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def _attr(attrs: str, name: str) -> str:
    m = re.search(rf'{name}="([^"]*)"', attrs)
    return m.group(1) if m else ""


def parse_ui_tree(xml: str) -> List[UIElement]:
    """Parse a uiautomator XML dump into a flat list of UIElements."""
    elements: List[UIElement] = []
    for node in _NODE_RE.finditer(xml):
        attrs = node.group(1)
        bounds_raw = _attr(attrs, "bounds")
        bm = _BOUNDS_RE.match(bounds_raw)
        if not bm:
            continue
        x1, y1, x2, y2 = (int(g) for g in bm.groups())
        if x1 == x2 or y1 == y2:
            continue
        elements.append(
            UIElement(
                text=_attr(attrs, "text").strip(),
                desc=_attr(attrs, "content-desc").strip(),
                resource_id=_attr(attrs, "resource-id").strip(),
                clickable=_attr(attrs, "clickable").lower() == "true",
                x1=x1, y1=y1, x2=x2, y2=y2,
            )
        )
    return elements


# --------------------------------------------------------------------------- #
# FastDecider — resolve the next action without an LLM call
# --------------------------------------------------------------------------- #

# Words that make an instruction too vague for pure text/UI-tree matching
# (icon-only buttons, spatial/visual descriptions the tree can't capture).
_VISUAL_HINTS = re.compile(
    r"\b(icon|avatar|profile pic|thumbnail|the (?:blue|red|green|yellow|"
    r"white|black|top[- ]left|top[- ]right|bottom[- ]left|bottom[- ]right)"
    r"[a-z ]*(?:button|icon)|circle|three dots|hamburger|gear icon)\b",
    re.IGNORECASE,
)

_OPEN_APP_RE = re.compile(r"\bopen\s+(?:the\s+)?([a-z0-9 .]+?)(?:\s+app)?$", re.IGNORECASE)
_TAP_RE = re.compile(r"\btap\s+(?:on\s+)?(?:the\s+)?[\"']?([^\"'.,]+?)[\"']?\s*$", re.IGNORECASE)
_TYPE_RE = re.compile(r"\btype\s+[\"']?(.+?)[\"']?\s*$", re.IGNORECASE)
_SEARCH_RE = re.compile(r"\bsearch(?:\s+for)?\s+[\"']?(.+?)[\"']?\s*$", re.IGNORECASE)
_SEND_HINT_RE = re.compile(r"\b(send|submit|search|go|confirm)\b", re.IGNORECASE)
_BACK_RE = re.compile(r"\b(go back|press back|navigate back)\b", re.IGNORECASE)
_HOME_RE = re.compile(r"\b(go home|press home|home screen)\b", re.IGNORECASE)
_SCROLL_RE = re.compile(r"\bscroll\s+(up|down|left|right)\b", re.IGNORECASE)
_WAIT_RE = re.compile(r"\bwait\b", re.IGNORECASE)

# Common button labels tried in priority order for a generic "submit this" step
_SUBMIT_LABELS = ("Send", "Search", "Go", "Submit", "Post", "Done", "OK", "Ok", "Next", "Continue")


class FastDecider:
    """
    Given a subtask description, screen text, and the parsed UI tree, tries
    to pick the single next action deterministically. Returns None when the
    instruction is ambiguous enough that an LLM/vision call is warranted.
    """

    def decide(
        self,
        subtask_description: str,
        history_actions: List[dict],
        ui_xml: str,
        already_typed: bool = False,
    ) -> Optional[dict]:
        desc = subtask_description.strip()

        # Never guess on visually-described targets — no text/bounds to match.
        if _VISUAL_HINTS.search(desc):
            return None

        elements = parse_ui_tree(ui_xml) if ui_xml else []
        last_action_names = [a.get("action") for a in history_actions[-3:]]

        # 1. Explicit "open <app>" instruction
        m = _OPEN_APP_RE.search(desc)
        if m and "open_app" not in last_action_names:
            app_name = m.group(1).strip()
            if app_name:
                return {"action": "open_app", "app": app_name}

        # 2. Explicit "type <text>" instruction, not yet done this subtask
        m = _TYPE_RE.search(desc)
        if m and "type" not in last_action_names:
            text = m.group(1).strip()
            if text:
                return {"action": "type", "text": text}

        # 3. Explicit "search for <query>" — type then rely on smart_send next turn
        m = _SEARCH_RE.search(desc)
        if m and not already_typed and "type" not in last_action_names:
            query = m.group(1).strip()
            if query:
                return {"action": "type", "text": query}

        # 4. If we just typed something and the instruction implies submitting
        if "type" in last_action_names and _SEND_HINT_RE.search(desc):
            if self._find_any_label(elements, _SUBMIT_LABELS):
                label = self._find_any_label(elements, _SUBMIT_LABELS)
                return {"action": "tap_text", "text": label}
            return {"action": "smart_send"}

        # 5. Explicit "tap <text>" instruction where the text is on screen
        m = _TAP_RE.search(desc)
        if m:
            target = m.group(1).strip()
            if target and self._label_exists(elements, target):
                return {"action": "tap_text", "text": target}

        # 6. Navigation shortcuts
        if _BACK_RE.search(desc):
            return {"action": "key", "key": "back"}
        if _HOME_RE.search(desc):
            return {"action": "key", "key": "home"}
        sm = _SCROLL_RE.search(desc)
        if sm:
            return {"action": "swipe", "direction": sm.group(1).lower()}
        if _WAIT_RE.search(desc):
            return {"action": "wait", "seconds": 1.5}

        # 7. Fallback: does ANY clickable element's label appear verbatim
        #    inside the subtask description? (covers "tap the Settings gear"
        #    style phrasing where our regexes above didn't match exactly)
        best = self._best_desc_overlap(elements, desc)
        if best:
            return {"action": "tap_text", "text": best}

        # Nothing confidently resolved — let the LLM/vision agent handle it.
        return None

    @staticmethod
    def _label_exists(elements: List[UIElement], target: str) -> bool:
        needle = target.lower()
        return any(needle in (e.label or "").lower() for e in elements if e.label)

    @staticmethod
    def _find_any_label(elements: List[UIElement], candidates: Tuple[str, ...]) -> Optional[str]:
        labels = {e.label.lower(): e.label for e in elements if e.label}
        for cand in candidates:
            if cand.lower() in labels:
                return labels[cand.lower()]
        return None

    @staticmethod
    def _best_desc_overlap(elements: List[UIElement], desc: str) -> Optional[str]:
        """Find a clickable element whose label is quoted or clearly named
        in the subtask description, preferring longer/more specific matches."""
        candidates = [e for e in elements if e.clickable and e.label and len(e.label) >= 3]
        best_label = None
        best_len = 0
        desc_lower = desc.lower()
        for e in candidates:
            lbl = e.label.lower()
            if lbl in desc_lower and len(lbl) > best_len:
                best_label = e.label
                best_len = len(lbl)
        return best_label


# --------------------------------------------------------------------------- #
# FastVerifier — decide complete/in_progress/stuck without an LLM call
# --------------------------------------------------------------------------- #

_DONE_KEYWORDS_BY_ACTION = {
    "open_app": None,  # handled specially: check current_app changed
}


class FastVerifier:
    """
    Rule-based success check. Returns (state, note) or None to defer to the
    LLM/vision verifier. Conservative: only returns "complete" when there's
    solid textual evidence; never returns "stuck" on its own (that's judged
    by the caller's repeat-loop detector), only "complete" or None.
    """

    def verify(
        self,
        subtask_description: str,
        success_criterion: str,
        action: dict,
        current_app_before: str,
        current_app_after: str,
        screen_text_before: str,
        screen_text_after: str,
    ) -> Optional[Tuple[str, str]]:
        name = action.get("action")

        # "open_app" succeeds once the foreground package actually changed
        # and isn't still the launcher/home screen. Covers both explicit
        # open_app actions AND atomic "open <app>" goals where the
        # requested app name appears in the now-foreground package.
        if name == "open_app":
            if current_app_after and current_app_after != current_app_before:
                if "launcher" not in current_app_after.lower():
                    return "complete", f"Foreground app changed to {current_app_after}"
            return None  # not yet confirmed, let loop continue / escalate later

        # key back/home/enter and swipe rarely complete a subtask alone —
        # defer, don't guess.
        if name in ("key", "swipe", "wait"):
            return None

        # type / tap_text / smart_send: check the success criterion's key
        # terms now appear on screen that weren't there before (strong
        # signal the action worked, e.g. sent message appears in a chat,
        # searched term now shown, typed text visible in a field).
        crit_terms = self._salient_terms(success_criterion)
        if crit_terms and screen_text_after:
            after_lower = screen_text_after.lower()
            if all(term in after_lower for term in crit_terms):
                return "complete", "Success criterion terms found on screen."

        # Generic fallback (no quoted criterion terms, e.g. the default
        # "Goal appears complete on screen" placeholder used for atomic
        # single-subtask plans): after a tap_text/smart_send that follows
        # a completed "type" action, if the screen changed meaningfully
        # from before this action, treat it as a good-faith completion for
        # simple one-shot goals only — NOT for multi-subtask plans, where
        # under-verifying would let the agent move on prematurely.
        if not crit_terms and name in ("tap_text", "smart_send") and screen_text_before != screen_text_after:
            desc_lower = subtask_description.lower()
            if any(w in desc_lower for w in ("send", "search", "submit", "post", "confirm")):
                return "complete", "Screen changed after submit action (fast-path heuristic)."

        return None

    @staticmethod
    def _salient_terms(criterion: str) -> List[str]:
        """Extract 1-3 short, distinctive quoted/keyword terms from a success
        criterion to check for verbatim on-screen presence. Deliberately
        strict (only quoted substrings) to avoid false positives."""
        quoted = re.findall(r'"([^"]{2,40})"|\'([^\']{2,40})\'', criterion)
        terms = [a or b for a, b in quoted]
        return [t.lower().strip() for t in terms if t.strip()]


# --------------------------------------------------------------------------- #
# FastPlanner helpers — instant single-subtask plans for obviously-simple goals
# --------------------------------------------------------------------------- #

# A goal is "obviously simple" if it's short and matches a single clear
# action pattern with no coordinating conjunctions implying multiple steps.
_MULTISTEP_HINTS = re.compile(
    r"\b(and then|after that|then\s|,\s*then|first .* then)\b", re.IGNORECASE
)


def looks_atomic(goal: str) -> bool:
    """
    True if a goal is clearly a single action (no multi-step language,
    reasonably short). Used to skip the classify_complexity + plan LLM
    round-trip entirely for the common case of simple one-shot commands.
    """
    g = goal.strip()
    if not g:
        return False
    if _MULTISTEP_HINTS.search(g):
        return False
    if len(g.split()) > 14:
        return False
    return True
