# Specialist Agents

This package adds single-app, exclusively-scoped autonomous agents on top
of the existing JARVIS pipeline, plus a `ToolSelectorAgent` that
automatically routes an utterance to the right one. Everything here runs
on the same ADB/phone-UI automation the rest of the project already
uses — no new API keys, no OAuth, no accounts.

## What's in here

| File | Purpose |
|---|---|
| `base_agent.py` | `SpecialistAgent` base class: command-vocabulary matching, wraps `VisualTaskAgent`/`TaskAgent` for execution, enforces an app-scope guard at runtime. |
| `whatsapp_agent.py` | `WhatsAppAgent` — the **only** agent allowed to touch WhatsApp. Find contact, read latest, reply, send PDF, find image, delete chat, archive chat. |
| `gmail_agent.py` | `GmailAgent` — inbox, search, compose, reply, summarize, archive. |
| `files_agent.py` | `FilesAgent` — find resume, delete zips, compress, extract, move files. Runs raw `adb shell` (`find`/`rm`/`mv`/`zip`/`unzip`) instead of tapping through a Files app UI — faster and more reliable for this kind of task. All paths are sandboxed under `/sdcard/`. |
| `tool_selector.py` | `ToolSelectorAgent` — decides which specialist agent (if any) owns an utterance; returns `None` if none do, so the caller falls through to the existing general pipeline unchanged. |

## How exclusivity is enforced ("no other agent touches WhatsApp")

Two independent layers:

1. **Routing layer** (`ToolSelectorAgent`): only `WhatsAppAgent`'s command
   vocabulary is ever offered WhatsApp-shaped phrasing. If the selector
   decides an utterance is WhatsApp's territory but `WhatsAppAgent`
   doesn't recognise the specific phrasing, the result is an honest "not
   supported" message — it never silently falls through to a different
   agent or to Gmail/Files.
2. **Runtime layer** (`SpecialistAgent._check_scope`): after every step
   the underlying vision agent takes, the agent checks the actual
   foreground app against its declared `APP_SCOPE`. If execution ever
   drifts outside that scope (e.g. a confused action opens Settings), the
   run is aborted and reported rather than continuing.

## How routing works

```
utterance
    │
    ▼
ToolSelectorAgent.select_agent_id()   -- cheap regex, near-zero latency
    │
    ├── None ─────────────► fall through to the existing
    │                        command_parser / llm_parser / VisualTaskAgent
    │                        pipeline, completely unchanged
    │
    └── "whatsapp" | "gmail" | "files"
             │
             ▼
        <Agent>.match_command()  -- which of THIS agent's ops matches?
             │
             ├── found  ──► run_op() ──► VisualTaskAgent / raw shell ──► AgentResult
             └── not found ──► honest "recognised as a <X> request but
                                no matching action" (does NOT reroute)
```

This is wired into `jarvis_gui.py`'s `_handle_command()`, right before
the general `parse_multi_command`/`parse_with_llm` path, via
`JarvisApp._try_specialist_agent()`.

## Dynamic reactor rings per agent

`jarvis_gui.py`'s `STATE_STYLE` dict gained three new ring states, each
with its own colour, rotation speed, pulse speed, and segment pattern —
not just a colour swap:

| State | Colour | Rotation | Segments | Feel |
|---|---|---|---|---|
| `AGENT_WHATSAPP` | WhatsApp green | fast (34) | 18 short segments | quick back-and-forth chat |
| `AGENT_GMAIL` | Gmail red | slower (16) | 8 thick segments | longer-form reading/composing |
| `AGENT_FILES` | amber | near-static (8) | 30 sparse tick-like segments | short, punchy, deterministic shell ops |

Each also gets a small filled hexagon badge at the top of the ring, in
the agent's own colour, as an extra non-text identifier. The ring swaps
into the right state the instant `ToolSelectorAgent` picks an agent
(before the first step even runs), via the `on_status` callback.

## Adding a new specialist agent

1. Subclass `SpecialistAgent`, set `AGENT_ID`, `DISPLAY_NAME`,
   `APP_SCOPE` (Android package name(s)), `APP_NAME`, `RING_STATE`, and a
   `COMMANDS` dict of `AgentOp`s (trigger regex patterns → a goal string
   template handed to the execution loop).
2. Register it in `tool_selector.py`'s `_AGENT_HINTS` (and
   `_SECONDARY_HINTS` if it needs command-shape-only detection) and
   `self._agent_classes`.
3. Add a ring state for it in `jarvis_gui.py`'s `STATE_STYLE` and the
   segment-pattern branch in `_animate_ring()`.
4. Export it from `agents/__init__.py`.

If most of the new agent's actions are better done via raw `adb shell`
than by tapping through a UI (like `FilesAgent`), override `run_op()`
directly instead of going through `VisualTaskAgent` — see
`files_agent.py` for the pattern.
