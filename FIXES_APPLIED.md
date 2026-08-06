# JARVIS Bug Fixes — August 2026

## Critical crash fix (this was your "mismatch" error)

**File:** `task_agent.py` — `VisualTaskAgent._decide()`

The method referenced `_HAS_GEMINI_VISION` (never defined anywhere in the
module) and caught `GeminiVisionError` (only imported on the Gemini-only
fallback path, not the Ollama-primary path you actually use). Since your
`.env` has Ollama as the primary vision backend, **every single vision
decision call threw a `NameError`**, silently caught by a bare
`except Exception`, causing JARVIS to fall back to a generic text
response instead of actually acting — exactly the "mismatch, but I can
open WhatsApp directly" behavior you saw.

Fixed: removed the broken exception branch, and fixed the matching
`_HAS_GEMINI_VISION` reference in `_perceive()`'s screenshot-encoding
step (now uses the correctly-imported `_HAS_VISION` flag).

This also explains a good chunk of the "very slow" complaint — every
vision call was silently failing and falling through to a *second*,
slower text-only Groq round-trip after already wasting time.

## Silent literal-`{arg}` bug (5 ops affected)

**File:** `agents/base_agent.py` — `SpecialistAgent.run_op()`

When a command was recognized but had no captured argument (e.g. "read
the latest message" with no name), the code only filled in the `{arg}`
placeholder in the goal template *if* an argument string was non-empty —
otherwise it left the literal text `"{arg}"` in the goal sent to the
vision model, confusing every one of these ops:

- WhatsApp: `read_latest`, `send_pdf`, `delete_chat`, `archive_chat`
- Gmail: `compose`

Fixed: added `SpecialistAgent._format_goal()`, a safe formatter that
always fills the placeholder — using a natural fallback phrase when no
argument was given. Also cleaned up Gmail's `compose` template/routing
to read naturally in both the "compose to X" and bare "compose" cases.

## Verification performed

- Full-tree `pyflakes` pass: zero undefined names, zero syntax errors
  (only pre-existing harmless unused-import warnings remain).
- Full-tree `py_compile`: all files compile.
- Runtime import test of every core module (task_agent, fast_engine,
  autonomous_planner, ollama_vision, gemini_vision, all agents,
  jarvis_brain, contacts, context_memory, adb_controller): clean.
- Regression test of all 9 previously-broken op/no-arg combinations
  across WhatsApp and Gmail agents: no leaked placeholders, natural
  phrasing confirmed for every case.

## What was already working well (no changes needed)

- `fast_engine.py` — deterministic regex/UI-tree fast path, skips the
  LLM entirely for most steps. Well designed, no bugs found.
- `agents/files_agent.py` — fully deterministic ADB shell driven, no
  vision/LLM dependency at all. No bugs found.
- `agents/recovery_agent.py` — crash detection and app-restart logic.
  No bugs found.
- `agents/tool_selector.py` — agent routing/exclusivity enforcement.
  No bugs found; verified WhatsApp vs Files vs Gmail routing collisions
  resolve correctly.
- `autonomous_planner.py` — atomic-goal short-circuit already skips
  planning LLM calls for simple one-shot commands.

## Recommended next steps for speed (not code bugs, but worth knowing)

Your `.env` uses `gemma3:4b` via Ollama on a GTX 1650-class or similar
setup context — vision calls to a 4B multimodal model are inherently
the slowest part of the pipeline (up to 45s timeout budgeted per call).
With the crash fixed, these calls will now actually succeed instead of
silently failing and falling back — so you should see both correctness
AND perceived speed improve immediately, since you're no longer paying
for a failed vision call AND a fallback Groq call on every escalation.
If it's still too slow after this fix, consider:
- A smaller/faster local vision model if your GPU is VRAM-constrained
- Increasing `num_predict` cap tightening (already done) further
- Checking `ollama ps` while JARVIS is idle to confirm the model stays
  resident (`keep_alive: "10m"` is already set in both ollama_vision.py
  and autonomous_planner.py)

---

# Follow-up fix — "no matching action" errors

## Root cause

Reported errors:
> "That sounded like a WhatsApp Agent request, but I don't have a matching
> action for it..."
> "That sounded like a Gmail Agent request, but I don't have a matching
> action for it..."

Two separate problems, now both fixed:

### 1. Specialist agents were hard-failing instead of falling back
`ToolSelectorAgent.route()` claimed any utterance mentioning "whatsapp"
or "gmail" and, if the phrasing didn't exactly match one of that agent's
fixed regex patterns, reported a hard error -- even when the OLD,
already-working `command_parser.py` pipeline (e.g. "open whatsapp and
send hi to mom") could have handled it perfectly well one layer down.
This made adding specialist agents a **regression** for phrasing that
used to work.

Fixed: when a specialist agent doesn't recognize the exact phrasing, it
now returns `None` (already-supported by the GUI's fallthrough logic)
instead of a failed result, letting the general command_parser/
llm_parser/VisualTaskAgent pipeline handle it. Specialist agents are now
strictly additive -- they only make things faster/more reliable, never
block a request that used to work.

### 2. WhatsApp had no "send a new message to X" action at all
The only WhatsApp ops were `find_contact` (opens a chat, doesn't send
anything) and `reply` (replies in an already-open chat). "send a message
to mom" / "message rahul saying ..." matched NEITHER.

Fixed: added a new `send_message` op with a deterministic fast path
(resolves the contact via contacts.json, then uses
`adb_controller.send_whatsapp_message()` -- the same reliable wa.me
deep-link + tap-send approach `find_contact` already uses) that falls
back to the visual agent only if the contact isn't saved.

### 3. Filler words broke otherwise-correct phrasing
Every specialist-agent pattern is fully anchored (`^...$`), so "can you
please find rahul" or "find rahul for me" failed to match even though
"find rahul" alone worked. Fixed: `SpecialistAgent.match_command()` now
strips a conservative set of leading/trailing filler ("can you", "please",
"for me", "thanks", etc.) before matching, without touching the middle
of the utterance (so it can't eat words that are part of a name/message).

### 4. "open whatsapp and X" left an orphaned "and"
Stripping just the word "whatsapp" from "open whatsapp and message
rahul" left "open and message rahul" -- not matched by anything. Fixed:
`ToolSelectorAgent._strip_app_name()` now swallows the whole "open
<app> and" phrase as a unit first.

## Verification

Ran a 26-case regression suite covering every previously-reported failure
plus edge cases (filler words, "open X and" phrasing, message bodies
containing the word "to", messages with no body) across WhatsApp, Gmail,
and Files agents -- all 26 now resolve correctly. Re-ran pyflakes +
py_compile + full module import check across the whole tree: clean.
