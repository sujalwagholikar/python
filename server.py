"""
server.py
=========
Minimal web backend for J.A.R.V.I.S., built for free-tier cloud hosting
(e.g. Render).

IMPORTANT — what this does and does NOT do
-------------------------------------------
The original project's phone-control features (adb_controller.py,
command_parser.py, task_agent.py, etc.) talk to a physical Android phone
over USB/WiFi ADB. A cloud host like Render has no USB port and no network
path to your phone, so those features CANNOT run in the cloud — there is
no way around this, it's a hardware access limitation, not a bug.

What DOES work fine in the cloud (pure network calls, no device needed):
  - jarvis_brain.jarvis_reply()   -> conversational chat (Groq LLM)
  - web_search.web_search()       -> DuckDuckGo / Google News / Tavily search

This server exposes exactly those two capabilities over a small JSON API,
plus a static/index.html chat UI, so the whole thing is one deployable
web service. If you also want the phone-control side, run main.py /
jarvis_launcher.pyw locally on the PC that's plugged into your phone —
that part is unchanged and still works exactly as before.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

from jarvis_brain import (
    JarvisBrainError,
    JarvisConversation,
    classify_utterance,
    jarvis_reply,
    jarvis_greeting,
)
from web_search import WebSearchError, web_search, summarize_for_speech

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="J.A.R.V.I.S. Web")

# Allow the frontend to call this API from any origin (simple single-service
# deploy on Render, so this is safe/simple; tighten if you split hosts).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation store, keyed by a client-supplied session id.
# Fine for a single small free-tier instance; resets on redeploy/restart.
_conversations: dict[str, JarvisConversation] = {}


def _get_conversation(session_id: str) -> JarvisConversation:
    convo = _conversations.get(session_id)
    if convo is None:
        convo = JarvisConversation()
        _conversations[session_id] = convo
    return convo


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    mode: str  # "chat" or "search"


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "groq_key_configured": bool(
            os.environ.get("GROQ_API_KEY_JARVIS") or os.environ.get("GROQ_API_KEY")
        ),
        "note": "Phone-control (ADB) features are not available on cloud hosting.",
    }


@app.get("/api/greeting")
def greeting() -> dict:
    return {"reply": jarvis_greeting()}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message must not be empty")

    convo = _get_conversation(req.session_id)

    try:
        mode = classify_utterance(text)
    except Exception:
        mode = "chat"

    if mode == "search":
        try:
            results = web_search(text)
            grounded_context = summarize_for_speech(text, results)
            reply = jarvis_reply(
                text,
                conversation=convo,
                device_context=f"Web search results:\n{grounded_context}",
            )
        except WebSearchError as e:
            reply = f"I couldn't complete that search, sir — {e}"
        except JarvisBrainError as e:
            reply = f"My reasoning core hit a snag, sir — {e}"
        return ChatResponse(reply=reply, mode="search")

    # "command" utterances can't be executed without a connected phone in
    # the cloud, so we still let JARVIS respond in character rather than
    # crash or silently ignore the message.
    try:
        device_note = (
            "No Android device is connected to this cloud session, so phone "
            "commands cannot be executed here — only conversation and web "
            "search work in this deployment. If the user's message sounds "
            "like a phone command, explain that briefly and offer to chat "
            "or search instead."
            if mode == "command"
            else ""
        )
        reply = jarvis_reply(text, conversation=convo, device_context=device_note)
    except JarvisBrainError as e:
        reply = f"I'm having trouble reaching my reasoning core, sir — {e}"

    return ChatResponse(reply=reply, mode=mode)


@app.post("/api/reset")
def reset(session_id: str = "default") -> dict:
    _conversations.pop(session_id, None)
    return {"status": "cleared"}


# --- Static frontend -------------------------------------------------------
# Serves static/index.html (and any assets alongside it) at "/", so the
# whole app — API + UI — is a single Render web service.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
