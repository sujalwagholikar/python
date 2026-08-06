# Deploying J.A.R.V.I.S. (web chat) to Render — free tier

This adds a small web layer on top of the existing project:

- `server.py` — FastAPI backend exposing `/api/chat`, `/api/greeting`,
  `/api/health`, `/api/reset`, and serving `static/index.html` at `/`.
- `static/index.html` — the chat UI (arc-reactor themed).
- `requirements-web.txt` — minimal deps for the cloud server only.
- `render.yaml` — one-click Render service definition.

**Nothing in the original project was changed.** `main.py`,
`jarvis_gui.py`, `jarvis_launcher.pyw`, `adb_controller.py`, and everything
else still work exactly as before, locally, for phone control.

## Why phone control isn't in the cloud version

`adb_controller.py` talks to your Android phone over a USB cable or local
WiFi ADB connection. Render's servers have no USB port and no network path
to your phone — that's a hardware limitation, not something any amount of
code can work around. So the cloud deployment only exposes JARVIS's
**conversation** and **web search** features (`jarvis_brain.py` +
`web_search.py`), which are pure network calls to Groq / DuckDuckGo /
Google News / Tavily and need no device connection.

If you want phone control, keep running `main.py` or
`jarvis_launcher.pyw` locally on the PC connected to your phone — that
part is untouched.

## Deploy steps

1. Push this whole folder to a GitHub repo.
2. On Render: **New → Blueprint**, point it at your repo. Render will read
   `render.yaml` and set up the service automatically.
   - Or manually: **New → Web Service**, connect the repo, then set:
     - Build command: `pip install -r requirements-web.txt`
     - Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
3. In the Render dashboard, set environment variables (Settings → Environment):
   - `GROQ_API_KEY_JARVIS` — your Groq API key (free at
     https://console.groq.com/keys). `GROQ_API_KEY` also works if you only
     have one key.
   - `TAVILY_API_KEY` — optional, improves web search quality.
4. Deploy. Render gives you a URL like `https://jarvis-web.onrender.com` —
   open it and the chat UI loads directly (`index.html` is served at `/`).

## Notes on the free tier

- Free Render web services spin down after ~15 minutes of no traffic and
  take a few seconds to wake back up on the next request — the UI will
  just look like it's "thinking" a bit longer on that first message.
- Conversation history is kept in memory per `session_id` (generated
  client-side) and resets whenever the service restarts or spins down.
