"""
check_ollama.py
================
Run this on YOUR machine (where Ollama is installed) to diagnose why
AutonomousPlanner reports:

    "Planning fallback: Neither Ollama nor Gemini API available for task planning"

Usage:
    python check_ollama.py

It checks, in order:
  1. Is anything listening on OLLAMA_HOST at all? (connection test)
  2. Does /api/tags list gemma3:4b (or your configured model)?
  3. Does an actual /api/chat call succeed?

Each step prints a clear PASS/FAIL with the real error message.
"""

import json
import os
import sys
import urllib.error
import urllib.request

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_VISION_MODEL", "gemma3:4b")


def step(n, title):
    print(f"\n[{n}] {title}")
    print("-" * 60)


def main():
    print(f"OLLAMA_HOST = {OLLAMA_HOST}")
    print(f"MODEL       = {MODEL}")

    # Step 1: raw connectivity
    step(1, "Checking connection to Ollama server")
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        print("PASS: Ollama server is reachable.")
    except urllib.error.URLError as e:
        print(f"FAIL: Cannot connect to {OLLAMA_HOST} — {e.reason}")
        print("\nFix:")
        print("  - Make sure Ollama is actually running: open a terminal and run `ollama serve`")
        print("    (on Windows/Mac, the Ollama app running in the tray usually does this for you)")
        print("  - If you changed the port, set OLLAMA_HOST in .env to match")
        print("  - If JARVIS runs inside WSL/Docker and Ollama runs on Windows host,")
        print("    'localhost' inside the container may NOT reach the Windows host.")
        print("    Try the host's LAN IP instead, e.g. OLLAMA_HOST=http://172.x.x.1:11434")
        sys.exit(1)
    except Exception as e:
        print(f"FAIL: Unexpected error — {type(e).__name__}: {e}")
        sys.exit(1)

    # Step 2: is the model actually pulled?
    step(2, f"Checking that model '{MODEL}' is installed")
    models = [m.get("name", "") for m in body.get("models", [])]
    print("Installed models:", models or "(none found)")
    if not any(m == MODEL or m.startswith(MODEL.split(":")[0] + ":") for m in models):
        print(f"FAIL: '{MODEL}' not found in `ollama list`.")
        print("\nFix:")
        print(f"  ollama pull {MODEL}")
        print("  (double check exact tag with `ollama list`, e.g. gemma3:4b vs gemma:3b vs gemma3)")
        sys.exit(1)
    print(f"PASS: '{MODEL}' is installed.")

    # Step 3: real chat call, same shape autonomous_planner.py uses
    step(3, "Sending a real /api/chat request (same call the planner makes)")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return only JSON: {\"ok\": true}"},
            {"role": "user", "content": "ping"},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 64},
    }
    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body.get("message", {}).get("content", "")
        print("PASS: Got a response from the model:")
        print(f"  {content!r}")
        print("\nOllama + Gemma are working correctly.")
        print("If AutonomousPlanner still reports the fallback error, the issue is")
        print("likely that a DIFFERENT process/venv is loading .env from elsewhere,")
        print("or OLLAMA_HOST/OLLAMA_VISION_MODEL is being overridden by a real")
        print("environment variable that differs from your .env file.")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"FAIL: HTTP {e.code} from Ollama — {detail[:500]}")
        sys.exit(1)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
