"""
jarvis_gui.py
=============
J.A.R.V.I.S. — full graphical, voice-driven Android control assistant.

Run this file directly (or via jarvis_launcher.pyw, which hides the
console window entirely on Windows). No terminal interaction is required
during normal use — everything happens through this window: click the
mic to talk, or type in the command box.

Architecture
------------
  Voice / typed input
        |
        v
  jarvis_brain.classify_utterance()  ---> "chat" ---> jarvis_brain.jarvis_reply()  (llama-3.3-70b-versatile)
        |                                    |
        |                                    +--> "search" ---> web_search.web_search()
        |                                                        (Tavily / DuckDuckGo / Google News RSS)
        v "command"
  context_memory (pronoun/reference resolution)                 spoken back via gTTS
        |
        v
  command_parser.parse_multi_command()  (fast offline regex)
        |  (ParseError)
        v
  llm_parser.parse_with_llm()  (llama-3.1-8b-instant, JSON intent)
        |
        v
  intent.name == "open_app_and_do"?  --yes--> task_agent.TaskAgent.run()
        |no                                    closed-loop: perceive real
        v                                      screen -> decide 1 action ->
  command_parser.execute_intent()               act -> verify against the
        ---> adb_controller.AndroidController   new real screen -> repeat
        |                                        until confirmed done/stuck
        v
  jarvis_brain.narrate_result()  (llama-3.3-70b-versatile paraphrase)
        |
        v
  spoken back via gTTS + shown in the transcript panel

Everything that can block (mic listening, TTS playback, ADB commands,
Groq calls, web search, system-stat polling, the task agent loop) runs
on background threads or lightweight `after()` polling; the Tk main
loop is never blocked, so the UI stays responsive and animated at all
times.

HUD additions in this version
------------------------------
- Multi-ring animated reactor core (tick ring, segmented spin ring,
  inner glow) that visibly changes color/speed/pattern per state:
  IDLE / LISTENING / THINKING / SEARCHING / AGENT / EXECUTING / SPEAKING / ERROR.
- Closed-loop, screen-verified task execution via `task_agent.py`: any
  multi-step goal is driven by perceive-decide-act-verify against the
  live screen and only reports success once actually confirmed.
- Three satellite gauge rings + a live sparkline graph, HUD corner
  brackets, and a vertical signal-bar strip — styled after a classic
  sci-fi heads-up-display.
- Top-left live telemetry overlay: CPU %, GPU %, RAM %, and real-time
  network throughput (KB/s or MB/s) sampled every second via
  `system_monitor.py` (uses `psutil` + `nvidia-smi` when available,
  degrades to "N/A" per-field otherwise — never crashes the UI).
- Real-time web search via `web_search.py`: DuckDuckGo HTML endpoint +
  Google News RSS (both keyless) plus Tavily (if `TAVILY_API_KEY` is
  set in `.env`), merged/deduplicated, routed to automatically whenever
  an utterance is classified as a search/lookup request.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import font as tkfont
from tkinter import messagebox, scrolledtext
from typing import Optional

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

from adb_controller import ADBBridge, ADBError, AndroidController, DeviceNotConnected
from command_parser import ParseError, execute_intent, parse_multi_command
from context_memory import ConversationMemory
from llm_parser import LLMParseError, parse_with_llm
from system_monitor import SystemMonitor
from task_agent import TaskAgent, VisualTaskAgent, TaskAgentError, StepRecord
from web_search import WebSearchError, summarize_for_speech, web_search
from agents import ToolSelectorAgent, AgentResult, LearningAgent
import jarvis_brain
import voice_io

try:
    from ollama_vision import OllamaVision
    _ollama = OllamaVision()
    _HAS_OLLAMA = _ollama.available
    _HAS_VISION = _HAS_OLLAMA or bool(os.environ.get("GEMINI_API_KEY", ""))
except ImportError:
    _HAS_OLLAMA = False
    _HAS_VISION = bool(os.environ.get("GEMINI_API_KEY", ""))

try:
    from autonomous_planner import AutonomousPlanner
    _HAS_PLANNER = True
except ImportError:
    _HAS_PLANNER = False

# --------------------------------------------------------------------------- #
# Palette — Iron-Man / arc-reactor HUD theme (deliberately not a generic
# "dark mode" default: carbon-fiber near-black base, arc-reactor cyan as
# the single accent, warm amber reserved only for alerts/errors, and a
# repeating hexagonal "reactor" motif rather than flat panels).
# --------------------------------------------------------------------------- #

BG_VOID = "#05070A"        # outermost background, almost pure black
BG_PANEL = "#0B1116"       # panel background
BG_PANEL_RAISED = "#101922"
LINE_DIM = "#1C2933"
ARC_CYAN = "#38E1FF"
ARC_CYAN_DIM = "#1B6E7F"
ARC_CYAN_GLOW = "#9FF3FF"
GOLD_ACCENT = "#E3B341"    # sparingly: wake word / listening pulse
SEARCH_VIOLET = "#B98CFF"  # web-search state accent
ALERT_RED = "#FF4D4D"
# Per-specialist-agent ring accents — each agent gets its own colour so the
# reactor ring is visually distinct depending on which agent has control.
WHATSAPP_GREEN = "#25D366"  # matches WhatsApp's own brand green
GMAIL_RED = "#EA4335"       # matches Gmail's own brand red
FILES_AMBER = "#F2A93B"     # warm amber for on-device file operations
TEXT_PRIMARY = "#E8F6FA"
TEXT_SECONDARY = "#6E8A96"
TEXT_FAINT = "#3E5560"

FONT_DISPLAY = ("Consolas", 20, "bold")
FONT_MONO = ("Consolas", 11)
FONT_MONO_SM = ("Consolas", 9)
FONT_MONO_XS = ("Consolas", 8)
FONT_LABEL = ("Consolas", 10, "bold")

# Per-state HUD styling: (ring color, rotation speed, pulse speed, label)
STATE_STYLE = {
    "IDLE":       (ARC_CYAN_DIM,  2,  3,  "STANDBY"),
    "LISTENING":  (GOLD_ACCENT,   6,  10, "LISTENING"),
    "THINKING":   (ARC_CYAN,      14, 16, "THINKING"),
    "PROCESSING": (ARC_CYAN,      18, 18, "PROCESSING"),
    "SEARCHING":  (SEARCH_VIOLET, 22, 20, "SCANNING WEB"),
    "PLANNING":   (SEARCH_VIOLET, 18, 14, "PLANNING TASK"),   # NEW: autonomous planner
    "EXECUTING":  (ARC_CYAN,      26, 22, "EXECUTING"),
    "AGENT":      (SEARCH_VIOLET, 30, 26, "AUTOMATING TASK"),
    # ---- specialist-agent states -------------------------------------
    # Each has a distinct colour AND a distinct rotation/pulse rhythm, so
    # which agent is active is readable from the ring alone, not just its
    # colour: WhatsApp spins fast/steady (quick chat back-and-forth),
    # Gmail spins slower/heavier (reading/composing takes longer per
    # step), Files barely rotates but pulses sharply (short, punchy,
    # deterministic shell ops rather than a multi-step visual loop).
    "AGENT_WHATSAPP": (WHATSAPP_GREEN, 34, 30, "WHATSAPP AGENT"),
    "AGENT_GMAIL":    (GMAIL_RED,      16, 12, "GMAIL AGENT"),
    "AGENT_FILES":    (FILES_AMBER,     8, 34, "FILES AGENT"),
    "AGENT_RECOVERY": (ALERT_RED,      40, 44, "RECOVERING"),  # fast spin + sharp pulse: urgent, short-lived
    "SPEAKING":   (ARC_CYAN_GLOW, 10, 14, "SPEAKING"),
    "ERROR":      (ALERT_RED,     4,  6,  "ERROR"),
}


@dataclass
class PipelineResult:
    kind: str          # "chat" | "command" | "error" | "system"
    user_text: str
    spoken_text: str
    detail: str = ""


class JarvisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("J.A.R.V.I.S. — Android Control Interface")
        self.root.geometry("980x680")
        self.root.minsize(820, 560)
        self.root.configure(bg=BG_VOID)

        # -- state -----------------------------------------------------
        self.memory = ConversationMemory()
        self.conversation = jarvis_brain.JarvisConversation()
        self.controller: Optional[AndroidController] = None
        self.listener: Optional[voice_io.SpeechListener] = None
        self.speaker: Optional[voice_io.SpeechSpeaker] = None
        self.learner = LearningAgent()  # persists usage patterns to learning_memory.json across sessions
        self.voice_enabled = tk.BooleanVar(value=True)
        self.always_listening = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="INITIALIZING")
        self.device_text = tk.StringVar(value="No device connected")
        self.busy = False
        self._listen_loop_stop = threading.Event()
        self._ui_queue: "queue.Queue[callable]" = queue.Queue()

        # -- telemetry (system stats + HUD animation state) ------------
        self._sys_monitor = SystemMonitor()
        self._latest_stats = None
        self._net_history = [0.0] * 40   # sparkline: recent download KB/s
        self._pulse_phase = 0.0
        self._ring_phase = 0.0

        self._build_layout()
        self._animate_ring()
        self._poll_ui_queue()
        self._poll_system_stats()

        # kick off connection + voice engine init off the main thread
        threading.Thread(target=self._startup_sequence, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def _build_layout(self) -> None:
        outer = tk.Frame(self.root, bg=BG_VOID)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        # ---- Header bar -------------------------------------------------
        header = tk.Frame(outer, bg=BG_VOID)
        header.pack(fill="x", pady=(0, 10))

        title = tk.Label(
            header, text="J . A . R . V . I . S .", fg=ARC_CYAN, bg=BG_VOID,
            font=FONT_DISPLAY,
        )
        title.pack(side="left")

        subtitle = tk.Label(
            header, text="ANDROID CONTROL INTERFACE",
            fg=TEXT_FAINT, bg=BG_VOID, font=FONT_MONO_SM,
        )
        subtitle.pack(side="left", padx=(14, 0), pady=(8, 0))

        self.device_label = tk.Label(
            header, textvariable=self.device_text, fg=TEXT_SECONDARY, bg=BG_VOID,
            font=FONT_MONO_SM, anchor="e",
        )
        self.device_label.pack(side="right")

        # ---- Telemetry strip: live CPU / GPU / RAM / network -------------
        self._build_telemetry_strip(outer)

        # ---- Body: left = reactor/status, right = transcript -----------
        body = tk.Frame(outer, bg=BG_VOID)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG_VOID, width=340)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG_VOID)
        right.pack(side="left", fill="both", expand=True)

        self._build_reactor_panel(left)
        self._build_transcript_panel(right)

        # ---- Input bar ---------------------------------------------------
        input_bar = tk.Frame(outer, bg=BG_VOID)
        input_bar.pack(fill="x", pady=(10, 0))

        self.entry = tk.Entry(
            input_bar, bg=BG_PANEL_RAISED, fg=TEXT_PRIMARY, insertbackground=ARC_CYAN,
            font=FONT_MONO, relief="flat", highlightthickness=1,
            highlightbackground=LINE_DIM, highlightcolor=ARC_CYAN,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self.entry.bind("<Return>", self._on_submit_text)
        self.entry.insert(0, "Type a command, or click the mic to speak…")
        self.entry.bind("<FocusIn>", self._clear_placeholder)
        self._placeholder_active = True

        self.send_btn = tk.Button(
            input_bar, text="SEND", command=self._on_submit_text,
            bg=BG_PANEL_RAISED, fg=ARC_CYAN, activebackground=ARC_CYAN_DIM,
            activeforeground=BG_VOID, relief="flat", font=FONT_LABEL,
            padx=16, cursor="hand2",
        )
        self.send_btn.pack(side="left", padx=(0, 8))

        self.mic_btn = tk.Button(
            input_bar, text="🎙  LISTEN", command=self._on_mic_click,
            bg=ARC_CYAN_DIM, fg=BG_VOID, activebackground=ARC_CYAN,
            activeforeground=BG_VOID, relief="flat", font=FONT_LABEL,
            padx=16, cursor="hand2",
        )
        self.mic_btn.pack(side="left")

        # ---- Footer: toggles ---------------------------------------------
        footer = tk.Frame(outer, bg=BG_VOID)
        footer.pack(fill="x", pady=(8, 0))

        self._make_checkbox(footer, "🔊 Voice replies", self.voice_enabled).pack(side="left")
        self._make_checkbox(footer, "👂 Always listening (wake word: "
                             f"'{voice_io.WAKE_WORD}')", self.always_listening,
                             command=self._on_toggle_always_listen).pack(side="left", padx=(20, 0))

        help_btn = tk.Button(
            footer, text="? HELP", command=self._show_help,
            bg=BG_VOID, fg=TEXT_FAINT, activebackground=BG_VOID,
            activeforeground=ARC_CYAN, relief="flat", font=FONT_MONO_SM,
            cursor="hand2", bd=0,
        )
        help_btn.pack(side="right")

    def _make_checkbox(self, parent, text, var, command=None):
        cb = tk.Checkbutton(
            parent, text=text, variable=var, bg=BG_VOID, fg=TEXT_SECONDARY,
            selectcolor=BG_PANEL_RAISED, activebackground=BG_VOID,
            activeforeground=ARC_CYAN, font=FONT_MONO_SM, relief="flat",
            highlightthickness=0, cursor="hand2", command=command,
        )
        return cb

    def _build_telemetry_strip(self, parent: tk.Frame) -> None:
        """Top-left-anchored live system telemetry: CPU / GPU / RAM / NET."""
        strip = tk.Frame(parent, bg=BG_PANEL, highlightbackground=LINE_DIM,
                          highlightthickness=1)
        strip.pack(fill="x", pady=(0, 10))

        inner = tk.Frame(strip, bg=BG_PANEL)
        inner.pack(side="left", padx=14, pady=6)

        def _stat_block(label: str) -> tk.Label:
            block = tk.Frame(inner, bg=BG_PANEL)
            block.pack(side="left", padx=(0, 22))
            tk.Label(block, text=label, fg=TEXT_FAINT, bg=BG_PANEL,
                      font=FONT_MONO_XS).pack(anchor="w")
            val = tk.Label(block, text="—", fg=ARC_CYAN, bg=BG_PANEL,
                            font=FONT_LABEL)
            val.pack(anchor="w")
            return val

        self.cpu_val_label = _stat_block("CPU")
        self.gpu_val_label = _stat_block("GPU")
        self.ram_val_label = _stat_block("RAM")
        self.net_down_label = _stat_block("NET ▼")
        self.net_up_label = _stat_block("NET ▲")

    def _build_reactor_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(parent, bg=BG_PANEL, highlightbackground=LINE_DIM,
                          highlightthickness=1)
        panel.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(panel, bg=BG_PANEL, highlightthickness=0,
                                 width=320, height=320)
        self.canvas.pack(pady=(20, 4))

        self.status_label = tk.Label(
            panel, textvariable=self.status_text, fg=ARC_CYAN, bg=BG_PANEL,
            font=FONT_LABEL,
        )
        self.status_label.pack(pady=(0, 4))

        divider = tk.Frame(panel, bg=LINE_DIM, height=1)
        divider.pack(fill="x", padx=20, pady=12)

        info_title = tk.Label(panel, text="SESSION MEMORY", fg=TEXT_FAINT,
                               bg=BG_PANEL, font=FONT_MONO_SM)
        info_title.pack(anchor="w", padx=20)

        self.mem_app = self._info_row(panel, "App")
        self.mem_contact = self._info_row(panel, "Contact")
        self.mem_intent = self._info_row(panel, "Last action")

        divider2 = tk.Frame(panel, bg=LINE_DIM, height=1)
        divider2.pack(fill="x", padx=20, pady=12)

        model_title = tk.Label(panel, text="MODELS ONLINE", fg=TEXT_FAINT,
                                bg=BG_PANEL, font=FONT_MONO_SM)
        model_title.pack(anchor="w", padx=20)
        tk.Label(panel, text="Brain:  llama-3.3-70b-versatile", fg=TEXT_SECONDARY,
                  bg=BG_PANEL, font=FONT_MONO_SM).pack(anchor="w", padx=20, pady=(4, 0))
        tk.Label(panel, text="Parser: llama-3.1-8b-instant", fg=TEXT_SECONDARY,
                  bg=BG_PANEL, font=FONT_MONO_SM).pack(anchor="w", padx=20, pady=(2, 0))
        tk.Label(panel, text="Search: DuckDuckGo + Google News"
                  + (" + Tavily" if os.environ.get("TAVILY_API_KEY") else ""),
                  fg=TEXT_SECONDARY, bg=BG_PANEL, font=FONT_MONO_SM
                  ).pack(anchor="w", padx=20, pady=(2, 0))
        # Visual agent status
        if _HAS_OLLAMA:
            vision_text = "Vision: Ollama gemma3:4b (Local) ✓"
            vision_color = ARC_CYAN
        elif _HAS_VISION:
            vision_text = "Vision: gemini-2.5-flash-lite ✓"
            vision_color = ARC_CYAN
        else:
            vision_text = "Vision: OFF (Ollama/Gemini)"
            vision_color = ALERT_RED

        tk.Label(panel, text=vision_text,
                  fg=vision_color, bg=BG_PANEL, font=FONT_MONO_SM
                  ).pack(anchor="w", padx=20, pady=(2, 0))
        tk.Label(panel, text="Agent:  visual + closed-loop ✓" if _HAS_VISION
                  else "Agent:  text-only, closed-loop",
                  fg=ARC_CYAN if _HAS_VISION else TEXT_SECONDARY,
                  bg=BG_PANEL, font=FONT_MONO_SM
                  ).pack(anchor="w", padx=20, pady=(2, 8))

    def _info_row(self, parent, label: str) -> tk.Label:
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill="x", padx=20, pady=2)
        tk.Label(row, text=f"{label}:", fg=TEXT_FAINT, bg=BG_PANEL,
                  font=FONT_MONO_SM, width=9, anchor="w").pack(side="left")
        val = tk.Label(row, text="—", fg=TEXT_SECONDARY, bg=BG_PANEL,
                        font=FONT_MONO_SM, anchor="w")
        val.pack(side="left", fill="x", expand=True)
        return val

    def _build_transcript_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(parent, bg=BG_PANEL, highlightbackground=LINE_DIM,
                          highlightthickness=1)
        panel.pack(fill="both", expand=True)

        header = tk.Label(panel, text="TRANSCRIPT", fg=TEXT_FAINT, bg=BG_PANEL,
                           font=FONT_MONO_SM, anchor="w")
        header.pack(fill="x", padx=14, pady=(10, 4))

        self.transcript = scrolledtext.ScrolledText(
            panel, bg=BG_PANEL, fg=TEXT_PRIMARY, font=FONT_MONO,
            relief="flat", wrap="word", state="disabled", padx=14, pady=8,
            insertbackground=ARC_CYAN,
        )
        self.transcript.pack(fill="both", expand=True, padx=2, pady=(0, 2))

        self.transcript.tag_config("user", foreground=ARC_CYAN_GLOW, font=FONT_MONO)
        self.transcript.tag_config("jarvis", foreground=ARC_CYAN, font=FONT_MONO)
        self.transcript.tag_config("system", foreground=TEXT_FAINT, font=FONT_MONO_SM)
        self.transcript.tag_config("error", foreground=ALERT_RED, font=FONT_MONO)
        self.transcript.tag_config("detail", foreground=TEXT_SECONDARY, font=FONT_MONO_SM)

    # ------------------------------------------------------------------ #
    # Reactor HUD animation — multi-ring core + satellite gauges +
    # sparkline + tick strip, styled after a sci-fi heads-up-display.
    # Every state (IDLE/LISTENING/THINKING/SEARCHING/EXECUTING/SPEAKING/
    # ERROR) gets a visibly distinct color, rotation speed, and ring
    # pattern via STATE_STYLE, so at a glance you can tell what JARVIS
    # is doing without reading the text label.
    # ------------------------------------------------------------------ #

    def _animate_ring(self) -> None:
        c = self.canvas
        c.delete("ring")
        cx, cy = 160, 148

        state = self.status_text.get()
        color, rot_speed, pulse_speed, _label = STATE_STYLE.get(
            state, STATE_STYLE["IDLE"])

        self._ring_phase = (self._ring_phase + rot_speed) % 360
        self._pulse_phase += pulse_speed
        pulse = (math.sin(self._pulse_phase / 20.0) + 1) / 2  # 0..1

        # ---- outer static tick ring (degree markings, like the ref image)
        self._draw_tick_ring(c, cx, cy, 138, 60, LINE_DIM, TEXT_FAINT)

        # ---- outer thin hex frame
        self._draw_hexagon(c, cx, cy, 122, LINE_DIM, width=1)

        # ---- rotating segmented ring — direction & density vary by state.
        # Specialist agents each get a visibly distinct SEGMENT PATTERN
        # (not just colour/speed), so the ring's "shape of motion" alone
        # tells you which agent is driving:
        #   WhatsApp -> many short quick segments (fast back-and-forth chat)
        #   Gmail    -> fewer, thicker, slower segments (longer-form reading)
        #   Files    -> sparse tick-like segments, near-static rotation
        #               (short deterministic shell ops, not a visual loop)
        if state == "AGENT_WHATSAPP":
            n_segments, seg_frac, width = 18, 0.55, 2
        elif state == "AGENT_GMAIL":
            n_segments, seg_frac, width = 8, 0.7, 4
        elif state == "AGENT_FILES":
            n_segments, seg_frac, width = 30, 0.18, 2
        elif state == "AGENT_RECOVERY":
            n_segments, seg_frac, width = 6, 0.4, 5  # sparse, thick, fast -- reads as "alert, actively fixing"
        elif state in ("SEARCHING", "EXECUTING"):
            n_segments, seg_frac, width = 14, 0.55, 2
        else:
            n_segments, seg_frac, width = 10, 0.55, 2
        self._draw_segments(c, cx, cy, 104, color, self._ring_phase,
                             n_segments, seg_frac=seg_frac, width=width)

        # ---- second counter-rotating dashed ring (only visibly active
        # outside idle — reinforces "something is happening")
        if state != "IDLE":
            self._draw_segments(c, cx, cy, 86, color,
                                 -self._ring_phase * 1.6, 24, seg_frac=0.3, width=2)

        # ---- error state gets an extra warning chevron ring
        if state == "ERROR":
            self._draw_segments(c, cx, cy, 116, ALERT_RED, self._ring_phase * 2.2,
                                 8, seg_frac=0.15, width=4)

        # ---- specialist-agent states get a small filled hex badge at the
        # ring's top, in the agent's own colour, so it's identifiable even
        # to someone glancing without reading the status label.
        if state in ("AGENT_WHATSAPP", "AGENT_GMAIL", "AGENT_FILES", "AGENT_RECOVERY"):
            self._draw_hexagon(c, cx, cy - 138, 9, color, width=2)

        # ---- core glow disc, radius pulses with state
        base_r = 42
        r = base_r + pulse * 10
        for rr in (r, r * 0.7):
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=color,
                          width=1, tags="ring")
        c.create_oval(cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55,
                       fill=color, outline="", tags="ring")
        c.create_oval(cx - r * 0.22, cy - r * 0.22, cx + r * 0.22, cy + r * 0.22,
                       fill="#FFFFFF", outline="", tags="ring")

        # ---- satellite gauge rings (top-right trio, like the ref image)
        self._draw_satellite_gauges(c, state, color)

        # ---- sparkline (network activity, bottom of canvas)
        self._draw_sparkline(c)

        # ---- HUD corner brackets
        self._draw_corner_brackets(c)

        self.root.after(45, self._animate_ring)

    def _draw_hexagon(self, c: tk.Canvas, cx, cy, r, color, width=1) -> None:
        pts = []
        for i in range(6):
            ang = math.pi / 3 * i - math.pi / 2
            pts.extend([cx + r * math.cos(ang), cy + r * math.sin(ang)])
        c.create_polygon(*pts, outline=color, fill="", width=width, tags="ring")

    def _draw_segments(self, c: tk.Canvas, cx, cy, r, color, phase, n,
                        seg_frac=0.55, width=3) -> None:
        seg_len = 360 / n * seg_frac
        for i in range(n):
            start = (360 / n * i + phase) % 360
            c.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=start, extent=seg_len, style="arc",
                outline=color, width=width, tags="ring",
            )

    def _draw_tick_ring(self, c: tk.Canvas, cx, cy, r, n, color, label_color) -> None:
        """Thin radial tick marks around the outer rim, every 5th one longer
        with a tiny numeric label — echoes the reference HUD's dial rim."""
        for i in range(n):
            ang = math.pi * 2 * i / n - math.pi / 2
            long_tick = (i % 5 == 0)
            inner_r = r - (14 if long_tick else 7)
            x1, y1 = cx + inner_r * math.cos(ang), cy + inner_r * math.sin(ang)
            x2, y2 = cx + r * math.cos(ang), cy + r * math.sin(ang)
            c.create_line(x1, y1, x2, y2, fill=color,
                          width=2 if long_tick else 1, tags="ring")

    def _draw_satellite_gauges(self, c: tk.Canvas, state: str, color: str) -> None:
        """Three small independently-spinning ring gauges, upper-right of
        the main reactor — decorative HUD detail matching the reference
        image's '01 / 02 / 03' dial cluster, each reflecting a different
        live signal so they're not purely decorative."""
        stats = self._latest_stats
        gx, gy0, spacing, rad = 292, 40, 46, 16

        vals = [
            (stats.cpu_percent if stats else None, ARC_CYAN),
            (stats.gpu_percent if stats else None, SEARCH_VIOLET),
            (stats.ram_percent if stats else None, GOLD_ACCENT),
        ]
        for i, (pct, gcolor) in enumerate(vals):
            gy = gy0 + i * spacing
            c.create_oval(gx - rad, gy - rad, gx + rad, gy + rad,
                          outline=LINE_DIM, width=1, tags="ring")
            frac = (pct or 0) / 100.0
            extent = -frac * 360
            active_color = gcolor if state != "IDLE" else ARC_CYAN_DIM
            c.create_arc(gx - rad, gy - rad, gx + rad, gy + rad,
                        start=90, extent=extent, style="arc",
                        outline=active_color, width=3, tags="ring")
            txt = f"{pct:.0f}" if pct is not None else "--"
            c.create_text(gx, gy, text=txt, fill=TEXT_SECONDARY,
                          font=FONT_MONO_XS, tags="ring")

    def _draw_sparkline(self, c: tk.Canvas) -> None:
        """Live network-download sparkline along the bottom of the canvas."""
        hist = self._net_history
        if not hist:
            return
        x0, y0, w, h = 14, 292, 300, 22
        peak = max(max(hist), 1.0)
        step = w / max(len(hist) - 1, 1)
        pts = []
        for i, v in enumerate(hist):
            x = x0 + i * step
            y = y0 - (v / peak) * h
            pts.extend([x, y])
        if len(pts) >= 4:
            c.create_line(*pts, fill=ARC_CYAN_DIM, width=1, tags="ring", smooth=True)

    def _draw_corner_brackets(self, c: tk.Canvas) -> None:
        """Thin L-shaped brackets in the canvas corners — classic HUD framing."""
        size = 16
        w, h = 320, 320
        pad = 4
        for (x, y, dx, dy) in (
            (pad, pad, 1, 1), (w - pad, pad, -1, 1),
            (pad, h - pad, 1, -1), (w - pad, h - pad, -1, -1),
        ):
            c.create_line(x, y, x + dx * size, y, fill=ARC_CYAN_DIM, width=2, tags="ring")
            c.create_line(x, y, x, y + dy * size, fill=ARC_CYAN_DIM, width=2, tags="ring")

    # ------------------------------------------------------------------ #
    # UI thread-safety helpers
    # ------------------------------------------------------------------ #

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                fn = self._ui_queue.get_nowait()
                fn()
        except queue.Empty:
            pass
        self.root.after(40, self._poll_ui_queue)

    def _post(self, fn) -> None:
        self._ui_queue.put(fn)

    def _poll_system_stats(self) -> None:
        """Background-sample CPU/GPU/RAM/network every second and push the
        result to the telemetry strip + reactor sparkline. Sampling runs
        off-thread so a slow `nvidia-smi` call never stalls the UI."""
        def _sample_and_post():
            try:
                stats = self._sys_monitor.sample()
            except Exception:
                stats = None
            if stats is not None:
                self._post(lambda: self._apply_system_stats(stats))
        threading.Thread(target=_sample_and_post, daemon=True).start()
        self.root.after(1000, self._poll_system_stats)

    def _apply_system_stats(self, stats) -> None:
        self._latest_stats = stats
        self.cpu_val_label.config(text=stats.cpu_str())
        self.gpu_val_label.config(text=stats.gpu_str())
        self.ram_val_label.config(text=stats.ram_str())
        self.net_down_label.config(text=stats.net_down_str())
        self.net_up_label.config(text=stats.net_up_str())

        down_kbps = (stats.net_down_bps or 0.0) / 1024.0
        self._net_history.append(down_kbps)
        self._net_history = self._net_history[-40:]

    def set_status(self, text: str) -> None:
        self._post(lambda: self.status_text.set(text))

    def append_transcript(self, speaker_label: str, text: str, tag: str) -> None:
        def _do():
            self.transcript.configure(state="normal")
            self.transcript.insert("end", f"{speaker_label}  ", (tag,))
            self.transcript.insert("end", f"{text}\n\n", (tag,))
            self.transcript.see("end")
            self.transcript.configure(state="disabled")
        self._post(_do)

    def append_detail(self, text: str) -> None:
        def _do():
            self.transcript.configure(state="normal")
            self.transcript.insert("end", f"   ↳ {text}\n\n", ("detail",))
            self.transcript.see("end")
            self.transcript.configure(state="disabled")
        self._post(_do)

    def update_memory_panel(self) -> None:
        def _do():
            self.mem_app.config(text=self.memory.last_app or "—")
            self.mem_contact.config(text=self.memory.last_contact or "—")
            self.mem_intent.config(text=self.memory.last_intent or "—")
        self._post(_do)

    def _clear_placeholder(self, _event=None) -> None:
        if self._placeholder_active:
            self.entry.delete(0, "end")
            self._placeholder_active = False

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #

    def _startup_sequence(self) -> None:
        self.set_status("INITIALIZING")
        self.append_transcript("SYSTEM", "Booting JARVIS interface…", "system")

        # Voice engine
        problems = voice_io.check_voice_dependencies()
        if problems:
            self.append_transcript("SYSTEM",
                "Voice features limited — " + "; ".join(problems), "system")
        else:
            try:
                self.listener = voice_io.SpeechListener()
                self.speaker = voice_io.SpeechSpeaker()
                self.append_transcript("SYSTEM", "Voice engine ready.", "system")
            except voice_io.VoiceIOError as e:
                self.append_transcript("SYSTEM", f"Voice engine unavailable: {e}", "system")

        # Phone connection (non-fatal if it fails — user can still chat / retry)
        try:
            bridge = ADBBridge(preferred="auto")
            self.controller = AndroidController(bridge)
            info = self.controller.connect()
            self._post(lambda: self.device_text.set(
                f"{info.model}  ·  Android {info.android_version}  ·  {info.connection.upper()}"))
            self.append_transcript("SYSTEM", f"Phone connected: {info.model}.", "system")
        except DeviceNotConnected as e:
            self._post(lambda: self.device_text.set("No device connected"))
            self.append_transcript("SYSTEM",
                "No Android device connected yet. Plug in via USB or use WiFi ADB — "
                "you can still chat with me in the meantime.", "system")
        except ADBError as e:
            self._post(lambda: self.device_text.set("ADB not available"))
            self.append_transcript("SYSTEM", f"ADB error: {e}", "system")

        self.set_status("IDLE")
        greeting = jarvis_brain.jarvis_greeting()
        self.append_transcript("JARVIS", greeting, "jarvis")
        self._speak_async(greeting)

        if self.always_listening.get():
            self._start_listen_loop()

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #

    def _on_submit_text(self, _event=None) -> None:
        if self._placeholder_active:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        threading.Thread(target=self._handle_utterance, args=(text,), daemon=True).start()

    def _on_mic_click(self) -> None:
        if self.busy:
            return
        threading.Thread(target=self._listen_and_handle_once, daemon=True).start()

    def _on_toggle_always_listen(self) -> None:
        if self.always_listening.get():
            self._start_listen_loop()
        else:
            self._listen_loop_stop.set()

    def _start_listen_loop(self) -> None:
        self._listen_loop_stop.clear()
        threading.Thread(target=self._always_listen_worker, daemon=True).start()

    def _always_listen_worker(self) -> None:
        if not self.listener:
            self.append_transcript("SYSTEM", "Microphone not available for always-listening mode.", "system")
            self._post(lambda: self.always_listening.set(False))
            return
        self.append_transcript("SYSTEM",
            f"Always-listening enabled. Say '{voice_io.WAKE_WORD}' followed by your request.",
            "system")
        while not self._listen_loop_stop.is_set():
            if self.busy:
                time.sleep(0.3)
                continue
            text, err = self.listener.listen_once(timeout=4.0, phrase_time_limit=8.0)
            if err in ("listen_timeout", "could_not_understand", None) and not text:
                continue
            if not text:
                continue
            lowered = text.lower()
            if voice_io.WAKE_WORD not in lowered:
                continue
            # strip the wake word out and handle the remainder (or, if
            # nothing follows, prompt the user for the actual request)
            remainder = lowered.split(voice_io.WAKE_WORD, 1)[-1].strip(" ,.")
            if not remainder:
                self._speak_async("Yes, sir?")
                follow, ferr = self.listener.listen_once(timeout=5.0, phrase_time_limit=10.0)
                if not follow:
                    continue
                remainder = follow
            self._handle_utterance(remainder)

    def _listen_and_handle_once(self) -> None:
        if not self.listener:
            self.append_transcript("SYSTEM",
                "Microphone/voice recognition isn't available. "
                "Check requirements.txt install and mic permissions.", "system")
            return
        self.busy = True
        self.set_status("LISTENING")
        self._post(lambda: self.mic_btn.config(text="🎙  LISTENING…", state="disabled"))
        text, err = self.listener.listen_once(timeout=6.0, phrase_time_limit=12.0)
        self._post(lambda: self.mic_btn.config(text="🎙  LISTEN", state="normal"))
        self.busy = False

        if err == "listen_timeout":
            self.set_status("IDLE")
            self.append_transcript("SYSTEM", "Didn't hear anything.", "system")
            return
        if err == "could_not_understand":
            self.set_status("IDLE")
            self.append_transcript("SYSTEM", "Sorry, I couldn't make that out.", "system")
            return
        if err:
            self.set_status("ERROR")
            self.append_transcript("SYSTEM", err, "system")
            self.root.after(1500, lambda: self.set_status("IDLE"))
            return

        self._handle_utterance(text)

    # ------------------------------------------------------------------ #
    # Core pipeline: text -> (chat | command) -> response
    # ------------------------------------------------------------------ #

    def _handle_utterance(self, text: str) -> None:
        self.busy = True
        self.append_transcript("YOU", text, "user")
        self.set_status("THINKING")

        try:
            route = jarvis_brain.classify_utterance(text)
        except Exception:
            route = "command"

        if route == "chat":
            self._handle_chat(text)
        elif route == "search":
            self._handle_search(text)
        else:
            self._handle_command(text)

        self.busy = False
        self.set_status("IDLE")

    def _handle_chat(self, text: str) -> None:
        try:
            device_ctx = self.memory.context_block_for_llm()
            reply = jarvis_brain.jarvis_reply(text, self.conversation, device_context=device_ctx)
        except jarvis_brain.JarvisBrainError as e:
            reply = ("I'm having trouble reaching my language core right now. "
                      "Double-check GROQ_API_KEY_JARVIS in your .env file.")
            self.append_detail(str(e))

        self.append_transcript("JARVIS", reply, "jarvis")
        self.set_status("SPEAKING")
        self._speak_async(reply)

    def _handle_search(self, text: str) -> None:
        """Real-time internet lookup: DuckDuckGo + Google News RSS (both
        keyless) plus Tavily if TAVILY_API_KEY is set — merged, then
        handed to the 70B brain to paraphrase into a short spoken answer
        grounded in the actual retrieved snippets."""
        self.set_status("SEARCHING")
        self.append_detail(f"searching the web for: {text!r}")
        try:
            results = web_search(text)
        except WebSearchError as e:
            msg = "I couldn't reach the web for that search just now, sir."
            self.append_transcript("JARVIS", msg, "jarvis")
            self.append_detail(str(e))
            self._speak_async(msg)
            return

        raw_summary = summarize_for_speech(text, results)
        self.append_detail(raw_summary)

        try:
            prompt = (
                f"The user asked JARVIS to look this up: {text!r}\n\n"
                f"Here are the live search results retrieved just now:\n"
                f"{raw_summary}\n\n"
                f"Answer the user's question in 2-4 short spoken sentences, "
                f"grounded ONLY in the results above. If they conflict or "
                f"are unclear, say so briefly. Do not invent facts not in "
                f"the results."
            )
            spoken = jarvis_brain.jarvis_reply(prompt, conversation=None)
        except jarvis_brain.JarvisBrainError:
            spoken = raw_summary

        self.append_transcript("JARVIS", spoken, "jarvis")
        self.set_status("SPEAKING")
        self._speak_async(spoken)

    def _handle_command(self, raw_text: str) -> None:
        resolved = self.memory.resolve_references(raw_text)
        if resolved != raw_text:
            self.append_detail(f"resolved: {resolved!r}")

        if not self.controller:
            msg = "I don't have a phone connected right now, sir. Plug one in via USB and try again."
            self.append_transcript("JARVIS", msg, "jarvis")
            self._speak_async(msg)
            return

        # ---- Specialist agent routing --------------------------------
        # Before the general command_parser/llm_parser pipeline gets a
        # shot at it, see if this is squarely a WhatsApp/Gmail/Files
        # request. If so, a single specialist agent owns it end-to-end
        # (no other agent touches WhatsApp, etc.) and the general
        # pipeline is skipped entirely for this utterance.
        agent_result = self._try_specialist_agent(resolved, raw_text)
        if agent_result is not None:
            return

        self.set_status("PROCESSING")
        intents = None
        try:
            intents = parse_multi_command(resolved)
        except ParseError:
            pass

        if intents is None:
            try:
                context_block = self.memory.context_block_for_llm()
                intent = parse_with_llm(resolved, context_block=context_block)
                intents = [intent]
            except LLMParseError as e:
                msg = "I couldn't work out how to do that on the phone."
                self.append_transcript("JARVIS", msg, "jarvis")
                self.append_detail(str(e))
                self._speak_async(msg)
                return
            except ValueError as e:
                msg = f"I hit a snag: {e}"
                self.append_transcript("JARVIS", msg, "jarvis")
                self._speak_async("I hit a snag trying to do that.")
                return

        self.set_status("EXECUTING")
        last_result = ""
        try:
            for intent in intents:
                intent = self.memory.enrich_intent(intent)

                # ---- Route complex/ambiguous multi-step goals to the
                # closed-loop TaskAgent instead of blindly firing a
                # pre-baked step list. This is the "track it to the end"
                # path: perceive -> decide -> act -> verify, repeated
                # against the REAL screen until the goal is confirmed
                # done (or honestly reported as failed/stuck).
                if intent.name == "open_app_and_do":
                    result = self._run_task_agent(raw_text, intent)
                else:
                    result = execute_intent(self.controller, intent)

                self.memory.update(raw_text, intent, result)
                last_result = result
                self.append_detail(f"[{intent.name}] {result}")
        except (ADBError, ParseError, ValueError, KeyError) as e:
            msg = f"That didn't go through: {e}"
            self.append_transcript("JARVIS", msg, "jarvis")
            self._speak_async("Sorry sir, that command didn't go through.")
            self.update_memory_panel()
            return

        self.update_memory_panel()

        try:
            spoken = jarvis_brain.narrate_result(raw_text, last_result)
        except Exception:
            spoken = last_result

        self.append_transcript("JARVIS", spoken, "jarvis")
        self.set_status("SPEAKING")
        self._speak_async(spoken)

    def _try_specialist_agent(self, resolved_text: str, raw_text: str) -> Optional[AgentResult]:
        """
        Route through ToolSelectorAgent first. Returns the AgentResult (and
        has already spoken/logged it) if a specialist agent claimed this
        utterance; returns None if no specialist agent's territory matched,
        so the caller falls through to the general command pipeline
        unchanged.
        """
        # Best-effort location signal for LearningAgent -- see the honesty
        # note in learning_agent.py: this is the WiFi SSID, not GPS.

        def _on_status(agent_id: str, ring_state: str) -> None:
            self.set_status(ring_state)
            self.append_detail(f"🔧 routed to {agent_id} agent")

        def _on_step(rec: StepRecord) -> None:
            icon = {"complete": "✓", "in_progress": "…", "stuck": "⚠"}.get(
                rec.verify_state, "…"
            )
            note = f" — {rec.verify_note}" if rec.verify_note else ""
            self.append_detail(
                f"  {icon} step {rec.step_num}: "
                f"{self._describe_agent_action(rec.action)}{note}"
            )

        selector = ToolSelectorAgent(
            self.controller, on_step=_on_step, on_status=_on_status,
            on_detail=self.append_detail, learner=self.learner,
        )

        # Cheap check first (no controller/agent instantiation) so
        # unrelated utterances ("what's the weather") pay zero extra cost.
        if selector.select_agent_id(resolved_text) is None:
            return None

        # Sampled here, not in the per-second telemetry loop, so it only
        # costs an extra ADB round-trip when we're already about to talk
        # to the device for a real specialist-agent command.
        try:
            ssid = self.controller.get_wifi_ssid()
            self.learner.record_network_seen(ssid)
        except Exception:
            pass

        try:
            result = selector.route(resolved_text)
        except ADBError as e:
            msg = f"That didn't go through: {e}"
            self.append_transcript("JARVIS", msg, "jarvis")
            self._speak_async("Sorry sir, that agent hit a device error.")
            self.set_status("ERROR")
            self.root.after(1500, lambda: self.set_status("IDLE"))
            return None  # let it surface as handled either way; nothing left to do

        if result is None:
            return None  # not actually a specialist-agent request after all

        # ConversationMemory.update() expects an Intent-shaped object
        # (name + params); specialist agents don't produce one, so build
        # a minimal duck-typed stand-in carrying the app name as context.
        class _AgentPseudoIntent:
            name = f"agent_{result.agent_id}"
            params = {"app": result.agent_id}

        self.memory.update(raw_text, _AgentPseudoIntent(), result.summary)
        self.update_memory_panel()

        try:
            spoken = jarvis_brain.narrate_result(raw_text, result.summary)
        except Exception:
            spoken = result.summary

        self.append_transcript("JARVIS", spoken, "jarvis")
        self.set_status("SPEAKING")
        self._speak_async(spoken)
        self.root.after(1200, lambda: self.set_status("IDLE"))
        return result

    def _run_task_agent(self, raw_text: str, intent) -> str:
        """
        Drive a visual, autonomously-planned perceive → decide → act → verify
        loop for a multi-step goal.

        Uses VisualTaskAgent when GEMINI_API_KEY is set:
          - AutonomousPlanner decomposes the goal into SubTasks first
          - GeminiVision sees real screenshots (not just UI text)
          - Visual verification confirms each step
          - tap_visual action can tap icon-only buttons

        Falls back to legacy text-only TaskAgent when Gemini is unavailable.
        """
        goal_text = intent.raw_text or raw_text

        self.set_status("PLANNING" if _HAS_VISION else "AGENT")
        self.append_detail(
            f"{'👁 visual agent' if _HAS_VISION else '📋 agent'} goal: "
            f"{goal_text!r} — planning…"
        )

        def _on_step(rec: StepRecord) -> None:
            icon = {"complete": "✓", "in_progress": "…", "stuck": "⚠"}.get(
                rec.verify_state, "…"
            )
            note = f" — {rec.verify_note}" if rec.verify_note else ""
            vis_icon = " 👁" if rec.screenshot_b64 else ""  # visual step indicator
            self.append_detail(
                f"  {icon} step {rec.step_num}: "
                f"{self._describe_agent_action(rec.action)}{vis_icon}{note}"
            )

        def _on_plan(plan) -> None:
            """Show the plan breakdown in the transcript."""
            self.set_status("AGENT")
            self.append_detail(
                f"📋 Plan ({plan.estimated_complexity}): "
                f"{len(plan.subtasks)} sub-task(s)"
            )
            for i, st in enumerate(plan.subtasks, 1):
                self.append_detail(f"   {i}. {st.description}")
            if plan.risk_notes:
                for r in plan.risk_notes:
                    self.append_detail(f"   ⚠ {r}")

        try:
            if _HAS_VISION:
                agent = VisualTaskAgent(
                    self.controller,
                    on_step=_on_step,
                    on_plan=_on_plan,
                )
            else:
                agent = TaskAgent(self.controller, on_step=_on_step)
            run = agent.run(goal_text)
        except TaskAgentError as e:
            return f"Task automation couldn't start: {e}"

        if run.success:
            return f"Task completed and verified: {run.final_summary}"
        else:
            steps_taken = len(run.steps)
            return (
                f"Task NOT completed after {steps_taken} step(s) — "
                f"{run.aborted_reason or 'goal was not confirmed done'}. "
                f"This was verified against the actual screen, not assumed."
            )

    @staticmethod
    def _describe_agent_action(action: dict) -> str:
        name = action.get("action", "?")
        if name == "tap_text":
            return f"tap '{action.get('text', '')}'"
        if name == "tap_xy":
            return f"tap ({action.get('x')},{action.get('y')})"
        if name == "tap_visual":
            return f"tap visual '{action.get('description', '')}'"
        if name == "type":
            return f"type {action.get('text', '')!r}"
        if name == "key":
            return f"press {action.get('key', '')}"
        if name == "swipe":
            return f"swipe {action.get('direction', '')}"
        if name == "wait":
            return f"wait {action.get('seconds', 1)}s"
        if name == "open_app":
            return f"open {action.get('app', '')}"
        if name == "done":
            return "✓ confirm done"
        if name == "give_up":
            return "✗ give up"
        return name

    # ------------------------------------------------------------------ #
    # TTS helper
    # ------------------------------------------------------------------ #

    def _speak_async(self, text: str) -> None:
        if not self.voice_enabled.get() or not self.speaker:
            return

        def _run():
            self.speaker.speak(text, blocking=True)
            self.set_status("IDLE")

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def _show_help(self) -> None:
        messagebox.showinfo(
            "J.A.R.V.I.S. — Help",
            "Speak naturally or type a request.\n\n"
            "Examples:\n"
            "  • \"open whatsapp and send hi to mom\"\n"
            "  • \"take a screenshot\"\n"
            "  • \"what's the weather like\" (general chat)\n"
            "  • \"call dad\"\n"
            "  • \"turn on the flashlight\"\n"
            "  • \"look up the latest news on the stock market\" "
            "(live web search)\n"
            "  • \"what's happening with the Mars rover\" (live web search)\n"
            "  • \"open instagram and follow the top post's creator\" "
            "(tracked multi-step task)\n\n"
            "Click the mic once for a single command, or enable "
            "'Always listening' and say the wake word "
            f"('{voice_io.WAKE_WORD}') followed by your request.\n\n"
            "The top-left strip shows live CPU / GPU / RAM / network "
            "throughput. The reactor ring's color and spin pattern change "
            "with what JARVIS is doing (listening, thinking, searching the "
            "web, automating a multi-step task, executing a phone command, "
            "speaking).\n\n"
            "Multi-step requests ('open X and do Y and Z') are handled by "
            "a closed-loop agent: it reads the REAL current screen before "
            "every action, decides the next single step, acts, then "
            "re-reads the screen to verify progress — repeating until the "
            "goal is actually confirmed done, or it honestly reports it "
            "got stuck/failed. Watch the transcript's indented lines for "
            "live step-by-step progress.\n\n"
            "Add contacts in contacts.json. API keys (Groq + optional "
            "Tavily for better search) go in the .env file — copy "
            ".env.example to .env and fill them in."
        )


def main() -> None:
    root = tk.Tk()
    try:
        root.iconbitmap(default="")
    except Exception:
        pass
    app = JarvisApp(root)

    def on_close():
        app._listen_loop_stop.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()