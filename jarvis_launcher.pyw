"""
jarvis_launcher.pyw
====================
Double-click entry point for JARVIS with ZERO terminal/console window.

On Windows, files with the ".pyw" extension are run by pythonw.exe
instead of python.exe, which does not open a console window at all —
this is the standard, correct way to launch a GUI-only Python app.

Just double-click this file (or make a desktop shortcut to it) to start
JARVIS. If something goes wrong before the GUI window can open (e.g. a
missing dependency), a message box is shown instead of silently failing
with no visible output.

If you're on macOS/Linux, you can also just run:
    python3 jarvis_gui.py
directly — a terminal is optional there, it just won't be *used* for
interaction once the window opens.
"""

from __future__ import annotations

import sys
import traceback


def _fatal_message_box(title: str, message: str) -> None:
    """Show an error dialog even if the main app never got far enough
    to build one itself (e.g. a missing stdlib/tkinter install)."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        # Absolute last resort — if even Tkinter itself is unavailable,
        # there is no GUI-safe way left to signal the user; write to a
        # small log file next to this launcher so it isn't silent.
        try:
            from pathlib import Path
            log_path = Path(__file__).parent / "jarvis_error.log"
            log_path.write_text(f"{title}\n\n{message}", encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    try:
        import jarvis_gui
        jarvis_gui.main()
    except ImportError as e:
        _fatal_message_box(
            "J.A.R.V.I.S. — Missing dependency",
            f"JARVIS couldn't start because a required package is missing:\n\n"
            f"{e}\n\n"
            f"Open a terminal in this folder once and run:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Then double-click this launcher again.",
        )
        sys.exit(1)
    except Exception as e:
        _fatal_message_box(
            "J.A.R.V.I.S. — Startup error",
            f"JARVIS hit an unexpected error on startup:\n\n{e}\n\n"
            f"{traceback.format_exc()}",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
