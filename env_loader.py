"""
env_loader.py
=============
Minimal, dependency-free ".env" file loader.

Reads a `.env` file (same folder as this file, i.e. the project root) and
sets any KEY=VALUE pairs found into os.environ, WITHOUT overwriting
variables that are already set in the real environment (so real env vars
always take priority over the .env file).

We intentionally don't require `python-dotenv` for this tiny bit of
parsing — one less dependency, zero extra install step.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"

_loaded = False


def load_env(path: Path | None = None) -> dict:
    """
    Parse the .env file and load any keys not already present in
    os.environ. Safe to call multiple times (only parses once per process
    unless a custom path is given).
    """
    global _loaded
    target = path or ENV_PATH
    if _loaded and path is None:
        return {}
    if not target.exists():
        _loaded = True
        return {}

    values: dict[str, str] = {}
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # strip matching surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        values[key] = value
        os.environ.setdefault(key, value)

    if path is None:
        _loaded = True
    return values


if __name__ == "__main__":
    loaded = load_env()
    print(f"Loaded {len(loaded)} variable(s) from {ENV_PATH}")
    for k in loaded:
        print(f"  {k}")
