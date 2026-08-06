"""
system_monitor.py
==================
Lightweight, dependency-tolerant live system telemetry for the JARVIS HUD:

  - CPU utilisation (%)
  - RAM utilisation (%)
  - GPU utilisation (%) — NVIDIA only, via `nvidia-smi` (no extra pip
    package required; falls back to "N/A" on any other GPU / no GPU)
  - Real-time network throughput, computed from the OS's cumulative
    byte counters sampled on an interval (the same technique Task
    Manager / `nload` / `iftop`-style tools use) — reported in KB/s or
    MB/s depending on magnitude.

Everything here is best-effort and never raises out of `sample()` —
a stat that can't be read comes back as None and the caller (the HUD)
just displays "N/A" for that field instead of crashing the app.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class SystemStats:
    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    gpu_percent: Optional[float] = None
    gpu_name: Optional[str] = None
    net_down_bps: Optional[float] = None   # bytes/sec, download
    net_up_bps: Optional[float] = None     # bytes/sec, upload

    def net_down_str(self) -> str:
        return _format_rate(self.net_down_bps)

    def net_up_str(self) -> str:
        return _format_rate(self.net_up_bps)

    def cpu_str(self) -> str:
        return f"{self.cpu_percent:.0f}%" if self.cpu_percent is not None else "N/A"

    def ram_str(self) -> str:
        return f"{self.ram_percent:.0f}%" if self.ram_percent is not None else "N/A"

    def gpu_str(self) -> str:
        return f"{self.gpu_percent:.0f}%" if self.gpu_percent is not None else "N/A"


def _format_rate(bytes_per_sec: Optional[float]) -> str:
    if bytes_per_sec is None:
        return "N/A"
    kbps = bytes_per_sec / 1024.0
    if kbps < 1024:
        return f"{kbps:.1f} KB/s"
    return f"{kbps / 1024.0:.2f} MB/s"


class SystemMonitor:
    """
    Call `sample()` on a timer (e.g. every 1s). Internally tracks the
    previous network byte counters + timestamp to compute an actual
    real-time rate rather than a cumulative total.
    """

    def __init__(self):
        self._last_net = None      # (bytes_recv, bytes_sent, timestamp)
        self._nvidia_smi_path = shutil.which("nvidia-smi")
        self._gpu_checked_unavailable = False

    # ------------------------------------------------------------------ #
    def sample(self) -> SystemStats:
        stats = SystemStats()

        # CPU + RAM
        if _HAS_PSUTIL:
            try:
                stats.cpu_percent = psutil.cpu_percent(interval=None)
            except Exception:
                pass
            try:
                stats.ram_percent = psutil.virtual_memory().percent
            except Exception:
                pass
        else:
            # Fallback: read /proc/loadavg as a crude proxy on Linux only.
            try:
                with open("/proc/loadavg") as f:
                    load1 = float(f.read().split()[0])
                cpu_count = _cpu_count_fallback()
                stats.cpu_percent = min(100.0, (load1 / max(cpu_count, 1)) * 100)
            except Exception:
                pass

        # Network throughput (real-time delta, not cumulative)
        stats.net_down_bps, stats.net_up_bps = self._sample_network()

        # GPU (NVIDIA only — degrades to N/A everywhere else)
        stats.gpu_percent, stats.gpu_name = self._sample_gpu()

        return stats

    # ------------------------------------------------------------------ #
    def _sample_network(self):
        if not _HAS_PSUTIL:
            return None, None
        try:
            counters = psutil.net_io_counters()
            now = time.monotonic()
            recv, sent = counters.bytes_recv, counters.bytes_sent
        except Exception:
            return None, None

        if self._last_net is None:
            self._last_net = (recv, sent, now)
            return None, None  # no delta yet on first sample

        prev_recv, prev_sent, prev_time = self._last_net
        dt = max(now - prev_time, 1e-6)
        down_bps = max(0.0, (recv - prev_recv) / dt)
        up_bps = max(0.0, (sent - prev_sent) / dt)
        self._last_net = (recv, sent, now)
        return down_bps, up_bps

    # ------------------------------------------------------------------ #
    def _sample_gpu(self):
        if self._gpu_checked_unavailable:
            return None, None
        if not self._nvidia_smi_path:
            self._gpu_checked_unavailable = True
            return None, None
        try:
            out = subprocess.run(
                [self._nvidia_smi_path,
                 "--query-gpu=utilization.gpu,name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode != 0 or not out.stdout.strip():
                self._gpu_checked_unavailable = True
                return None, None
            first_line = out.stdout.strip().splitlines()[0]
            util_str, _, name = first_line.partition(",")
            return float(util_str.strip()), name.strip()
        except Exception:
            self._gpu_checked_unavailable = True
            return None, None


def _cpu_count_fallback() -> int:
    try:
        import os
        return os.cpu_count() or 1
    except Exception:
        return 1
