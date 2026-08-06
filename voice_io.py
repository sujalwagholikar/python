"""
voice_io.py
===========
Voice input/output layer for JARVIS.

- Speech-to-text: SpeechRecognition, using Google's free web speech API
  (via recognizer.recognize_google) through the default microphone.
- Text-to-speech: gTTS (Google Text-to-Speech) renders an MP3, played back
  with pygame's mixer (cross-platform, no external "mpg123"/"ffplay"
  binary required, unlike playsound on some platforms).

Both directions are wrapped so they NEVER raise into GUI code:
  - listen_once()  returns (text, error) — error is None on success.
  - speak(text)    returns True/False and never throws.

Everything that touches the microphone or plays audio runs in whatever
thread calls it — the GUI layer is responsible for running these off the
Tkinter main thread (see jarvis_gui.py), so the UI never freezes while
listening or talking.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# Lazy imports — SpeechRecognition/gTTS/pygame are somewhat heavy and we
# want a clear, actionable error message if they're missing rather than a
# crash at module import time before the GUI can even show a message box.
# --------------------------------------------------------------------------- #

_sr = None
_gtts = None
_pygame = None
_import_errors: dict[str, str] = {}


def _ensure_imports() -> None:
    global _sr, _gtts, _pygame
    if _sr is None:
        try:
            import speech_recognition as sr  # type: ignore
            _sr = sr
        except ImportError as e:
            _import_errors["speech_recognition"] = str(e)
    if _gtts is None:
        try:
            from gtts import gTTS  # type: ignore
            _gtts = gTTS
        except ImportError as e:
            _import_errors["gtts"] = str(e)
    if _pygame is None:
        try:
            import pygame  # type: ignore
            _pygame = pygame
        except ImportError as e:
            _import_errors["pygame"] = str(e)


VOICE_LANG = os.environ.get("JARVIS_VOICE_LANG", "en")
VOICE_TLD = os.environ.get("JARVIS_VOICE_TLD", "co.uk")  # co.uk accent reads a bit more "JARVIS"
WAKE_WORD = os.environ.get("JARVIS_WAKE_WORD", "jarvis").strip().lower()


class VoiceIOError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Speech-to-text
# --------------------------------------------------------------------------- #

class SpeechListener:
    """
    Thin wrapper around SpeechRecognition's Microphone + Recognizer, with
    ambient-noise calibration done once and reused, and clear, specific
    error strings for the most common failure modes (no mic, no internet,
    couldn't understand audio, timeout).
    """

    def __init__(self, energy_threshold: Optional[int] = None):
        _ensure_imports()
        if _sr is None:
            raise VoiceIOError(
                "speech_recognition package not installed. "
                "Run: pip install SpeechRecognition"
            )
        self._sr = _sr
        self.recognizer = _sr.Recognizer()
        if energy_threshold is not None:
            self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self._mic_lock = threading.Lock()
        self._calibrated = False

    def calibrate(self, duration: float = 1.0) -> None:
        try:
            with self._sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            self._calibrated = True
        except OSError as e:
            raise VoiceIOError(
                f"No microphone found / accessible: {e}. "
                f"Check OS microphone permissions and that a mic is connected."
            ) from e

    def listen_once(
        self,
        timeout: float = 6.0,
        phrase_time_limit: float = 12.0,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Listen for a single utterance and transcribe it.
        Returns (text, error). Exactly one of the two is None on return.
        Never raises.
        """
        with self._mic_lock:
            try:
                if not self._calibrated:
                    self.calibrate()
                with self._sr.Microphone() as source:
                    audio = self.recognizer.listen(
                        source, timeout=timeout, phrase_time_limit=phrase_time_limit
                    )
            except OSError as e:
                return None, f"Microphone error: {e}"
            except self._sr.WaitTimeoutError:
                return None, "listen_timeout"

        try:
            text = self.recognizer.recognize_google(audio, language="en-US")
            return text, None
        except self._sr.UnknownValueError:
            return None, "could_not_understand"
        except self._sr.RequestError as e:
            return None, f"Speech recognition service error (check internet): {e}"


# --------------------------------------------------------------------------- #
# Text-to-speech
# --------------------------------------------------------------------------- #

class SpeechSpeaker:
    """
    Renders text to speech via gTTS and plays it back with pygame's
    mixer. A per-instance lock prevents overlapping playback if speak()
    is called again before the previous utterance finished (callers can
    also call stop() to interrupt immediately, e.g. if the user starts
    talking over JARVIS).
    """

    def __init__(self, lang: str = VOICE_LANG, tld: str = VOICE_TLD):
        _ensure_imports()
        missing = [k for k in ("gtts", "pygame") if k in _import_errors]
        if missing:
            raise VoiceIOError(
                f"Missing packages for text-to-speech: {', '.join(missing)}. "
                f"Run: pip install gTTS pygame"
            )
        self.lang = lang
        self.tld = tld
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._mixer_ready = False
        self._tmp_dir = Path(tempfile.gettempdir()) / "jarvis_tts"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_mixer(self) -> None:
        if not self._mixer_ready:
            _pygame.mixer.init()
            self._mixer_ready = True

    def speak(self, text: str, blocking: bool = True) -> bool:
        """
        Synthesize and play `text`. Returns True on success, False on any
        failure (never raises, so a TTS hiccup never crashes the app).
        If blocking=False, playback happens on a background thread and
        this returns True immediately once playback has started.
        """
        text = (text or "").strip()
        if not text:
            return False

        def _run() -> bool:
            with self._lock:
                self._stop_flag.clear()
                mp3_path = self._tmp_dir / f"utt_{int(time.time() * 1000)}.mp3"
                try:
                    tts = _gtts(text=text, lang=self.lang, tld=self.tld)
                    tts.save(str(mp3_path))
                    self._ensure_mixer()
                    _pygame.mixer.music.load(str(mp3_path))
                    _pygame.mixer.music.play()
                    while _pygame.mixer.music.get_busy():
                        if self._stop_flag.is_set():
                            _pygame.mixer.music.stop()
                            break
                        time.sleep(0.05)
                    return True
                except Exception:
                    return False
                finally:
                    try:
                        _pygame.mixer.music.unload()
                    except Exception:
                        pass
                    try:
                        mp3_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        if blocking:
            return _run()
        threading.Thread(target=_run, daemon=True).start()
        return True

    def stop(self) -> None:
        """Interrupt whatever is currently being spoken, if anything."""
        self._stop_flag.set()


# --------------------------------------------------------------------------- #
# Convenience: dependency check the GUI can call at startup
# --------------------------------------------------------------------------- #

def check_voice_dependencies() -> list[str]:
    """Returns a list of human-readable missing-dependency messages (empty if all OK)."""
    _ensure_imports()
    problems = []
    if "speech_recognition" in _import_errors:
        problems.append("SpeechRecognition is not installed (pip install SpeechRecognition)")
    if "gtts" in _import_errors:
        problems.append("gTTS is not installed (pip install gTTS)")
    if "pygame" in _import_errors:
        problems.append("pygame is not installed (pip install pygame)")
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        try:
            import pyaudio2  # type: ignore  # noqa: F401
        except ImportError:
            problems.append(
                "PyAudio is not installed — required for microphone access "
                "(pip install PyAudio; on Windows this installs as a prebuilt "
                "wheel automatically with recent pip)"
            )
    return problems
