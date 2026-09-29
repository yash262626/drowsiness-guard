"""
alarm.py  —  Multi-channel Alarm System for Drowsiness Guard
==============================================================
Channels:
  1. pygame audio    — plays alarm.wav (loud, persistent)
  2. OpenCV flash    — red screen flash overlay
  3. System beep     — os.system fallback if pygame unavailable
"""

import os
import time
import threading
import numpy as np
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
ALARM_PATH = BASE_DIR / "sounds" / "alarm.wav"

_alarm_thread = None
_stop_event   = threading.Event()
_pygame_ok    = False

# ─── Init pygame ─────────────────────────────────────────────────────────────

def init_audio():
    global _pygame_ok
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        _pygame_ok = True

        if not ALARM_PATH.exists():
            _generate_alarm_wav()
    except Exception as e:
        _pygame_ok = False
        print(f"⚠️  pygame audio unavailable ({e}). Will use system beep.")


def _generate_alarm_wav():
    """Programmatically generate a loud alarm .wav (no external file needed)."""
    try:
        import pygame
        import wave
        import struct
        import math

        ALARM_PATH.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = 44100
        duration    = 2.0       # seconds per cycle
        frequency   = 880       # Hz (A5 — piercing)
        amplitude   = 28000

        n_samples = int(sample_rate * duration)
        samples   = []

        for i in range(n_samples):
            t      = i / sample_rate
            # Two-tone siren: alternate between 880 Hz and 660 Hz every 0.5s
            freq   = frequency if int(t / 0.5) % 2 == 0 else 660
            val    = int(amplitude * math.sin(2 * math.pi * freq * t))
            samples.append(struct.pack("<h", val))   # little-endian signed short

        # Stereo: duplicate channel
        stereo = b"".join([s + s for s in samples])

        with wave.open(str(ALARM_PATH), "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(stereo)

        print(f"🔔  Alarm sound generated → {ALARM_PATH}")
    except Exception as e:
        print(f"⚠️  Could not generate alarm.wav: {e}")


# ─── Alarm Control ───────────────────────────────────────────────────────────

def start_alarm():
    """Start the alarm (non-blocking — runs in background thread)."""
    global _alarm_thread, _stop_event

    if _alarm_thread and _alarm_thread.is_alive():
        return   # already running

    _stop_event.clear()
    _alarm_thread = threading.Thread(target=_alarm_loop, daemon=True)
    _alarm_thread.start()


def stop_alarm():
    """Stop the alarm."""
    _stop_event.set()
    if _pygame_ok:
        try:
            import pygame
            pygame.mixer.stop()
        except Exception:
            pass


def is_alarm_active() -> bool:
    return bool(_alarm_thread and _alarm_thread.is_alive() and not _stop_event.is_set())


def _alarm_loop():
    """Background thread: loops alarm sound until stopped."""
    if _pygame_ok and ALARM_PATH.exists():
        try:
            import pygame
            sound = pygame.mixer.Sound(str(ALARM_PATH))
            sound.set_volume(1.0)
            while not _stop_event.is_set():
                sound.play()
                time.sleep(2.1)
            sound.stop()
            return
        except Exception:
            pass

    # Fallback: system beep
    while not _stop_event.is_set():
        try:
            os.system("echo -e '\a'")   # Linux/Mac terminal bell
        except Exception:
            pass
        time.sleep(1.0)


# ─── Visual Flash Overlay ────────────────────────────────────────────────────

_flash_state = {"on": False, "last_toggle": 0.0, "interval": 0.3}


def apply_alarm_overlay(frame: np.ndarray) -> np.ndarray:
    """Apply flashing red overlay to frame when alarm is active."""
    now = time.time()
    if now - _flash_state["last_toggle"] > _flash_state["interval"]:
        _flash_state["on"]          = not _flash_state["on"]
        _flash_state["last_toggle"] = now

    vis = frame.copy()
    if _flash_state["on"]:
        overlay       = vis.copy()
        overlay[:, :] = (0, 0, 200)   # BGR red
        import cv2
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        import cv2
        cv2.putText(vis, "⚠  WAKE UP!  ⚠", (vis.shape[1]//2 - 150, vis.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 3)

    return vis


# ─── Auto-init on import ─────────────────────────────────────────────────────
init_audio()
