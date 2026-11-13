"""Non-blocking cue sounds: a rising double beep when recording starts, a falling low beep when it stops.

Short sine-tone WAVs are generated once into a temp dir and played with the platform player
(`afplay` on macOS, `paplay`/`aplay` on Linux) in a background process, so the control loop never
waits on audio. The terminal bell is the fallback if no player exists.
"""

import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050

# (frequency Hz, duration s) segments; 0 Hz = silence.
CUES: dict[str, list[tuple[float, float]]] = {
    "start": [(880.0, 0.08), (0.0, 0.04), (1320.0, 0.12)],  # high, rising double beep
    "stop": [(660.0, 0.10), (0.0, 0.03), (330.0, 0.22)],  # falling, lower and longer
}


def synth(segments: list[tuple[float, float]], rate: int = SAMPLE_RATE, volume: float = 0.4) -> np.ndarray:
    """Int16 mono samples for the given tone segments, with 5 ms fades to avoid clicks."""
    parts = []
    for freq, dur in segments:
        n = int(rate * dur)
        if freq <= 0:
            parts.append(np.zeros(n))
            continue
        t = np.arange(n) / rate
        tone = np.sin(2 * np.pi * freq * t)
        fade = min(n // 2, int(0.005 * rate))
        if fade:
            ramp = np.linspace(0.0, 1.0, fade)
            tone[:fade] *= ramp
            tone[-fade:] *= ramp[::-1]
        parts.append(tone)
    return (np.concatenate(parts) * volume * 32767).astype(np.int16)


def _player() -> list[str] | None:
    if sys.platform == "darwin" and shutil.which("afplay"):
        return ["afplay"]
    for cmd in (["paplay"], ["aplay", "-q"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


class Cues:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.player = _player() if enabled else None
        self.files: dict[str, Path] = {}
        if self.player:
            d = Path(tempfile.mkdtemp(prefix="dlsj_cues_"))
            for name, segments in CUES.items():
                path = d / f"{name}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(synth(segments).tobytes())
                self.files[name] = path

    def play(self, name: str) -> None:
        if not self.enabled:
            return
        if self.player and name in self.files:
            try:
                subprocess.Popen([*self.player, str(self.files[name])],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                pass
        sys.stdout.write("\a")
        sys.stdout.flush()
