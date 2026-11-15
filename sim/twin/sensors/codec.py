"""The dataset's video codec round-trip for rendered frames: H.264 (libx264) crf 30, g 2, yuv420p, preset medium.

The recorded LeRobot dataset (meta/info.json) stores every camera as H.264 High, yuv420p, GOP 2 (I P I P ...),
crf 30, libx264's default preset, encoded through PyAV (LeRobot `RGBEncoderConfig(vcodec="h264")`). The policy was
trained on the DECODED frames, so the twin sends its frames through the same encoder:

    from sim.twin.sensors.codec import H264
    codec = H264()                          # spawns one PyAV worker subprocess on first use (never PyAV in Isaac)
    out = codec(img, "top")                 # uint8 (H, W, 3) RGB -> decoded uint8 (H, W, 3)
    codec.close()                           # also registered with atexit; the worker exits on EOF of its stdin

GOP phase: per stream (camera) the frames alternate I, P like the dataset. An I frame is encoded alone; a P frame is
encoded as the 2-frame GOP (previous input, this input) and the decoded second frame is returned, so a static region
of a P frame is the previous decoded I frame (the real P frames are skip blocks there: measure.py, I->P temporal
noise 0.0 DN) and moving regions get P-frame quantisation.

If PyAV or the subprocess is unavailable, `emulate()` approximates the artefacts in-process with cv2 JPEG (4:2:0,
quality fitted to the H.264 I frames) and the same P-frame hold of static regions.
"""
from __future__ import annotations

import atexit
import os
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SETTINGS = {"codec": "libx264", "crf": 30, "g": 2, "pix_fmt": "yuv420p", "preset": "medium"}
_HDR = struct.Struct("<4sIIIII")  # magic, n_frames (1 = I, 2 = I+P), height, width, crf, return index
_MAGIC = b"H264"
VENV_PY = "/opt/isaaclab-venv/bin/python"  # the twin image's venv (has PyAV); DLSJ_CODEC_PYTHON overrides


def roundtrip(frames, crf: int = SETTINGS["crf"], preset: str = SETTINGS["preset"], g: int = SETTINGS["g"]):
    """Encode RGB uint8 frames (list / (N, H, W, 3)) as one H.264 stream with the dataset's settings and decode them
    again (PyAV, plain python only). Returns (decoded frames (N, H, W, 3) uint8, packet sizes in bytes)."""
    from fractions import Fraction

    import av

    frames = [np.ascontiguousarray(f[..., :3]) for f in frames]
    H, W = frames[0].shape[:2]
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = W, H, SETTINGS["pix_fmt"]
    enc.time_base = Fraction(1, 30)
    enc.framerate = Fraction(30, 1)
    enc.options = {"g": str(g), "crf": str(crf), "preset": preset}
    pkts = []
    for i, f in enumerate(frames):
        vf = av.VideoFrame.from_ndarray(f, format="rgb24")
        vf.pts = i
        pkts += enc.encode(vf)
    pkts += enc.encode(None)
    dec = av.CodecContext.create("h264", "r")
    out = []
    for p in pkts:
        out += [fr.to_ndarray(format="rgb24") for fr in dec.decode(p)]
    out += [fr.to_ndarray(format="rgb24") for fr in dec.decode(None)]
    return np.stack(out[: len(frames)]), [p.size for p in pkts]


def _read_exact(f, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        b = f.read(n - len(buf))
        if not b:
            return None
        buf += b
    return bytes(buf)


def worker_main():
    """Worker loop (runs in its own process): header + raw RGB frames on stdin -> one decoded frame on stdout."""
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        hdr = _read_exact(fin, _HDR.size)
        if hdr is None:
            return
        magic, n, H, W, crf, ret = _HDR.unpack(hdr)
        if magic != _MAGIC:
            return
        raw = _read_exact(fin, n * H * W * 3)
        if raw is None:
            return
        fr = np.frombuffer(raw, np.uint8).reshape(n, H, W, 3)
        out, _ = roundtrip(list(fr), crf=crf)
        fout.write(np.ascontiguousarray(out[ret]).tobytes())
        fout.flush()


class H264:
    """Client of the PyAV worker subprocess with per-stream GOP phase (I, P, I, P, ...)."""

    def __init__(self, crf: int = SETTINGS["crf"]):
        self.crf = crf
        self.proc: subprocess.Popen | None = None
        self.failed: str | None = None
        self.prev: dict[str, np.ndarray] = {}  # last INPUT frame per stream (the I frame a P frame refers to)
        self.phase: dict[str, int] = {}
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _start(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        exe = os.environ.get("DLSJ_CODEC_PYTHON") or (VENV_PY if Path(VENV_PY).exists() else sys.executable)
        self.proc = subprocess.Popen(
            [exe, "-c", "from sim.twin.sensors.codec import worker_main; worker_main()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=str(REPO))

    def reset(self, stream: str | None = None):
        """Start a new GOP sequence (next frame is an I frame) for one stream or all."""
        for d in (self.prev, self.phase):
            if stream is None:
                d.clear()
            else:
                d.pop(stream, None)

    def __call__(self, img: np.ndarray, stream: str, crf: int | None = None, keyframe: bool | None = None):
        """Round-trip one RGB uint8 frame of `stream`; keyframe None = follow the GOP phase."""
        img = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
        k = self.phase.get(stream, 0)
        is_i = (k % 2 == 0) if keyframe is None else keyframe
        prev = self.prev.get(stream)
        if prev is None or prev.shape != img.shape:
            is_i = True
        frames = [img] if is_i else [prev, img]
        out = self._send(frames, int(self.crf if crf is None else crf))
        self.prev[stream] = img
        self.phase[stream] = 1 if is_i else 0
        return out

    def _send(self, frames, crf: int) -> np.ndarray:
        H, W = frames[0].shape[:2]
        with self._lock:
            if self.failed is None:
                try:
                    if self.proc is None or self.proc.poll() is not None:
                        self._start()
                    self.proc.stdin.write(_HDR.pack(_MAGIC, len(frames), H, W, crf, len(frames) - 1))
                    for f in frames:
                        self.proc.stdin.write(f.tobytes())
                    self.proc.stdin.flush()
                    raw = _read_exact(self.proc.stdout, H * W * 3)
                    if raw is not None:
                        return np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy()
                    self.failed = "worker closed its output"
                except (OSError, ValueError) as e:
                    self.failed = f"{type(e).__name__}: {e}"
                print(f"[sensors.codec] H.264 worker unavailable ({self.failed}); emulating", flush=True)
                self.close()
        return emulate(frames[-1], frames[0] if len(frames) == 2 else None)

    def close(self):
        """Stop the worker (EOF on its stdin) and wait for it."""
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.close()
            p.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            p.kill()
            p.wait()
        finally:
            if p.stdout:
                p.stdout.close()


def emulate(img: np.ndarray, prev: np.ndarray | None = None, quality: int = 72, hold_thr: float = 2.0) -> np.ndarray:
    """In-process stand-in for the H.264 round-trip: JPEG 4:2:0 at `quality` (I frame); for a P frame (prev given) the
    8 px blocks that did not change (mean |diff| < hold_thr DN) keep the previous frame's decoded pixels."""
    import cv2

    def jpg(x):
        ok, buf = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1] if ok else x

    out = jpg(img)
    if prev is None:
        return out
    pj = jpg(prev)
    d = np.abs(img.astype(np.int16) - prev.astype(np.int16)).mean(-1)
    H, W = d.shape
    b = cv2.resize(d[: H // 8 * 8, : W // 8 * 8], (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    hold = cv2.resize((b < hold_thr).astype(np.uint8), (W // 8 * 8, H // 8 * 8), interpolation=cv2.INTER_NEAREST)
    m = np.zeros((H, W), bool)
    m[: H // 8 * 8, : W // 8 * 8] = hold > 0
    out[m] = pj[m]
    return out
