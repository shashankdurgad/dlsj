"""Intel RealSense colour + depth as a `teleop_web.CameraStream`-compatible stream.

Uses pyrealsense2 rather than V4L2, so colour and depth come from the same frameset and depth is aligned
to the colour image, pixel for pixel. Exposes:
* `.frame`: BGR uint8
* `.depth`: uint16 millimetres, (H, W), 0 = no reading
* `.jpeg`, `.seq`, `.cond`, `.error`, `.info()`, `.stop()`, as teleop_web's web handler and the recorders expect

Cameras are chosen by serial number (`rs-enumerate-devices`, or `list_serials()`).
"""

import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


def list_serials() -> list[str]:
    import pyrealsense2 as rs

    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]


def depth_preview(depth_mm: np.ndarray, max_mm: float = 1500.0) -> np.ndarray:
    """Colourised depth (BGR) for viewing: near = blue, far = red, no reading = black."""
    vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_mm, alpha=255.0 / max_mm), cv2.COLORMAP_TURBO)
    vis[depth_mm == 0] = 0
    return vis


class RealSenseStream:
    def __init__(self, name: str, serial: str, width: int = 640, height: int = 480, fps: int = 30,
                 depth: bool = True, quality: int = 80):
        self.name, self.serial = name, str(serial)
        self.width, self.height, self.fps = width, height, fps
        self.use_depth, self.quality = depth, quality
        self.index = f"rs:{self.serial}"
        self.frame: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.seq = 0
        self.measured_fps = 0.0
        self.shape: tuple[int, int] | None = None
        self.error: str | None = None
        self.cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"rs-{name}", daemon=True)

    def start(self) -> "RealSenseStream":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _pipeline(self):
        import pyrealsense2 as rs

        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.use_depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        pipe = rs.pipeline()
        pipe.start(cfg)
        align = rs.align(rs.stream.color) if self.use_depth else None
        return pipe, align

    def _run(self) -> None:
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        while not self._stop.is_set():
            try:
                pipe, align = self._pipeline()
            except Exception as e:  # noqa: BLE001  camera unplugged or busy: retry
                self.error = f"cannot open RealSense {self.serial}: {e}"
                time.sleep(2.0)
                continue
            self.error = None
            t_last, n = time.perf_counter(), 0
            try:
                while not self._stop.is_set():
                    fs = pipe.wait_for_frames(2000)
                    if align is not None:
                        fs = align.process(fs)
                    c = fs.get_color_frame()
                    if not c:
                        continue
                    color = np.asanyarray(c.get_data()).copy()
                    depth = None
                    if self.use_depth:
                        d = fs.get_depth_frame()
                        depth = np.asanyarray(d.get_data()).copy() if d else None
                    ok, buf = cv2.imencode(".jpg", color, params)
                    n += 1
                    now = time.perf_counter()
                    if now - t_last >= 1.0:
                        self.measured_fps, t_last, n = n / (now - t_last), now, 0
                    with self.cond:
                        self.frame, self.depth, self.shape = color, depth, color.shape[:2]
                        if ok:
                            self.jpeg = buf.tobytes()
                        self.seq += 1
                        self.cond.notify_all()
            except Exception as e:  # noqa: BLE001
                self.error = f"RealSense {self.serial}: {e}"
                log.warning(self.error)
            finally:
                try:
                    pipe.stop()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.5)

    def info(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "fps": round(self.measured_fps, 1),
            "size": f"{self.shape[1]}x{self.shape[0]}" if self.shape else None,
            "error": self.error,
            "depth": self.use_depth,
        }
