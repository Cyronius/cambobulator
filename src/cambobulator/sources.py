"""Frame sources: real cameras, video files and a synthetic test pattern.

A source spec is a string:

* ``"0"``, ``"1"``, ``"camera:1"`` -- a camera by index
* ``"synthetic"`` -- a moving test pattern (no hardware needed)
* anything else -- a path to a video file, which loops
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class SourceError(RuntimeError):
    pass


class FrameSource(ABC):
    name: str = "source"
    width: int = 0
    height: int = 0
    fps: float = 30.0

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Block until the next frame and return it (BGR uint8), or None on failure."""

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _camera_backends() -> list[int]:
    # DirectShow opens fast and honours resolution requests on Windows; MSMF
    # can take several seconds to open. AVFoundation is the only option on macOS.
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


@contextmanager
def quiet_opencv():
    """Silence OpenCV's console warnings while probing cameras that may not exist."""
    logging_mod = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging_mod is None:
        yield
        return
    previous = logging_mod.getLogLevel()
    logging_mod.setLogLevel(logging_mod.LOG_LEVEL_SILENT)
    try:
        yield
    finally:
        logging_mod.setLogLevel(previous)


class CameraSource(FrameSource):
    def __init__(self, index: int, width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        self.index = index
        self.name = f"camera {index}"
        cap = None
        with quiet_opencv():
            for backend in _camera_backends():
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    break
                cap.release()
                cap = None
        if cap is None:
            raise SourceError(
                f"Could not open camera {index}. Check that it is plugged in, that no other app "
                "(Teams, Zoom, the Camera app) is using it, and that this app has camera permission. "
                "Run 'cambobulator cameras' to list cameras."
            )
        # MJPG lets most USB webcams do 720p/1080p at 30 fps; raw YUY2 is often capped lower.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always hand us the newest frame
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise SourceError(
                f"Camera {index} opened but returned no image. It may be in use by another app, "
                "or it may be a virtual camera with nothing feeding it."
            )
        self._cap = cap
        self._first: np.ndarray | None = frame
        self.height, self.width = frame.shape[:2]
        self.fps = cap.get(cv2.CAP_PROP_FPS) or fps
        if (self.width, self.height) != (width, height):
            log.info("Camera %d gave %dx%d instead of the requested %dx%d", index, self.width, self.height, width, height)

    def read(self) -> np.ndarray | None:
        if self._first is not None:
            frame, self._first = self._first, None
            return frame
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class _Paced(FrameSource):
    """Sleeps so frames come out at ``fps``, like a real camera would."""

    def __init__(self, fps: float, realtime: bool) -> None:
        self.fps = fps
        self.realtime = realtime
        self._next = time.monotonic()

    def _pace(self) -> None:
        if not self.realtime:
            return
        self._next += 1.0 / self.fps
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.25:  # fell far behind; don't try to catch up in a burst
            self._next = time.monotonic()


class VideoFileSource(_Paced):
    def __init__(self, path: str | Path, width: int | None = None, height: int | None = None,
                 loop: bool = True, realtime: bool = True) -> None:
        path = Path(path)
        if not path.is_file():
            raise SourceError(f"Video file not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise SourceError(f"Could not open video file {path} (unsupported format?)")
        super().__init__(self._cap.get(cv2.CAP_PROP_FPS) or 30.0, realtime)
        self.name = f"file {path.name}"
        self.loop = loop
        self._size = (width, height) if width and height else None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width, self.height = self._size or (w, h)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            return None
        self._pace()
        if self._size and frame.shape[1::-1] != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        self._cap.release()


SYNTHETIC_PERSON_BGR = (60, 80, 200)


def synthetic_person_mask(frame: np.ndarray) -> np.ndarray:
    """Exact person mask for frames made by :class:`SyntheticSource` (matches its flat colour)."""
    return np.all(frame == np.array(SYNTHETIC_PERSON_BGR, np.uint8), axis=2).astype(np.float32)


class SyntheticSource(_Paced):
    """A textured "room" with a person-shaped blob wandering around.

    ``person_mask()`` returns the ground-truth mask for the last frame, which
    makes a perfect fake segmenter for demos and tests.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0, realtime: bool = True) -> None:
        super().__init__(fps, realtime)
        self.name = "synthetic"
        self.width, self.height = width, height
        self._t = 0
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        room = np.empty((height, width, 3), np.float32)
        room[..., 0] = 90 + 60 * xx / width
        room[..., 1] = 110 + 40 * yy / height
        room[..., 2] = 140 + 30 * np.sin(xx / 37.0) * np.cos(yy / 29.0)
        # "Shelves": a few rectangles so blending errors are visible.
        for i in range(5):
            x0 = int(width * (0.08 + 0.18 * i))
            y0 = int(height * (0.15 + 0.1 * (i % 2)))
            cv2.rectangle(room, (x0, y0), (x0 + width // 10, y0 + height // 5), (40 + 30 * i, 60, 160 - 20 * i), -1)
        self._room = room.astype(np.uint8)
        self._mask = np.zeros((height, width), np.float32)

    def person_mask(self) -> np.ndarray:
        return self._mask.copy()

    def read(self) -> np.ndarray | None:
        t = self._t / self.fps
        self._t += 1
        w, h = self.width, self.height
        frame = self._room.copy()
        mask = np.zeros((h, w), np.uint8)
        cx = int(w * (0.5 + 0.25 * math.sin(t * 0.7)))
        head_r = max(4, h // 9)
        cy = int(h * 0.38 + 6 * math.sin(t * 2.0))
        cv2.circle(mask, (cx, cy), head_r, 255, -1)
        cv2.ellipse(mask, (cx, h), (int(head_r * 2.2), int(h * 0.45)), 0, 180, 360, 255, -1)
        frame[mask > 0] = SYNTHETIC_PERSON_BGR
        cv2.putText(frame, f"{t:6.2f}s", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, h / 900), (255, 255, 255), 1)
        self._mask = (mask > 0).astype(np.float32)
        self._pace()
        return frame


def open_source(spec: str | int, width: int = 1280, height: int = 720, fps: float = 30.0) -> FrameSource:
    spec = str(spec).strip()
    if spec.lower().startswith("camera:"):
        spec = spec.split(":", 1)[1]
    if spec.isdigit():
        return CameraSource(int(spec), width, height, fps)
    if spec.lower() in ("synthetic", "test", "demo"):
        return SyntheticSource(width, height, fps)
    return VideoFileSource(spec, width, height)


class CaptureThread:
    """Reads a source on its own thread and keeps only the newest frame.

    Dropping stale frames (instead of queueing them) keeps end-to-end latency
    at one frame even when filters are momentarily slow.
    """

    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(self, source: FrameSource) -> None:
        self.source = source
        self.error: str | None = None
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._timestamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"capture-{source.name}", daemon=True)

    def start(self) -> "CaptureThread":
        self._thread.start()
        return self

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                frame = self.source.read()
            except Exception as exc:  # driver errors surface as exceptions on some platforms
                log.exception("Capture error")
                frame, self.error = None, str(exc)
            if frame is None:
                failures += 1
                if failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self.error = self.error or f"{self.source.name} stopped delivering frames"
                    log.error(self.error)
                    with self._cond:
                        self._cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            failures = 0
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._timestamp = time.monotonic()
                self._cond.notify_all()

    def wait_frame(self, after_seq: int, timeout: float = 1.0) -> tuple[int, float, np.ndarray | None]:
        """Wait for a frame newer than ``after_seq``. Returns (seq, timestamp, frame)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq or self.error is not None or self._stop.is_set(),
                                timeout=timeout)
            if self._seq > after_seq:
                return self._seq, self._timestamp, self._frame
            return after_seq, 0.0, None

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        self.source.close()
