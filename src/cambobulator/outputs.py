"""Where processed frames go: the virtual camera, a video file, or a list (tests)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class Output(Protocol):
    def send(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


class VirtualCameraError(RuntimeError):
    pass


def virtual_camera_help() -> str:
    if sys.platform == "win32" or sys.platform == "darwin":
        return (
            "Cambobulator sends video through the OBS Virtual Camera driver.\n"
            "  1. Install OBS Studio (version 28 or newer) from https://obsproject.com\n"
            "  2. Start OBS once and click 'Start Virtual Camera', then 'Stop Virtual Camera' and close OBS.\n"
            "     This registers the driver" + (" (on macOS, approve the system extension when asked)." if sys.platform == "darwin" else ".") + "\n"
            "  3. Make sure OBS itself is not running its virtual camera while Cambobulator is.\n"
            "  4. Run Cambobulator again, then pick 'OBS Virtual Camera' in Teams/Discord/Zoom."
        )
    return (
        "Cambobulator sends video through a v4l2loopback device.\n"
        "  sudo apt install v4l2loopback-dkms   (or your distro's equivalent)\n"
        "  sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label=\"Cambobulator\"\n"
        "Then run Cambobulator again and pick 'Cambobulator' in your video app."
    )


class VirtualCameraOutput:
    def __init__(self, width: int, height: int, fps: float, backend: str | None = None,
                 device: str | None = None) -> None:
        try:
            import pyvirtualcam
        except ImportError as exc:
            raise VirtualCameraError("pyvirtualcam is not installed (pip install pyvirtualcam).") from exc
        try:
            self._cam = pyvirtualcam.Camera(
                width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR, backend=backend, device=device
            )
        except Exception as exc:  # pyvirtualcam raises RuntimeError with a short reason
            raise VirtualCameraError(f"Could not start the virtual camera: {exc}\n\n{virtual_camera_help()}") from exc
        self.width, self.height = width, height
        self.device: str = getattr(self._cam, "device", "virtual camera")

    def send(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        self._cam.send(frame)

    def close(self) -> None:
        self._cam.close()


class VideoFileOutput:
    """Record the processed stream to a file (.mp4 or .avi)."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float) -> None:
        path = Path(path)
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if path.suffix.lower() == ".avi" else "mp4v"))
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open {path} for writing")
        self.size = (width, height)
        self.frames = 0

    def send(self, frame: np.ndarray) -> None:
        if frame.shape[1::-1] != self.size:
            frame = cv2.resize(frame, self.size)
        self._writer.write(frame)
        self.frames += 1

    def close(self) -> None:
        self._writer.release()


class CollectOutput:
    """Keeps frames in memory; for tests."""

    def __init__(self, limit: int | None = None) -> None:
        self.frames: list[np.ndarray] = []
        self.limit = limit
        self.closed = False

    def send(self, frame: np.ndarray) -> None:
        if self.limit is None or len(self.frames) < self.limit:
            self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True
