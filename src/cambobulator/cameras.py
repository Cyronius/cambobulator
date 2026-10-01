"""List the cameras on this machine."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Never feed a virtual camera back into itself.
VIRTUAL_CAMERA_HINTS = ("obs virtual", "obs-camera", "virtual cam", "unitycapture", "cambobulator", "v4l2loopback",
                        "dummy video device")


@dataclass
class CameraInfo:
    index: int
    name: str
    is_virtual: bool = False

    def to_dict(self) -> dict:
        return {"index": self.index, "name": self.name, "is_virtual": self.is_virtual, "source": str(self.index)}


def _looks_virtual(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in VIRTUAL_CAMERA_HINTS)


def _windows_names() -> list[str] | None:
    # pygrabber lists DirectShow devices in the same order OpenCV's CAP_DSHOW uses.
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return None
    try:
        return list(FilterGraph().get_input_devices())
    except Exception as exc:
        log.debug("pygrabber failed: %s", exc)
        return None


def _linux_cameras() -> list[CameraInfo]:
    cams = []
    for dev in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:] or 0)):
        index = int(dev.name[5:])
        try:
            name = (dev / "name").read_text().strip()
        except OSError:
            name = dev.name
        # Each UVC camera exposes a second "metadata" node that cannot capture.
        try:
            if (dev / "index").read_text().strip() not in ("0", ""):
                continue
        except OSError:
            pass
        cams.append(CameraInfo(index, name, _looks_virtual(name)))
    return cams


def _probe(max_index: int) -> list[CameraInfo]:
    import cv2

    from cambobulator.sources import _camera_backends, quiet_opencv

    backend = _camera_backends()[0]
    found = []
    with quiet_opencv():
        for i in range(max_index):
            cap = cv2.VideoCapture(i, backend)
            try:
                if cap.isOpened():
                    found.append(CameraInfo(i, f"Camera {i}"))
            finally:
                cap.release()
    return found


def list_cameras(max_index: int = 8) -> list[CameraInfo]:
    """Cameras by index, with friendly names where the OS gives them to us."""
    if sys.platform == "win32":
        names = _windows_names()
        if names is not None:
            return [CameraInfo(i, n, _looks_virtual(n)) for i, n in enumerate(names)]
    elif sys.platform.startswith("linux") and Path("/sys/class/video4linux").is_dir():
        return _linux_cameras()
    return _probe(max_index)
