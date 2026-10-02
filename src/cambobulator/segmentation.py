"""Person segmentation.

A segmenter turns a BGR frame into a float32 mask of shape (H, W) with values
in [0, 1], where 1 means "this pixel is the person". Returning ``None`` means
"I can't tell" and filters should fall back to something sensible.

Filters receive their segmenter through the constructor, so tests can pass a
stub instead of loading a real model.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "selfie_segmenter_landscape.onnx"
MODEL_SIZE = (256, 144)  # (width, height) the model takes


class Segmenter(Protocol):
    name: str

    def segment(self, frame: np.ndarray) -> np.ndarray | None: ...

    def close(self) -> None: ...


class NullSegmenter:
    """Never finds anyone."""

    name = "none"

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        return None

    def close(self) -> None:
        pass


class CallableSegmenter:
    """Wrap a plain function ``frame -> mask``. Handy for tests and demos."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray | None], name: str = "callable") -> None:
        self.fn = fn
        self.name = name

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        return self.fn(frame)

    def close(self) -> None:
        pass


class SelfieSegmenter:
    """MediaPipe's selfie segmentation model (bundled as ONNX), run by OpenCV's dnn module.

    The model sees a 256x144 RGB image, so each frame is shrunk to that first
    (frames that aren't 16:9 get stretched, and so does the mask on the way
    back). That keeps the cost to a few milliseconds per frame on a laptop CPU.
    """

    name = "selfie segmenter"

    def __init__(self, model_path: str | os.PathLike | None = None) -> None:
        path = Path(model_path).expanduser() if model_path else MODEL_PATH
        if path.suffix.lower() == ".tflite":
            # Configs from before the ONNX switch pointed at MediaPipe's .tflite download.
            log.warning("Ignoring segmenter_model %s: the bundled ONNX model replaces it", path)
            path = MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Segmentation model not found: {path}")
        self._net = cv2.dnn.readNetFromONNX(str(path))

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, MODEL_SIZE, interpolation=cv2.INTER_AREA)
        self._net.setInput(cv2.dnn.blobFromImage(small, 1.0 / 255.0, MODEL_SIZE, swapRB=True))
        mask = self._net.forward().reshape(MODEL_SIZE[1], MODEL_SIZE[0])
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0, out=mask)

    def close(self) -> None:
        self._net = None


class LazySegmenter:
    """Builds the real segmenter on first use, in whatever thread calls it.

    If construction fails, the error is logged once and it behaves like a
    :class:`NullSegmenter`, so the filter can fall back instead of crashing.
    """

    def __init__(self, factory: Callable[[], Segmenter], label: str = "auto") -> None:
        self._factory = factory
        self._label = label
        self._inner: Segmenter | None = None
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def name(self) -> str:
        if self._inner is None:
            return f"{self._label} (not loaded)"
        return self._inner.name

    def _get(self) -> Segmenter:
        with self._lock:
            if self._inner is None:
                try:
                    self._inner = self._factory()
                except Exception as exc:
                    self.error = str(exc)
                    log.warning("Person segmentation unavailable: %s", exc)
                    self._inner = NullSegmenter(reason=str(exc))
            return self._inner

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        return self._get().segment(frame)

    def close(self) -> None:
        with self._lock:
            if self._inner is not None:
                self._inner.close()
            self._inner = None


def make_segmenter(kind: str = "auto", model_path: str | None = None) -> Segmenter:
    """``auto`` loads the bundled selfie segmenter lazily; ``none`` disables segmentation;
    ``synthetic`` finds the person in the synthetic test pattern (for demos)."""
    kind = (kind or "auto").lower()
    if kind == "none":
        return NullSegmenter(reason="disabled in settings")
    if kind == "synthetic":
        from cambobulator.sources import synthetic_person_mask

        return CallableSegmenter(synthetic_person_mask, name="synthetic test pattern")
    if kind in ("auto", "mediapipe"):  # "mediapipe": configs from before the ONNX switch
        return LazySegmenter(lambda: SelfieSegmenter(model_path), label="selfie segmenter")
    raise ValueError(f"Unknown segmenter {kind!r} (use auto, none or synthetic)")
