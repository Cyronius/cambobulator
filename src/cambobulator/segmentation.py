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
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"
)
MODEL_FILENAME = "selfie_segmenter_landscape.tflite"


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


def cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "cambobulator"


def ensure_model(path: str | os.PathLike | None = None) -> Path:
    """Return a local path to the segmentation model, downloading it once if needed."""
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Segmentation model not found: {p}")
        return p
    p = cache_dir() / MODEL_FILENAME
    if p.is_file() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading person-segmentation model to %s", p)
    tmp = p.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
        tmp.replace(p)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the segmentation model from {MODEL_URL} ({exc}). "
            f"Download it by hand and set 'segmenter_model' in the config, or put it at {p}."
        ) from exc
    return p


class MediaPipeSegmenter:
    """MediaPipe selfie segmentation (Tasks API, with the legacy API as fallback).

    The model works on a small image (256x144), so we downscale first; that
    keeps the cost to a few milliseconds per frame on a laptop CPU.
    """

    name = "mediapipe"

    def __init__(self, model_path: str | None = None, work_width: int = 256) -> None:
        self.work_width = work_width
        self._tasks = None
        self._legacy = None
        try:
            import mediapipe as mp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("mediapipe is not installed (pip install mediapipe)") from exc
        try:
            from mediapipe.tasks.python import BaseOptions, vision

            options = vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
                running_mode=vision.RunningMode.IMAGE,
                output_confidence_masks=True,
                output_category_mask=False,
            )
            self._tasks = vision.ImageSegmenter.create_from_options(options)
        except Exception as tasks_exc:
            # Older mediapipe builds ship the legacy "solutions" API.
            solutions = getattr(mp, "solutions", None)
            if solutions is None or not hasattr(solutions, "selfie_segmentation"):
                raise RuntimeError(f"Could not start MediaPipe segmentation: {tasks_exc}") from tasks_exc
            log.info("MediaPipe Tasks API unavailable (%s); using legacy selfie_segmentation", tasks_exc)
            self._legacy = solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        import mediapipe as mp

        h, w = frame.shape[:2]
        scale = min(1.0, self.work_width / w)
        small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if self._tasks is not None:
            result = self._tasks.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.confidence_masks:
                return None
            mask = np.asarray(result.confidence_masks[-1].numpy_view(), dtype=np.float32)
        else:
            result = self._legacy.process(rgb)
            if result.segmentation_mask is None:
                return None
            mask = np.asarray(result.segmentation_mask, dtype=np.float32)
        mask = mask.reshape(mask.shape[0], mask.shape[1])
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0, out=mask)

    def close(self) -> None:
        for obj in (self._tasks, self._legacy):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._tasks = self._legacy = None


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
    """``auto`` / ``mediapipe`` load MediaPipe lazily; ``none`` disables segmentation;
    ``synthetic`` finds the person in the synthetic test pattern (for demos)."""
    kind = (kind or "auto").lower()
    if kind == "none":
        return NullSegmenter(reason="disabled in settings")
    if kind == "synthetic":
        from cambobulator.sources import synthetic_person_mask

        return CallableSegmenter(synthetic_person_mask, name="synthetic test pattern")
    if kind in ("auto", "mediapipe"):
        return LazySegmenter(lambda: MediaPipeSegmenter(model_path), label="mediapipe")
    raise ValueError(f"Unknown segmenter {kind!r} (use auto, mediapipe, none or synthetic)")
