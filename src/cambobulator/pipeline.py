"""Source -> filters (in practice the one effect) -> outputs."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cambobulator.filters import Filter, FrameContext, create_filter
from cambobulator.outputs import Output
from cambobulator.segmentation import Segmenter
from cambobulator.sources import CaptureThread, FrameSource

log = logging.getLogger(__name__)


@dataclass
class FilterSlot:
    filter: Filter
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    error: str | None = None

    def describe(self) -> dict[str, Any]:
        d = self.filter.describe()
        d.update(id=self.id, enabled=self.enabled, error=self.error)
        return d


class Pipeline:
    """Owns the filters, the capture thread and the processing thread.

    Parameter changes and frame processing happen under ``self.lock``, so the UI
    thread can change parameters while frames flow.
    """

    def __init__(self, segmenter: Segmenter | None = None) -> None:
        self.lock = threading.RLock()
        self.segmenter = segmenter
        self.slots: list[FilterSlot] = []
        self.outputs: dict[str, Output] = {}
        self.output_errors: dict[str, str] = {}
        self.version = 0  # bumped on structural changes so UIs know to re-render
        self.source: FrameSource | None = None
        self._capture: CaptureThread | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._preview_cond = threading.Condition()
        self._preview_seq = 0
        self._preview_raw: np.ndarray | None = None
        self._preview_out: np.ndarray | None = None
        self._frames = 0
        self._fps = 0.0
        self._process_ms = 0.0

    # -- filters ----------------------------------------------------------
    def _changed(self) -> None:
        self.version += 1

    def add_filter(self, type_name: str, enabled: bool = True, params: dict[str, Any] | None = None,
                   slot_id: str | None = None) -> FilterSlot:
        f = create_filter(type_name, segmenter=self.segmenter)
        for name, value in (params or {}).items():
            try:
                f.set_param(name, value)
            except (KeyError, ValueError) as exc:
                log.warning("Ignoring setting %s=%r for %s: %s", name, value, type_name, exc)
        slot = FilterSlot(f, enabled, slot_id or uuid.uuid4().hex[:8])
        with self.lock:
            self.slots.append(slot)
            self._changed()
        return slot

    def get(self, slot_id: str) -> FilterSlot:
        with self.lock:
            for slot in self.slots:
                if slot.id == slot_id:
                    return slot
        raise KeyError(f"No filter with id {slot_id!r}")

    def set_enabled(self, slot_id: str, enabled: bool) -> None:
        with self.lock:
            slot = self.get(slot_id)
            slot.enabled = bool(enabled)
            slot.error = None
            self._changed()

    def set_param(self, slot_id: str, name: str, value: Any) -> Any:
        with self.lock:
            return self.get(slot_id).filter.set_param(name, value)

    def run_action(self, slot_id: str, action: str) -> None:
        with self.lock:
            self.get(slot_id).filter.run_action(action)

    def describe(self) -> list[dict[str, Any]]:
        with self.lock:
            return [s.describe() for s in self.slots]

    # -- processing -------------------------------------------------------
    def process_frame(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        """Run one frame through the enabled filters, in order.

        A filter that raises is skipped (the frame passes through untouched)
        and its error is shown in the UI, so one bug can't kill the camera feed.
        """
        if ctx is None:
            ctx = FrameContext.now(raw=frame)
        with self.lock:
            for slot in self.slots:
                if not slot.enabled:
                    continue
                try:
                    result = slot.filter.process(frame, ctx)
                except Exception as exc:
                    if slot.error is None:
                        log.exception("Filter %s failed", slot.filter.NAME)
                    slot.error = f"{type(exc).__name__}: {exc}"
                    continue
                if result is None or result.shape != frame.shape or result.dtype != np.uint8:
                    slot.error = "filter returned a frame of the wrong shape or type"
                    continue
                slot.error = None
                frame = result
        return frame

    # -- outputs ------------------------------------------------------------
    def add_output(self, name: str, output: Output) -> None:
        with self.lock:
            old = self.outputs.pop(name, None)
            self.outputs[name] = output
            self.output_errors.pop(name, None)
        if old is not None:
            old.close()

    def remove_output(self, name: str) -> None:
        with self.lock:
            out = self.outputs.pop(name, None)
        if out is not None:
            out.close()

    def _send(self, frame: np.ndarray) -> None:
        with self.lock:
            outputs = list(self.outputs.items())
        for name, out in outputs:
            try:
                out.send(frame)
            except Exception as exc:
                log.error("Output %s failed and was stopped: %s", name, exc)
                self.output_errors[name] = str(exc)
                with self.lock:
                    if self.outputs.get(name) is out:
                        del self.outputs[name]
                try:
                    out.close()
                except Exception:
                    pass

    # -- running ----------------------------------------------------------
    @property
    def source_error(self) -> str | None:
        cap = self._capture
        return cap.error if cap is not None else None

    def set_source(self, source: FrameSource | None, reset_filters: bool = True) -> None:
        with self.lock:
            old, self._capture = self._capture, None
            self.source = source
            if reset_filters:
                for slot in self.slots:
                    slot.filter.reset()
        if old is not None:
            old.stop()
        if source is not None:
            cap = CaptureThread(source).start()
            with self.lock:
                self._capture = cap

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        seq = 0
        current: CaptureThread | None = None
        window_start, window_frames = time.monotonic(), 0
        while not self._stop.is_set():
            cap = self._capture
            if cap is None:
                time.sleep(0.02)
                continue
            if cap is not current:
                current, seq = cap, 0
            new_seq, timestamp, frame = cap.wait_frame(seq, timeout=0.5)
            if frame is None:
                if cap.error:
                    time.sleep(0.05)
                continue
            seq = new_seq
            t0 = time.perf_counter()
            ctx = FrameContext(timestamp=timestamp, index=self._frames, raw=frame)
            out = self.process_frame(frame.copy(), ctx)
            self._send(out)
            self._process_ms = 0.9 * self._process_ms + 0.1 * (time.perf_counter() - t0) * 1000.0
            self._frames += 1
            window_frames += 1
            now = time.monotonic()
            if now - window_start >= 1.0:
                self._fps = window_frames / (now - window_start)
                window_start, window_frames = now, 0
            with self._preview_cond:
                self._preview_raw, self._preview_out = frame, out
                self._preview_seq += 1
                self._preview_cond.notify_all()

    def wait_preview(self, after_seq: int, timeout: float = 1.0) -> tuple[int, np.ndarray | None, np.ndarray | None]:
        with self._preview_cond:
            self._preview_cond.wait_for(lambda: self._preview_seq > after_seq or self._stop.is_set(), timeout=timeout)
            return self._preview_seq, self._preview_raw, self._preview_out

    def stats(self) -> dict[str, Any]:
        src = self.source
        return {
            "frames": self._frames,
            "fps": round(self._fps, 1),
            "process_ms": round(self._process_ms, 1),
            "source": src.name if src else None,
            "source_size": [src.width, src.height] if src else None,
            "source_error": self.source_error,
        }

    def stop(self) -> None:
        self._stop.set()
        with self._preview_cond:
            self._preview_cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.set_source(None, reset_filters=False)
        with self.lock:
            outputs, self.outputs = list(self.outputs.values()), {}
        for out in outputs:
            try:
                out.close()
            except Exception:
                log.exception("Error closing output")
        if self.segmenter is not None:
            self.segmenter.close()
