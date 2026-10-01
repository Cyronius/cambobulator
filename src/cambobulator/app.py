"""Glue between saved settings, the running pipeline and the virtual camera.

Both ``cambobulator run`` (headless) and ``cambobulator ui`` drive a Controller.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from cambobulator.config import FilterConfig, Settings
from cambobulator.filters import available_filters
from cambobulator.outputs import VirtualCameraError, VirtualCameraOutput
from cambobulator.pipeline import Pipeline
from cambobulator.segmentation import Segmenter, make_segmenter
from cambobulator.sources import FrameSource, SourceError, open_source

log = logging.getLogger(__name__)

VCAM = "virtual_camera"


class Controller:
    def __init__(self, settings: Settings, config_path: Path | None = None,
                 segmenter: Segmenter | None = None, autosave_delay: float = 1.0) -> None:
        self.settings = settings
        self.config_path = config_path
        self.autosave_delay = autosave_delay
        self.pipeline = Pipeline(segmenter or make_segmenter(settings.segmenter, settings.segmenter_model))
        self.vcam_error: str | None = None
        self.vcam_device: str | None = None
        self.source_error: str | None = None
        self._save_timer: threading.Timer | None = None
        self._save_lock = threading.RLock()
        self._save_due = 0.0
        for fc in settings.filters:
            try:
                slot = self.pipeline.add_filter(fc.type, fc.enabled, fc.params, slot_id=fc.id)
            except KeyError as exc:
                log.warning("Skipping filter from config: %s", exc)
                continue
            if self.assets_dir is not None:
                slot.filter.load_assets(self.assets_dir, f"{slot.id}-")

    @property
    def assets_dir(self) -> Path | None:
        return self.config_path.parent if self.config_path else None

    # -- source -------------------------------------------------------------
    def open_source(self, spec: str | None = None, width: int | None = None, height: int | None = None,
                    fps: int | None = None, strict: bool = False, source: FrameSource | None = None) -> bool:
        """Switch to a new source. Returns False (and records the error) if it can't be opened."""
        s = self.settings
        spec = s.source if spec is None else str(spec)
        new_size = (width or s.width, height or s.height, fps or s.fps)
        changed_scene = spec != s.source
        try:
            if source is None:
                # Release the old camera first: on Windows a camera can only be opened once.
                self.pipeline.set_source(None, reset_filters=False)
                source = open_source(spec, *new_size)
        except SourceError as exc:
            self.source_error = str(exc)
            if strict:
                raise
            log.error("%s", exc)
            return False
        self.source_error = None
        s.source = spec
        size_changed = new_size != (s.width, s.height, s.fps)
        s.width, s.height, s.fps = new_size
        self.pipeline.set_source(source, reset_filters=changed_scene)
        if size_changed and VCAM in self.pipeline.outputs:
            self.set_virtual_camera(True)  # reopen at the new size
        self.schedule_save()
        return True

    # -- virtual camera -------------------------------------------------------
    def set_virtual_camera(self, enabled: bool, strict: bool = False) -> bool:
        s = self.settings
        self.pipeline.remove_output(VCAM)
        self.vcam_device = None
        ok = True
        if enabled:
            try:
                out = VirtualCameraOutput(s.width, s.height, s.fps, s.vcam_backend, s.vcam_device)
            except VirtualCameraError as exc:
                self.vcam_error = str(exc)
                if strict:
                    raise
                log.error("%s", exc)
                ok = False
            else:
                self.vcam_error = None
                self.vcam_device = out.device
                self.pipeline.add_output(VCAM, out)
                log.info("Virtual camera running: %s", out.device)
        else:
            self.vcam_error = None
        s.virtual_camera = bool(enabled)
        self.schedule_save()
        return ok

    @property
    def vcam_running(self) -> bool:
        return VCAM in self.pipeline.outputs

    # -- filter edits (all of these autosave) ------------------------------
    def add_filter(self, type_name: str) -> str:
        slot = self.pipeline.add_filter(type_name)
        self.schedule_save()
        return slot.id

    def remove_filter(self, slot_id: str) -> None:
        self.pipeline.remove_filter(slot_id)
        self.schedule_save()

    def move_filter(self, slot_id: str, delta: int) -> None:
        self.pipeline.move(slot_id, delta)
        self.schedule_save()

    def reorder_filters(self, ids: list[str]) -> None:
        self.pipeline.reorder(ids)
        self.schedule_save()

    def set_enabled(self, slot_id: str, enabled: bool) -> None:
        self.pipeline.set_enabled(slot_id, enabled)
        self.schedule_save()

    def set_param(self, slot_id: str, name: str, value: Any) -> Any:
        v = self.pipeline.set_param(slot_id, name, value)
        self.schedule_save()
        return v

    def run_action(self, slot_id: str, action: str) -> None:
        self.pipeline.run_action(slot_id, action)
        # Captures happen on a later frame (maybe after a countdown), so save a bit later too.
        delay = self.pipeline.get(slot_id).filter.values.get("capture_delay", 0) if "delayed" in action else 0
        self.schedule_save(delay=self.autosave_delay + float(delay) + 1.0)

    # -- persistence ----------------------------------------------------------
    def sync_settings(self) -> Settings:
        with self.pipeline.lock:
            self.settings.filters = [
                FilterConfig(s.filter.NAME, s.enabled, dict(s.filter.values), s.id) for s in self.pipeline.slots
            ]
        return self.settings

    def save(self) -> None:
        if self.config_path is None:
            return
        with self._save_lock:
            self.sync_settings().save(self.config_path)
            with self.pipeline.lock:
                for slot in self.pipeline.slots:
                    slot.filter.save_assets(self.assets_dir, f"{slot.id}-")

    def schedule_save(self, delay: float | None = None) -> None:
        """Save after a quiet period. A pending later save is never pulled earlier,
        so a slider tweak can't save before a countdown capture has happened."""
        if self.config_path is None:
            return
        due = time.monotonic() + (self.autosave_delay if delay is None else delay)
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                due = max(due, self._save_due)
            self._save_due = due
            self._save_timer = threading.Timer(max(0.0, due - time.monotonic()), self._save_quietly)
            self._save_timer.daemon = True
            self._save_timer.start()

    def _save_quietly(self) -> None:
        with self._save_lock:
            self._save_timer = None
        try:
            self.save()
        except Exception:
            log.exception("Could not save settings to %s", self.config_path)

    # -- lifecycle ------------------------------------------------------------
    def start(self, virtual_camera: bool | None = None, strict: bool = False) -> None:
        self.pipeline.start()
        self.open_source(strict=strict)
        if virtual_camera if virtual_camera is not None else self.settings.virtual_camera:
            self.set_virtual_camera(True, strict=strict)

    def close(self) -> None:
        if self._save_timer is not None:
            self._save_timer.cancel()
        self._save_quietly()
        self.pipeline.stop()

    def state(self) -> dict[str, Any]:
        s = self.settings
        return {
            "settings": {k: v for k, v in s.to_dict().items() if k != "filters"},
            "chain": self.pipeline.describe(),
            "chain_version": self.pipeline.version,
            "available_filters": available_filters(),
            "virtual_camera": {
                "enabled": s.virtual_camera,
                "running": self.vcam_running,
                "device": self.vcam_device,
                "error": self.vcam_error or self.pipeline.output_errors.get(VCAM),
            },
            "source_error": self.source_error or self.pipeline.source_error,
            "stats": self.pipeline.stats(),
            "config_path": str(self.config_path) if self.config_path else None,
        }
