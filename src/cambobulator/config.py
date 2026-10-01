"""Settings, stored as JSON so `cambobulator run` can reuse what the UI set up."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def default_config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "cambobulator"


def default_config_path() -> Path:
    return default_config_dir() / "config.json"


@dataclass
class FilterConfig:
    type: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FilterConfig":
        return cls(type=str(d["type"]), enabled=bool(d.get("enabled", True)),
                   params=dict(d.get("params") or {}), id=d.get("id"))


def _default_filters() -> list[FilterConfig]:
    return [FilterConfig("fade_into_background")]


@dataclass
class Settings:
    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    virtual_camera: bool = True
    vcam_backend: str | None = None
    vcam_device: str | None = None
    segmenter: str = "auto"
    segmenter_model: str | None = None
    ui_host: str = "127.0.0.1"
    ui_port: int = 8765
    filters: list[FilterConfig] = field(default_factory=_default_filters)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in d.items() if k in known and k != "filters"}
        settings = cls(**values)
        if "filters" in d:
            settings.filters = []
            for item in d["filters"] or []:
                try:
                    settings.filters.append(FilterConfig.from_dict(item))
                except (KeyError, TypeError):
                    log.warning("Ignoring malformed filter entry in config: %r", item)
        settings.source = str(settings.source)
        settings.width, settings.height, settings.fps = int(settings.width), int(settings.height), int(settings.fps)
        return settings

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings; a missing or unreadable file gives defaults (and is never overwritten silently)."""
        if not path.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            backup = path.with_suffix(path.suffix + ".bad")
            log.warning("Could not read %s (%s); using defaults. The old file was moved to %s", path, exc, backup)
            try:
                path.replace(backup)
            except OSError:
                pass
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)
