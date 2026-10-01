"""Filter base class."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from cambobulator.params import Action, Param

if TYPE_CHECKING:
    from cambobulator.segmentation import Segmenter


@dataclass
class FrameContext:
    """Per-frame information handed to every filter in the chain."""

    timestamp: float
    """Seconds on a monotonic clock. Filters use this instead of time.time()
    so tests can drive time deterministically."""
    index: int = 0
    raw: np.ndarray | None = None
    """The frame as captured, before any filter ran."""

    @classmethod
    def now(cls, index: int = 0, raw: np.ndarray | None = None) -> "FrameContext":
        return cls(timestamp=time.monotonic(), index=index, raw=raw)


class Filter:
    """Base class for all filters.

    Subclasses set ``NAME`` (a unique registry id), ``LABEL``, ``DESCRIPTION``,
    ``PARAMS`` and optionally ``ACTIONS``, then implement :meth:`process`.

    Frames are ``numpy.uint8`` arrays in OpenCV's BGR order, shape (H, W, 3).
    ``process`` must return a frame of the same shape and dtype; it may modify
    the input in place.
    """

    NAME: ClassVar[str] = ""
    LABEL: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    PARAMS: ClassVar[tuple[Param, ...]] = ()
    ACTIONS: ClassVar[tuple[Action, ...]] = ()
    USES_SEGMENTER: ClassVar[bool] = False

    def __init__(self, segmenter: "Segmenter | None" = None, **values: Any) -> None:
        self.segmenter = segmenter
        self.values: dict[str, Any] = {p.name: p.default for p in self.PARAMS}
        for name, value in values.items():
            self.set_param(name, value)

    # -- params -----------------------------------------------------------
    def param(self, name: str) -> Param:
        for p in self.PARAMS:
            if p.name == name:
                return p
        raise KeyError(f"{self.NAME}: no parameter named {name!r}")

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def set_param(self, name: str, value: Any) -> Any:
        coerced = self.param(name).coerce(value)
        self.values[name] = coerced
        self.on_param_changed(name, coerced)
        return coerced

    def on_param_changed(self, name: str, value: Any) -> None:
        """Hook for subclasses that cache derived values."""

    # -- actions ----------------------------------------------------------
    def run_action(self, name: str) -> None:
        if not any(a.name == name for a in self.ACTIONS):
            raise KeyError(f"{self.NAME}: no action named {name!r}")
        getattr(self, f"action_{name}")()

    # -- lifecycle --------------------------------------------------------
    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Called when the source changes (new camera, new resolution)."""

    def status(self) -> dict[str, Any]:
        """Small, JSON-safe dict shown in the UI under the filter's controls."""
        return {}

    def save_assets(self, directory: Path, prefix: str) -> None:
        """Persist large state (images) next to the config file."""

    def load_assets(self, directory: Path, prefix: str) -> None:
        """Restore what :meth:`save_assets` wrote. Missing files are fine."""

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.NAME,
            "label": self.LABEL or self.NAME,
            "description": self.DESCRIPTION,
            "params": [dict(p.to_dict(), value=self.values[p.name]) for p in self.PARAMS],
            "actions": [a.to_dict() for a in self.ACTIONS],
            "status": self.status(),
        }
