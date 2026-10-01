"""Small general-purpose filters. They also serve as examples."""

from __future__ import annotations

import cv2
import numpy as np

from cambobulator.filters import register_filter
from cambobulator.filters.base import Filter, FrameContext
from cambobulator.params import Param


@register_filter
class Mirror(Filter):
    NAME = "mirror"
    LABEL = "Mirror"
    DESCRIPTION = "Flip the image."
    PARAMS = (
        Param("direction", "choice", "horizontal", choices=("horizontal", "vertical", "both"), label="Direction"),
    )

    _CODES = {"horizontal": 1, "vertical": 0, "both": -1}

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        return cv2.flip(frame, self._CODES[self["direction"]])


@register_filter
class Adjust(Filter):
    NAME = "adjust"
    LABEL = "Brightness / Contrast / Saturation"
    DESCRIPTION = "Basic colour correction."
    PARAMS = (
        Param("brightness", "float", 0.0, -100, 100, 1, "Brightness"),
        Param("contrast", "float", 1.0, 0.2, 3.0, 0.05, "Contrast"),
        Param("saturation", "float", 1.0, 0.0, 3.0, 0.05, "Saturation"),
    )

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        b, c, s = self["brightness"], self["contrast"], self["saturation"]
        if s != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=s)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if b != 0.0 or c != 1.0:
            # Contrast pivots around mid-grey so it does not also shift brightness.
            frame = cv2.convertScaleAbs(frame, alpha=c, beta=b + 128.0 * (1.0 - c))
        return frame


@register_filter
class Pixelate(Filter):
    NAME = "pixelate"
    LABEL = "Pixelate"
    DESCRIPTION = "Chunky pixels."
    PARAMS = (Param("block", "int", 16, 2, 96, 1, "Block size (px)"),)

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        h, w = frame.shape[:2]
        block = self["block"]
        small = cv2.resize(frame, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
