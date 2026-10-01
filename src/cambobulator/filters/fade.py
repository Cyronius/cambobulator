"""Fade into Background: blend the person toward a clean background plate.

    out = lerp(frame, background, fade * person_mask)

The background plate comes from one of three places, best first:

1. A captured plate: the user steps out of frame and presses "capture".
2. A learned plate: every pixel the segmenter is sure is *not* the person is
   copied into the plate the first time it is seen.
3. An inpainted guess, for pixels behind the person that have never been seen.

Captured and learned plates keep adapting slowly (``bg_learn_rate``) from
non-person pixels, so slow lighting drift does not reveal the plate.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cambobulator.filters import register_filter
from cambobulator.filters.base import Filter, FrameContext
from cambobulator.params import Action, Param

# Mask morphology runs at this width at most; masks are smooth, so this is
# plenty and keeps big feather/grow kernels cheap at 720p/1080p.
MASK_WORK_WIDTH = 480
# Pixels whose person-mask value is below this count as background.
BACKGROUND_CUTOFF = 0.02
# The "moving" side of auto-fade drops to 0 in this many seconds.
AUTO_DROP_SECONDS = 0.35


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _resize(img: np.ndarray, size: tuple[int, int], interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == size:
        return img
    return cv2.resize(img, size, interpolation=interp)


def refine_mask(raw: np.ndarray, threshold: float, grow: int, feather: int) -> np.ndarray:
    """Soft-threshold, grow and feather a raw person probability mask.

    ``grow`` and ``feather`` are in full-resolution pixels. Returns float32
    (H, W) in [0, 1].
    """
    h, w = raw.shape[:2]
    scale = min(1.0, MASK_WORK_WIDTH / w)
    work = (max(1, round(w * scale)), max(1, round(h * scale)))
    m = _resize(raw.astype(np.float32, copy=False), work, cv2.INTER_AREA)
    # A narrow ramp around the threshold instead of a hard step avoids
    # flicker when the model hovers near the threshold.
    m = np.clip((m - threshold) / 0.2 + 0.5, 0.0, 1.0)
    # The feather ramp is pushed outward (extra dilation by ``feather``) so the
    # mask is still ~1 at the person's real edge; otherwise a faint outline
    # of the person survives at full fade.
    g = int(round((grow + feather) * scale))
    if g > 0:
        m = cv2.dilate(m, _ellipse(g))
    sigma = feather * scale / 2.0
    if sigma > 0.3:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma)
    return _resize(m, (w, h))  # linear resize of [0, 1] data stays in [0, 1]


def difference_mask(frame: np.ndarray, plate: np.ndarray, level: float = 18.0) -> np.ndarray:
    """Fallback "segmenter": whatever differs from the background plate is the person."""
    h, w = frame.shape[:2]
    scale = min(1.0, 320 / w)
    work = (max(1, round(w * scale)), max(1, round(h * scale)))
    a = cv2.GaussianBlur(_resize(frame.astype(np.float32, copy=False), work, cv2.INTER_AREA), (5, 5), 0)
    b = cv2.GaussianBlur(_resize(plate, work, cv2.INTER_AREA), (5, 5), 0)
    diff = cv2.absdiff(a, b).max(axis=2)
    m = np.clip((diff - level) / level, 0.0, 1.0).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(3))
    return _resize(m, (w, h))


@register_filter
class FadeIntoBackground(Filter):
    NAME = "fade_into_background"
    LABEL = "Fade into Background"
    DESCRIPTION = "For people who would rather not be perceived. Blends you into your own background."
    USES_SEGMENTER = True
    PARAMS = (
        Param("fade", "float", 0.85, 0.0, 1.0, 0.01, "Fade", "0 = normal, 1 = fully invisible"),
        Param("threshold", "float", 0.5, 0.05, 0.95, 0.01, "Mask threshold",
              "How sure the segmenter must be that a pixel is you"),
        Param("grow", "int", 6, 0, 60, 1, "Grow mask (px)", "Expand the mask to hide edge halos"),
        Param("feather", "int", 14, 0, 80, 1, "Edge feather (px)", "Soften the mask edge (the soft ramp sits outside you)"),
        Param("bg_learn_rate", "float", 0.01, 0.0, 0.2, 0.005, "Background adapt rate",
              "How fast the plate follows lighting changes (0 = frozen)"),
        Param("capture_delay", "float", 3.0, 0.0, 15.0, 0.5, "Capture countdown (s)"),
        Param("auto_fade", "bool", False, label="Auto-fade when still",
              help="Fade in while you sit still; snap back when you move or talk"),
        Param("auto_ramp_seconds", "float", 6.0, 0.5, 60.0, 0.5, "Auto-fade ramp (s)",
              "Seconds of stillness to reach full fade"),
        Param("motion_sensitivity", "float", 0.6, 0.0, 1.0, 0.01, "Motion sensitivity"),
        Param("shimmer", "float", 0.0, 0.0, 1.0, 0.01, "Predator shimmer",
              "Refraction ripple, strongest at half fade"),
    )
    ACTIONS = (
        Action("capture_background", "Capture background now", "Step out of frame first"),
        Action("capture_background_delayed", "Capture after countdown", "Gives you time to step out"),
        Action("clear_background", "Forget background"),
    )

    def __init__(self, segmenter=None, **values: Any) -> None:
        super().__init__(segmenter=segmenter, **values)
        self._grid_cache: tuple[tuple[int, int], np.ndarray, np.ndarray] | None = None
        self._clear_state()
        self._capture_now = False
        self._countdown_requested = False
        self._capture_at: float | None = None
        self._last_now: float | None = None

    def _clear_state(self) -> None:
        self._shape: tuple[int, int] | None = None
        self._plate: np.ndarray | None = None  # float32 (H, W, 3)
        self._valid: np.ndarray | None = None  # bool (H, W): plate pixel has been seen
        self._all_valid = False
        self._known_fraction = 0.0
        self._captured = False
        self._prev_gray: np.ndarray | None = None
        self._auto_level = 0.0
        self._effective_fade = 0.0
        self._mode = "waiting"
        self._warning = ""

    # -- actions ------------------------------------------------------------
    def action_capture_background(self) -> None:
        self._capture_now = True

    def action_capture_background_delayed(self) -> None:
        self._countdown_requested = True

    def action_clear_background(self) -> None:
        self._capture_now = self._countdown_requested = False
        self._capture_at = None
        self._clear_state()

    def reset(self) -> None:
        # A new source usually means a new scene: the old plate is useless.
        self.action_clear_background()

    @property
    def has_captured_background(self) -> bool:
        return self._captured

    @property
    def background(self) -> np.ndarray | None:
        return None if self._plate is None else self._plate.copy()

    def set_background(self, plate: np.ndarray) -> None:
        """Use ``plate`` (BGR uint8 or float) as a captured background."""
        self._plate = plate.astype(np.float32).copy()
        h, w = plate.shape[:2]
        self._shape = (h, w)
        self._valid = np.ones((h, w), bool)
        self._all_valid = True
        self._known_fraction = 1.0
        self._captured = True
        self._warning = ""

    # -- persistence ----------------------------------------------------------
    def _asset_path(self, directory: Path, prefix: str) -> Path:
        return Path(directory) / f"{prefix}background.png"

    def save_assets(self, directory: Path, prefix: str) -> None:
        path = self._asset_path(directory, prefix)
        if self._captured and self._plate is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.clip(self._plate, 0, 255).astype(np.uint8))
        elif path.exists():
            path.unlink()

    def load_assets(self, directory: Path, prefix: str) -> None:
        path = self._asset_path(directory, prefix)
        if path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                self.set_background(img)

    # -- status -------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        if self._captured:
            background = "captured"
        elif self._all_valid:
            background = "learned automatically (capture one for best results)"
        elif self._plate is not None and self._known_fraction > 0:
            background = f"learning ({math.floor(self._known_fraction * 100)}% seen; capture for best results)"
        else:
            background = "none yet"
        countdown = None
        if self._capture_at is not None and self._last_now is not None:
            countdown = round(max(0.0, self._capture_at - self._last_now), 1)
        elif self._countdown_requested:
            countdown = float(self["capture_delay"])
        return {
            "background": background,
            "mask_source": self._mode,
            "countdown": countdown,
            "effective_fade": round(self._effective_fade, 2),
            "auto_level": round(self._auto_level, 2) if self["auto_fade"] else None,
            "warning": self._warning,
        }

    # -- processing -----------------------------------------------------------
    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        now = ctx.timestamp if ctx is not None else time.monotonic()
        dt = 0.0 if self._last_now is None else min(max(now - self._last_now, 0.0), 0.5)
        self._last_now = now
        h, w = frame.shape[:2]

        if self._shape != (h, w):
            self._on_new_shape(h, w)
        if self._countdown_requested:
            self._countdown_requested = False
            self._capture_at = now + self["capture_delay"]

        raw = self.segmenter.segment(frame) if self.segmenter is not None else None
        if raw is not None:
            self._mode = getattr(self.segmenter, "name", "segmenter")
        elif self._captured:
            raw = difference_mask(frame, self._plate)
            self._mode = "difference from background (no segmenter)"
        else:
            self._mode = "unavailable"

        if self._capture_now or (self._capture_at is not None and now >= self._capture_at):
            self._capture(frame, raw)
            return frame

        if raw is None:
            self._warning = (
                "Person segmentation is unavailable. Capture a background plate "
                "and Cambobulator will find you by comparing against it."
            )
            self._effective_fade = 0.0
            return frame
        if self._warning.startswith("Person segmentation"):
            self._warning = ""

        person = refine_mask(raw, self["threshold"], self["grow"], self["feather"])
        self._learn_background(frame, person)

        fade = self["fade"]
        if self["auto_fade"]:
            self._update_auto_level(frame, person, dt)
            fade *= self._auto_level
        self._effective_fade = fade
        if fade <= 0.0:
            return frame

        if self._all_valid:
            background = cv2.convertScaleAbs(self._plate)  # float -> uint8, rounded and saturated
        else:
            background = self._estimated_background(frame, person)
        if self["shimmer"] > 0.0:
            background = self._refract(background, person, fade, now)

        alpha = person * np.float32(fade)
        return cv2.blendLinear(background, frame, alpha, 1.0 - alpha)

    def _on_new_shape(self, h: int, w: int) -> None:
        if self._captured and self._plate is not None:
            self.set_background(cv2.resize(self._plate, (w, h), interpolation=cv2.INTER_LINEAR))
        else:
            self._plate = np.zeros((h, w, 3), np.float32)
            self._valid = np.zeros((h, w), bool)
            self._all_valid = False
            self._known_fraction = 0.0
        self._shape = (h, w)
        self._prev_gray = None

    def _capture(self, frame: np.ndarray, raw: np.ndarray | None) -> None:
        self._capture_now = False
        self._capture_at = None
        self.set_background(frame)
        # Only trust a real segmenter here; the difference mask was computed
        # against the *old* plate.
        if raw is not None and not self._mode.startswith("difference"):
            if float((raw > self["threshold"]).mean()) > 0.03:
                self._warning = (
                    "Someone was in frame during capture, so the background may contain a ghost. "
                    "Step out of view and capture again."
                )

    def _learn_background(self, frame: np.ndarray, person: np.ndarray) -> None:
        background_px = person < BACKGROUND_CUTOFF
        if not self._all_valid:
            fill = background_px & ~self._valid
            if fill.any():
                self._plate[fill] = frame[fill]
                self._valid |= fill
            self._known_fraction = float(self._valid.mean())
            self._all_valid = self._known_fraction >= 1.0
        rate = self["bg_learn_rate"]
        if rate > 0.0:
            cv2.accumulateWeighted(frame, self._plate, rate, mask=background_px.astype(np.uint8))

    def _estimated_background(self, frame: np.ndarray, person: np.ndarray) -> np.ndarray:
        """Plate where known; an inpainted, blurred guess for never-seen pixels. Returns uint8."""
        h, w = frame.shape[:2]
        scale = min(1.0, 112 / w)
        work = (max(1, round(w * scale)), max(1, round(h * scale)))
        # INTER_LINEAR rather than INTER_AREA: much faster at odd scale factors,
        # and the guess gets blurred anyway.
        valid_s = _resize(self._valid.view(np.uint8), work).astype(np.float32)
        plate_s = _resize(self._plate, work)
        frame_s = _resize(frame, work).astype(np.float32)
        known = valid_s[..., None] * plate_s + (1.0 - valid_s[..., None]) * frame_s
        small = np.clip(known, 0, 255).astype(np.uint8)
        hole = (_resize(person, work) > BACKGROUND_CUTOFF) & (valid_s < 0.99)
        hole = cv2.dilate(hole.astype(np.uint8) * 255, _ellipse(1))
        if hole.any():
            small = cv2.inpaint(small, hole, 3, cv2.INPAINT_TELEA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=1.5)
        guess = _resize(small, (w, h))
        # Soften the seam between real plate pixels and the guess, but never
        # give weight to plate pixels that were never seen.
        weight = _resize(np.minimum(cv2.GaussianBlur(valid_s, (0, 0), sigmaX=1.5), valid_s), (w, h))
        np.multiply(weight, self._valid, out=weight)
        return cv2.blendLinear(cv2.convertScaleAbs(self._plate), guess, weight, 1.0 - weight)

    def _update_auto_level(self, frame: np.ndarray, person: np.ndarray, dt: float) -> None:
        h, w = frame.shape[:2]
        scale = min(1.0, 160 / w)
        work = (max(1, round(w * scale)), max(1, round(h * scale)))
        gray = cv2.GaussianBlur(cv2.cvtColor(_resize(frame, work, cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (3, 3), 0)
        prev, self._prev_gray = self._prev_gray, gray
        if prev is None:
            return
        region = _resize(person, work, cv2.INTER_AREA) > 0.3
        moving_fraction = 0.0
        if region.sum() > 20:
            moving_fraction = float((cv2.absdiff(gray, prev)[region] > 10).mean())
        # sensitivity 0 -> 5% of the body must move; 1 -> 0.1% (a mouth is enough)
        limit = 0.05 * 0.02 ** self["motion_sensitivity"]
        if moving_fraction > limit:
            self._auto_level -= dt / AUTO_DROP_SECONDS
        else:
            self._auto_level += dt / self["auto_ramp_seconds"]
        self._auto_level = min(1.0, max(0.0, self._auto_level))

    def _refract(self, background: np.ndarray, person: np.ndarray, fade: float, now: float) -> np.ndarray:
        """Predator-style ripple: look at the background through a wobbly lens."""
        h, w = background.shape[:2]
        amplitude = self["shimmer"] * 10.0 * (w / 640.0) * math.sin(math.pi * fade)
        if amplitude < 0.2:
            return background
        if self._grid_cache is None or self._grid_cache[0] != (h, w):
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid_cache = ((h, w), gx, gy)
        _, gx, gy = self._grid_cache
        sw, sh = max(1, w // 4), max(1, h // 4)
        x, y = np.meshgrid(np.linspace(0, w, sw, dtype=np.float32), np.linspace(0, h, sh, dtype=np.float32))
        t = now % 1000.0
        dx = np.sin(y / 23.0 + t * 3.1) + 0.5 * np.sin((x + y) / 11.0 - t * 4.7)
        dy = np.cos(x / 19.0 - t * 2.3) + 0.5 * np.cos((x - y) / 13.0 + t * 3.9)
        lens = _resize(person, (sw, sh), cv2.INTER_AREA) * amplitude
        dx = _resize((dx * lens).astype(np.float32), (w, h))
        dy = _resize((dy * lens).astype(np.float32), (w, h))
        return cv2.remap(background, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
