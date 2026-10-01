import numpy as np
import pytest

from cambobulator.segmentation import CallableSegmenter

W, H = 160, 120
BOX = (slice(40, 90), slice(60, 110))  # rows, cols of the fake "person"


def make_background(w: int = W, h: int = H) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.zeros((h, w, 3), np.uint8)
    bg[..., 0] = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    bg[..., 1] = (yy * 255 // max(1, h - 1)).astype(np.uint8)
    bg[..., 2] = 100
    return bg


def make_person_mask(w: int = W, h: int = H, box=BOX) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    m[box] = 1.0
    return m


def make_frame(bg: np.ndarray, mask: np.ndarray, colour=(10, 200, 30)) -> np.ndarray:
    frame = bg.copy()
    frame[mask > 0.5] = colour
    return frame


class MaskBox:
    """A fake segmenter whose mask the test can move around."""

    name = "fake"

    def __init__(self, mask: np.ndarray | None) -> None:
        self.mask = mask
        self.calls = 0

    def segment(self, frame):
        self.calls += 1
        return None if self.mask is None else self.mask.copy()

    def close(self):
        pass


@pytest.fixture
def scene():
    bg = make_background()
    mask = make_person_mask()
    return bg, mask, make_frame(bg, mask)


@pytest.fixture
def fake_segmenter(scene):
    return MaskBox(scene[1])


@pytest.fixture
def callable_segmenter(scene):
    return CallableSegmenter(lambda f: scene[1].copy(), name="fake")
