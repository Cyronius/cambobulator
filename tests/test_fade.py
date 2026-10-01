import numpy as np
import pytest

from cambobulator.filters import FrameContext, create_filter
from cambobulator.filters.fade import FadeIntoBackground, difference_mask, refine_mask
from conftest import BOX, MaskBox, make_background, make_frame, make_person_mask

INSIDE = (slice(50, 80), slice(70, 100))  # well inside BOX
OUTSIDE = (slice(0, 30), slice(0, 40))  # far from BOX


def sharp(seg, **kw):
    """A fade filter with no mask softening, so the math is exact."""
    params = dict(feather=0, grow=0, bg_learn_rate=0.0, threshold=0.5)
    params.update(kw)
    return FadeIntoBackground(segmenter=seg, **params)


def ctx(t: float, i: int = 0) -> FrameContext:
    return FrameContext(timestamp=t, index=i)


def test_fade_zero_is_identity(scene, fake_segmenter):
    bg, _, frame = scene
    f = sharp(fake_segmenter, fade=0.0)
    f.set_background(bg)
    out = f.process(frame.copy(), ctx(0))
    assert np.array_equal(out, frame)


def test_full_fade_replaces_person_with_plate(scene, fake_segmenter):
    bg, _, frame = scene
    f = sharp(fake_segmenter, fade=1.0)
    f.set_background(bg)
    out = f.process(frame.copy(), ctx(0))
    assert np.array_equal(out[BOX], bg[BOX])  # person gone
    assert np.array_equal(out[OUTSIDE], frame[OUTSIDE])  # background untouched
    assert out.dtype == np.uint8 and out.shape == frame.shape


def test_half_fade_is_linear_blend(scene, fake_segmenter):
    bg, _, frame = scene
    f = sharp(fake_segmenter, fade=0.5)
    f.set_background(bg)
    out = f.process(frame.copy(), ctx(0)).astype(int)
    expected = (frame.astype(float) * 0.5 + bg.astype(float) * 0.5)[BOX]
    assert np.abs(out[BOX] - expected).max() <= 1


def test_threshold_controls_what_counts_as_person(scene):
    bg, mask, frame = scene
    seg = MaskBox(mask * 0.4)  # an unsure segmenter
    f = sharp(seg, fade=1.0, threshold=0.7)
    f.set_background(bg)
    assert np.array_equal(f.process(frame.copy(), ctx(0)), frame)
    f.set_param("threshold", 0.2)
    assert np.array_equal(f.process(frame.copy(), ctx(1))[INSIDE], bg[INSIDE])


def test_feather_makes_soft_edges():
    raw = make_person_mask()
    hard = refine_mask(raw, 0.5, grow=0, feather=0)
    soft = refine_mask(raw, 0.5, grow=0, feather=12)
    assert set(np.unique(hard)) <= {0.0, 1.0}
    edge_row = soft[65, 50:120]
    assert ((edge_row > 0.05) & (edge_row < 0.95)).sum() > 4  # a gradient, not a step
    assert soft[65, 85] > 0.95 and soft[65, 0] < 0.01


def test_grow_expands_mask():
    raw = make_person_mask()
    grown = refine_mask(raw, 0.5, grow=5, feather=0)
    assert grown[65, 57] == 1.0  # 3 px left of the box
    assert refine_mask(raw, 0.5, grow=0, feather=0)[65, 57] == 0.0


def test_rolling_update_tracks_lighting_outside_person(scene, fake_segmenter):
    bg, mask, _ = scene
    f = sharp(fake_segmenter, fade=1.0, bg_learn_rate=0.2)
    f.set_background(bg)
    brighter = np.clip(bg.astype(int) + 40, 0, 255).astype(np.uint8)
    frame = make_frame(brighter, mask)
    for i in range(40):
        f.process(frame.copy(), ctx(i / 30, i))
    plate = f.background
    assert np.abs(plate[OUTSIDE] - brighter[OUTSIDE]).max() < 2  # followed the lighting
    assert np.abs(plate[INSIDE] - bg[INSIDE]).max() < 1  # did not learn the person


def test_without_plate_learns_background_and_inpaints_hole(fake_segmenter):
    # Uniform wall: the inpainted guess behind the person should match it.
    wall = np.full((120, 160, 3), (90, 140, 200), np.uint8)
    mask = make_person_mask()
    frame = make_frame(wall, mask)
    f = sharp(fake_segmenter, fade=1.0)
    out = f.process(frame.copy(), ctx(0))
    assert not f.has_captured_background
    assert "learning" in f.status()["background"]
    assert np.abs(out[INSIDE].astype(int) - wall[INSIDE]).max() <= 6
    assert np.array_equal(out[OUTSIDE], frame[OUTSIDE])


def test_learned_plate_fills_in_when_person_moves():
    bg = make_background()
    seg = MaskBox(make_person_mask())
    f = sharp(seg, fade=1.0)
    f.process(make_frame(bg, seg.mask), ctx(0))
    assert f.status()["background"].startswith("learning")
    moved = make_person_mask(box=(slice(40, 90), slice(0, 40)))
    seg.mask = moved
    f.process(make_frame(bg, moved), ctx(0.1))
    assert f.status()["background"].startswith("learned automatically")
    seg.mask = make_person_mask()
    out = f.process(make_frame(bg, seg.mask), ctx(0.2))
    assert np.array_equal(out[BOX], bg[BOX])  # exact, from the learned plate


def test_capture_now(scene, fake_segmenter):
    bg, mask, frame = scene
    fake_segmenter.mask = np.zeros_like(mask)  # nobody in frame
    f = sharp(fake_segmenter, fade=1.0)
    f.run_action("capture_background")
    f.process(bg.copy(), ctx(0))
    assert f.has_captured_background
    assert f.status()["warning"] == ""
    fake_segmenter.mask = mask
    assert np.array_equal(f.process(frame.copy(), ctx(1))[BOX], bg[BOX])


def test_capture_warns_if_person_present(scene, fake_segmenter):
    _, _, frame = scene
    f = sharp(fake_segmenter)
    f.run_action("capture_background")
    f.process(frame.copy(), ctx(0))
    assert "ghost" in f.status()["warning"]


def test_delayed_capture_uses_frame_timestamps(scene, fake_segmenter):
    bg, mask, frame = scene
    f = sharp(fake_segmenter, capture_delay=3.0)
    f.run_action("capture_background_delayed")
    f.process(frame.copy(), ctx(10.0))
    assert f.status()["countdown"] == 3.0
    f.process(frame.copy(), ctx(12.0))
    assert not f.has_captured_background
    assert f.status()["countdown"] == 1.0
    fake_segmenter.mask = np.zeros_like(mask)
    f.process(bg.copy(), ctx(13.01))
    assert f.has_captured_background
    assert f.status()["countdown"] is None


def test_clear_background(scene, fake_segmenter):
    bg, _, _ = scene
    f = sharp(fake_segmenter)
    f.set_background(bg)
    f.run_action("clear_background")
    assert not f.has_captured_background and f.background is None


def test_difference_fallback_when_segmenter_unavailable(scene):
    bg, _, frame = scene
    f = sharp(MaskBox(None), fade=1.0)
    out = f.process(frame.copy(), ctx(0))
    assert np.array_equal(out, frame)  # nothing to go on yet
    assert "unavailable" in f.status()["warning"]
    f.set_background(bg)
    out = f.process(frame.copy(), ctx(1))
    assert f.status()["mask_source"].startswith("difference")
    assert np.abs(out[INSIDE].astype(int) - bg[INSIDE]).max() <= 2
    assert np.array_equal(out[OUTSIDE], frame[OUTSIDE])


def test_difference_mask_finds_changed_pixels(scene):
    bg, mask, frame = scene
    m = difference_mask(frame, bg.astype(np.float32))
    assert m[INSIDE].min() > 0.9 and m[OUTSIDE].max() < 0.1


def test_no_segmenter_at_all_passes_through(scene):
    _, _, frame = scene
    f = FadeIntoBackground(segmenter=None)
    assert np.array_equal(f.process(frame.copy(), ctx(0)), frame)


def test_auto_fade_ramps_when_still_and_drops_on_motion(scene, fake_segmenter):
    bg, mask, frame = scene
    f = sharp(fake_segmenter, fade=1.0, auto_fade=True, auto_ramp_seconds=2.0)
    f.set_background(bg)
    t = 0.0
    for i in range(70):  # ~2.3 s of a perfectly still person
        f.process(frame.copy(), ctx(t, i))
        t += 1 / 30
    assert f.status()["auto_level"] == 1.0
    assert np.array_equal(f.process(frame.copy(), ctx(t))[BOX], bg[BOX])
    # Now he waves: the person region changes a lot every frame.
    rng = np.random.default_rng(0)
    for i in range(10):
        t += 1 / 30
        moving = frame.copy()
        moving[BOX] = rng.integers(0, 255, moving[BOX].shape, dtype=np.uint8)
        f.process(moving, ctx(t, i))
    assert f.status()["auto_level"] < 0.3
    assert f.status()["effective_fade"] < 0.3


def test_shimmer_runs_and_vanishes_at_full_fade(scene, fake_segmenter):
    bg, _, frame = scene
    plain = sharp(fake_segmenter, fade=1.0)
    shimmer = sharp(fake_segmenter, fade=1.0, shimmer=1.0)
    for f in (plain, shimmer):
        f.set_background(bg)
    assert np.array_equal(plain.process(frame.copy(), ctx(1)), shimmer.process(frame.copy(), ctx(1)))
    shimmer.set_param("fade", 0.5)
    plain.set_param("fade", 0.5)
    a, b = plain.process(frame.copy(), ctx(2)), shimmer.process(frame.copy(), ctx(2))
    assert a.shape == b.shape and not np.array_equal(a, b)
    assert np.array_equal(a[OUTSIDE], b[OUTSIDE])  # only the person refracts


def test_resolution_change_resizes_captured_plate(scene, fake_segmenter):
    bg, _, _ = scene
    f = sharp(fake_segmenter, fade=1.0)
    f.set_background(bg)
    big_bg = make_background(320, 240)
    big_mask = make_person_mask(320, 240, box=(slice(80, 180), slice(120, 220)))
    fake_segmenter.mask = big_mask
    out = f.process(make_frame(big_bg, big_mask), ctx(0))
    assert out.shape == (240, 320, 3)
    assert f.has_captured_background


def test_save_and_load_assets(tmp_path, scene, fake_segmenter):
    bg, _, frame = scene
    f = sharp(fake_segmenter)
    f.set_background(bg)
    f.save_assets(tmp_path, "abc-")
    assert (tmp_path / "abc-background.png").is_file()
    g = sharp(fake_segmenter, fade=1.0)
    g.load_assets(tmp_path, "abc-")
    assert g.has_captured_background
    assert np.array_equal(g.process(frame.copy(), ctx(0))[BOX], bg[BOX])
    g.run_action("clear_background")
    g.save_assets(tmp_path, "abc-")
    assert not (tmp_path / "abc-background.png").exists()


def test_registry_creates_fade_with_defaults(fake_segmenter):
    f = create_filter("fade_into_background", segmenter=fake_segmenter, fade=2.0)
    assert f["fade"] == 1.0  # clamped
    assert {p["name"] for p in f.describe()["params"]} >= {"fade", "feather", "threshold"}
    with pytest.raises(KeyError):
        f.run_action("nope")


def test_realistic_defaults_hide_person(scene, fake_segmenter):
    """With default grow/feather the person interior is fully replaced."""
    bg, _, frame = scene
    f = FadeIntoBackground(segmenter=fake_segmenter, fade=1.0)
    f.set_background(bg)
    out = f.process(frame.copy(), ctx(0))
    assert np.abs(out[INSIDE].astype(int) - bg[INSIDE]).max() <= 1
