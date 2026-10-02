import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from cambobulator import cli
from cambobulator.app import Controller
from cambobulator.config import FilterConfig, Settings
from cambobulator.driver import write_format_hint  # the real one; conftest stubs the module attribute
from cambobulator.filters import (
    Filter,
    available_filters,
    create_filter,
    get_filter_class,
    register_filter,
    unregister_filter,
)
from cambobulator.outputs import VirtualCameraError, VirtualCameraOutput, virtual_camera_help
from cambobulator.params import Param
from cambobulator.segmentation import LazySegmenter, NullSegmenter, SelfieSegmenter, make_segmenter
from cambobulator.sources import SourceError, SyntheticSource, VideoFileSource, open_source

DATA = Path(__file__).parent / "data"


# -- params -------------------------------------------------------------------
def test_param_coercion():
    p = Param("x", "float", 0.5, 0.0, 1.0)
    assert p.coerce("0.25") == 0.25
    assert p.coerce(5) == 1.0 and p.coerce(-1) == 0.0
    assert Param("n", "int", 1, 0, 10).coerce(3.6) == 4
    b = Param("b", "bool", False)
    assert b.coerce("true") is True and b.coerce("0") is False and b.coerce(1) is True
    c = Param("c", "choice", "a", choices=("a", "b"))
    assert c.coerce("b") == "b"
    with pytest.raises(ValueError):
        c.coerce("z")
    with pytest.raises(ValueError):
        p.coerce("abc")
    with pytest.raises(ValueError):
        Param("bad", "complex", 0)


# -- registry -------------------------------------------------------------------
def test_registry_knows_only_the_effect():
    assert [f["type"] for f in available_filters()] == ["fade_into_background"]
    with pytest.raises(KeyError):
        get_filter_class("nope")


def test_custom_filter_registration():
    @register_filter
    class Invert(Filter):
        NAME = "test_invert"
        LABEL = "Invert"

        def process(self, frame, ctx=None):
            return 255 - frame

    try:
        assert create_filter("test_invert").process(np.zeros((2, 2, 3), np.uint8))[0, 0, 0] == 255

        class Clash(Filter):
            NAME = "test_invert"

        with pytest.raises(ValueError):
            register_filter(Clash)
    finally:
        unregister_filter("test_invert")


# -- config -------------------------------------------------------------------
def test_settings_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    s = Settings(source="synthetic", width=640, height=360,
                 filters=[FilterConfig("mirror", False, {"direction": "vertical"}, "m1")])
    s.save(path)
    loaded = Settings.load(path)
    assert loaded.source == "synthetic" and loaded.width == 640
    assert loaded.filters[0].type == "mirror" and loaded.filters[0].params == {"direction": "vertical"}
    assert loaded.filters[0].enabled is False and loaded.filters[0].id == "m1"


def test_settings_defaults_and_bad_files(tmp_path):
    assert Settings.load(tmp_path / "missing.json").filters[0].type == "fade_into_background"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert Settings.load(bad).source == "0"
    assert (tmp_path / "bad.json.bad").exists()
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"source": 2, "future_option": True, "filters": [{"nope": 1}]}))
    s = Settings.load(extra)
    assert s.source == "2" and s.filters == []


def test_old_filter_chain_config_loads_as_the_one_effect():
    old = Settings(source="synthetic", filters=[
        FilterConfig("mirror"),
        FilterConfig("fade_into_background", params={"fade": 0.4, "feather": 30, "capture_delay": 3}, id="f1"),
        FilterConfig("pixelate", params={"block": 8}),
    ])
    slots = Controller(old, None, segmenter=NullSegmenter()).pipeline.slots
    assert [(s.id, s.filter.NAME, s.filter["fade"]) for s in slots] == [("f1", "fade_into_background", 0.4)]
    assert Controller(Settings(filters=[]), None, segmenter=NullSegmenter()).pipeline.slots[0].filter.NAME \
        == "fade_into_background"


# -- sources -------------------------------------------------------------------
def test_synthetic_source_and_segmenter():
    src = SyntheticSource(320, 240, fps=30, realtime=False)
    frame = src.read()
    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
    assert src.person_mask().sum() > 0
    found, truth = make_segmenter("synthetic").segment(frame), src.person_mask()
    # The timestamp text can cover a few person pixels; nothing else may differ.
    assert found[truth == 0].max() == 0
    assert found.sum() > 0.95 * truth.sum()


def test_video_file_source_loops(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, np.uint8))
    writer.release()
    src = open_source(str(path), width=64, height=48)
    assert isinstance(src, VideoFileSource)
    src.realtime = False
    frames = [src.read() for _ in range(8)]  # more than the file has: must loop
    assert all(f is not None and f.shape == (48, 64, 3) for f in frames)
    src.close()
    resized = open_source(str(path), width=128, height=96)  # files follow the configured size
    assert resized.read().shape == (96, 128, 3)
    resized.close()


def test_missing_sources_give_clear_errors(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        open_source(str(tmp_path / "nope.mp4"))
    with pytest.raises(SourceError, match="Could not open camera"):
        open_source("camera:57")


# -- segmentation ---------------------------------------------------------------
def test_lazy_segmenter_falls_back_on_failure():
    def broken():
        raise RuntimeError("no model here")

    seg = LazySegmenter(broken)
    assert seg.segment(np.zeros((4, 4, 3), np.uint8)) is None
    assert "no model" in seg.error
    assert isinstance(make_segmenter("none"), NullSegmenter)
    with pytest.raises(ValueError):
        make_segmenter("magic")


def test_selfie_segmenter_matches_mediapipe():
    # The bundled ONNX model must give the same mask MediaPipe gave for this photo (see tests/data/README.txt).
    frame = cv2.imread(str(DATA / "portrait_256x144.png"))
    expected = cv2.imread(str(DATA / "portrait_256x144_mediapipe_mask.png"), cv2.IMREAD_GRAYSCALE) / 255.0
    seg = SelfieSegmenter()
    mask = seg.segment(frame)
    assert mask.shape == (144, 256) and mask.dtype == np.float32
    assert np.abs(mask - expected).max() < 0.01
    # A full-size frame is shrunk for the model and the mask comes back at frame size.
    big = seg.segment(cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC))
    assert big.shape == (720, 1280)
    person, truth = big > 0.5, cv2.resize(expected, (1280, 720)) > 0.5
    assert (person & truth).sum() / (person | truth).sum() > 0.97


# -- virtual camera errors ---------------------------------------------------------
def test_virtual_camera_error_explains_setup(monkeypatch):
    fake = types.ModuleType("pyvirtualcam")

    class PixelFormat:
        BGR = "bgr"

    def camera(*a, **kw):
        raise RuntimeError("'obs' backend: virtual camera output could not be started")

    fake.PixelFormat = PixelFormat
    fake.Camera = camera
    monkeypatch.setitem(sys.modules, "pyvirtualcam", fake)
    with pytest.raises(VirtualCameraError) as err:
        VirtualCameraOutput(640, 480, 30)
    assert "could not be started" in str(err.value)
    assert virtual_camera_help() in str(err.value)


def test_format_hint_uses_obs_file_format(tmp_path):
    # OBS's module sscanf's "%ux%ux%llu": width x height x frame interval in 100 ns units.
    path = tmp_path / "obs-virtualcam.txt"
    write_format_hint(1280, 720, 30, path)
    assert path.read_text() == "1280x720x333333"


def test_virtual_camera_help_mentions_platform_driver(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert "OBS" in virtual_camera_help()
    monkeypatch.setattr(sys, "platform", "linux")
    assert "v4l2loopback" in virtual_camera_help()


# -- CLI ------------------------------------------------------------------------
def test_cli_run_headless_records_synthetic(tmp_path, capsys):
    out = tmp_path / "out.avi"
    code = cli.main(["run", "--config", str(tmp_path / "c.json"), "--source", "synthetic", "--width", "160",
                     "--height", "120", "--segmenter", "none", "--no-vcam", "--record", str(out),
                     "--max-frames", "10"])
    assert code == 0
    assert out.stat().st_size > 0
    saved = json.loads((tmp_path / "c.json").read_text())
    assert saved["source"] == "synthetic" and saved["filters"][0]["type"] == "fade_into_background"


def test_cli_run_reports_missing_camera(tmp_path, capsys):
    code = cli.main(["run", "--config", str(tmp_path / "c.json"), "--source", "57", "--no-vcam",
                     "--segmenter", "none"])
    assert code == 2
    assert "Could not open camera 57" in capsys.readouterr().err


def test_cli_filters_lists_params(capsys):
    assert cli.main(["filters"]) == 0
    text = capsys.readouterr().out
    assert "fade_into_background" in text and "capture_background" in text
