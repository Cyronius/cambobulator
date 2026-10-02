import time

import numpy as np
import pytest

from cambobulator.filters import Filter, register_filter, unregister_filter
from cambobulator.outputs import CollectOutput
from cambobulator.params import Param
from cambobulator.pipeline import Pipeline
from cambobulator.sources import SyntheticSource


class AddN(Filter):
    NAME = "test_add"
    PARAMS = (Param("n", "int", 1, 0, 100),)

    def process(self, frame, ctx=None):
        return (frame.astype(np.int32) + self["n"]).clip(0, 255).astype(np.uint8)


class Double(Filter):
    NAME = "test_double"

    def process(self, frame, ctx=None):
        return (frame.astype(np.int32) * 2).clip(0, 255).astype(np.uint8)


class Boom(Filter):
    NAME = "test_boom"

    def process(self, frame, ctx=None):
        raise RuntimeError("kaboom")


@pytest.fixture(autouse=True)
def test_filters():
    for cls in (AddN, Double, Boom):
        register_filter(cls)
    yield
    for cls in (AddN, Double, Boom):
        unregister_filter(cls.NAME)


def px(frame):
    return int(frame[0, 0, 0])


def frame_of(v=10):
    return np.full((4, 4, 3), v, np.uint8)


def test_filters_run_in_the_order_added():
    p = Pipeline()
    p.add_filter("test_add", params={"n": 1})
    p.add_filter("test_double")
    assert px(p.process_frame(frame_of())) == 22  # (10 + 1) * 2


def test_disabled_filters_are_skipped_and_params_apply():
    p = Pipeline()
    add = p.add_filter("test_add")
    p.set_param(add.id, "n", 5)
    assert px(p.process_frame(frame_of())) == 15
    p.set_enabled(add.id, False)
    assert px(p.process_frame(frame_of())) == 10


def test_failing_filter_passes_frame_through_and_reports():
    p = Pipeline()
    boom = p.add_filter("test_boom")
    p.add_filter("test_add")
    assert px(p.process_frame(frame_of())) == 11
    assert "kaboom" in p.describe()[0]["error"]
    assert boom.error


def test_unknown_params_from_config_are_ignored():
    p = Pipeline()
    slot = p.add_filter("test_add", params={"n": 3, "bogus": 1})
    assert slot.filter["n"] == 3


def test_threaded_run_with_synthetic_source():
    p = Pipeline()
    p.add_filter("test_add", params={"n": 1})
    out = CollectOutput()
    p.add_output("collect", out)
    src = SyntheticSource(64, 48, fps=200)
    p.start()
    p.set_source(src)
    deadline = time.monotonic() + 5
    while len(out.frames) < 10 and time.monotonic() < deadline:
        time.sleep(0.01)
    seq, raw, processed = p.wait_preview(0, timeout=1)
    p.stop()
    assert len(out.frames) >= 10
    assert out.closed
    assert raw.shape == processed.shape == (48, 64, 3)
    assert np.array_equal(processed, np.minimum(raw.astype(int) + 1, 255))  # the filter ran
    assert p.stats()["frames"] >= 10


def test_failing_output_is_dropped_but_pipeline_keeps_going():
    class BadOutput:
        def send(self, frame):
            raise OSError("driver went away")

        def close(self):
            pass

    p = Pipeline()
    good = CollectOutput()
    p.add_output("bad", BadOutput())
    p.add_output("good", good)
    p.start()
    p.set_source(SyntheticSource(32, 24, fps=200))
    deadline = time.monotonic() + 5
    while len(good.frames) < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    p.stop()
    assert len(good.frames) >= 5
    assert "driver went away" in p.output_errors["bad"]
