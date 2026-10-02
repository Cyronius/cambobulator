import time

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from cambobulator.app import Controller  # noqa: E402
from cambobulator.config import FilterConfig, Settings  # noqa: E402
from cambobulator.web.server import create_app  # noqa: E402
from conftest import MaskBox  # noqa: E402


@pytest.fixture
def controller(tmp_path):
    settings = Settings(source="synthetic", width=160, height=120, fps=60, virtual_camera=False,
                        filters=[FilterConfig("fade_into_background")])
    c = Controller(settings, tmp_path / "config.json", segmenter=MaskBox(None), autosave_delay=0.05)
    c.start()
    yield c
    c.close()


@pytest.fixture
def client(controller):
    return TestClient(create_app(controller))


def test_state_and_index(client):
    assert "Cambobulator" in client.get("/").text
    st = client.get("/api/state").json()
    assert [s["type"] for s in st["chain"]] == ["fade_into_background"]
    assert st["virtual_camera"]["running"] is False


def test_edit_effect_over_http(client, tmp_path):
    fade_id = client.get("/api/state").json()["chain"][0]["id"]
    r = client.post(f"/api/filters/{fade_id}/params", json={"fade": 0.3, "pixelate": 8, "auto_fade": "true"})
    assert r.json()["values"] == {"fade": 0.3, "pixelate": 8, "auto_fade": True}
    assert client.post(f"/api/filters/{fade_id}/params", json={"nope": 1}).status_code == 400
    assert client.post(f"/api/filters/{fade_id}/enabled", json={"enabled": False}).json()["ok"]
    assert client.get("/api/state").json()["chain"][0]["enabled"] is False

    # Autosave writes the config.
    config = tmp_path / "config.json"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not (
            config.exists() and Settings.load(config).filters[0].params.get("pixelate") == 8):
        time.sleep(0.05)
    saved = Settings.load(config)
    assert len(saved.filters) == 1 and saved.filters[0].enabled is False
    assert saved.filters[0].params["fade"] == 0.3 and saved.filters[0].params["pixelate"] == 8

    assert client.post(f"/api/filters/{fade_id}/actions/capture_background").json()["ok"]
    assert 0 < client.get("/api/state").json()["chain"][0]["status"]["countdown"] <= 5.0
    assert client.post(f"/api/filters/{fade_id}/actions/bogus").status_code == 400


def test_source_switch_errors_are_reported(client):
    r = client.post("/api/source", json={"source": "/no/such/file.mp4"}).json()
    assert r["ok"] is False and "not found" in r["error"]
    assert "not found" in client.get("/api/state").json()["source_error"]
    assert client.post("/api/source", json={"source": "synthetic"}).json()["ok"] is True


def test_virtual_camera_toggle_reports_driver_problem(client, monkeypatch):
    import cambobulator.app as app_mod
    from cambobulator.outputs import VirtualCameraError

    def no_driver(*a, **kw):
        raise VirtualCameraError("no driver; install OBS")

    monkeypatch.setattr(app_mod, "VirtualCameraOutput", no_driver)
    r = client.post("/api/virtual-camera", json={"enabled": True}).json()
    assert r["ok"] is False and "install OBS" in r["error"]
    assert client.get("/api/state").json()["virtual_camera"]["running"] is False


def test_saved_background_reloads(tmp_path, controller):
    import numpy as np

    slot = controller.pipeline.slots[0]
    slot.filter.set_background(np.full((120, 160, 3), 77, np.uint8))
    controller.save()
    again = Controller(Settings.load(tmp_path / "config.json"), tmp_path / "config.json", segmenter=MaskBox(None))
    assert again.pipeline.slots[0].filter.has_captured_background
    again.pipeline.stop()
