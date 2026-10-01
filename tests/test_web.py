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
    assert st["chain"][0]["type"] == "fade_into_background"
    assert st["virtual_camera"]["running"] is False
    assert any(f["type"] == "mirror" for f in st["available_filters"])


def test_edit_chain_over_http(client, controller, tmp_path):
    fade_id = client.get("/api/state").json()["chain"][0]["id"]
    r = client.post(f"/api/filters/{fade_id}/params", json={"fade": 0.3, "auto_fade": "true"})
    assert r.json()["values"] == {"fade": 0.3, "auto_fade": True}
    assert client.post(f"/api/filters/{fade_id}/params", json={"nope": 1}).status_code == 400

    mirror_id = client.post("/api/filters", json={"type": "mirror"}).json()["id"]
    client.post(f"/api/filters/{mirror_id}/move", json={"delta": -1})
    assert [s["id"] for s in client.get("/api/state").json()["chain"]] == [mirror_id, fade_id]
    client.post("/api/filters/order", json={"ids": [fade_id, mirror_id]})
    client.post(f"/api/filters/{mirror_id}/enabled", json={"enabled": False})
    chain = client.get("/api/state").json()["chain"]
    assert [s["id"] for s in chain] == [fade_id, mirror_id] and chain[1]["enabled"] is False

    assert client.post(f"/api/filters/{fade_id}/actions/capture_background").json()["ok"]
    assert client.post(f"/api/filters/{fade_id}/actions/bogus").status_code == 400
    assert client.delete(f"/api/filters/{mirror_id}").json()["ok"]
    assert client.delete("/api/filters/missing").status_code == 404

    # Autosave writes the config (and the captured background plate).
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not (tmp_path / f"{fade_id}-background.png").exists():
        time.sleep(0.05)
    controller.save()
    saved = Settings.load(tmp_path / "config.json")
    assert saved.filters[0].params["fade"] == 0.3 and len(saved.filters) == 1
    assert (tmp_path / f"{fade_id}-background.png").exists()


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
