"""Local control panel: FastAPI JSON API + MJPEG preview streams."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any, Iterator

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from cambobulator.app import Controller
from cambobulator.cameras import list_cameras

PREVIEW_WIDTH = 640
BOUNDARY = "frame"


def _placeholder(text: str, width: int = PREVIEW_WIDTH, height: int = 360) -> np.ndarray:
    img = np.full((height, width, 3), 32, np.uint8)
    y = 40
    for line in text.splitlines()[:12]:
        for chunk in [line[i:i + 70] for i in range(0, max(len(line), 1), 70)]:
            cv2.putText(img, chunk, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            y += 22
    return img


class _JpegCache:
    """Encode each preview frame once, however many browser tabs are watching."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[int, bytes]] = {}

    def get(self, kind: str, seq: int, frame: np.ndarray) -> bytes:
        with self._lock:
            hit = self._cache.get(kind)
            if hit and hit[0] == seq:
                return hit[1]
        h, w = frame.shape[:2]
        if w > PREVIEW_WIDTH:
            frame = cv2.resize(frame, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = buf.tobytes() if ok else b""
        with self._lock:
            self._cache[kind] = (seq, data)
        return data


def create_app(controller: Controller) -> FastAPI:
    jpegs = _JpegCache()
    stopping = threading.Event()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        stopping.set()  # end MJPEG streams so the server can exit

    app = FastAPI(title="Cambobulator", lifespan=lifespan)
    app.state.stopping = stopping
    app.state.controller = controller

    def slot_or_404(slot_id: str):
        try:
            return controller.pipeline.get(slot_id)
        except KeyError:
            raise HTTPException(404, f"no filter {slot_id}") from None

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("cambobulator.web").joinpath("index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return controller.state()

    @app.get("/api/cameras")
    def cameras() -> list[dict[str, Any]]:
        return [c.to_dict() for c in list_cameras()]

    @app.post("/api/source")
    def set_source(body: dict = Body(...)) -> dict[str, Any]:
        ok = controller.open_source(
            str(body.get("source", controller.settings.source)),
            int(body["width"]) if body.get("width") else None,
            int(body["height"]) if body.get("height") else None,
            int(body["fps"]) if body.get("fps") else None,
        )
        return {"ok": ok, "error": controller.source_error}

    @app.post("/api/virtual-camera")
    def set_vcam(body: dict = Body(...)) -> dict[str, Any]:
        ok = controller.set_virtual_camera(bool(body.get("enabled")))
        return {"ok": ok, "error": controller.vcam_error, "device": controller.vcam_device}

    @app.post("/api/filters")
    def add_filter(body: dict = Body(...)) -> dict[str, Any]:
        try:
            return {"id": controller.add_filter(str(body["type"]))}
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/filters/order")
    def reorder(body: dict = Body(...)) -> dict[str, Any]:
        try:
            controller.reorder_filters(list(body["ids"]))
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from None
        return {"ok": True}

    @app.delete("/api/filters/{slot_id}")
    def remove_filter(slot_id: str) -> dict[str, Any]:
        slot_or_404(slot_id)
        controller.remove_filter(slot_id)
        return {"ok": True}

    @app.post("/api/filters/{slot_id}/enabled")
    def set_enabled(slot_id: str, body: dict = Body(...)) -> dict[str, Any]:
        slot_or_404(slot_id)
        controller.set_enabled(slot_id, bool(body.get("enabled")))
        return {"ok": True}

    @app.post("/api/filters/{slot_id}/move")
    def move(slot_id: str, body: dict = Body(...)) -> dict[str, Any]:
        slot_or_404(slot_id)
        controller.move_filter(slot_id, int(body.get("delta", 0)))
        return {"ok": True}

    @app.post("/api/filters/{slot_id}/params")
    def set_params(slot_id: str, body: dict = Body(...)) -> dict[str, Any]:
        slot_or_404(slot_id)
        values = {}
        for name, value in body.items():
            try:
                values[name] = controller.set_param(slot_id, name, value)
            except (KeyError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from None
        return {"ok": True, "values": values}

    @app.post("/api/filters/{slot_id}/actions/{action}")
    def run_action(slot_id: str, action: str) -> dict[str, Any]:
        slot_or_404(slot_id)
        try:
            controller.run_action(slot_id, action)
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"ok": True}

    @app.post("/api/save")
    def save() -> dict[str, Any]:
        controller.save()
        return {"ok": True, "path": str(controller.config_path)}

    def mjpeg(kind: str) -> Iterator[bytes]:
        seq = 0
        while not stopping.is_set():
            seq, raw, out = controller.pipeline.wait_preview(seq, timeout=1.0)
            frame = raw if kind == "raw" else out
            if frame is None:
                error = controller.source_error or controller.pipeline.source_error
                text = error or "Waiting for camera..."
                data = jpegs.get(f"placeholder-{kind}", hash(text), _placeholder(text))
            else:
                data = jpegs.get(kind, seq, frame)
            yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n").encode() + data + b"\r\n"

    @app.get("/stream/{kind}.mjpg")
    def stream(kind: str) -> StreamingResponse:
        if kind not in ("raw", "out"):
            raise HTTPException(404)
        return StreamingResponse(mjpeg(kind), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    return app
