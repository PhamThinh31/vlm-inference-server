"""Integration tests against the real FastAPI app with a faked engine.

Covered:
  1. /healthz is unconditional (liveness)
  2. /readyz reflects engine state
  3. request_id middleware: echoed header + accepts inbound override
  4. Task 1 happy path via base64 upload
  5. Task 1 parse_failed when model omits a key (regression: no
     positional fallback)
  6. Path-traversal rejection on server-path inputs

Not covered here (out of scope for this PR): cancellation, GIS pipeline
end-to-end, concurrent-session race. Those belong in Phase 2.
"""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image


def _png_b64(size=(8, 8), color=(255, 0, 0)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.asyncio
async def test_healthz_always_ok(client):
    # Liveness must never depend on engine state. Even if the engine
    # were unready, this endpoint returns 200 — otherwise K8s would
    # SIGKILL the pod during a slow cold-start.
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}


@pytest.mark.asyncio
async def test_readyz_reports_engine_ready(client, fake_engine):
    r = await client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["engine_initialized"] is True
    assert body["gpu_available"] is True


@pytest.mark.asyncio
async def test_readyz_returns_503_when_engine_not_ready(client, fake_engine, monkeypatch):
    async def _down():
        return {"initialized": False, "model_loaded": False, "gpu_available": False}

    monkeypatch.setattr(fake_engine, "health_check", _down)
    r = await client.get("/readyz")
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_request_id_header_roundtrip(client):
    # No inbound id → server generates one and echoes it.
    r1 = await client.get("/healthz")
    generated = r1.headers.get("x-request-id")
    assert generated and len(generated) >= 16

    # Inbound id → server honours it (for edge-proxy correlation).
    r2 = await client.get("/healthz", headers={"X-Request-Id": "trace-abc-123"})
    assert r2.headers["x-request-id"] == "trace-abc-123"


@pytest.mark.asyncio
async def test_task1_classify_happy_path(client, fake_engine):
    fake_engine.responses.append('{"img_000": {"class": "garment", "view": "front"}}')

    payload = {
        "image_data_list": [
            {"filename": "shirt.jpg", "data": _png_b64()},
        ]
    }
    r = await client.post("/task/classify", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_images"] == 1
    cls = body["classifications"][0]
    assert cls["class"] == "garment"
    assert cls["parse_failed"] is False


@pytest.mark.asyncio
async def test_task1_missing_key_does_not_positionally_fallback(client, fake_engine):
    # Model response names only the first image. The second must come
    # back as parse_failed=True with class=unknown — NOT silently
    # assigned the first image's classification. This is the regression
    # guard for the old Strategy 3 index-based fallback.
    fake_engine.responses.append('{"img_000": {"class": "garment"}}')

    payload = {
        "image_data_list": [
            {"filename": "a.jpg", "data": _png_b64()},
            {"filename": "b.jpg", "data": _png_b64()},
        ]
    }
    r = await client.post("/task/classify", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_images"] == 2

    # Order is preserved by the router; second row must be parse_failed.
    second = body["classifications"][1]
    assert second["parse_failed"] is True
    assert second["class"] == "unknown"


@pytest.mark.asyncio
async def test_path_traversal_rejected(client):
    # Server-path input outside ALLOWED_IMAGE_ROOTS must 4xx, not 500
    # and definitely not successfully read the file.
    r = await client.post(
        "/task/classify",
        json={"image_paths_list": ["/etc/passwd"]},
    )
    assert r.status_code in (400, 403)
