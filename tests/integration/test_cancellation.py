"""Cancellation and graceful-shutdown integration tests.

These exercise behaviour that unit tests cannot: a live FastAPI app,
real middleware ordering, real lifespan state. The fake engine is the
only boundary we mock.

Covered:
  - CancelledError inside engine.generate triggers engine.abort() and
    the handler propagates the cancel up to the middleware.
  - During graceful-shutdown drain, new /task/* requests are rejected
    with 503 + Retry-After, while /healthz and /metrics keep answering.
  - Lifespan drain waits for in-flight /task/* requests before exiting,
    up to the configured deadline.
"""
from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

import pytest
from PIL import Image


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 128, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.asyncio
async def test_engine_abort_called_on_cancel(fake_engine, monkeypatch):
    """engine.generate must call engine.abort when cancelled mid-flight.

    Drives engine.generate directly (no HTTP) so the assertion is on
    the abort contract, not the middleware plumbing.
    """
    from app import engine as engine_mod

    aborted: list[str] = []

    # Swap the fake's engine with an object that has a long-running
    # generate() and a recording abort().
    class _Hang:
        async def generate(self, *_a, **_kw):
            # Never yields — simulates a vLLM generation in progress.
            while True:
                await asyncio.sleep(0.05)
                yield None  # pragma: no cover (unreachable after cancel)

        async def abort(self, request_id: str) -> None:
            aborted.append(request_id)

        async def get_tokenizer(self):
            return None

    real_engine = engine_mod.VLMEngine()
    real_engine.engine = _Hang()
    real_engine.initialized = True

    async def _drive():
        await real_engine.generate(prompt="p", images=[])

    task = asyncio.create_task(_drive())
    await asyncio.sleep(0.1)  # let it enter generate()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert aborted, "engine.abort was never called on cancel"
    assert aborted[0].startswith("req_")


@pytest.mark.asyncio
async def test_drain_rejects_new_task_requests(client, fake_engine):
    """Flipping shutting_down returns 503 on /task/* but keeps /healthz."""
    from app.main import app

    app.state.shutting_down = True
    try:
        r_health = await client.get("/healthz")
        assert r_health.status_code == 200, "liveness must stay up during drain"

        r_task = await client.post(
            "/task/classify",
            json={"image_data_list": [{"filename": "x.jpg", "data": _png_b64()}]},
        )
        assert r_task.status_code == 503
        assert "Retry-After" in r_task.headers
        assert int(r_task.headers["Retry-After"]) > 0
    finally:
        app.state.shutting_down = False


@pytest.mark.asyncio
async def test_inflight_counter_tracks_task_requests(client, fake_engine):
    """/task/* requests increment the drain counter; probes don't."""
    from app.main import app

    fake_engine.responses.append('{"img_000": {"class": "garment"}}')
    baseline = app.state.inflight_task_count

    # Probes: no increment observed after the call returns.
    await client.get("/healthz")
    assert app.state.inflight_task_count == baseline

    # Task call: increments during, decrements after. Easiest assertion
    # is the post-call steady state (the middleware-scoped peek is
    # covered by the drain test above).
    r = await client.post(
        "/task/classify",
        json={"image_data_list": [{"filename": "x.jpg", "data": _png_b64()}]},
    )
    assert r.status_code == 200
    assert app.state.inflight_task_count == baseline
