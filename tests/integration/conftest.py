"""Integration-test fixtures.

Strategy: run the *real* FastAPI app in-process via `httpx.ASGITransport`,
mocking only at the vLLM engine boundary. That keeps middleware, routes,
session manager, parsers, and schema validation all honest — the thing
under test is the composition, not individual functions.

What's stubbed:
  - `engine.initialize()` → no-op (we don't load a real VLM in CI)
  - `engine.generate()`   → returns a queue of canned responses so
                            tests can assert on specific parser paths
  - `engine.health_check / get_cache_stats` → cheap dicts
  - `engine.build_prompt` → identity-ish passthrough

Everything else runs as shipped. If a test breaks because of a change
outside the engine boundary, that's the test doing its job.
"""
from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator, Deque

import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeEngine:
    """Minimal stand-in for `app.engine.VLMEngine`.

    Tests push canned responses onto `responses` before making a request;
    the router exhausts the queue in FIFO order. When the queue is empty
    `generate` returns an empty string — a deliberate parse-failure so
    tests notice if they under-specified their fixture.
    """

    def __init__(self) -> None:
        self.initialized = True
        self.tokenizer = None
        self.responses: Deque[str] = deque()
        self.generate_calls: list[dict[str, Any]] = []

    async def initialize(self, *_a, **_kw) -> None:
        self.initialized = True

    def build_prompt(self, messages, images) -> str:
        return "<prompt>"

    async def generate(self, prompt, images, **kwargs) -> str:
        self.generate_calls.append({"prompt": prompt, "n_images": len(images), **kwargs})
        if self.responses:
            return self.responses.popleft()
        return ""

    async def health_check(self) -> dict[str, Any]:
        return {"initialized": True, "model_loaded": True, "gpu_available": True}

    async def get_cache_stats(self) -> dict[str, Any]:
        return {"available": False, "reason": "fake engine"}


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped loop so the lifespan context survives across tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def fake_engine(monkeypatch) -> FakeEngine:
    """Patch the module-level engine singleton before the app imports it."""
    from app import engine as engine_mod
    from app.pipelines import task1_classifier, task2_attributes, task3_validation

    fe = FakeEngine()
    # The pipeline modules imported `engine` by name at import time, so
    # patching `engine_mod.engine` alone is not enough — we must also
    # rebind the name inside each consumer module.
    monkeypatch.setattr(engine_mod, "engine", fe)
    monkeypatch.setattr(task1_classifier, "engine", fe)
    monkeypatch.setattr(task2_attributes, "engine", fe)
    monkeypatch.setattr(task3_validation, "engine", fe)
    return fe


@pytest_asyncio.fixture
async def client(fake_engine) -> AsyncIterator:
    """In-process ASGI client with lifespan events driven manually."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Manually run lifespan startup so app.state is populated.
        async with app.router.lifespan_context(app):
            yield ac
