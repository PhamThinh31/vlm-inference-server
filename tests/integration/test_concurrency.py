"""Concurrency stress tests.

These exist to *find bugs*, not to hit a coverage line. Each test
codifies a specific failure mode the codebase is now claimed to be
immune to:

  - request_id collisions under N concurrent requests (regression
    guard for the old `self._counter += 1` race in engine.py)
  - SessionManager dict mutation while another coroutine iterates it
  - decoded-PIL-handle leak when concurrent inferences hit the same
    session (would-be OOM at 50+ concurrent sessions on H100s)

If a future refactor reintroduces any of these, these tests should
fail loudly. If they ever go flaky, that IS the signal — the race is
real, not the test.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
from collections import Counter

import pytest
from PIL import Image


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 200, 60)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _png_file(tmpdir: str, name: str) -> str:
    path = os.path.join(tmpdir, name)
    Image.new("RGB", (16, 16), (200, 10, 80)).save(path, format="PNG")
    return path


@pytest.mark.asyncio
async def test_50_concurrent_classify_no_request_id_collision(client, fake_engine):
    """50 concurrent /task/classify must yield 50 distinct X-Request-Ids.

    The fake engine returns a different canned response per call so the
    middleware exercises the full request lifecycle (no shortcuts via
    early validation errors).
    """
    N = 50
    fake_engine.responses.extend(
        f'{{"img_000": {{"class": "garment", "view": "front", "n": {i}}}}}'
        for i in range(N)
    )

    payload = {"image_data_list": [{"filename": "x.jpg", "data": _png_b64()}]}

    async def one() -> str:
        r = await client.post("/task/classify", json=payload)
        assert r.status_code == 200, r.text
        return r.headers["x-request-id"]

    ids = await asyncio.gather(*(one() for _ in range(N)))

    counts = Counter(ids)
    dupes = {rid: c for rid, c in counts.items() if c > 1}
    assert not dupes, f"duplicate request ids under load: {dupes}"
    assert len(ids) == N


@pytest.mark.asyncio
async def test_concurrent_engine_request_ids_unique(fake_engine):
    """Inside engine.generate, request_id is drawn from itertools.count.

    Direct exercise of the counter under high concurrency — bypasses
    the HTTP layer so a regression to `self._counter += 1` would show
    up here even if middleware happened to mask it elsewhere.
    """
    from app import engine as engine_mod

    seen: list[str] = []

    class _Recorder:
        async def generate(self, _inputs, sampling_params, request_id):
            seen.append(request_id)
            # Yield one immediate "final" output so generate returns.
            class _Out:
                outputs = [type("o", (), {"text": ""})()]
            yield _Out()

        async def get_tokenizer(self):
            return None

    real_engine = engine_mod.VLMEngine()
    real_engine.engine = _Recorder()
    real_engine.initialized = True

    N = 200
    await asyncio.gather(*(real_engine.generate("p", []) for _ in range(N)))

    assert len(seen) == N
    assert len(set(seen)) == N, f"id collisions: {len(seen) - len(set(seen))} dupes"


@pytest.mark.asyncio
async def test_concurrent_session_upload_and_read_no_crash(client, fake_engine):
    """Uploading to a session while another coroutine reads it must not crash.

    The historical risk: `get_images_for_inference` iterates
    `session.images.items()` while `upload_image` mutates the dict.
    Today the mutation is guarded by `session.lock`, but the iteration
    is not — so this test is also a tripwire for any future change
    that drops the lock or moves the iteration outside it.
    """
    # Create session
    r = await client.post("/session/create", json={})
    assert r.status_code == 200
    sid = r.json()["session_id"]

    # Pre-stage a corpus on disk so the upload endpoint has something
    # to read. Use the server-path mode would require ALLOWED_IMAGE_ROOTS;
    # use the multipart upload instead.
    with tempfile.TemporaryDirectory() as tmp:
        files = [_png_file(tmp, f"img_{i:03d}.png") for i in range(20)]

        async def upload_one(path: str):
            with open(path, "rb") as f:
                data = f.read()
            r = await client.post(
                f"/session/{sid}/upload",
                files={"files": (os.path.basename(path), data, "image/png")},
            )
            assert r.status_code == 200, r.text

        async def read_session():
            # Hammer the GET while uploads are in flight. We don't
            # assert image_count here (it races with uploads, which is
            # the point — we assert the endpoint never crashes).
            for _ in range(20):
                r = await client.get(f"/session/{sid}")
                assert r.status_code == 200
                await asyncio.sleep(0.005)

        # Interleave 20 uploads with 20 reads.
        await asyncio.gather(
            *(upload_one(p) for p in files),
            read_session(),
        )

    # Final state: session exists, has all 20 images, no leaked decoded
    # handles. The uploaded count being exactly 20 proves no dict
    # mutation was lost to a race.
    r = await client.get(f"/session/{sid}")
    assert r.status_code == 200
    assert r.json()["image_count"] == 20


@pytest.mark.asyncio
async def test_concurrent_inference_does_not_leak_decoded_images(client, fake_engine):
    """After two parallel inferences on the same session, decoded PIL
    handles must be evicted — not pinned in memory.

    This guards the `inference_window` finally-block contract
    (`evict_decoded_images`). A regression that skipped eviction would
    not crash any test individually, but at scale it silently grows
    the heap until OOM. So we assert the post-condition explicitly.
    """
    from app.session import session_manager

    # Create + upload
    r = await client.post("/session/create", json={})
    sid = r.json()["session_id"]
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(4):
            path = _png_file(tmp, f"img_{i}.png")
            with open(path, "rb") as f:
                await client.post(
                    f"/session/{sid}/upload",
                    files={"files": (f"img_{i}.png", f.read(), "image/png")},
                )

    # Two parallel classify calls against the session. Each call does
    # a decode→inference→evict cycle.
    fake_engine.responses.extend([
        '{"img_000": {"class": "garment"}, "img_001": {"class": "body"},'
        '"img_002": {"class": "garment"}, "img_003": {"class": "body"}}'
    ] * 2)

    payload = {"session_id": sid}
    r1, r2 = await asyncio.gather(
        client.post("/task/classify", json=payload),
        client.post("/task/classify", json=payload),
    )
    assert r1.status_code == 200 and r2.status_code == 200

    # Post-condition: every ImageData.image is None (evicted). If any
    # handle is still resident, the eviction contract is broken.
    session = await session_manager.get_session(sid)
    pinned = [name for name, d in session.images.items() if d.image is not None]
    assert not pinned, f"decoded image handles leaked after inference: {pinned}"
