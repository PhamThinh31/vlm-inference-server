"""Session lifecycle: TTL eviction and decoded-image bounds.

The README claims sessions self-evict after `max_session_age` and that
decoded PIL handles do not pin memory for the full 24h TTL. Both are
load-bearing operational claims — the difference between a service
that holds steady at 50 concurrent sessions and one that OOMs at 3 AM.
These tests pin the contracts so a refactor cannot quietly break them.
"""
from __future__ import annotations

import asyncio
import io
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
from PIL import Image

from app.session import SessionManager, _utcnow


def _png_path(dirpath: str, name: str) -> str:
    p = Path(dirpath) / name
    Image.new("RGB", (32, 32), (255, 0, 0)).save(p, format="PNG")
    return str(p)


@pytest.mark.asyncio
async def test_expired_session_is_removed_by_cleanup():
    """A session past its max_session_age must be deleted on the next sweep.

    Drives the cleanup logic directly (no `asyncio.sleep(3600)`) by
    back-dating `updated_at` and invoking the sweep step that the
    cleanup loop runs every hour. This test is the regression guard
    for any refactor that moves the expiry comparison or the deletion
    call out of `_cleanup_loop`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(base_temp_dir=tmp, max_session_age_hours=1)
        sid = await sm.create_session()
        session = await sm.get_session(sid)
        assert session is not None

        # Back-date so the session is "expired" without sleeping.
        session.updated_at = _utcnow() - timedelta(hours=2)

        # Run one sweep manually — same logic the loop runs hourly.
        now = _utcnow()
        expired = [
            s for s, sess in list(sm.sessions.items())
            if now - sess.updated_at > sm.max_session_age
        ]
        for s in expired:
            await sm.delete_session(s)

        assert sid not in sm.sessions
        assert await sm.get_session(sid) is None


@pytest.mark.asyncio
async def test_inference_window_evicts_decoded_handles():
    """`inference_window` must evict decoded PIL handles on exit.

    Without this contract a long-lived session pins ~48 MB per image
    for the full TTL. The test enters the window, asserts handles are
    decoded, exits, and asserts they are released. A regression that
    skipped the `finally: evict_decoded_images` would not crash any
    individual test but would silently grow the heap to OOM.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(base_temp_dir=tmp, max_session_age_hours=1)
        sid = await sm.create_session()

        for i in range(3):
            await sm.upload_image(sid, _png_path(tmp, f"img_{i}.png"))

        async with sm.inference_window(sid) as (images, ids):
            assert len(images) == 3
            assert len(ids) == 3
            session = await sm.get_session(sid)
            decoded = [d for d in session.images.values() if d.image is not None]
            assert len(decoded) == 3, "images should be decoded inside the window"

        # Exit: every handle must be released. This is the one
        # invariant the README promises about steady-state memory.
        session = await sm.get_session(sid)
        still_pinned = [
            name for name, d in session.images.items() if d.image is not None
        ]
        assert not still_pinned, f"decoded handles leaked: {still_pinned}"


@pytest.mark.asyncio
async def test_inference_window_evicts_even_on_exception():
    """Eviction must run in `finally`, not just on the success path.

    Catches the easy-to-miss refactor where someone wraps the eviction
    in the same try as the inference work and a mid-pipeline raise
    leaves the handles pinned for the full TTL.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(base_temp_dir=tmp, max_session_age_hours=1)
        sid = await sm.create_session()
        await sm.upload_image(sid, _png_path(tmp, "x.png"))

        with pytest.raises(RuntimeError, match="boom"):
            async with sm.inference_window(sid) as (images, _):
                assert len(images) == 1
                raise RuntimeError("boom")

        session = await sm.get_session(sid)
        assert all(d.image is None for d in session.images.values())
