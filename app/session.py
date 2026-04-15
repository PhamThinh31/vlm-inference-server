"""
Session manager for stateful, multi-step image workflows.

Each session owns a temp directory on disk and an in-memory image map.
Per-session locks allow independent sessions to upload concurrently.
"""
import asyncio
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from PIL import Image

from .config import settings
from .utils.image_mapping import ImageNameMapper

logger = logging.getLogger(__name__)

# Bounded pool for blocking PIL decodes. Unbounded `asyncio.to_thread`
# creates as many OS threads as concurrent callers; at 50-session fan-out
# that thrashes the scheduler and starves vLLM's worker loop.
_decode_sem = asyncio.Semaphore(settings.image_decode_workers)


async def _decode_rgb(path: str) -> Image.Image:
    """Off-loop RGB decode, bounded by _decode_sem."""
    def _do() -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB")
    async with _decode_sem:
        return await asyncio.to_thread(_do)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ImageData:
    """Metadata for an uploaded image.

    `image` is a cached decoded PIL handle. It's optional on purpose:
    at modest concurrency (50 sessions x 10 images x 4096^2 RGB) pinning
    decoded frames for the whole 24h TTL will OOM the heap long before
    the GPU. Use `SessionManager.load_image()` to decode lazily and
    `evict_decoded()` after the inference window closes.
    """
    filename: str
    path: str
    image: Optional[Image.Image] = None
    resolution: Tuple[int, int] = (0, 0)
    file_size: int = 0
    classification: Optional[Dict] = None


@dataclass
class Session:
    session_id: str
    created_at: datetime
    updated_at: datetime
    images: Dict[str, ImageData] = field(default_factory=dict)
    temp_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    results: Dict[str, Any] = field(default_factory=dict)
    name_mapper: ImageNameMapper = field(default_factory=ImageNameMapper)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def get_image_by_name(self, filename: str) -> Optional[ImageData]:
        if filename in self.images:
            return self.images[filename]
        original = self.name_mapper.get_original_name(filename)
        if original and original in self.images:
            return self.images[original]
        base = os.path.splitext(filename)[0]
        for name, data in self.images.items():
            if os.path.splitext(name)[0] == base:
                return data
        return None


class SessionManager:
    def __init__(
        self,
        base_temp_dir: str = "/tmp/vlm4gis",
        max_session_age_hours: int = 24,
        max_image_size: Tuple[int, int] = (1024, 1024),
    ) -> None:
        self.sessions: Dict[str, Session] = {}
        self.base_temp_dir = Path(base_temp_dir)
        self.base_temp_dir.mkdir(exist_ok=True, parents=True)
        self.max_session_age = timedelta(hours=max_session_age_hours)
        self.max_image_size = max_image_size
        self._registry_lock = asyncio.Lock()  # protects the `sessions` dict
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def create_session(
        self,
        session_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        async with self._registry_lock:
            session_id = str(uuid.uuid4())
            temp_dir = self.base_temp_dir / session_id
            temp_dir.mkdir(exist_ok=True, parents=True)
            now = _utcnow()
            session = Session(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                temp_dir=str(temp_dir),
                metadata=dict(metadata or {}),
                status="created",
            )
            if session_name:
                session.metadata["name"] = session_name
            self.sessions[session_id] = session
        logger.info("created session %s", session_id)
        return session_id

    async def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    async def upload_image(
        self,
        session_id: str,
        file_path: str,
        filename: Optional[str] = None,
    ) -> ImageData:
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"session {session_id} not found")

        async with session.lock:
            filename = filename or os.path.basename(file_path)
            dest = Path(session.temp_dir) / filename

            # Decode + resize + write-back to disk all happen in a worker
            # thread. Doing this inline on the event loop pins an entire
            # FastAPI worker for ~tens-of-ms per image at 4K resolutions.
            def _ingest() -> Tuple[int, int]:
                with Image.open(file_path) as raw:
                    img = raw if raw.mode == "RGB" else raw.convert("RGB")
                    if (
                        img.size[0] > self.max_image_size[0]
                        or img.size[1] > self.max_image_size[1]
                    ):
                        img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                    img.save(str(dest), "JPEG", quality=95)
                    return img.size

            async with _decode_sem:
                size = await asyncio.to_thread(_ingest)

            data = ImageData(
                filename=filename,
                path=str(dest),
                image=None,
                resolution=size,
                file_size=dest.stat().st_size,
            )
            session.images[filename] = data
            session.updated_at = _utcnow()
            return data

    async def upload_images_batch(
        self, session_id: str, file_paths: List[str]
    ) -> Dict[str, Any]:
        uploaded: List[str] = []
        failed: List[str] = []
        for path in file_paths:
            try:
                data = await self.upload_image(session_id, path)
                uploaded.append(data.filename)
            except Exception as exc:  # noqa: BLE001 - boundary logging
                logger.error("upload failed for %s: %s", path, exc)
                failed.append(path)
        return {"uploaded": uploaded, "failed": failed, "total": len(uploaded)}

    async def get_images_for_inference(
        self,
        session_id: str,
        filenames: Optional[List[str]] = None,
        use_simple_names: bool = True,
    ) -> Tuple[List[Image.Image], List[str]]:
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"session {session_id} not found")

        images: List[Image.Image] = []
        image_ids: List[str] = []
        filenames = filenames or list(session.images.keys())

        if use_simple_names:
            session.name_mapper.reset()

        # Decode concurrently (bounded by _decode_sem) instead of sequentially
        # walking the list. On a 12-image session this drops wall-time from
        # ~12*decode_ms to ~ceil(12/workers)*decode_ms.
        resolved: List[ImageData] = []
        for filename in filenames:
            data = session.get_image_by_name(filename)
            if not data:
                logger.warning("image %s not in session %s", filename, session_id)
                continue
            resolved.append(data)

        async def _ensure_decoded(d: ImageData) -> None:
            if d.image is None:
                d.image = await _decode_rgb(d.path)

        await asyncio.gather(*(_ensure_decoded(d) for d in resolved))

        for data in resolved:
            images.append(data.image)
            image_ids.append(
                session.name_mapper.add_mapping(data.filename)
                if use_simple_names
                else data.filename
            )
        return images, image_ids

    @asynccontextmanager
    async def inference_window(
        self,
        session_id: str,
        filenames: Optional[List[str]] = None,
        use_simple_names: bool = True,
    ) -> AsyncIterator[Tuple[List[Image.Image], List[str]]]:
        """Scoped decode/evict lifecycle.

        Replaces the implicit "callers are expected to evict after use"
        contract with an explicit `async with`. Eviction happens even if
        the caller raises — previously a mid-pipeline exception pinned
        every decoded frame for the rest of the session TTL.
        """
        images, ids = await self.get_images_for_inference(
            session_id, filenames=filenames, use_simple_names=use_simple_names
        )
        try:
            yield images, ids
        finally:
            await self.evict_decoded_images(session_id)

    async def evict_decoded_images(self, session_id: str) -> int:
        """Drop in-memory PIL handles; files on disk stay intact.

        Call this once an inference window closes so the ~48 MB/image heap
        cost doesn't linger for the full 24h session TTL.
        """
        session = await self.get_session(session_id)
        if not session:
            return 0
        count = 0
        async with session.lock:
            for data in session.images.values():
                if data.image is not None:
                    data.image = None
                    count += 1
        return count

    async def update_session_status(self, session_id: str, status: str) -> None:
        session = await self.get_session(session_id)
        if session:
            session.status = status
            session.updated_at = _utcnow()

    async def store_results(
        self, session_id: str, task_name: str, results: Any
    ) -> None:
        session = await self.get_session(session_id)
        if session:
            session.results[task_name] = results
            session.updated_at = _utcnow()

    async def delete_session(self, session_id: str) -> None:
        async with self._registry_lock:
            session = self.sessions.pop(session_id, None)
        if session and session.temp_dir and os.path.exists(session.temp_dir):
            shutil.rmtree(session.temp_dir, ignore_errors=True)
        if session:
            logger.info("deleted session %s", session_id)

    async def _cleanup_loop(self) -> None:
        # Inner try/except so one bad sweep doesn't kill the loop forever.
        # Previously: a single raised exception left sessions accumulating
        # until the process was restarted — classic silent-failure mode.
        while True:
            try:
                await asyncio.sleep(3600)
                now = _utcnow()
                expired = [
                    sid
                    for sid, s in list(self.sessions.items())
                    if now - s.updated_at > self.max_session_age
                ]
                for sid in expired:
                    try:
                        await self.delete_session(sid)
                        logger.info("expired session %s removed", sid)
                    except Exception:
                        logger.exception("failed to delete expired session %s", sid)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("cleanup sweep failed, continuing")

    async def get_all_sessions(self) -> List[Dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "image_count": len(s.images),
                "status": s.status,
                "metadata": s.metadata,
            }
            for s in self.sessions.values()
        ]


session_manager = SessionManager()
