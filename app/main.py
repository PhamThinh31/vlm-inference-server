# app/main.py
"""
VLM4GIS FastAPI Application
High-performance VLM inference server for GIS processing
"""

import asyncio
import base64
import io
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Iterator, NamedTuple

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

from .config import settings
from .engine import EngineConfig, engine
from .logging_config import bind_context, clear_context, configure_logging
from .metrics import inflight_requests, metrics_response, observe_request
from .pipelines.gis_processor import GISProcessor
from .pipelines.task1_classifier import Task1Classifier
from .pipelines.task2_attributes import Task2AttributesExtractor
from .pipelines.task3_validation import Task3Validator
from .schemas import (
    ErrorCode,
    GISPipelineRequest,
    GISPipelineResponse,
    HealthStatus,
    ImageClassification,
    ImageUpload,
    SessionCreate,
    SessionInfo,
    Task1Request,
    Task1Response,
    Task2Request,
    Task2Response,
    Task2Result,
    Task3Request,
    Task3Response,
    TaskError,
    ValidationResult,
)
from .session import session_manager
from .utils.image_mapping import ImageNameMapper

# Structured JSON logging. `log_json=False` in the env flips to a
# human-readable console renderer for local dev; production always emits
# JSON so log aggregators (Loki/CloudWatch/ES) can index fields directly.
configure_logging(level=settings.log_level, json=getattr(settings, "log_json", True))
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image resolution helpers — single place to decode / load images so that
# every endpoint stays thin and the logic is never duplicated.
# ---------------------------------------------------------------------------

def _decode_base64_image(data_str: str) -> Image.Image:
    """Decode a base64-encoded string into a PIL RGB image."""
    return Image.open(io.BytesIO(base64.b64decode(data_str))).convert("RGB")


_ALLOWED_IMAGE_ROOTS = tuple(
    os.path.realpath(p) for p in settings.allowed_image_roots.split(",") if p.strip()
)
_ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_path_image(path: str) -> Image.Image:
    """Load an image from a server-local filesystem path.

    Guarded against path-traversal: the resolved real path must live
    under one of the roots listed in ALLOWED_IMAGE_ROOTS, and the
    extension must be in the image allowlist. Without this a caller
    could read arbitrary server files by passing e.g. '/etc/passwd'.
    """
    real = os.path.realpath(path)
    # commonpath operates on components so "/data" vs "/data.bak" is safe.
    # A naive real.startswith(root) would let "/data.bak/etc" through when
    # only "/data" is allowed.
    def _under(root: str) -> bool:
        try:
            return os.path.commonpath([real, root]) == root
        except ValueError:
            return False  # different drives on Windows, or relative vs abs
    if _ALLOWED_IMAGE_ROOTS and not any(_under(r) for r in _ALLOWED_IMAGE_ROOTS):
        raise HTTPException(403, f"Path not under any allowed root: {path}")
    if os.path.splitext(real)[1].lower() not in _ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(400, f"Unsupported image extension: {path}")
    if not os.path.exists(real):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(real).convert("RGB")


async def _resolve_classify_images(
    request: Task1Request,
) -> tuple[list[Image.Image], list[str], ImageNameMapper, str | None]:
    """
    Normalise Task 1 input into *(images, ids, mapper, session_id)*
    regardless of whether the caller sent base64 data, server paths, or a
    session reference.
    """
    mapper = ImageNameMapper()

    # 1. Direct data upload (base64)
    if request.image_data_list:
        images, ids = [], []
        for item in request.image_data_list:
            try:
                img = _decode_base64_image(item.data)
                simple = mapper.add_mapping(os.path.basename(item.filename))
                images.append(img)
                ids.append(simple)
            except Exception:
                logger.exception("Failed to decode image: %s", item.filename)
        if not images:
            raise HTTPException(400, "No valid images in payload")
        return images, ids, mapper, None

    # 2. Server-side file paths
    if request.image_paths_list:
        images, ids = [], []
        for path in request.image_paths_list:
            try:
                img = _load_path_image(path)
                simple = mapper.add_mapping(os.path.basename(path))
                images.append(img)
                ids.append(simple)
            except Exception:
                logger.exception("Failed to load image: %s", path)
        if not images:
            raise HTTPException(400, "No valid images at provided paths")
        return images, ids, mapper, "direct_inference"

    # 3. Session-based
    if request.session_id:
        session = await session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(404, f"Session {request.session_id} not found")
        images, ids = await session_manager.get_images_for_inference(
            session_id=request.session_id,
            filenames=request.image_files,
            use_simple_names=True,
        )
        if not images:
            raise HTTPException(400, "No images found in session")
        await session_manager.update_session_status(
            request.session_id, "processing"
        )
        return images, ids, session.name_mapper, request.session_id

    raise HTTPException(
        400, "Provide one of: session_id, image_paths_list, or image_data_list"
    )


class _DirectPair(NamedTuple):
    """A single pair resolved from direct (non-session) input.

    `inference_id` is the filename the pipeline sees in the prompt —
    short, cache-friendly, predictable. `display_name` is what the
    response reports to the client — the original filename or full
    path, so the caller can correlate results back to their request.
    They are usually equal; they differ for server-path input where
    we synthesise a cache-friendly inference id.
    """
    garment_image: Image.Image
    body_image: Image.Image
    garment_inference_id: str
    body_inference_id: str
    garment_display_name: str
    body_display_name: str


def _iter_direct_pairs(
    request: "Task2Request | Task3Request",
) -> Iterator[_DirectPair]:
    """Yield pairs for base64 and server-path request modes.

    Previously Tasks 2 and 3 each had ~40 lines inlining these two
    branches with the same try/decode/load dance, and the server-path
    branch tacked on a `result.garment_file = path` post-hoc mutation
    in both places. This generator collapses both branches, yielding
    the display name alongside the inference id so the caller can
    assign the right value directly instead of mutating after the fact.
    """
    if request.image_data_pairs:
        for pair in request.image_data_pairs:
            yield _DirectPair(
                garment_image=_decode_base64_image(pair.garment.data),
                body_image=_decode_base64_image(pair.body.data),
                garment_inference_id=pair.garment.filename,
                body_inference_id=pair.body.filename,
                garment_display_name=pair.garment.filename,
                body_display_name=pair.body.filename,
            )
        return

    if request.image_pairs:
        for garment_path, body_path in request.image_pairs:
            g_base = os.path.splitext(os.path.basename(garment_path))[0]
            b_base = os.path.splitext(os.path.basename(body_path))[0]
            yield _DirectPair(
                garment_image=_load_path_image(garment_path),
                body_image=_load_path_image(body_path),
                garment_inference_id=f"garment_{g_base}.jpg",
                body_inference_id=f"body_{b_base}.jpg",
                garment_display_name=garment_path,
                body_display_name=body_path,
            )


async def _resolve_session_pairs(
    session_id: str,
    explicit_pairs: list[tuple[str, str]] | None,
    default_pair_deriver,
) -> tuple[list[tuple[str, str]], dict[str, Image.Image]]:
    """Resolve pairs for a session-based task call.

    Returns `(pairs, images_dict)` ready for the batch pipeline call.
    If the caller supplied `explicit_pairs`, we use them verbatim;
    otherwise we pull Task 1 results off the session and hand them to
    `default_pair_deriver` (each task has its own: `select_best_pairs`
    for Task 2, `generate_all_pairs` for Task 3). Raises 404/400 with
    the same messages the inline code used to produce.
    """
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    images_dict = {
        fname: data.image
        for fname, data in session.images.items()
        if data.image
    }

    pairs = explicit_pairs
    if not pairs:
        task1_results = session.results.get("task1", [])
        if not task1_results:
            raise HTTPException(
                400, "Run classification first (no Task 1 results)"
            )
        classifications = [ImageClassification(**r) for r in task1_results]
        pairs = default_pair_deriver(classifications)

    if not pairs:
        raise HTTPException(400, "No valid pairs found")
    return pairs, images_dict


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown manager."""
    logger.info("Starting VLM4GIS server...")

    config = EngineConfig(
        model_path=settings.model_path,
        max_model_len=settings.max_model_len,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        max_num_seqs=settings.max_num_seqs,
        max_num_batched_tokens=settings.max_num_batched_tokens,
        enable_prefix_caching=settings.enable_prefix_caching,
        quantization=settings.vllm_quantization,
        speculative_model=settings.vllm_speculative_model,
        num_speculative_tokens=settings.vllm_num_speculative_tokens,
    )
    await engine.initialize(config)
    await session_manager.initialize()

    # batch_size sourced from settings.task1_batch_size. The previous
    # hardcoded 5 silently overrode TASK1_BATCH_SIZE env vars — ops could
    # "tune" it in config.py all day and nothing would change.
    app.state.task1 = Task1Classifier(version="v1.0")
    app.state.task2 = Task2AttributesExtractor(version="v1.1")
    app.state.task3 = Task3Validator(version="v1.1")
    app.state.gis = GISProcessor()
    app.state.start_time = time.time()

    app.state.shutting_down = False
    app.state.inflight_task_count = 0
    logger.info("VLM4GIS server started successfully")
    yield

    logger.info("Shutting down VLM4GIS server...")
    # Step 1: flip the drain flag. New /task/* requests now 503 with a
    # Retry-After header so load balancers route elsewhere. Non-task
    # endpoints (/healthz, /metrics) keep answering so probes and
    # scrapers stay sane during the drain.
    app.state.shutting_down = True

    # Step 2: wait for in-flight work to finish, up to the deadline.
    # Polling is fine — this is a one-shot, not a hot path.
    deadline = time.time() + settings.graceful_shutdown_timeout
    while app.state.inflight_task_count > 0 and time.time() < deadline:
        logger.info(
            "draining: %d in-flight task requests",
            app.state.inflight_task_count,
        )
        await asyncio.sleep(0.5)
    if app.state.inflight_task_count > 0:
        logger.warning(
            "drain deadline exceeded with %d in-flight; proceeding",
            app.state.inflight_task_count,
        )

    await session_manager.shutdown()
    # Best-effort cleanup — a stuck delete on one session must not hang
    # the shutdown hook and block FastAPI from exiting. Any survivors
    # will be reaped by the cleanup loop on next startup (temp_dir is
    # persistent across runs).
    for sid in list(session_manager.sessions):
        try:
            await asyncio.wait_for(session_manager.delete_session(sid), timeout=5.0)
        except Exception:
            logger.exception("shutdown: delete_session(%s) failed; continuing", sid)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VLM4GIS",
    description="Vision-Language Model for GIS - High-performance inference server",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_REQUEST_BYTES = 50 * 1024 * 1024


def _is_task_route(path: str) -> bool:
    # Only /task/* endpoints hold the GPU. Probes, metrics, session
    # admin, and docs must keep answering during a drain so K8s doesn't
    # SIGKILL the pod and scrapers don't alarm.
    return path.startswith("/task/")


@app.middleware("http")
async def body_size_and_metrics(request: Request, call_next):
    """Reject oversized payloads, bind request_id, drain-gate, cancel on disconnect.

    Responsibilities (in order):
      1. 413 on oversized body.
      2. 503 + Retry-After on /task/* while the server is draining.
      3. Bind request_id into the logging contextvar.
      4. Race the handler against a client-disconnect poller; on
         disconnect, cancel the handler coroutine, which propagates
         CancelledError into engine.generate(), which calls
         engine.abort(request_id) — the GPU stops generating tokens
         for a request nobody is listening for.
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BYTES:
        return JSONResponse({"detail": "request body too large"}, status_code=413)

    if _is_task_route(request.url.path) and getattr(app.state, "shutting_down", False):
        # Retry-After is advisory; load balancers and well-behaved
        # clients back off for that many seconds.
        return JSONResponse(
            {"detail": "server is draining; retry shortly"},
            status_code=503,
            headers={"Retry-After": str(int(settings.graceful_shutdown_timeout))},
        )

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    bind_context(request_id=request_id, path=request.url.path, method=request.method)

    is_task = _is_task_route(request.url.path)
    if is_task:
        app.state.inflight_task_count += 1

    inflight_requests.inc()
    start = time.perf_counter()
    try:
        response = await _run_with_disconnect_watchdog(request, call_next)
        response.headers["X-Request-Id"] = request_id
        observe_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration=time.perf_counter() - start,
        )
        return response
    except asyncio.CancelledError:
        # Client disconnected mid-flight. Return a 499-equivalent so
        # downstream logs/metrics see the abandonment; the socket is
        # already closed so the body never reaches the client.
        logger.info("request cancelled by client disconnect")
        observe_request(
            method=request.method,
            path=request.url.path,
            status=499,
            duration=time.perf_counter() - start,
        )
        return JSONResponse({"detail": "client disconnected"}, status_code=499)
    finally:
        inflight_requests.dec()
        if is_task:
            app.state.inflight_task_count -= 1
        clear_context()


async def _run_with_disconnect_watchdog(request: Request, call_next):
    """Execute `call_next`, cancelling it if the client disconnects.

    We run the handler in a task and poll `request.is_disconnected()`
    in parallel. On disconnect the handler task is cancelled — that
    CancelledError travels all the way into `engine.generate`, which
    aborts the vLLM request. Both tasks are always awaited/cancelled
    so we don't leak coroutines.
    """
    handler_task = asyncio.create_task(call_next(request))

    async def _watch() -> None:
        while not handler_task.done():
            try:
                if await request.is_disconnected():
                    handler_task.cancel()
                    return
            except Exception:  # noqa: BLE001 - receive() after close raises
                return
            await asyncio.sleep(settings.disconnect_poll_interval)

    watch_task = asyncio.create_task(_watch())
    try:
        return await handler_task
    finally:
        watch_task.cancel()
        # Suppress the watchdog's own cancel — it's internal bookkeeping.
        try:
            await watch_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus scrape endpoint."""
    return metrics_response()


# ===== Health & Status =====================================================


@app.get("/", response_model=dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "sessions": "/session/list",
            "tasks": {
                "task1": "/task/classify",
                "task2": "/task/attributes",
                "task3": "/task/validate",
                "pipeline": "/task/gis-pipeline",
            },
        },
    }


@app.get("/healthz")
async def liveness():
    """Liveness probe: the process is up and the event loop is responsive.

    Deliberately does NOT touch the engine or GPU. K8s uses this to
    decide whether to SIGKILL the pod; a stuck vLLM engine must not
    cause the pod to get killed (that just loses in-flight work and
    the replacement hits the same stuck state).
    """
    return {"status": "alive"}


@app.get("/readyz", response_model=HealthStatus)
async def readiness():
    """Readiness probe: engine initialised and GPU reachable.

    K8s uses this to decide whether to route traffic to the pod. A
    503 here drains the pod from the service without killing it —
    correct behaviour for a warm-up or transient GPU hiccup.
    """
    info = await engine.health_check()
    ready = info["initialized"] and info["gpu_available"]
    payload = HealthStatus(
        status="healthy" if ready else "degraded",
        engine_initialized=info["initialized"],
        gpu_available=info["gpu_available"],
        gpu_memory_usage=info.get("gpu_memory_allocated"),
        active_sessions=len(session_manager.sessions),
        uptime_seconds=time.time() - app.state.start_time,
    )
    if not ready:
        return JSONResponse(payload.model_dump(mode="json"), status_code=503)
    return payload


# /health kept as a transition alias — delete once all probes migrate.
@app.get("/health", response_model=HealthStatus, deprecated=True)
async def health_check():
    return await readiness()


# ===== Session Management ==================================================


@app.post("/session/create", response_model=SessionInfo)
async def create_session(request: SessionCreate):
    """Create a new processing session."""
    session_id = await session_manager.create_session(
        session_name=request.session_name,
        metadata=request.metadata,
    )
    session = await session_manager.get_session(session_id)
    return SessionInfo(
        session_id=session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        image_count=0,
        status=session.status,
        metadata=session.metadata,
    )


@app.post("/session/{session_id}/upload", response_model=ImageUpload)
async def upload_images(session_id: str, files: list[UploadFile] = File(...)):
    """Upload images to a session."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    uploaded, failed = [], []
    for file in files:
        try:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            await session_manager.upload_image(session_id, tmp_path, file.filename)
            uploaded.append(file.filename)
            os.unlink(tmp_path)
        except Exception:
            logger.exception("Failed to upload %s", file.filename)
            failed.append(file.filename)

    return ImageUpload(
        session_id=session_id,
        uploaded_files=uploaded,
        failed_files=failed,
        total_uploaded=len(uploaded),
    )


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    return SessionInfo(
        session_id=session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        image_count=len(session.images),
        status=session.status,
        metadata=session.metadata,
    )


@app.get("/session/list", response_model=list[SessionInfo])
async def list_sessions():
    """List all active sessions."""
    raw = await session_manager.get_all_sessions()
    return [
        SessionInfo(
            session_id=s["session_id"],
            created_at=datetime.fromisoformat(s["created_at"]),
            updated_at=datetime.fromisoformat(s["updated_at"]),
            image_count=s["image_count"],
            status=s["status"],
            metadata=s.get("metadata"),
        )
        for s in raw
    ]


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    await session_manager.delete_session(session_id)
    return {"message": f"Session {session_id} deleted successfully"}


# ===== Task Endpoints ======================================================


@app.post("/task/classify", response_model=Task1Response)
async def classify_images(request: Task1Request):
    """Task 1: Classify images as garment / body / unknown."""
    images, image_ids, mapper, session_id = await _resolve_classify_images(request)

    start_time = time.time()
    classifications, raw = await app.state.task1.classify_images(
        images=images, image_ids=image_ids, use_cache=True,
    )

    # Map simplified inference names back to originals
    for cls in classifications:
        original = mapper.get_original_name(cls.filename)
        if original:
            cls.filename = original

    processing_time = time.time() - start_time

    # Persist results for session-based requests
    if request.session_id:
        await session_manager.store_results(
            request.session_id, "task1", [c.dict() for c in classifications],
        )
        await session_manager.update_session_status(request.session_id, "completed")

    return Task1Response(
        session_id=session_id,
        classifications=classifications,
        processing_time=processing_time,
        total_images=len(classifications),
        raw_response=raw,
    )


@app.post("/task/attributes", response_model=Task2Response)
async def extract_attributes(request: Task2Request):
    """Task 2: Extract garment attributes from image pairs."""
    # Direct input (base64 or server paths) — iterate one pair at a time.
    if request.image_data_pairs or request.image_pairs:
        results: list[Task2Result] = []
        total_time = 0.0
        for pair in _iter_direct_pairs(request):
            start = time.time()
            result = await app.state.task2.extract_attributes(
                garment_image=pair.garment_image, body_image=pair.body_image,
                garment_id=pair.garment_inference_id,
                body_id=pair.body_inference_id,
                use_cache=True,
            )
            total_time += time.time() - start
            # Report the caller's original path/filename, not the
            # cache-friendly inference id we synthesised.
            result.garment_file = pair.garment_display_name
            result.body_file = pair.body_display_name
            results.append(result)
        return Task2Response(
            session_id="direct_inference" if request.image_pairs else None,
            results=results,
            processing_time=total_time, pairs_processed=len(results),
        )

    if request.session_id:
        pairs, images_dict = await _resolve_session_pairs(
            request.session_id,
            explicit_pairs=request.pairs,
            default_pair_deriver=lambda cls: app.state.task2.select_best_pairs(
                cls, max_pairs=5,
            ),
        )
        start_time = time.time()
        results = await app.state.task2.extract_attributes_batch(
            pairs=pairs, images_dict=images_dict, use_cache=True,
        )
        processing_time = time.time() - start_time
        await session_manager.store_results(
            request.session_id, "task2", [r.dict() for r in results],
        )
        return Task2Response(
            session_id=request.session_id, results=results,
            processing_time=processing_time, pairs_processed=len(results),
        )

    raise HTTPException(
        400, "Provide one of: session_id, image_pairs, or image_data_pairs"
    )


def _task3_response(
    session_id: str | None,
    validations: list[ValidationResult],
    processing_time: float,
) -> Task3Response:
    """Bundle the valid/invalid counting that every Task 3 return path does."""
    valid_count = sum(1 for v in validations if v.is_valid)
    return Task3Response(
        session_id=session_id,
        validations=validations,
        processing_time=processing_time,
        total_validated=len(validations),
        valid_count=valid_count,
        invalid_count=len(validations) - valid_count,
    )


@app.post("/task/validate", response_model=Task3Response)
async def validate_pairs(request: Task3Request):
    """Task 3: Validate garment-body pair compatibility."""
    # Direct input (base64 or server paths). Each pair is wrapped in
    # try/except so one bad pair yields a parse_failed ValidationResult
    # rather than tearing down the whole request — matches the
    # return_exceptions=True contract in validate_pairs_batch.
    if request.image_data_pairs or request.image_pairs:
        validations: list[ValidationResult] = []
        total_time = 0.0
        for pair in _iter_direct_pairs(request):
            try:
                start = time.time()
                result = await app.state.task3.validate_pair(
                    garment_image=pair.garment_image, body_image=pair.body_image,
                    garment_id=pair.garment_inference_id,
                    body_id=pair.body_inference_id,
                    use_cache=True,
                )
                total_time += time.time() - start
                result.garment_file = pair.garment_display_name
                result.body_file = pair.body_display_name
                validations.append(result)
            except Exception as exc:
                logger.exception(
                    "Error validating pair (%s, %s)",
                    pair.garment_display_name, pair.body_display_name,
                )
                validations.append(ValidationResult(
                    garment_file=pair.garment_display_name,
                    body_file=pair.body_display_name,
                    is_valid=False, confidence=None, reasoning=None,
                    parse_failed=True,
                    error=TaskError(code=ErrorCode.PROCESSING_ERROR, detail=str(exc)),
                ))
        return _task3_response(
            session_id="direct_inference" if request.image_pairs else None,
            validations=validations,
            processing_time=total_time,
        )

    if request.session_id:
        pairs, images_dict = await _resolve_session_pairs(
            request.session_id,
            explicit_pairs=request.pairs,
            default_pair_deriver=lambda cls: app.state.task3.generate_all_pairs(
                cls, max_pairs=request.max_pairs,
            ),
        )
        start_time = time.time()
        validations = await app.state.task3.validate_pairs_batch(
            pairs=pairs, images_dict=images_dict, use_cache=True,
        )
        processing_time = time.time() - start_time
        await session_manager.store_results(
            request.session_id, "task3", [v.dict() for v in validations],
        )
        return _task3_response(
            session_id=request.session_id,
            validations=validations,
            processing_time=processing_time,
        )

    raise HTTPException(
        400, "Provide one of: session_id, image_pairs, or image_data_pairs"
    )


@app.post("/task/gis-pipeline", response_model=GISPipelineResponse)
async def run_gis_pipeline(request: GISPipelineRequest):
    """Run the complete GIS pipeline on a session."""
    session = await session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(404, f"Session {request.session_id} not found")
    if not session.images:
        raise HTTPException(400, "No images in session")

    await session_manager.update_session_status(request.session_id, "processing")
    try:
        response = await app.state.gis.process_folder(
            session=session,
            max_pairs_task2=request.max_pairs_task2,
            max_pairs_task3=request.max_pairs_task3,
            use_cache=request.use_cache,
            priority_weights=request.priority_weights,
        )
        await session_manager.store_results(
            request.session_id, "gis_pipeline", response.dict(),
        )
        await session_manager.update_session_status(request.session_id, "completed")
        return response
    except Exception:
        await session_manager.update_session_status(request.session_id, "error")
        logger.exception("GIS pipeline error")
        raise
    finally:
        # Free decoded pixels; the session (results, metadata) lives on.
        evicted = await session_manager.evict_decoded_images(request.session_id)
        if evicted:
            logger.debug("evicted %d decoded images from %s", evicted, request.session_id)


# ===== Main Entry Point ====================================================

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
