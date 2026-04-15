# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI: ruff lint + format check, unit + integration
  tests with coverage, and a Docker build job. Concurrency-cancel on
  superseded runs. No GPU deps installed — the integration suite uses
  the in-process FakeEngine.
- `app/pipelines/_runner.py` — shared `run_vlm()` + `build_prompt_text()`
  helpers. Removes three copies of resize-then-build-prompt-then-call.
- `tests/integration/test_session_lifecycle.py` — pins the TTL
  expiry contract and the `inference_window` eviction guarantee
  (including the on-exception path).
- README "Operational limits" section flagging in-process session
  state, decoded-image RAM ceiling, and the no-supervisor caveat.
- **Client-disconnect cancellation.** The middleware races each
  request against a disconnect poller; on disconnect, the handler
  coroutine is cancelled and `engine.abort(request_id)` is called so
  vLLM stops generating tokens for a request nobody is listening for.
- **Graceful shutdown drain.** SIGTERM flips `app.state.shutting_down`;
  new `/task/*` requests get 503 + `Retry-After` while probes and
  metrics keep answering. Lifespan waits up to
  `GRACEFUL_SHUTDOWN_TIMEOUT` (default 30s) for in-flight task work
  before tearing down the engine.
- `ErrorCode` enum and `TaskError` model in `app/schemas.py` —
  structured, machine-readable error reporting on task result rows.

### Added
- Structured JSON logging via `structlog` with `request_id` propagated
  through a `contextvars.ContextVar` to all downstream log lines.
- `X-Request-Id` header: server generates one per request or honours an
  inbound value, echoed on every response.
- `/healthz` (liveness) and `/readyz` (readiness) split. `/health`
  kept as a deprecated alias.
- Integration test harness under `tests/integration/` using
  `httpx.ASGITransport` with a `FakeEngine` stub at the vLLM boundary.
- Contract tests under `tests/unit/test_pipeline_contracts.py` guarding
  the "no fabricated defaults" invariants.
- `docs/architecture.md`, `docs/runbook.md`, `docs/performance.md`.
- `Makefile` with `dev`, `test`, `load-test`, `bench`, `lint` targets.
- `.pre-commit-config.yaml` (ruff, mypy, detect-secrets, pytest-fast).
- Config knobs for quantization and speculative decoding
  (`VLLM_QUANTIZATION`, `VLLM_SPECULATIVE_MODEL`,
  `VLLM_NUM_SPECULATIVE_TOKENS`).
- Multi-stage Dockerfile with non-root `vlm` user and persistent
  session volume (`/var/lib/vlm4gis`).

### Changed
- **BREAKING:** `GISPipelineResponse.cache_hits: int` removed; replaced
  by `cache_hit_rate: Optional[float]`. None means the engine did not
  expose the metric — never fabricated as zero, which would be
  indistinguishable from a real cache miss in dashboards.
- **BREAKING:** `Task2Result.error` and `ValidationResult.error` are
  now `Optional[TaskError]` (was `Optional[str]`). Downstream consumers
  must branch on `error.code` — an `ErrorCode` enum — rather than
  substring-matching a free-text string. Renaming a code is a breaking
  schema change; adding a new one is deliberate.
- **BREAKING:** `Task2Result.category` is now `Optional[str]`. Failure
  rows have `category=None` and a populated `error` field instead of
  sentinel strings like `"unknown"` or `"error"`.
- **BREAKING:** `ValidationResult.confidence` is now `Optional[float]`.
  Unreported confidence is `None`; the previous `0.5` default was
  fabricated data.
- `Task1Classifier.classify_images` return annotation corrected from
  `List[ImageClassification]` to `Tuple[List[ImageClassification], str]`
  to match the long-standing implementation. Callers (`main.py`,
  `gis_processor.py`) already destructured the tuple; the annotation
  was the lie.
- `Task1Classifier._parse_response` no longer applies a positional
  index-based fallback. Missing model keys produce `parse_failed=True`
  rows instead of silently mis-labelled ones.
- Engine request ID generator switched to `itertools.count` (atomic
  under the GIL) to prevent duplicate IDs under concurrent load.
- `AsyncLLMEngine.from_engine_args` now invoked via `asyncio.to_thread`
  so model init does not block the event loop.
- Removed hardcoded `batch_size=5` in `main.py` lifespan; respects
  `TASK1_BATCH_SIZE` env var.

### Removed
- `safe_parse_json` is no longer used as a fallback-of-a-fallback in
  Task 1 parsing. It remains available for direct use.
