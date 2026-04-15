 **Notice**
>
> This repository is a public, sanitized extract of a VLM inference
> server originally built for a private industrial project. Prompt
> templates, model checkpoints, dataset paths, and customer-specific
> logic have been replaced with placeholders or removed.
>
> The architecture, serving stack, pipeline composition, and tests
> are the production design. The redacted prompt templates in
> `app/pipelines/prompt_templates.py` are intentional — reach out
> directly if you want to discuss the real ones.
>
> AI assistance (Claude) was used for documentation, refactors, and
> the hardening passes described in `CHANGELOG.md`. Design decisions
> and the original implementation are mine.

# VLM Inference Server

A high-throughput, async inference server for Vision-Language Models (VLMs), built on **FastAPI** and **vLLM**. Designed for production workloads that need stateful, multi-step image reasoning — garment classification, attribute extraction, and pair validation for Virtual Try-On (VTON) pipelines.

> This project demonstrates a production-style VLM serving stack: an async engine singleton, vision-encoding reuse across tasks, session-scoped state, concurrent batching, and a reproducible stress-test harness.

---

## Operational limits — read before deploying

This service is **single-node by design**. Several pieces of state are
in-process and must be understood before scaling out:

| Concern | Limit | What happens at the limit | Mitigation |
|---|---|---|---|
| Session state | In-memory `dict` in `SessionManager` | A second replica cannot serve a session created on the first → 404 / lost uploads | Pin clients to one replica via session affinity, OR migrate session state to Redis (see `docs/architecture.md` → "What this does NOT do yet") |
| Decoded image RAM | ~48 MB per 4K RGB image, pinned for the inference window only | At 50 sessions × 10 images held simultaneously ≈ 24 GB heap | `inference_window()` evicts on exit; lower `SESSION_MAX_AGE_HOURS` for tighter bounds |
| GPU memory | `GPU_MEMORY_UTIL` (default 0.6) of the device | OOM under spike, retried then surfaced as 5xx | Drop on shared nodes; scale up the device class; see `docs/runbook.md` § "GPU OOM" |
| vLLM engine | One per process | A stuck engine fails `/readyz` (drains traffic) but does not auto-restart | Manual / K8s pod restart today; supervisor is Phase 3 work |
| Cleanup loop | One sweep per replica | At N replicas you do N sweeps of the same disk | Run a single replica, OR add leader election |

Operating this on more than one replica without addressing the
session-state row above will silently corrupt user-visible state.
That is not a future-improvement bullet; it is the boundary of what
the current code supports.

For health/probe wiring see `docs/runbook.md`. For tuning knobs see
`docs/performance.md`.

---

## Highlights

- **Async all the way down** — FastAPI + `vllm.AsyncLLMEngine`, continuous batching, prefix caching.
- **Session-based state** — uploaded images are stored once per session and reused across the multi-task pipeline, so the client never re-uploads the same file between tasks.
- **Prefix-cache friendly prompts** — task prompts share a large common prefix, which vLLM's prefix caching exploits across sequential requests in the pipeline.
- **Three composable tasks**
  - **Task 1 — Classification**: `garment` / `body` / `unknown`
  - **Task 2 — Attribute extraction**: structured garment attributes
  - **Task 3 — Pair validation**: VTON-style garment↔body compatibility
- **Pipeline endpoint** — run all three on a folder in one call.
- **Stress-test harness** — reproducible load tests with metrics export (`tests/`).

---

## Architecture

```
                   ┌──────────────────────────────────────────┐
                   │               FastAPI App                │
                   │  /session  /task/classify  /task/attrs   │
                   │  /task/validate  /task/gis-pipeline      │
                   └──────────────┬───────────────────────────┘
                                  │
                   ┌──────────────▼───────────────┐
                   │   SessionManager (in-memory) │
                   │  - image store per session   │
                   │  - vision-encoding cache     │
                   │  - TTL cleanup               │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────▼───────────────┐
                   │   VLMEngine  (singleton)     │
                   │   vllm.AsyncLLMEngine        │
                   │   - continuous batching      │
                   │   - prefix caching           │
                   │   - chunked prefill          │
                   └──────────────────────────────┘
```

Code layout:

| Path | What's in it |
|------|--------------|
| [app/main.py](app/main.py) | FastAPI routes, lifespan, CORS |
| [app/engine.py](app/engine.py) | vLLM engine singleton + `EngineConfig` |
| [app/session.py](app/session.py) | Session lifecycle, image store, cache |
| [app/schemas.py](app/schemas.py) | Pydantic request/response models |
| [app/config.py](app/config.py) | Env-driven settings |
| [app/pipelines/](app/pipelines) | Task 1 / 2 / 3 + GIS pipeline |
| [client/vlm4gis_client.py](client/vlm4gis_client.py) | Async Python client |
| [tests/](tests) | Stress-test harness & metrics |

---

## Benchmarks

Reproducible via `tests/process_folders_with_metrics.py`. Full raw data in [stress_test_metrics.json](stress_test_metrics.json).

**End-to-end folder processing** — classification + pair validation on 65 folders (298 images), `max_concurrent_tasks=20`, single GPU.

| Metric | Value |
|---|---|
| Folders processed | 65 / 65 (100% success) |
| Total images | 298 |
| Total wall time | **39.37 s** |
| Throughput | **0.53 img/s** sustained end-to-end pipeline |
| Peak folder throughput | 1.20 img/s |
| Median folder throughput | 0.54 img/s |
| Avg folder latency | 9.77 s |
| Pair validations | 208 |

Throughput here measures the *full multi-task pipeline* (classify → validate pairs), not single-forward-pass latency. Single-task classification is substantially faster; see `tests/quick_load_test.py`.

---

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.1+ (tested on a single 24 GB+ GPU)
- Python 3.10+
- A Qwen-VL-style checkpoint at `MODEL_PATH`

### Install

```bash
git clone https://github.com/PhamThinh31/vlm-inference-server.git
cd vlm-inference-server

# Install vLLM first (version-specific wheel)
pip install vllm==0.11.2 --extra-index-url https://wheels.vllm.ai/0.11.2/

# Then the rest
pip install -r requirements.txt

cp .template.env .env
# edit .env — at minimum set MODEL_PATH
```

### Run

```bash
# Local
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Docker
docker-compose up -d
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive OpenAPI UI.

---

## API Overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`    | `/health`                  | Engine + session health |
| `POST`   | `/session/create`          | Create a processing session |
| `POST`   | `/session/{id}/upload`     | Upload one or more images |
| `GET`    | `/session/{id}`            | Session info |
| `DELETE` | `/session/{id}`            | Drop session & cached encodings |
| `POST`   | `/task/classify`           | Task 1: classify images |
| `POST`   | `/task/attributes`         | Task 2: extract attributes |
| `POST`   | `/task/validate`           | Task 3: validate garment↔body pairs |
| `POST`   | `/task/gis-pipeline`       | Run full pipeline on a folder |

### Example — full pipeline from the Python client

```python
import asyncio
from client.vlm4gis_client import VLM4GISClient

async def main():
    async with VLM4GISClient(base_url="http://localhost:8000") as client:
        result = await client.process_folder(
            folder_path="./test_images/sample_folder",
            session_name="demo-run",
        )
        print(result["summary"])

asyncio.run(main())
```

### Example — curl

```bash
# Health
curl http://localhost:8000/health

# Create session
SESSION=$(curl -s -X POST http://localhost:8000/session/create \
    -H "Content-Type: application/json" \
    -d '{"session_name":"demo"}' | jq -r .session_id)

# Upload
curl -X POST "http://localhost:8000/session/$SESSION/upload" \
    -F "files=@test_images/example.jpg"

# Classify
curl -X POST http://localhost:8000/task/classify \
    -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SESSION\"}"
```

---

## Configuration

All settings are env-driven ([`.template.env`](.template.env)). Key knobs:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_PATH` | — | Path to the VLM checkpoint (required) |
| `GPU_MEMORY_UTIL` | `0.6` | Fraction of GPU memory vLLM may reserve |
| `MAX_MODEL_LEN` | `8192` | Max context window |
| `MAX_NUM_SEQS` | `256` | Concurrent sequences for batching |
| `ENABLE_PREFIX_CACHING` | `true` | Reuse KV cache for shared prefixes |
| `MAX_CONCURRENT_VALIDATIONS` | `5` | Semaphore for Task 3 pair checks |
| `CORS_ORIGINS` | `*` | Tighten in production |
| `API_KEY_REQUIRED` | `false` | Enable to require `x-api-key` |

---

## Stress Testing

```bash
# Quick load test against a running server
python tests/quick_load_test.py

# Full folder pipeline with metrics export
python tests/process_folders_with_metrics.py \
    --data-dir ./test_images \
    --concurrency 20 \
    --out stress_test_metrics.json
```

See [tests/STRESS_TEST_README.md](tests/STRESS_TEST_README.md) for the full harness.

---

## Engineering Notes

- **Why a module-level engine?** vLLM's `AsyncLLMEngine` holds GPU state; we instantiate it once on startup via FastAPI's `lifespan` and share it across routes.
- **Why session-scoped state?** The three tasks operate on the same images. Uploading once and keying downstream calls by `session_id` avoids re-upload round-trips.
- **Why continuous batching + prefix caching?** VLM prompts share large prefixes (system + task instructions); prefix caching gives a meaningful latency win when the same prompt template runs across many images.
- **Concurrency limits**: Task 3 is CPU/GPU-bound per pair; a `max_concurrent_validations` semaphore prevents scheduler thrash.

---

## Roadmap

- [ ] Unit-test coverage for pipelines (currently only stress tests)
- [ ] Prometheus metrics endpoint
- [ ] Redis-backed session store (Docker Compose already provisions Redis)
- [ ] API-key middleware enforcement
- [ ] Multi-GPU tensor parallel config