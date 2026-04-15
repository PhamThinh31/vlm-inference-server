# Performance Tuning Guide

Tune based on *observed* workload, not guessed load. Every knob below
has a Prometheus signal to watch before and after you change it.

## Quantization

vLLM can load AWQ or GPTQ-quantised checkpoints with no code changes,
just env:

```
MODEL_PATH=/models/qwen-vl-awq
VLLM_QUANTIZATION=awq   # or gptq, gptq_marlin, fp8
```

Wired via `EngineConfig.quantization` → `AsyncEngineArgs(quantization=...)`.

**When to use**
- VRAM-constrained nodes (A10, L4, 4090).
- Throughput-bound workloads where per-token latency is already adequate.

**When NOT to use**
- H100/H200 with plenty of VRAM — fp16 is often faster because the
  dequantization path is not free.
- First pilot of a new model — validate accuracy unquantised first.

## Speculative decoding

```
VLLM_SPECULATIVE_MODEL=/models/qwen-vl-draft-1b
VLLM_NUM_SPECULATIVE_TOKENS=5
```

Wired via `EngineConfig.speculative_model`. Only helps if the draft
model's acceptance rate (check vLLM logs for `accept_rate`) is above
~0.6. Below that, you pay the draft inference cost with no benefit.

## Dynamic batching

```
MAX_NUM_SEQS=256              # concurrent sequences
MAX_NUM_BATCHED_TOKENS=8192   # per-iteration token budget
```

**Tuning procedure**
1. Start from defaults.
2. Ramp concurrent clients with `tests/stress_test_with_data.py`.
3. Watch Grafana:
   - `http_request_duration_seconds` p50/p99
   - vLLM's `num_running_reqs` vs `num_waiting_reqs`
4. If `num_waiting_reqs` stays high and p99 is acceptable → raise `MAX_NUM_SEQS`.
5. If `num_waiting_reqs` is low but GPU utilisation is low → raise `MAX_NUM_BATCHED_TOKENS`.
6. If VRAM OOMs → lower `MAX_NUM_SEQS`, lower `GPU_MEMORY_UTIL`.

Keep a log of the numbers per hardware SKU; this is node-specific.

## Prefix caching — measure, don't hope

`enable_prefix_caching=True` is only useful if the prompt *prefix* is
stable across requests. The prompt templates in
`app/pipelines/prompt_templates.py` are structured so the invariant
system instructions live at the top and per-request variables
(image IDs, filenames) are interpolated at the bottom.

**Guard rails**
- Do not concatenate a session ID, timestamp, or user-specific string
  into the template head. It will zero your hit rate silently.
- Check `summary.cache_stats.after.gpu_prefix_cache_hit_rate` on a real
  workload. Target > 0.5 for session-style traffic.

## Image preprocessing — why thread pool, not process pool

PIL/Pillow releases the GIL during C-level decode, convert, and resize
calls. The existing `asyncio.to_thread` + `IMAGE_DECODE_WORKERS`
semaphore gets real parallelism on multi-core hosts.

A `ProcessPoolExecutor` would require pickling `PIL.Image` objects
across process boundaries (~MB per image, copied twice), and you lose
the shared decode buffer. Benchmarks from similar workloads show
process pools **slower** than threads for decode-heavy paths at any
image size above ~512px.

**The real CPU-bound hotspot is JSON / regex parsing**, not decode.
If you see that in a profile, the fix is to replace the regex cascade
in `task2_attributes._parse_response` with a single compiled regex
or a Pydantic `TypeAdapter` — not to move decode to processes.

## Benchmarking

```bash
make bench            # single-shot perf snapshot
make load-test        # sustained load with metrics capture
```

The baseline lives in `stress_test_metrics.json`. CI regression gate is
Phase 3 work; until then, compare manually.
