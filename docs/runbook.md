# Runbook

Concrete diagnostics for the incidents we have actually seen. If you
add a new failure mode to this runbook, include the symptom, the probe
command, and the remediation — not just a narrative.

---

## 1. GPU Out-Of-Memory (OOM)

**Symptom**
- `engine.generate` retries then raises `RuntimeError: CUDA out of memory`
- Grafana: `http_request_duration_seconds` p99 spikes, `http_inflight_requests` climbs, 5xx on `/task/*`.

**Probe**
```bash
kubectl exec -it <pod> -- nvidia-smi
curl -s localhost:8000/readyz | jq .gpu_memory_usage
```

**Root causes (in descending frequency)**
1. **Session heap pressure.** 50 sessions × 10 images × ~48 MB decoded RGB = ~24 GB pinned. The 24h TTL is the worst case.
   **Fix:** verify `evict_decoded_images` runs in the route's `finally` (it does in GIS pipeline; spot-check new routes). Lower `SESSION_MAX_AGE_HOURS`.
2. **`GPU_MEMORY_UTIL` too high.** vLLM tries to claim e.g. 0.9 of VRAM on startup; anything else on the GPU starves.
   **Fix:** drop to 0.6–0.7 for shared nodes.
3. **A run-away `limit_mm_per_prompt`** — someone raised the per-prompt image cap from 10.
   **Fix:** revert in `EngineConfig`.

**Remediation**
- Rolling-restart the pod; readiness will keep traffic off it until vLLM warms.
- If OOM recurs within minutes of restart, the steady-state load exceeds capacity — scale out or shrink `max_num_seqs`.

---

## 2. Stuck / Unresponsive Engine

**Symptom**
- `/readyz` returns 200 but `/task/*` hangs past `ENGINE_RETRY_BASE_DELAY`.
- `http_inflight_requests` monotonically climbs; GPU utilisation pegged at 100% with zero tokens/sec.

**Probe**
```bash
# Is the event loop alive?
curl -sS localhost:8000/healthz
# Is vLLM scheduling anything?
kubectl logs <pod> | rg -i 'scheduler|iteration' | tail -20
# Any Python-level deadlock?
kubectl exec -it <pod> -- py-spy dump --pid 1
```

**Root causes**
1. **CUDA async error storm** — one request corrupted the stream, subsequent ones inherit the bad state.
2. **Prefix cache corruption** after an OOM recovery — rare but observed once on vllm 0.11.1.

**Remediation**
- There is no supervisor (Phase 2 work). Today: delete the pod. K8s will recreate it, `/readyz` gates traffic until warm.
- If this recurs more than once per week, prioritise the engine supervisor in Phase 2.

---

## 3. Parse-Failed Spikes

**Symptom**
- `parse_failed_pairs` in GIS pipeline summary > 10% over a window.
- Grafana (once added): `task1_parse_failed_total` slope climbs.

**Probe**
```bash
kubectl logs <pod> | rg 'task[123].*parse' | tail -50
# Correlate with a recent deploy
git log --since='24h ago' -- app/pipelines/
```

**Root causes**
1. **Prompt drift.** A prompt_templates edit changed the output schema the model emits; parser regex no longer matches.
   **Fix:** revert prompt; or update parser; never ship prompt + parser changes in separate PRs.
2. **Model checkpoint swap.** A new VLM weight behaves differently on hedge cases ("the pair is yes-adjacent to a match...").
   **Fix:** gate model rollouts behind a canary (Phase 4 item).
3. **Tokenizer truncation.** Response hits `DEFAULT_MAX_TOKENS` before the closing `}`; JSON extractor returns None.
   **Fix:** bump `DEFAULT_MAX_TOKENS`; check `raw_response` in logs — truncated JSON is the tell.

**Remediation**
- Parse failures are *reported*, not swallowed. Check `summary.data_quality` on the last GIS response for counts per task.

---

## 4. Prefix-Cache Miss (Low Hit Rate)

**Symptom**
- `summary.cache_stats.after.gpu_prefix_cache_hit_rate` < 0.1 during the steady state of a session.
- Wall-clock per task ~unchanged after warmup.

**Probe**
```bash
curl -s localhost:8000/task/gis-pipeline -d '...' | jq .summary.cache_stats
```

**Root causes**
1. **Per-request image IDs in the prompt prefix.** The simple-name mapper injects `image_001, image_002, ...` into the prompt — cache-friendly. If a new prompt version concatenates filenames or session IDs into the *prefix* portion of the prompt, every request has a unique prefix → 0% hit rate.
   **Fix:** keep invariant system prompt at the top of `prompt_templates.py`; only per-request variables go in the tail.
2. **Chunked prefill disabled.** `ENABLE_CHUNKED_PREFILL` must be True for prefix caching to amortise long prompts.
3. **`gpu_prefix_cache_hit_rate` field unavailable** on the installed vLLM build.
   **Fix:** upgrade vLLM or check `engine.get_cache_stats().available`.

---

## 5. Silent Data Corruption (historical — must not regress)

**Symptom**
- Downstream modeling team reports a spike of `category="unknown"` rows or `confidence=0.5` rows.

**This is NOT a runtime incident; it is a regression of the parser contract.**

The codebase forbids:
- `Task2Result.category == "error" | "unknown"` as a failure sentinel — failure is `category=None` + `error="..."`.
- `ValidationResult.confidence == 0.5` as a default — unreported confidence is `None`.
- Positional-index fallback in Task 1 parsing — missing keys produce `parse_failed=True`.

**Probe**
```bash
pytest tests/unit/test_pipeline_contracts.py -v
```

**Remediation**
- Revert the PR that reintroduced the default. Do not patch downstream to filter — that normalises the lie.

---

## Escalation

- Engine-level incidents (OOM, stuck, GPU driver): on-call ML-platform.
- Parse-failed spikes: team that owns `app/pipelines/prompt_templates.py`.
- API-level (5xx, auth, rate limit): API team on-call.
