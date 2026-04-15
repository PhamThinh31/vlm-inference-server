"""
Async vLLM engine wrapper.

A single module-level `engine` instance is shared across FastAPI routes.
Initialisation is idempotent and guarded by an asyncio lock.
"""
import asyncio
import itertools
import logging
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .config import settings

logger = logging.getLogger(__name__)


# Error types vLLM raises that are worth retrying. OOM and CUDA async
# errors usually clear once the current request completes; anything else
# (tokeniser failure, invalid prompt, validation error) is terminal and
# retrying just wastes GPU minutes.
_RETRYABLE_EXCEPTIONS = (
    asyncio.TimeoutError,
    RuntimeError,  # narrows to CUDA OOM / engine transient — see _is_transient
)


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        tag in msg
        for tag in ("out of memory", "cuda", "engine is stopped", "timeout")
    )


@dataclass
class EngineConfig:
    model_path: str = "/path/to/your/model/checkpoint"
    trust_remote_code: bool = True  # required by most VLM checkpoints
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.6
    max_num_seqs: int = 256
    disable_log_stats: bool = True
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    limit_mm_per_prompt: Dict[str, int] = field(
        default_factory=lambda: {"image": 10}
    )
    # Optional perf knobs. None = upstream vLLM default; set via env
    # (see docs/performance.md). Passed straight through to
    # AsyncEngineArgs so we inherit vLLM's validation.
    quantization: Optional[str] = None
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5


class VLMEngine:
    """Thin async wrapper around `vllm.AsyncLLMEngine`."""

    def __init__(self) -> None:
        self.engine: Optional[AsyncLLMEngine] = None
        self.tokenizer: Any = None
        self.config: Optional[EngineConfig] = None
        self._init_lock = asyncio.Lock()
        # itertools.count.__next__ is atomic under the GIL, so concurrent
        # coroutines cannot collide on the same request ID. The previous
        # `+= 1` was a read-modify-write and could produce duplicate IDs
        # (vLLM silently dedupes by ID — you'd lose requests, not crash).
        self._request_counter = itertools.count(1)
        self.initialized = False

    async def initialize(self, config: Optional[EngineConfig] = None) -> None:
        async with self._init_lock:
            if self.initialized:
                return
            self.config = config or EngineConfig()
            logger.info("initialising vLLM engine: %s", self.config.model_path)

            kwargs: Dict[str, Any] = dict(
                model=self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_seqs=self.config.max_num_seqs,
                limit_mm_per_prompt=self.config.limit_mm_per_prompt,
                disable_log_stats=self.config.disable_log_stats,
                enable_prefix_caching=self.config.enable_prefix_caching,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
            )
            # Only pass optional knobs when set so we inherit vLLM's
            # evolving defaults instead of pinning them to None here.
            if self.config.quantization:
                kwargs["quantization"] = self.config.quantization
            if self.config.speculative_model:
                kwargs["speculative_model"] = self.config.speculative_model
                kwargs["num_speculative_tokens"] = self.config.num_speculative_tokens
            args = AsyncEngineArgs(**kwargs)
            # `from_engine_args` loads weights, spins up the executor, and
            # touches CUDA — multi-second blocking work. Running it inline
            # stalls the event loop and every in-flight health probe times
            # out. Offload to a worker thread so the loop stays responsive.
            self.engine = await asyncio.to_thread(AsyncLLMEngine.from_engine_args, args)
            self.tokenizer = await self.engine.get_tokenizer()
            self.initialized = True

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info("engine ready on %s (%.1fGB)", name, mem)

    def _next_request_id(self) -> str:
        return f"req_{next(self._request_counter)}"

    async def generate(
        self,
        prompt: str,
        images: List[Any],
        sampling_params: Optional[SamplingParams] = None,
        image_ids: Optional[List[str]] = None,  # reserved for future KV-cache keying
        use_cache: bool = True,  # prefix caching is configured at engine level via enable_prefix_caching
    ) -> str:
        if not self.initialized:
            raise RuntimeError("engine not initialised; call initialize() first")

        sampling_params = sampling_params or SamplingParams(
            temperature=settings.default_temperature,
            max_tokens=settings.default_max_tokens,
            stop=["</s>", "<|endoftext|>"],
        )
        inputs = {"prompt": prompt, "multi_modal_data": {"image": images}}

        attempts = max(1, settings.engine_retry_attempts + 1)
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            request_id = self._next_request_id()
            try:
                final = None
                async for out in self.engine.generate(
                    inputs,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ):
                    final = out
                return final.outputs[0].text if final else ""
            except asyncio.CancelledError:
                # Client disconnected or the request was aborted upstream.
                # Tell vLLM to stop scheduling this request — otherwise the
                # GPU keeps generating tokens nobody is listening for.
                # engine.abort is best-effort; swallow its own errors so we
                # don't mask the original cancellation.
                try:
                    await self.engine.abort(request_id)
                except Exception:  # noqa: BLE001
                    logger.debug("engine.abort(%s) failed during cancel", request_id)
                logger.info("engine.generate cancelled (request_id=%s)", request_id)
                raise
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if not _is_transient(exc) or attempt == attempts - 1:
                    raise
                # Exponential backoff with jitter; jitter matters because
                # concurrent requests retrying in lock-step re-trigger the
                # same OOM they just hit.
                delay = settings.engine_retry_base_delay * (2 ** attempt)
                delay *= 0.5 + random.random()
                logger.warning(
                    "engine.generate transient failure (%s); retry %d/%d in %.2fs",
                    exc, attempt + 1, attempts - 1, delay,
                )
                await asyncio.sleep(delay)
        # Unreachable: loop either returns or raises. Keeps the type checker happy.
        raise last_exc  # type: ignore[misc]

    def build_prompt(
        self,
        messages: List[Dict[str, Any]],
        images: List[Any],
    ) -> str:
        """Apply the tokenizer chat template with image placeholders."""
        content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
        if messages and isinstance(messages[-1].get("content"), str):
            content.append({"type": "text", "text": messages[-1]["content"]})
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Best-effort read of vLLM's prefix-cache counters.

        vLLM's stats API has shifted between minor versions; we probe the
        most common accessors and gracefully fall back to a 'not available'
        marker instead of lying to callers (see `GISProcessor.cache_hits`
        history — we used to fabricate this number).
        """
        if not self.initialized or self.engine is None:
            return {"available": False, "reason": "engine not initialized"}
        try:
            get_stats = getattr(self.engine, "get_stats", None) or getattr(
                getattr(self.engine, "engine", None), "get_stats", None
            )
            if get_stats is None:
                return {"available": False, "reason": "engine.get_stats missing"}
            stats = get_stats() if not asyncio.iscoroutinefunction(get_stats) else await get_stats()
            # Fields we care about across vLLM versions:
            return {
                "available": True,
                "gpu_prefix_cache_hit_rate": getattr(stats, "gpu_prefix_cache_hit_rate", None),
                "num_prompt_tokens": getattr(stats, "num_prompt_tokens", None),
                "num_generation_tokens": getattr(stats, "num_generation_tokens", None),
            }
        except Exception as exc:  # noqa: BLE001 - surface the reason, don't crash
            logger.debug("cache stats probe failed: %s", exc)
            return {"available": False, "reason": str(exc)}

    async def health_check(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "initialized": self.initialized,
            "model_loaded": self.engine is not None,
            "gpu_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1024**3, 2
            )
            status["gpu_memory_reserved_gb"] = round(
                torch.cuda.memory_reserved() / 1024**3, 2
            )
        return status


# Module-level singleton.
engine = VLMEngine()


@asynccontextmanager
async def get_engine():
    if not engine.initialized:
        await engine.initialize()
    yield engine
