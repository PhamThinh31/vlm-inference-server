# app/config.py
"""
VLM4GIS Configuration Module

All settings are driven by environment variables (or a .env file).
Import the ``settings`` singleton wherever you need configuration values.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — single source of truth for configuration."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Server
    app_name: str = "VLM4GIS"
    app_version: str = "1.0.0"
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")

    # Model
    model_path: str = Field(
        default="/path/to/your/model/checkpoint", alias="MODEL_PATH"
    )
    model_name: str = Field(default="qwen-vl", alias="MODEL_NAME")

    # Engine
    gpu_memory_utilization: float = Field(default=0.6, alias="GPU_MEMORY_UTIL")
    max_model_len: int = Field(default=8192, alias="MAX_MODEL_LEN")
    max_num_seqs: int = Field(default=256, alias="MAX_NUM_SEQS")
    max_num_batched_tokens: int = Field(default=8192, alias="MAX_NUM_BATCHED_TOKENS")
    enable_prefix_caching: bool = Field(default=True, alias="ENABLE_PREFIX_CACHING")

    # Quantization: None | "awq" | "gptq" | "gptq_marlin" | "fp8".
    # Must match the checkpoint you're pointing MODEL_PATH at — a
    # mismatch raises at engine init, which is what we want (fail loud
    # at startup, not silently at inference time).
    vllm_quantization: Optional[str] = Field(default=None, alias="VLLM_QUANTIZATION")

    # Speculative decoding. Only helps when the draft model's
    # acceptance rate on your traffic is ~>0.6; below that, it's a net
    # loss. See docs/performance.md.
    vllm_speculative_model: Optional[str] = Field(
        default=None, alias="VLLM_SPECULATIVE_MODEL"
    )
    vllm_num_speculative_tokens: int = Field(
        default=5, alias="VLLM_NUM_SPECULATIVE_TOKENS"
    )

    # Session
    session_temp_dir: str = Field(default="/tmp/vlm4gis", alias="SESSION_TEMP_DIR")
    session_max_age_hours: int = Field(default=24, alias="SESSION_MAX_AGE_HOURS")
    max_image_size: int = Field(default=4096, alias="MAX_IMAGE_SIZE")

    # Task defaults
    default_temperature: float = Field(default=0.1, alias="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=512, alias="DEFAULT_MAX_TOKENS")
    max_concurrent_validations: int = Field(
        default=5, alias="MAX_CONCURRENT_VALIDATIONS"
    )

    # Per-task batching. Task 1 (multi-image classification) is bounded by
    # the prompt-token budget: batch_size=5 keeps ~10 image tokens * 512 per
    # image under max_model_len with headroom for the generation. Bumping
    # this past 7 caused 5454>4096 overflows in practice — hence the default.
    task1_batch_size: int = Field(default=5, alias="TASK1_BATCH_SIZE")
    # Inference-time downscale. Qwen-VL internally tiles at 448px, so
    # anything above ~1024 is wasted pixels + extra vision tokens.
    vlm_input_max_side: int = Field(default=1024, alias="VLM_INPUT_MAX_SIDE")

    # Transient-failure retry budget for engine.generate().
    engine_retry_attempts: int = Field(default=2, alias="ENGINE_RETRY_ATTEMPTS")
    engine_retry_base_delay: float = Field(default=0.5, alias="ENGINE_RETRY_BASE_DELAY")

    # Bounded concurrency for blocking PIL decodes inside the async loop.
    # Tuned to roughly match CPU count; going higher just context-switches.
    image_decode_workers: int = Field(default=4, alias="IMAGE_DECODE_WORKERS")

    # Graceful-shutdown deadline. On SIGTERM we flip a flag that makes
    # new /task/* requests 503 with Retry-After, then wait up to this
    # many seconds for in-flight work to drain before proceeding with
    # teardown. Longer than this and K8s SIGKILLs us anyway.
    graceful_shutdown_timeout: float = Field(
        default=30.0, alias="GRACEFUL_SHUTDOWN_TIMEOUT"
    )
    # How often the disconnect watchdog polls request.is_disconnected().
    # 0.5s is a compromise between cancellation latency and per-request
    # overhead; on 100 concurrent requests that's 200 syscalls/sec total.
    disconnect_poll_interval: float = Field(
        default=0.5, alias="DISCONNECT_POLL_INTERVAL"
    )

    # API
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    api_key_required: bool = Field(default=False, alias="API_KEY_REQUIRED")
    api_keys: Optional[str] = Field(default=None, alias="API_KEYS")

    # Comma-separated list of absolute roots from which /task/* server-path
    # inputs may read. Empty disables the /task/* server-path mode entirely.
    # Default mirrors the docker-compose mount.
    allowed_image_roots: str = Field(
        default="/data,/tmp/vlm4gis", alias="ALLOWED_IMAGE_ROOTS"
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")
    # JSON output in prod; set LOG_JSON=false for a human-readable console
    # renderer during local dev.
    log_json: bool = Field(default=True, alias="LOG_JSON")


settings = Settings()