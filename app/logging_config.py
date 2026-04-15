"""Structured logging configuration.

Single source of truth for how log records are formatted and what
contextual fields are bound to every line. Configured once at process
start (from lifespan in main.py) so every `logging.getLogger(__name__)`
call automatically produces JSON lines with the active request's
`request_id`, `session_id`, and `task` fields.

Why not stdlib `logging` only: we still emit via stdlib (so third-party
libs like vLLM, uvicorn, FastAPI flow through the same pipeline), but
structlog wraps the final rendering so unstructured third-party logs
get the same JSON envelope as our own structured calls.
"""
from __future__ import annotations

import contextvars
import logging
import sys
from typing import Any, Mapping

import structlog

# Context var holding fields bound for the current async task — think
# "request-scoped MDC." Middleware sets request_id here; every log line
# emitted downstream (including from pipeline code that has no awareness
# of the request) automatically carries it.
_log_context: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)


def bind_context(**fields: Any) -> None:
    """Merge `fields` into the current task's log context."""
    current = dict(_log_context.get())
    current.update(fields)
    _log_context.set(current)


def clear_context() -> None:
    _log_context.set({})


def _inject_context(logger, method_name, event_dict):  # noqa: ARG001
    """structlog processor: merge contextvar fields into each log event."""
    ctx = _log_context.get()
    for k, v in ctx.items():
        event_dict.setdefault(k, v)
    return event_dict


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    """Configure stdlib + structlog. Idempotent.

    `json=False` gives human-readable output for local dev; `json=True`
    is the production format — one JSON object per line, one field per
    key, no multi-line tracebacks (they're serialised into the `exception`
    key instead).
    """
    numeric = getattr(logging, level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    # Route stdlib logging through structlog's formatter so vLLM/uvicorn
    # records get the same JSON envelope as our own.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(numeric)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
