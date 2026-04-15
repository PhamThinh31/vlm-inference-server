"""
Prometheus metrics. Kept in its own module so the middleware and any
pipeline code can import without pulling the full FastAPI app graph.
"""
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests by method, path and status.",
    ["method", "path", "status"],
)

request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

inflight_requests = Gauge(
    "http_inflight_requests",
    "In-flight HTTP requests currently being served.",
)


def _label_path(path: str) -> str:
    """
    Collapse high-cardinality path segments (UUIDs, numeric ids) so the
    metric doesn't explode one series per session id.
    """
    parts = []
    for seg in path.split("/"):
        if not seg:
            parts.append(seg)
            continue
        if seg.isdigit() or (len(seg) >= 16 and "-" in seg):
            parts.append(":id")
        else:
            parts.append(seg)
    return "/".join(parts)


def observe_request(method: str, path: str, status: int, duration: float) -> None:
    labelled = _label_path(path)
    requests_total.labels(method=method, path=labelled, status=str(status)).inc()
    request_duration_seconds.labels(method=method, path=labelled).observe(duration)


def metrics_response() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
