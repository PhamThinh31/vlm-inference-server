# syntax=docker/dockerfile:1.6
#
# Multi-stage build: a fat `builder` stage that compiles wheels, and a
# lean `runtime` stage that only ships the installed site-packages and
# the application code. Cuts image size and attack surface.

# ---------- builder --------------------------------------------------------
FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv git \
    && rm -rf /var/lib/apt/lists/*

# Install into a venv so we can copy the whole tree into the runtime stage.
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------- runtime --------------------------------------------------------
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 vlm \
    && useradd  --system --uid 1000 --gid vlm --home /app --shell /usr/sbin/nologin vlm

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=vlm:vlm app/    ./app/
COPY --chown=vlm:vlm client/ ./client/

# Sessions need a writable mountpoint; declare it so `docker run -v` can
# attach persistent storage instead of the ephemeral container layer.
RUN mkdir -p /var/lib/vlm4gis && chown vlm:vlm /var/lib/vlm4gis
VOLUME ["/var/lib/vlm4gis"]
ENV SESSION_TEMP_DIR=/var/lib/vlm4gis

USER vlm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
