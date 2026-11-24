# syntax=docker/dockerfile:1.7
#
# The API and the worker (M12.3). One image, two commands.
#
#   build   - resolves dependencies, CPU-only torch, downloads the models
#   runtime - the virtualenv and the models, no compiler, no uv, non-root
#
# Secrets are never baked in: everything sensitive arrives as an environment
# variable at run time (see deploy/README and docker-compose.prod.yml).

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH
WORKDIR /app

# Dependencies first, from the lock alone, so a source change does not
# reinstall them. torch comes from PyTorch's CPU index: the PyPI wheel for
# Linux x86_64 carries CUDA libraries, gigabytes this server cannot use.
COPY pyproject.toml uv.lock ./
RUN uv venv /app/.venv \
 && uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt \
      | grep -vE '^(torch|triton|nvidia-)' > /tmp/requirements.txt \
 && TORCH=$(grep -A1 '^name = "torch"' uv.lock | sed -n 's/^version = "\(.*\)"/\1/p') \
 && uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}" \
 && uv pip install --no-deps -r /tmp/requirements.txt

COPY src ./src
COPY README.md ./
RUN uv pip install --no-deps .

# The models, into the image. `MEMOS_HF_HOME` is absolute so the adapters
# resolve it the same way at build and at run time.
ENV MEMOS_HF_HOME=/app/models
COPY scripts/prefetch_models.py ./scripts/
RUN python scripts/prefetch_models.py


FROM python:${PYTHON_VERSION}-slim AS runtime
RUN groupadd --system --gid 10001 memo \
 && useradd --system --uid 10001 --gid memo --home-dir /app --shell /usr/sbin/nologin memo
WORKDIR /app
ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MEMOS_HF_HOME=/app/models \
    MEMOS_BLOB_ROOT=/data/blobs \
    MEMOS_LOG_JSON=true \
    # The models are in the image; never reach the Hub from a running server.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --from=build --chown=memo:memo /app/.venv /app/.venv
COPY --from=build --chown=memo:memo /app/models /app/models
COPY --chown=memo:memo alembic.ini ./
COPY --chown=memo:memo migrations ./migrations
COPY --chown=memo:memo deploy/healthcheck.py ./deploy/

RUN mkdir -p /data/blobs && chown memo:memo /data/blobs
USER memo
EXPOSE 8000

# Ready, not merely alive: `/health/ready` answers 200 only when Postgres
# answers and pgvector is installed.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "deploy/healthcheck.py"]

# Behind the reverse proxy, the client's address arrives in X-Forwarded-For.
# It is trusted only from the proxy network (docker-compose.prod.yml), which is
# what keeps the login rate limiter per-client rather than one shared bucket.
CMD ["sh", "-c", "exec uvicorn memoryos.api.app:create_app --factory --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips \"${MEMOS_TRUSTED_PROXIES:-127.0.0.1}\""]
