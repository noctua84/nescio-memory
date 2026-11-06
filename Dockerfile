# syntax=docker/dockerfile:1

# Multi-stage build following the uv Docker guide
# (https://docs.astral.sh/uv/guides/integration/docker/): dependencies are
# installed from uv.lock in a builder stage, and only the finished virtualenv
# is copied into the runtime stage, so uv and the build cache are not shipped.
#
# Build arg:
#   EXTRAS  empty (default) -> the Ollama image: small, no torch.
#           any value        -> also installs the `local-embeddings` extra
#                               (sentence-transformers + torch, roughly 530 MB
#                               of wheels). Set EMBEDDING_BACKEND=local to use it.
#   docker build --build-arg EXTRAS=local-embeddings .
#
# Base images are pinned by digest. Dependabot's docker ecosystem updates the
# tag and the digest together.

# uv is a named stage so Dependabot can track its version.
FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv

FROM python:3.12-slim-trixie@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30 AS builder

COPY --from=uv /uv /uvx /bin/

ARG EXTRAS=""

# COMPILE_BYTECODE: faster container start. LINK_MODE=copy: the cache mount is
# on a different filesystem, so hardlinks would only produce a warning.
# PYTHON_DOWNLOADS=0: use the base image's Python, never fetch another one.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first, so this layer is rebuilt only when the lockfile changes.
# --locked fails the build if uv.lock is stale. --no-build forbids building
# sdists, matching CI: every locked package ships a wheel.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-build --no-dev --no-install-project ${EXTRAS:+--extra local-embeddings}

COPY pyproject.toml uv.lock README.md ./
COPY app ./app

# The project itself. It has no [build-system], so uv treats it as a virtual
# project and installs nothing more here; the app runs from /app/app. The step
# is kept so that adding a build backend later does not silently ship an image
# without the project installed.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-build --no-dev --no-editable ${EXTRAS:+--extra local-embeddings}


FROM python:3.12-slim-trixie@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30

# Unprivileged user with a real home directory: the local backend caches
# Hugging Face weights under $HOME on first use.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app

# Code is root-owned and read-only to the app user; only /home/app is writable.
COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts ./scripts

# PYTHONPATH: the project is not installed into the venv, and running a script
# (python scripts/create_api_key.py) puts scripts/ rather than /app on sys.path,
# so `import app` would fail without it.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

EXPOSE 8080

# python-slim has no curl, so probe with the standard library. /health is the
# liveness endpoint: it touches no dependency, so a database or Ollama outage
# cannot mark the container unhealthy. Readiness (/ready) is probed by compose.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"]

# HOST and PORT from .env are not applied by the app, so they are set here.
# Migrations are deliberately NOT run on start: run `alembic upgrade head` as
# its own step (the compose `migrate` service does).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
