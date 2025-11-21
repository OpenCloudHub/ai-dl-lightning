# syntax=docker/dockerfile:1
#==============================================================================#
# Build arguments
ARG RAY_VERSION=2.48.0
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG DISTRO=bookworm
ARG RAY_PY_TAG=py${PYTHON_MAJOR}${PYTHON_MINOR}
ARG UV_PY_TAG=python${PYTHON_MAJOR}.${PYTHON_MINOR}-${DISTRO}

#==============================================================================#
# Stage: Base with UV (tooling layer)
FROM ghcr.io/astral-sh/uv:${UV_PY_TAG} AS uv_base
WORKDIR /workspace/project

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

#==============================================================================#
# Stage: Development (for devcontainer)
FROM uv_base AS dev
COPY pyproject.toml uv.lock ./
# Don't create venv - let devcontainer handle it at runtime
ENV ENVIRONMENT=development

#==============================================================================#
# Stage: TRAINING (production training image)
FROM rayproject/ray:${RAY_VERSION}-${RAY_PY_TAG} AS training
WORKDIR /workspace/project

# Switch to ray user for all operations
USER ray

# Copy UV binary
COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv

# Copy dependency files with proper ownership
COPY --chown=ray:ray pyproject.toml uv.lock ./

# Install dependencies with caching
RUN --mount=type=cache,target=/home/ray/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra training --no-dev --no-install-project

# Copy source code
COPY --chown=ray:ray src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=training

#==============================================================================#
# Stage: SERVING (production serving image)
FROM python:3.12-slim-bookworm AS serving
WORKDIR /workspace/project

# Install system dependencies including wget for health probes
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user matching Ray conventions
RUN groupadd -g 1000 ray && \
    useradd -m -u 1000 -g 1000 -s /bin/bash ray && \
    chown -R ray:ray /workspace/project

# Switch to non-root user
USER ray

# Copy UV binary
COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv

# Copy dependency files with proper ownership
COPY --chown=ray:ray pyproject.toml uv.lock ./

# Install dependencies with caching
RUN --mount=type=cache,target=/home/ray/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra serving --no-dev --no-install-project

# Copy source code
COPY --chown=ray:ray src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=production