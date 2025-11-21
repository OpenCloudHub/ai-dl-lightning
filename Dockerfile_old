# syntax=docker/dockerfile:1
#==============================================================================#
# Build arguments
ARG RAY_VERSION=2.48.0
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG DISTRO=bookworm
# Compose tags
ARG RAY_PY_TAG=py${PYTHON_MAJOR}${PYTHON_MINOR}
ARG UV_PY_TAG=python${PYTHON_MAJOR}.${PYTHON_MINOR}-${DISTRO}
#==============================================================================#
# Stage: UV base with build tools
FROM ghcr.io/astral-sh/uv:${UV_PY_TAG} AS uv_base

WORKDIR /workspace/project

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project:/workspace/project/src"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#==============================================================================#
# Stage: Development environment (all extras)
FROM uv_base AS dev

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --all-groups --no-install-project

ENV ENVIRONMENT=development

#==============================================================================#
# Stage: Training image
FROM rayproject/ray:${RAY_VERSION}-${RAY_PY_TAG} AS training

WORKDIR /workspace/project

# Copy UV binary
COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

# Install base + training dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra training --no-dev --no-install-project

COPY --chown=ray:users src/ /workspace/project/src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project:/workspace/project/src" \
    ENVIRONMENT=training

#==============================================================================#
# Stage: Serving image (minimal)
FROM rayproject/ray:${RAY_VERSION}-${RAY_PY_TAG} AS serving

WORKDIR /workspace/project

# Copy UV binary
COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

# Install base + serving dependencies only
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra serving --no-dev --no-install-project

COPY --chown=ray:users src/ /workspace/project/src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project:/workspace/project/src" \
    ENVIRONMENT=production
