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
# Stage: UV binaries
FROM ghcr.io/astral-sh/uv:${UV_PY_TAG} AS uv

WORKDIR /workspace/project

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#==============================================================================#
# Stage: Development environment
FROM uv AS dev

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --dev

ENV ENVIRONMENT=development

#==============================================================================#
# Stage: Training - Ray base with training code
# Serves both for training and serving, could be seperated if needed
FROM rayproject/ray:${RAY_VERSION}-${RAY_PY_TAG} AS prod

ARG MLFLOW_TRACKING_URI
ARG MODEL_URI

WORKDIR /workspace/project

COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

COPY src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=production

# Set MLflow tracking URI and model details as environment variables
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    MODEL_URI=${MODEL_URI}