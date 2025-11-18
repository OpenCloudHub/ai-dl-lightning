# syntax=docker/dockerfile:1
#==============================================================================#
# Build arguments
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG DISTRO=bookworm
ARG NVIDIA_TAG=25.06-py3
#==============================================================================#
# Stage: NVIDIA PyTorch base
FROM nvcr.io/nvidia/pytorch:${NVIDIA_TAG} AS nvidia_base
WORKDIR /workspace/project

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project:/workspace/project/src"

#==============================================================================#
# Stage: UV + build tools
FROM nvidia_base AS uv_base

COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /uvx /usr/local/bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git curl ninja-build \
    ffmpeg libavutil-dev libavcodec-dev libavformat-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

#==============================================================================#
# Stage: DEVELOPMENT (everything)
FROM uv_base AS dev

COPY pyproject.toml uv.lock ./

ENV MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE

# Install all extras + dev dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --all-groups --no-install-project

ENV ENVIRONMENT=development

#==============================================================================#
# Stage: TRAINING image (with flash-attn)
FROM uv_base AS training

COPY pyproject.toml uv.lock ./

# Conservative settings for flash-attn compilation
ENV MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE

# Install base + training dependencies (includes flash-attn)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra training --no-dev --no-install-project

COPY src/ /workspace/project/src/

ENV ENVIRONMENT=training

#==============================================================================#
# Stage: SERVING image (NO flash-attn, smaller)
FROM nvidia_base AS serving

# Copy UV but skip build tools (cleaner image)
COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

# Install base + serving dependencies (NO flash-attn)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra serving --no-dev --no-install-project

COPY src/ /workspace/project/src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project:/workspace/project/src" \
    ENVIRONMENT=production
