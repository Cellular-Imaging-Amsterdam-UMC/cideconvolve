# ===========================================================================
# CIDeconvolve — Bilayers-compatible GPU-enabled Docker image
# ===========================================================================
# Base: Python slim. GPU support comes from the CUDA-enabled PyTorch wheel
# plus the host NVIDIA driver mounted by NVIDIA Container Toolkit.
#
# Bilayers convention: images in /data/in and results in /data/out.
# The entrypoint is wrapper.py, which parses parameters from config.yaml
# and then delegates to deconvolve.py.
# ===========================================================================

ARG CUDA_TOOLKIT_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu22.04
ARG CUDA_HOME_PATH=/usr/local/cuda-13.2
ARG PYTORCH_VERSION=2.13.0
ARG PYTORCH_CUDA=cu132

FROM ${CUDA_TOOLKIT_IMAGE} AS cuda-toolkit

FROM python:3.11-slim-bookworm AS optimized-extension-builder

ARG CUDA_HOME_PATH
ARG PYTORCH_VERSION
ARG PYTORCH_CUDA
ENV CUDA_HOME=${CUDA_HOME_PATH}
ENV CUDA_PATH=${CUDA_HOME_PATH}
ENV PATH=${CUDA_HOME_PATH}/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;10.0;12.0+PTX"

COPY --from=cuda-toolkit ${CUDA_HOME_PATH} ${CUDA_HOME_PATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "torch==${PYTORCH_VERSION}+${PYTORCH_CUDA}" \
        --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
COPY core/optimized_cuda/ /build/core/optimized_cuda/
RUN PYTHONPATH=/build python /build/core/optimized_cuda/build_prebuilt.py --output /optimized-extension

FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_VERSION
ARG PYTORCH_CUDA

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CIDECONVOLVE_PYTORCH_VERSION=${PYTORCH_VERSION}
ENV CIDECONVOLVE_PYTORCH_CUDA=${PYTORCH_CUDA}

# --- System packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        fonts-dejavu-core \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
COPY requirements_docker.txt /app/requirements_docker.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-compile \
        "torch==${PYTORCH_VERSION}+${PYTORCH_CUDA}" \
        --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}" \
    && python -m pip install --no-compile -r requirements_docker.txt

LABEL org.cideconvolve.pytorch.version="${PYTORCH_VERSION}" \
      org.cideconvolve.pytorch.cuda="${PYTORCH_CUDA}"

# --- Application code ---
COPY cideconvolve_io/ /app/cideconvolve_io/
COPY core/ /app/core/
COPY --from=optimized-extension-builder /optimized-extension/_optimized_cuda*.so /app/core/
COPY wrapper.py /app/wrapper.py
COPY bilayers_cli.py /app/bilayers_cli.py
COPY config.yaml /app/config.yaml
COPY docker/cuda_smoke.py /app/cuda_smoke.py

# --- Bilayers data directories ---
RUN mkdir -p /data/in /data/out

# Expose NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["python", "/app/wrapper.py"]
