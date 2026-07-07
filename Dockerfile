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

FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

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
    && python -m pip install --no-compile -r requirements_docker.txt

# --- Application code ---
COPY cideconvolve_io/ /app/cideconvolve_io/
COPY core/ /app/core/
COPY wrapper.py /app/wrapper.py
COPY bilayers_cli.py /app/bilayers_cli.py
COPY config.yaml /app/config.yaml

# --- Bilayers data directories ---
RUN mkdir -p /data/in /data/out

# Expose NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["python", "/app/wrapper.py"]
