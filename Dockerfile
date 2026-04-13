# ===========================================================================
# CIDeconvolve — BIAFLOWS-compatible GPU-enabled Docker image
# ===========================================================================
# Base: NVIDIA CUDA 12.6 runtime (PyTorch GPU support).
# Includes: Python 3.11 and all Python dependencies.
#
# BIAFLOWS convention: images in /data/in, results in /data/out,
# ground truth in /data/gt.  The entrypoint is wrapper.py which
# parses --infolder / --outfolder / --gtfolder and descriptor.json
# parameters, then delegates to deconvolve.py.
# ===========================================================================

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# --- System packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# --- Python dependencies ---
COPY requirements_docker.txt /app/requirements_docker.txt
RUN pip install --no-cache-dir -r requirements_docker.txt

# --- Application code ---
COPY deconvolve.py /app/deconvolve.py
COPY deconvolve_ci.py /app/deconvolve_ci.py
COPY bioflows_local.py /app/bioflows_local.py
COPY wrapper.py /app/wrapper.py
COPY descriptor.json /app/descriptor.json

# --- BIAFLOWS data directories ---
RUN mkdir -p /data/in /data/out /data/gt

# Expose NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["python", "/app/wrapper.py"]
