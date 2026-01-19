# RunPod Serverless Dockerfile for LongLive Video Generation
# Based on NVIDIA's LongLive: Real-time Interactive Long Video Generation

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Use --ignore-installed to bypass distutils conflicts with base image packages
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt && \
    pip install --no-cache-dir runpod

# Install flash-attn from pre-built wheel (much faster than compiling from source)
# This avoids the 20+ minute compilation that causes build timeouts
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu124torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Copy the LongLive codebase
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model cache directory (will be mounted as network volume in RunPod)
ENV HF_HOME=/runpod-volume/huggingface
ENV MODEL_CACHE_DIR=/runpod-volume/models

# Entry point
CMD ["python", "-u", "handler.py"]
