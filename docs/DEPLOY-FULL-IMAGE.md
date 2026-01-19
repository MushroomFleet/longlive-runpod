# LongLive RunPod Deployment - Full Docker Image Method

> **TINS (This Is Not a Story)**: Technical documentation for deploying LongLive to RunPod using a fully-baked Docker image with all dependencies included.

## Overview

This method builds a complete Docker image (~15-20GB) containing all Python dependencies and compiled extensions. Models are still downloaded at runtime via lazy loading, but the build process requires significant local resources.

**Requirements:**
- Windows with Docker Desktop installed
- 40GB+ free disk space
- High-speed internet connection (for pulling base images)
- 16GB+ RAM recommended

## Prerequisites

### 1. Install Docker Desktop for Windows

1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
2. Run the installer and follow prompts
3. Enable WSL 2 backend when prompted (recommended)
4. Restart your computer after installation
5. Launch Docker Desktop and wait for it to fully start

### 2. Verify Docker Installation

Open PowerShell or Command Prompt:

```powershell
docker --version
docker info
```

You should see Docker version and system information.

### 3. Configure Docker Resources

Docker Desktop → Settings → Resources:
- **Memory**: 8GB minimum, 16GB recommended
- **Disk image size**: 60GB+ recommended
- **CPUs**: 4+ recommended for faster builds

## Project Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/MushroomFleet/LongLive.git
cd LongLive
```

### 2. Create the Dockerfile

Create a file named `Dockerfile` in the project root:

```dockerfile
# RunPod Serverless Dockerfile for LongLive Video Generation
# Full image build - includes all compiled dependencies

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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod

# Install flash-attn (this takes 20-30 minutes to compile)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

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
```

### 3. Create .dockerignore

Create a file named `.dockerignore` in the project root:

```
# Git
.git
.gitignore

# Documentation
*.md
docs/
assets/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Training artifacts (not needed for inference)
wandb/
outputs/
videos/
checkpoints/
*.ckpt

# Training scripts
train*.py
train*.sh
trainer/

# Logs
*.log
logs/

# OS files
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
nul
```

### 4. Create the Handler

Create `handler.py` with the RunPod serverless handler (see the main repository for the full implementation).

## Building the Docker Image

### 1. Build the Image

This process takes 30-60 minutes depending on your system:

```powershell
docker build -t longlive-runpod:latest .
```

**Expected output stages:**
1. Pulling base image (~7GB) - 5-10 minutes
2. Installing apt packages - 1-2 minutes
3. Installing Python requirements - 5-10 minutes
4. Compiling flash-attn - 20-30 minutes
5. Copying application files - 1 minute

### 2. Verify the Build

```powershell
docker images longlive-runpod
```

Expected size: 15-20GB

### 3. Test Locally (Optional)

```powershell
# Run container interactively
docker run -it --rm longlive-runpod:latest /bin/bash

# Inside container, verify imports
python -c "import torch; print(torch.cuda.is_available())"
python -c "import flash_attn; print('flash-attn OK')"
python -c "from pipeline.causal_inference import CausalInferencePipeline; print('Pipeline OK')"
```

## Pushing to Docker Registry

### Option A: Docker Hub

```powershell
# Login to Docker Hub
docker login

# Tag the image
docker tag longlive-runpod:latest YOUR_DOCKERHUB_USERNAME/longlive-runpod:latest

# Push to registry
docker push YOUR_DOCKERHUB_USERNAME/longlive-runpod:latest
```

### Option B: RunPod Container Registry

```powershell
# Login to RunPod registry (get credentials from RunPod Console)
docker login https://registry.runpod.io

# Tag for RunPod registry
docker tag longlive-runpod:latest registry.runpod.io/YOUR_NAMESPACE/longlive-runpod:latest

# Push to registry
docker push registry.runpod.io/YOUR_NAMESPACE/longlive-runpod:latest
```

## RunPod Deployment

### 1. Create Network Volume

1. Go to RunPod Console → Storage → Network Volumes
2. Click "Create Network Volume"
3. Configure:
   - **Name**: `longlive-models`
   - **Size**: 50GB minimum (100GB recommended)
   - **Region**: Choose based on GPU availability
4. Click Create

### 2. Create Serverless Endpoint

1. Go to RunPod Console → Serverless → New Endpoint
2. Select **"Use a Docker Image"** (not GitHub integration)
3. Enter your image URL:
   - Docker Hub: `YOUR_DOCKERHUB_USERNAME/longlive-runpod:latest`
   - RunPod Registry: `registry.runpod.io/YOUR_NAMESPACE/longlive-runpod:latest`
4. Configure:
   - **GPU Type**: A100-80GB or H100
   - **Min Workers**: 0 (scale to zero)
   - **Max Workers**: 1-3 (based on needs)
   - **Idle Timeout**: 60 seconds
   - **Network Volume**: Select your volume, mount at `/runpod-volume`
5. Click Deploy

### 3. First Request (Model Download)

The first request will trigger model downloads (~15GB total):

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "A cat playing with yarn", "num_frames": 30}}'
```

This will return a job ID. Check status:

```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

First request takes 10-15 minutes (model download + inference).

### 4. Subsequent Requests

After models are cached, requests complete in 1-3 minutes:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A majestic eagle soaring over mountains at sunset",
      "num_frames": 120,
      "seed": 42
    }
  }'
```

## Troubleshooting

### Docker Build Fails - Out of Memory

Increase Docker Desktop memory allocation:
- Settings → Resources → Memory → 16GB+

### Docker Build Fails - flash-attn Compilation

If flash-attn fails to compile, use the pre-built wheel approach:

```dockerfile
# Replace the flash-attn line with:
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu124torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### Push Fails - Image Too Large

Docker Hub free tier has a 10GB layer limit. Options:
1. Use RunPod's container registry
2. Use a paid Docker Hub plan
3. Use GitHub Container Registry (ghcr.io)

### RunPod Worker Crashes - OOM

- Ensure using A100-80GB or H100 (40GB+ VRAM required)
- Reduce `num_frames` in requests

## Cost Estimation

| Component | Cost |
|-----------|------|
| A100-80GB | ~$2.00/hr |
| Network Volume (100GB) | ~$0.07/hr |
| First request | ~$0.50 (15 min) |
| Subsequent requests | ~$0.07 (2 min) |

## Summary

This full-image method:
- **Pros**: Faster cold starts (no pip installs at runtime), reliable builds
- **Cons**: Large image size (~20GB), long build times, requires local Docker

For bandwidth-limited environments, see `DEPLOY-LIGHTWEIGHT.md` for the GitHub integration approach.
