# LongLive RunPod Deployment - Lightweight GitHub Integration Method

> **TINS (This Is Not a Story)**: Technical documentation for deploying LongLive to RunPod using GitHub integration with lazy model loading. Optimized for bandwidth-limited environments.

## Overview

This method uses RunPod's GitHub integration to build Docker images remotely, avoiding large local downloads. The Docker image is kept small (<2GB) by downloading models at runtime.

**Requirements:**
- GitHub account
- RunPod account with credits
- No local Docker required
- Works with limited bandwidth (<1GB upload)

**Architecture:**
```
1. Push code to GitHub (~1MB)
2. RunPod builds image remotely (~2GB)
3. First request downloads models to network volume (~15GB)
4. Subsequent requests use cached models
```

## Development Progress Log

This document captures the iterative development process and solutions to issues encountered.

### Issue 1: Base Image Size (7GB+)

**Problem:** The `runpod/pytorch` base image is ~7GB, exceeding bandwidth limits for local Docker builds.

**Solution:** Use RunPod's GitHub integration - they have base images cached on their build servers, so only your code is uploaded.

### Issue 2: pip Install Conflicts

**Problem:** Build failed with distutils error:
```
Cannot uninstall 'blinker'. It is a distutils installed project...
```

**Solution:** Added `--ignore-installed` flag to pip install:
```dockerfile
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt
```

### Issue 3: Build Timeout (30 minutes)

**Problem:** flash-attn compilation takes 20+ minutes, causing RunPod's 30-minute build limit to be exceeded.

**Solution:** Use pre-built wheel with fallback:
```dockerfile
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu124torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## Final Implementation

### Step 1: Repository Setup

Create or fork the repository at: https://github.com/MushroomFleet/longlive-runpod

### Step 2: Dockerfile (Final Version)

```dockerfile
# RunPod Serverless Dockerfile for LongLive Video Generation
# Lightweight build - models downloaded at runtime

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
```

### Step 3: Handler with Lazy Model Loading

Key additions to `handler.py`:

```python
from huggingface_hub import snapshot_download

# Model paths - use network volume for persistence across cold starts
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/models")
WAN_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "Wan2.1-T2V-1.3B")
LONGLIVE_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "longlive")


def ensure_models_downloaded():
    """
    Download models on first run if not present on the network volume.
    This enables lazy loading - keeping Docker image small while models
    persist across cold starts via RunPod network volumes.
    """
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Download Wan2.1-T2V-1.3B if not present
    wan_marker = os.path.join(WAN_MODEL_PATH, "config.json")
    if not os.path.exists(wan_marker):
        print("[LongLive] Downloading Wan2.1-T2V-1.3B model (~5-10GB)...")
        print("[LongLive] This may take 5-10 minutes on first run...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=WAN_MODEL_PATH,
            local_dir_use_symlinks=False
        )
        print("[LongLive] Wan model downloaded successfully")
    else:
        print("[LongLive] Wan model found in cache")

    # Download LongLive checkpoints if not present
    longlive_marker = os.path.join(LONGLIVE_MODEL_PATH, "models", "longlive_base.pt")
    if not os.path.exists(longlive_marker):
        print("[LongLive] Downloading LongLive checkpoints (~5GB)...")
        print("[LongLive] This may take 3-5 minutes on first run...")
        snapshot_download(
            repo_id="Efficient-Large-Model/LongLive",
            local_dir=LONGLIVE_MODEL_PATH,
            local_dir_use_symlinks=False
        )
        print("[LongLive] LongLive checkpoints downloaded successfully")
    else:
        print("[LongLive] LongLive checkpoints found in cache")

    # Create symlinks to expected paths
    symlinks = [
        ("wan_models/Wan2.1-T2V-1.3B", WAN_MODEL_PATH),
        ("longlive_models", LONGLIVE_MODEL_PATH)
    ]
    for link_name, target in symlinks:
        link_path = os.path.join("/app", link_name)
        if not os.path.exists(link_path):
            parent_dir = os.path.dirname(link_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            try:
                os.symlink(target, link_path)
                print(f"[LongLive] Created symlink: {link_path} -> {target}")
            except OSError as e:
                print(f"[LongLive] Warning: Could not create symlink {link_path}: {e}")
```

Update `get_default_config()` to use dynamic paths:

```python
def get_default_config():
    return {
        # ... other config ...
        "generator_ckpt": os.path.join(LONGLIVE_MODEL_PATH, "models", "longlive_base.pt"),
        "lora_ckpt": os.path.join(LONGLIVE_MODEL_PATH, "models", "lora.pt"),
        # ...
    }
```

Startup sequence at module level:

```python
# Ensure models are downloaded (lazy loading on first run)
print("[LongLive] Checking/downloading models...")
try:
    ensure_models_downloaded()
    print("[LongLive] Models ready")
except Exception as e:
    print(f"[LongLive] Error downloading models: {e}")
    traceback.print_exc()

# Pre-load pipeline on cold start
print("[LongLive] Pre-loading pipeline...")
try:
    load_pipeline()
    print("[LongLive] Pipeline pre-loaded successfully")
except Exception as e:
    print(f"[LongLive] Warning: Failed to pre-load pipeline: {e}")
    traceback.print_exc()

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
```

### Step 4: .dockerignore

```
.git
.gitignore
*.md
docs/
assets/
__pycache__/
*.py[cod]
.env
.venv
venv/
.idea/
.vscode/
wandb/
outputs/
videos/
checkpoints/
*.ckpt
train*.py
train*.sh
trainer/
*.log
logs/
.DS_Store
Thumbs.db
*.tmp
nul
.claude/
```

### Step 5: Push to GitHub

```bash
# Configure git identity
git config user.name "YourUsername"
git config user.email "your@email.com"

# Add files
git add Dockerfile handler.py .dockerignore

# Commit
git commit -m "Add RunPod serverless deployment with lazy model loading"

# Push to your repository
git push origin main
```

### Step 6: RunPod Network Volume Setup

1. Go to RunPod Console → **Storage** → **Network Volumes**
2. Click **"Create Network Volume"**
3. Configure:
   - **Name**: `longlive-models`
   - **Size**: 50GB minimum, 100GB recommended
   - **Region**: Same region as your endpoint (e.g., US-KS-2, EU-RO-1)
4. Click **Create**

**Important:** Note the volume ID and region for the next step.

### Step 7: RunPod Endpoint Deployment

1. Go to RunPod Console → **Serverless** → **New Endpoint**

2. Select **"Import Git Repository"**

3. Connect GitHub:
   - Click "Connect GitHub"
   - Authorize RunPod to access your repositories
   - Select `longlive-runpod` repository

4. Configure Build:
   - **Branch**: `main`
   - **Dockerfile Path**: `Dockerfile`

5. Configure Worker:
   - **GPU Type**: A100-80GB or H100 (40GB+ VRAM required)
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: 1 (increase for production)
   - **Idle Timeout**: 60 seconds

6. Attach Network Volume:
   - Select your `longlive-models` volume
   - **Mount Path**: `/runpod-volume`

7. Click **Deploy**

### Step 8: Monitor Build Progress

1. Go to your endpoint page
2. Click **"Builds"** tab
3. Watch for:
   - **Pending**: Waiting in queue
   - **Building**: Docker build in progress
   - **Complete**: Ready to use
   - **Failed**: Check logs for errors

Build typically takes 10-15 minutes.

### Step 9: First Request (Model Download)

Use async endpoint for first request (sync will timeout):

```bash
# Start job
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cat playing with yarn",
      "num_frames": 30,
      "seed": 42
    }
  }'
```

Response:
```json
{
  "id": "abc123-job-id",
  "status": "IN_QUEUE"
}
```

Check status (repeat until completed):
```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/abc123-job-id" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

First request takes 10-15 minutes:
- Worker startup: ~30 seconds
- Model download: ~10-12 minutes
- Pipeline loading: ~1-2 minutes
- Inference: ~30 seconds (for 30 frames)

### Step 10: Subsequent Requests

After models are cached, use sync endpoint:

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

Response time: ~2 minutes total
- Cold start (if worker scaled down): ~90 seconds
- Inference: ~30-40 seconds for 120 frames

## Monitoring and Debugging

### View Worker Logs

1. Endpoint page → **"Logs"** tab
2. Filter by Worker ID
3. Look for `[LongLive]` prefixed messages:
   ```
   [LongLive] Checking/downloading models...
   [LongLive] Downloading Wan2.1-T2V-1.3B model (~5-10GB)...
   [LongLive] Wan model downloaded successfully
   [LongLive] LongLive checkpoints found in cache
   [LongLive] Models ready
   [LongLive] Pre-loading pipeline...
   [LongLive] Pipeline pre-loaded successfully
   ```

### View Build Logs

1. Endpoint page → **"Builds"** tab
2. Click on a build
3. View full Docker build output

### Common Issues

**Issue: Build stays "Pending"**
- RunPod build queue congestion
- Wait 5-10 minutes, or try during off-peak hours

**Issue: Build fails with pip conflicts**
- Ensure Dockerfile has `--ignore-installed` flag
- Push fix and trigger rebuild

**Issue: Build timeout (30 min limit)**
- Ensure flash-attn uses pre-built wheel
- Push fix and trigger rebuild

**Issue: Worker initializing but never ready**
- Build not complete yet
- Check Builds tab for status

**Issue: Models re-downloading every cold start**
- Network volume not mounted
- Check endpoint settings → Volume mount path is `/runpod-volume`

**Issue: Out of memory**
- Using wrong GPU type (needs 40GB+ VRAM)
- Reduce `num_frames` in request

## API Reference

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description of the video |
| `num_frames` | int | No | 120 | Number of frames (3-1050, must be divisible by 3) |
| `seed` | int | No | random | Random seed for reproducibility |
| `fps` | int | No | 16 | Output video frame rate |

### Output Format

```json
{
  "status": "success",
  "video_base64": "<base64-encoded-mp4>",
  "format": "mp4",
  "fps": 16,
  "num_frames": 120,
  "resolution": "832x480",
  "seed": 42
}
```

### Error Responses

```json
{
  "status": "error",
  "error_type": "validation_error",
  "message": "'prompt' is required"
}
```

```json
{
  "status": "error",
  "error_type": "out_of_memory",
  "message": "GPU ran out of memory. Try reducing num_frames."
}
```

## Cost Summary

| Phase | Duration | Cost (A100-80GB @ $2/hr) |
|-------|----------|--------------------------|
| Build | Free | $0 |
| First request | ~15 min | ~$0.50 |
| Subsequent (cold) | ~2 min | ~$0.07 |
| Subsequent (warm) | ~40 sec | ~$0.02 |
| Network volume (100GB) | Per hour | ~$0.07/hr |

## Comparison: Full Image vs Lightweight

| Aspect | Full Image | Lightweight (This Method) |
|--------|------------|---------------------------|
| Local bandwidth needed | ~20GB | <1MB |
| Build location | Local Docker | RunPod servers |
| Build time | 30-60 min | 10-15 min |
| Image size | ~20GB | ~2GB (+ models on volume) |
| First request time | ~3 min | ~15 min |
| Cold start time | ~90 sec | ~90 sec |
| Model updates | Rebuild image | Delete volume cache |

## Files Summary

```
longlive-runpod/
├── Dockerfile           # Lightweight build config
├── handler.py           # RunPod serverless handler with lazy loading
├── .dockerignore        # Excludes unnecessary files from build
├── requirements.txt     # Python dependencies (from original repo)
├── pipeline/            # LongLive inference pipeline
├── utils/               # Utility functions
└── docs/
    ├── DEPLOY-FULL-IMAGE.md    # Full Docker image method
    └── DEPLOY-LIGHTWEIGHT.md   # This document
```

## Repository

- **Source**: https://github.com/MushroomFleet/longlive-runpod
- **Original LongLive**: https://github.com/NVlabs/LongLive
- **Models**: https://huggingface.co/Efficient-Large-Model/LongLive
