"""
RunPod Serverless Handler for LongLive Video Generation

This handler provides a serverless endpoint for NVIDIA's LongLive
real-time video generation framework.

Input schema:
{
    "prompt": str,           # Required: Text prompt for video generation
    "num_frames": int,       # Optional: Number of frames (default: 120, ~7.5s at 16fps)
    "seed": int,             # Optional: Random seed (default: random)
    "fps": int,              # Optional: Output FPS (default: 16)
}

Output schema:
{
    "status": "success",
    "video_base64": str,     # Base64-encoded MP4 video
    "num_frames": int,
    "resolution": "832x480",
    "seed": int,
}
"""
import os
import sys
import base64
import tempfile
import traceback
import uuid

import torch
import runpod
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange
from huggingface_hub import snapshot_download

# Add project to path
sys.path.insert(0, '/app')

from pipeline.causal_inference import CausalInferencePipeline
from utils.lora_utils import configure_lora_for_model
from utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
import peft


# Global pipeline instance (loaded once, reused across requests)
PIPELINE = None
CONFIG = None

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

    # Create symlinks to expected paths (so existing code paths work unchanged)
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
                # Fallback: copy might be needed on some systems
                pass


def get_default_config():
    """Returns default inference configuration matching longlive_inference.yaml."""
    return {
        "denoising_step_list": [1000, 750, 500, 250],
        "warp_denoising_step": True,
        "num_frame_per_block": 3,
        "model_name": "Wan2.1-T2V-1.3B",
        "model_kwargs": {
            "local_attn_size": 12,
            "timestep_shift": 5.0,
            "sink_size": 3,
        },
        "num_output_frames": 120,
        "use_ema": False,
        "seed": 0,
        "num_samples": 1,
        "global_sink": True,
        "context_noise": 0,
        "generator_ckpt": os.path.join(LONGLIVE_MODEL_PATH, "models", "longlive_base.pt"),
        "lora_ckpt": os.path.join(LONGLIVE_MODEL_PATH, "models", "lora.pt"),
        "adapter": {
            "type": "lora",
            "rank": 256,
            "alpha": 256,
            "dropout": 0.0,
            "dtype": "bfloat16",
            "verbose": False,
        },
    }


def load_pipeline():
    """
    Load and cache the inference pipeline.
    Uses global caching to avoid reloading on each request.
    """
    global PIPELINE, CONFIG

    if PIPELINE is not None:
        return PIPELINE, CONFIG

    print("[LongLive] Loading pipeline...")

    # Create config from defaults
    config_dict = get_default_config()
    config = OmegaConf.create(config_dict)
    config.distributed = False

    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    # Check available VRAM
    free_memory = get_cuda_free_memory_gb(device)
    print(f"[LongLive] Available VRAM: {free_memory:.2f} GB")
    low_memory = free_memory < 40

    # Initialize pipeline
    pipeline = CausalInferencePipeline(config, device=device)

    # Load generator checkpoint
    if config.generator_ckpt and os.path.exists(config.generator_ckpt):
        print(f"[LongLive] Loading generator checkpoint from {config.generator_ckpt}")
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")

        pipeline.generator.load_state_dict(raw_gen_state_dict)
        print("[LongLive] Generator checkpoint loaded")

    # Apply LoRA
    pipeline.is_lora_enabled = False
    if getattr(config, "adapter", None):
        print("[LongLive] Applying LoRA...")
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=True,
        )

        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path and os.path.exists(lora_ckpt_path):
            print(f"[LongLive] Loading LoRA weights from {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
            print("[LongLive] LoRA weights loaded")

        pipeline.is_lora_enabled = True

    # Move to device with appropriate precision
    pipeline = pipeline.to(dtype=torch.bfloat16)

    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)

    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # Cache the pipeline
    PIPELINE = pipeline
    CONFIG = config

    print("[LongLive] Pipeline loaded successfully")
    return pipeline, config


def validate_input(job_input):
    """
    Validate and normalize input parameters.
    """
    # Check for prompt
    prompt = job_input.get("prompt")
    if not prompt:
        raise ValueError("'prompt' is required")

    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        raise ValueError("'prompt' must be a non-empty string")

    # Validate num_frames
    num_frames = job_input.get("num_frames", 120)
    if not isinstance(num_frames, int) or num_frames < 3 or num_frames > 1050:
        raise ValueError("num_frames must be an integer between 3 and 1050")

    # Ensure num_frames is divisible by num_frame_per_block (3)
    if num_frames % 3 != 0:
        num_frames = (num_frames // 3) * 3
        print(f"[LongLive] Adjusted num_frames to {num_frames} (must be divisible by 3)")

    # Validate seed
    seed = job_input.get("seed")
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    elif not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    # Validate FPS
    fps = job_input.get("fps", 16)
    if not isinstance(fps, int) or fps < 1 or fps > 60:
        raise ValueError("fps must be an integer between 1 and 60")

    return {
        "prompt": prompt.strip(),
        "num_frames": num_frames,
        "seed": seed,
        "fps": fps,
    }


def run_inference(validated_input):
    """
    Run video generation inference.
    Returns video tensor of shape (B, T, H, W, C) with values in [0, 255].
    """
    prompt = validated_input["prompt"]
    num_frames = validated_input["num_frames"]
    seed = validated_input["seed"]

    # Load pipeline
    pipeline, config = load_pipeline()

    # Set seed for reproducibility
    torch.manual_seed(seed)

    device = torch.device("cuda")

    # Check available memory for low_memory mode
    low_memory = get_cuda_free_memory_gb(device) < 40

    # Create noise tensor
    # Shape: [batch_size, num_frames, channels, height, width]
    # Fixed resolution: 60x104 (latent space for 480x832 video)
    sampled_noise = torch.randn(
        [1, num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    # Run inference
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=[prompt],
        return_latents=True,
        low_memory=low_memory,
    )

    # Convert from (B, T, C, H, W) to (B, T, H, W, C)
    video = rearrange(video, 'b t c h w -> b t h w c')

    # Scale to [0, 255] range
    video = (video * 255.0).clamp(0, 255).cpu()

    # Clear VAE cache to free memory
    pipeline.vae.model.clear_cache()
    torch.cuda.empty_cache()

    return video


def encode_video_to_base64(video_tensor, fps):
    """
    Encode video tensor to base64-encoded MP4.
    video_tensor: Shape (B, T, H, W, C), values in [0, 255]
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Take first batch item
        video = video_tensor[0].to(torch.uint8)
        write_video(tmp_path, video, fps=fps)

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()

        return base64.b64encode(video_bytes).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def handler(job):
    """
    RunPod serverless handler function.
    """
    job_input = job.get("input", {})

    try:
        # Validate input
        validated_input = validate_input(job_input)

        print(f"[LongLive] Generating video with {validated_input['num_frames']} frames")
        print(f"[LongLive] Prompt: {validated_input['prompt'][:100]}...")
        print(f"[LongLive] Seed: {validated_input['seed']}")

        # Run inference
        video_tensor = run_inference(validated_input)

        # Encode to base64
        fps = validated_input["fps"]
        video_data = encode_video_to_base64(video_tensor, fps)

        return {
            "status": "success",
            "video_base64": video_data,
            "format": "mp4",
            "fps": fps,
            "num_frames": validated_input["num_frames"],
            "resolution": "832x480",
            "seed": validated_input["seed"],
        }

    except ValueError as e:
        return {
            "status": "error",
            "error_type": "validation_error",
            "message": str(e),
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "status": "error",
            "error_type": "out_of_memory",
            "message": "GPU ran out of memory. Try reducing num_frames.",
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "error_type": "internal_error",
            "message": str(e),
        }


# Ensure models are downloaded (lazy loading on first run)
print("[LongLive] Checking/downloading models...")
try:
    ensure_models_downloaded()
    print("[LongLive] Models ready")
except Exception as e:
    print(f"[LongLive] Error downloading models: {e}")
    traceback.print_exc()
    # Continue anyway - will fail gracefully during inference if models missing

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
