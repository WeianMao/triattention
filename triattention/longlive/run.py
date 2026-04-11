# SPDX-License-Identifier: Apache-2.0
"""
Entry point for LongLive inference with TriAttention KV compression.

Usage:
    python -m triattention.longlive.run --config_path triattention/longlive/configs/triattention_120f.yaml
    python -m triattention.longlive --config_path triattention/longlive/configs/triattention_120f.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from einops import rearrange
from torchvision.io import write_video

# Apply patches BEFORE importing anything from LongLive.
from triattention.longlive.patches import apply_patches

apply_patches()

from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.memory import get_cuda_free_memory_gb, gpu
from utils.misc import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongLive inference with TriAttention KV compression"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # Initialize distributed inference
    if "LOCAL_RANK" in os.environ:
        os.environ["NCCL_CROSS_NIC"] = "1"
        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
        os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.constants.default_pg_timeout,
            )
        set_seed(config.seed + local_rank)
        config.distributed = True
        if rank == 0:
            print(
                f"[Rank {rank}] Initialized distributed processing on device {device}"
            )
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda")
        set_seed(config.seed)
        config.distributed = False
        print(f"Single GPU mode on device {device}")

    print(f"Free VRAM {get_cuda_free_memory_gb(device)} GB")
    low_memory = get_cuda_free_memory_gb(device) < 40
    low_memory = True

    torch.set_grad_enabled(False)

    # Initialize pipeline (patches are already applied).
    pipeline = CausalInferencePipeline(config, device=device)

    # Load generator checkpoint.
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict[
                "generator_ema" if config.use_ema else "generator"
            ]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(
                f"Generator state dict not found in {config.generator_ckpt}"
            )

        # Strip "model." prefix if present.
        gen_state_dict = {}
        for k, v in raw_gen_state_dict.items():
            new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
            gen_state_dict[new_key] = v

        pipeline.generator.model.load_state_dict(gen_state_dict, strict=False)
        print(f"Loaded generator checkpoint from {config.generator_ckpt}")

    # Load LoRA if configured.
    pipeline.is_lora_enabled = False
    if getattr(config, "adapter", None):
        import peft
        from utils.lora_utils import configure_lora_for_model

        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_main:
            print(f"LoRA enabled with config: {config.adapter}")
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=is_main,
        )

        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            if is_main:
                print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
            if is_main:
                print("LoRA weights loaded for generator")

        pipeline.is_lora_enabled = True

    # Move pipeline to appropriate dtype and device.
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline = pipeline.to(device)

    # Set up dataset and dataloader.
    dataset = TextDataset(prompt_path=config.data_path)
    num_prompts = len(dataset)
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler
    )

    os.makedirs(config.output_folder, exist_ok=True)

    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data["idx"].item()

        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]

        prompt = batch["prompts"][0]
        extended_prompt = (
            batch["extended_prompts"][0] if "extended_prompts" in batch else None
        )
        if extended_prompt is not None:
            prompts = [extended_prompt] * config.num_samples
        else:
            prompts = [prompt] * config.num_samples

        sampled_noise = torch.randn(
            [config.num_samples, config.num_output_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
        )

        print("sampled_noise.device", sampled_noise.device)
        print("prompts", prompts)

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
            profile=False,
        )
        current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
        video_out = 255.0 * current_video

        # Clear VAE cache.
        pipeline.vae.model.clear_cache()

        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if idx < num_prompts:
            if hasattr(pipeline, "is_lora_enabled") and pipeline.is_lora_enabled:
                model_type = "lora"
            elif getattr(config, "use_ema", False):
                model_type = "ema"
            else:
                model_type = "regular"

            for seed_idx in range(config.num_samples):
                if config.save_with_index:
                    output_path = os.path.join(
                        config.output_folder,
                        f"rank{rank}-{idx}-{seed_idx}_{model_type}.mp4",
                    )
                else:
                    output_path = os.path.join(
                        config.output_folder,
                        f"rank{rank}-{prompt[:100]}-{seed_idx}.mp4",
                    )
                write_video(output_path, video_out[seed_idx], fps=16)

        if config.inference_iter != -1 and i >= config.inference_iter:
            break

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
