# SPDX-License-Identifier: Apache-2.0
"""
Entry point for LongLive interactive (multi-prompt) inference with
TriAttention KV compression.

Usage:
    python -m longlive.run_interactive \
        --config_path longlive/configs/triattention_interactive.yaml
"""
from __future__ import annotations

import argparse
import os
from typing import List

import torch
import torch.distributed as dist
from einops import rearrange
from torchvision.io import write_video

# Apply patches BEFORE importing anything from LongLive.
from longlive.patches import apply_patches

apply_patches(interactive_mode="compress")

from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
from utils.dataset import MultiTextDataset
from utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
from utils.misc import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongLive interactive inference with TriAttention KV compression"
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

    torch.set_grad_enabled(False)

    # Initialize pipeline (patches are already applied).
    pipeline = InteractiveCausalInferencePipeline(config, device=device)

    # Load generator checkpoint.
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        raw_gen_state_dict = state_dict[
            "generator_ema" if config.use_ema else "generator"
        ]

        if config.use_ema:
            def _clean_key(name: str) -> str:
                return name.replace("_fsdp_wrapped_module.", "")

            cleaned_state_dict = {
                _clean_key(k): v for k, v in raw_gen_state_dict.items()
            }
            missing, unexpected = pipeline.generator.load_state_dict(
                cleaned_state_dict, strict=False
            )
            if local_rank == 0:
                if missing:
                    print(
                        f"[Warning] {len(missing)} parameters missing: "
                        f"{missing[:8]} ..."
                    )
                if unexpected:
                    print(
                        f"[Warning] {len(unexpected)} unexpected params: "
                        f"{unexpected[:8]} ..."
                    )
        else:
            pipeline.generator.load_state_dict(raw_gen_state_dict)

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
                peft.set_peft_model_state_dict(
                    pipeline.generator.model, lora_checkpoint["generator_lora"]
                )
            else:
                peft.set_peft_model_state_dict(
                    pipeline.generator.model, lora_checkpoint
                )
            if is_main:
                print("LoRA weights loaded for generator")

        pipeline.is_lora_enabled = True

    # Move pipeline to appropriate dtype and device.
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # Parse switch_frame_indices from config (comma-separated or int).
    if isinstance(config.switch_frame_indices, int):
        switch_frame_indices: List[int] = [int(config.switch_frame_indices)]
    else:
        switch_frame_indices = [
            int(x)
            for x in str(config.switch_frame_indices).split(",")
            if str(x).strip()
        ]

    # Build dataset/dataloader.
    dataset = MultiTextDataset(config.data_path)
    num_segments = len(dataset[0]["prompts_list"])
    assert len(switch_frame_indices) == num_segments - 1, (
        f"switch_frame_indices length {len(switch_frame_indices)} does not "
        f"match num_segments - 1 ({num_segments - 1})"
    )
    if local_rank == 0:
        print(f"Number of segments: {num_segments}")
        print(f"Switch frame indices: {switch_frame_indices}")

    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False, drop_last=True
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False
    )

    if local_rank == 0:
        os.makedirs(config.output_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # Inference loop.
    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data["idx"].item()
        prompts_list: List[str] = batch_data["prompts_list"]

        sampled_noise = torch.randn(
            [config.num_samples, config.num_output_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
        )

        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts_list=prompts_list,
            switch_frame_indices=switch_frame_indices,
            return_latents=False,
        )

        current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0

        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

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
                short_name = prompts_list[0][0][:100].replace("/", "_")
                output_path = os.path.join(
                    config.output_folder,
                    f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4",
                )
            write_video(output_path, current_video[seed_idx].to(torch.uint8), fps=16)
            if local_rank == 0:
                print(f"Saved video to {output_path}")

        if config.inference_iter != -1 and i >= config.inference_iter:
            break

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
