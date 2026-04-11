# SPDX-License-Identifier: Apache-2.0
"""
Monkey patches for LongLive to enable TriAttention KV compression.

Patches are applied at runtime before any LongLive module is imported.
The three integration points are:
  1. WanDiffusionWrapper.__init__  -- accept **extra_kwargs (kv_* params)
  2. CausalInferencePipeline.__init__ -- add compressor/calibrator setup
  3. CausalInferencePipeline.inference -- add maybe_compress + calibrator save
  4. CausalInferencePipeline._initialize_kv_cache -- compression-aware cache sizing
  5. CausalInferencePipeline._set_all_modules_max_attention_size -- compression-aware sizing
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.distributed as dist


def apply_patches(longlive_root: str | Path | None = None) -> None:
    """Apply TriAttention KV compression patches to LongLive.

    Args:
        longlive_root: Path to LongLive repo root. If None, uses the
                       bundled submodule at ``triattention/longlive/longlive``.
    """
    if longlive_root is None:
        longlive_root = Path(__file__).parent / "longlive"

    longlive_root = Path(longlive_root)

    # Add LongLive to sys.path so its internal imports resolve.
    longlive_str = str(longlive_root)
    if longlive_str not in sys.path:
        sys.path.insert(0, longlive_str)

    # Inject our kv_compression module BEFORE LongLive tries to import it.
    from triattention.longlive import kv_compression

    sys.modules.setdefault("utils.kv_compression", kv_compression)

    # Now patch individual classes.
    _patch_wan_wrapper()
    _patch_causal_inference()


# ---------------------------------------------------------------------------
# WanDiffusionWrapper patch
# ---------------------------------------------------------------------------

def _patch_wan_wrapper() -> None:
    """Add **extra_kwargs passthrough to WanDiffusionWrapper.__init__."""
    from utils.wan_wrapper import WanDiffusionWrapper

    original_init = WanDiffusionWrapper.__init__

    def patched_init(self, *args, **kwargs):
        # Separate kv_* and grid_* kwargs that the original __init__ does not accept.
        extra = {
            k: v
            for k, v in kwargs.items()
            if k.startswith("kv_") or k in ("grid_h", "grid_w")
        }
        filtered = {k: v for k, v in kwargs.items() if k not in extra}
        original_init(self, *args, **filtered)
        self.extra_kwargs = extra

    WanDiffusionWrapper.__init__ = patched_init


# ---------------------------------------------------------------------------
# CausalInferencePipeline patches
# ---------------------------------------------------------------------------

def _get_kw(model_kwargs: Any, key: str, default: Any) -> Any:
    """Helper to read from dict or namespace-like object."""
    if isinstance(model_kwargs, dict):
        return model_kwargs.get(key, default)
    return getattr(model_kwargs, key, default)


def _patch_causal_inference() -> None:
    """Patch CausalInferencePipeline for KV compression support."""
    from pipeline.causal_inference import CausalInferencePipeline
    from triattention.longlive.kv_compression import (
        KVCompressionConfig,
        LongLiveKVCompressor,
        QStatsAccumulator,
    )

    # Attach the static helper to the class.
    CausalInferencePipeline._get_kw = staticmethod(_get_kw)

    # ---- 1. Wrap __init__ to add compressor/calibrator after original init ----
    _original_init = CausalInferencePipeline.__init__

    def _patched_init(self, args, device, generator=None, text_encoder=None, vae=None):
        _original_init(self, args, device, generator=generator,
                       text_encoder=text_encoder, vae=vae)

        model_kwargs = getattr(args, "model_kwargs", {})
        kv_mode = str(_get_kw(model_kwargs, "kv_compression_mode", "off")).lower()
        if kv_mode == "off" and bool(_get_kw(model_kwargs, "use_kv_compression", False)):
            kv_mode = "compress"
        self.kv_compression_mode = kv_mode
        self.kv_stats_path = Path(
            _get_kw(
                model_kwargs,
                "kv_stats_path",
                "longlive_models/kv_stats/normal_q_stats.pt",
            )
        )
        self.kv_compressor: LongLiveKVCompressor | None = None
        self.kv_calibrator: QStatsAccumulator | None = None

        if self.kv_compression_mode == "compress":
            default_budget = (
                int(self.local_attn_size) * self.frame_seq_length
                if int(self.local_attn_size) > 0
                else 21 * self.frame_seq_length
            )
            budget_tokens = int(
                _get_kw(model_kwargs, "kv_budget_tokens", default_budget)
            )
            compress_every_n_frames = int(
                _get_kw(model_kwargs, "kv_compress_every_n_frames", 10)
            )
            keep_last_frames = int(
                _get_kw(model_kwargs, "kv_keep_last_frames", self.num_frame_per_block)
            )
            pruning_mode = str(
                _get_kw(model_kwargs, "kv_pruning_mode", "perhead")
            ).lower()
            score_aggregation = str(
                _get_kw(model_kwargs, "kv_score_aggregation", "mean")
            ).lower()
            perhead_layer_aggregation = str(
                _get_kw(model_kwargs, "kv_perhead_layer_aggregation", "mean_of_layer_max")
            ).lower()
            offset_max_frames = int(
                _get_kw(model_kwargs, "kv_offset_max_frames", 128)
            )
            normalize_scores = bool(
                _get_kw(model_kwargs, "kv_normalize_scores", True)
            )
            tie_break_noise = bool(
                _get_kw(model_kwargs, "kv_tie_break_noise", True)
            )
            tie_break_noise_scale = float(
                _get_kw(model_kwargs, "kv_tie_break_noise_scale", 1e-6)
            )
            random_seed = int(_get_kw(model_kwargs, "kv_random_seed", 0))
            self.kv_compressor = LongLiveKVCompressor(
                KVCompressionConfig(
                    stats_path=self.kv_stats_path,
                    budget_tokens=budget_tokens,
                    compress_every_n_frames=compress_every_n_frames,
                    keep_last_frames=keep_last_frames,
                    frame_seq_length=self.frame_seq_length,
                    mode="compress",
                    pruning_mode=pruning_mode,
                    score_aggregation=score_aggregation,
                    perhead_layer_aggregation=perhead_layer_aggregation,
                    offset_max_frames=offset_max_frames,
                    normalize_scores=normalize_scores,
                    tie_break_noise=tie_break_noise,
                    tie_break_noise_scale=tie_break_noise_scale,
                    random_seed=random_seed,
                    grid_h=30,
                    grid_w=52,
                )
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    f"[KV-Compress] mode=compress budget_tokens={budget_tokens} "
                    f"every_n_frames={compress_every_n_frames} keep_last_frames={keep_last_frames} "
                    f"pruning_mode={pruning_mode} score_agg={score_aggregation} "
                    f"layer_agg={perhead_layer_aggregation} offset_max_frames={offset_max_frames} "
                    f"normalize_scores={normalize_scores} tie_break_noise={tie_break_noise} "
                    f"stats={self.kv_stats_path}"
                )
        elif self.kv_compression_mode == "calibrate":
            first_attn = self.generator.model.blocks[0].self_attn
            self.kv_calibrator = QStatsAccumulator(
                num_layers=self.num_transformer_blocks,
                num_heads=first_attn.num_heads,
                head_dim=first_attn.head_dim,
            )
            self.kv_calibrator.attach(self.generator.model)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[KV-Calib] mode=calibrate stats_out={self.kv_stats_path}")

        # When compression is active, local attention window must cover the entire
        # video to prevent native sliding-window eviction from conflicting with
        # the compressor's position tracking.
        if self.kv_compression_mode in ("compress", "calibrate"):
            local_attn = getattr(args.model_kwargs, "local_attn_size", -1)
            if isinstance(local_attn, int) and local_attn != -1:
                num_frames = getattr(args, "num_output_frames", 0)
                if num_frames > 0 and local_attn < num_frames:
                    raise ValueError(
                        f"KV compression requires local_attn_size >= num_output_frames "
                        f"to prevent sliding-window eviction, but got "
                        f"local_attn_size={local_attn} < num_output_frames={num_frames}. "
                        f"Set local_attn_size to -1 or >= {num_frames}."
                    )

    CausalInferencePipeline.__init__ = _patched_init

    # ---- 2. Replace inference method with compression-aware version ----
    def _patched_inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """Inference with TriAttention KV compression support."""
        from utils.memory import (
            gpu,
            get_cuda_free_memory_gb,
            move_model_to_device_with_memory_preservation,
        )

        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device("cpu") if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype,
        )

        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # KV cache sizing with compression awareness
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        elif self.kv_compression_mode == "compress" and self.kv_compressor is not None:
            headroom = (
                self.kv_compressor.config.compress_every_n_frames
                * self.frame_seq_length
            )
            kv_cache_size = self.kv_compressor.config.budget_tokens + headroom
            kv_policy = (
                f"compress (budget={self.kv_compressor.config.budget_tokens}"
                f"+headroom={headroom})"
            )
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(
            f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, "
            f"frame_seq_length: {self.frame_seq_length}, "
            f"num_output_frames: {num_output_frames})"
        )

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(
            f"[inference] local_attn_size set on model: "
            f"{self.generator.model.local_attn_size}"
        )
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]

            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred.to(output.device)

            # Rerun with timestep zero to update KV cache using clean context.
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            # KV compression trigger point
            if self.kv_compressor is not None:
                current_end_frame = current_start_frame + current_num_frames
                self.kv_compressor.maybe_compress(
                    self.kv_cache1, current_end_frame=current_end_frame
                )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Decode the output
        if getattr(self.args.model_kwargs, "use_infinite_attention", False):
            video = self.vae.decode_to_pixel_chunk(
                output.to(noise.device), use_cache=False
            )
        else:
            video = self.vae.decode_to_pixel(
                output.to(noise.device), use_cache=False
            )
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            print("Profiling results:")
            print(
                f"  - Initialization/caching time: {init_time:.2f} ms "
                f"({100 * init_time / total_time:.2f}%)"
            )
            print(
                f"  - Diffusion generation time: {diffusion_time:.2f} ms "
                f"({100 * diffusion_time / total_time:.2f}%)"
            )
            for i, bt in enumerate(block_times):
                print(
                    f"    - Block {i} generation time: {bt:.2f} ms "
                    f"({100 * bt / diffusion_time:.2f}% of diffusion)"
                )
            print(
                f"  - VAE decoding time: {vae_time:.2f} ms "
                f"({100 * vae_time / total_time:.2f}%)"
            )
            print(f"  - Total time: {total_time:.2f} ms")

        # Save calibration stats if calibrating
        if self.kv_calibrator is not None:
            metadata = {
                "model_name": _get_kw(
                    getattr(self.args, "model_kwargs", {}),
                    "model_name",
                    "Wan2.1-T2V-1.3B",
                ),
                "num_frames": int(num_output_frames),
                "num_blocks": int(self.num_transformer_blocks),
                "num_frame_per_block": int(self.num_frame_per_block),
                "denoising_step_list": [
                    int(x) for x in self.denoising_step_list.tolist()
                ],
                "source": "LongLive-normal-causal",
            }
            self.kv_calibrator.save(self.kv_stats_path, metadata)

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video

    CausalInferencePipeline.inference = _patched_inference

    # ---- 3. Replace _initialize_kv_cache ----
    def _patched_initialize_kv_cache(
        self, batch_size, dtype, device, kv_cache_size_override: int | None = None
    ):
        """Initialize a Per-GPU KV cache for the Wan model."""
        kv_cache1 = []
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            entry = {
                "k": torch.zeros(
                    [batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device
                ),
                "v": torch.zeros(
                    [batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            kv_cache1.append(entry)

        self.kv_cache1 = kv_cache1

    CausalInferencePipeline._initialize_kv_cache = _patched_initialize_kv_cache

    # ---- 4. Replace _set_all_modules_max_attention_size ----
    def _patched_set_all_modules_max_attention_size(
        self, local_attn_size_value: int
    ):
        """Set max_attention_size on all submodules that define it."""
        if local_attn_size_value == -1:
            if (
                self.kv_compression_mode == "compress"
                and self.kv_compressor is not None
            ):
                headroom = (
                    self.kv_compressor.config.compress_every_n_frames
                    * self.frame_seq_length
                )
                target_size = self.kv_compressor.config.budget_tokens + headroom
            else:
                target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(
                        name if name else module.__class__.__name__
                    )
                except Exception:
                    pass

    CausalInferencePipeline._set_all_modules_max_attention_size = (
        _patched_set_all_modules_max_attention_size
    )
