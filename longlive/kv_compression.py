# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist


def _is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _to_complex_pairs(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Head dim must be even, got {x.shape[-1]}")
    return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())


def _build_geometric_offsets(max_len: int, device: torch.device) -> torch.Tensor:
    if max_len < 1:
        return torch.tensor([1.0], device=device, dtype=torch.float32)
    offsets: List[float] = []
    value = 1
    while value <= max_len:
        offsets.append(float(value))
        value *= 2
    if len(offsets) == 0:
        offsets.append(1.0)
    return torch.tensor(offsets, device=device, dtype=torch.float32)


class QStatsAccumulator:
    """Collect pre-RoPE Q statistics from CausalWanSelfAttention modules."""

    def __init__(self, num_layers: int, num_heads: int, head_dim: int) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.freq_dim = head_dim // 2
        self.q_sum = torch.zeros(
            num_layers, num_heads, self.freq_dim, dtype=torch.complex64
        )
        self.q_abs_sum = torch.zeros(
            num_layers, num_heads, self.freq_dim, dtype=torch.float32
        )
        self.token_count = torch.zeros(num_layers, dtype=torch.float64)
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model) -> None:
        if not hasattr(model, "blocks"):
            raise ValueError("Model has no blocks; cannot attach Q stats collector.")
        for layer_idx, block in enumerate(model.blocks):
            attn = block.self_attn
            handle = attn.register_forward_pre_hook(
                self._make_hook(layer_idx), with_kwargs=True
            )
            self._handles.append(handle)
        if _is_main_process():
            print(
                f"[KV-Calib] Attached pre-hooks to {len(self._handles)} attention layers."
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            x = kwargs.get("x")
            if x is None and args:
                x = args[0]
            if x is None:
                return
            if x.dim() != 3:
                return
            with torch.no_grad():
                bsz, seq_len, _ = x.shape
                q = module.norm_q(module.q(x)).view(
                    bsz, seq_len, module.num_heads, module.head_dim
                )
                q_complex = _to_complex_pairs(q.to(dtype=torch.float32))
                q_sum = q_complex.sum(dim=(0, 1)).to(device="cpu", dtype=torch.complex64)
                q_abs_sum = q_complex.abs().sum(dim=(0, 1)).to(
                    device="cpu", dtype=torch.float32
                )
                self.q_sum[layer_idx] += q_sum
                self.q_abs_sum[layer_idx] += q_abs_sum
                self.token_count[layer_idx] += float(bsz * seq_len)

        return hook

    def save(self, output_path: Path, metadata: Dict[str, object]) -> None:
        if (self.token_count <= 0).all():
            raise RuntimeError("No Q statistics collected; nothing to save.")
        denom = self.token_count.clamp_min(1.0).view(-1, 1, 1).to(torch.float32)
        q_mean = self.q_sum / denom
        q_abs_mean = self.q_abs_sum / denom
        payload = {
            "metadata": metadata,
            "q_mean_real": q_mean.real.to(torch.float32),
            "q_mean_imag": q_mean.imag.to(torch.float32),
            "q_abs_mean": q_abs_mean.to(torch.float32),
            "token_count": self.token_count.to(torch.float64),
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        if _is_main_process():
            total_tokens = int(self.token_count.sum().item())
            print(f"[KV-Calib] Saved stats to {output_path} (tokens={total_tokens})")


@dataclass
class KVCompressionConfig:
    stats_path: Path
    budget_tokens: int
    compress_every_n_frames: int
    keep_last_frames: int
    frame_seq_length: int
    mode: str = "compress"
    pruning_mode: str = "perhead"
    score_aggregation: str = "mean"
    perhead_layer_aggregation: str = "mean_of_layer_max"
    offset_max_frames: int = 128
    normalize_scores: bool = True
    tie_break_noise: bool = True
    tie_break_noise_scale: float = 1e-6
    random_seed: int = 0
    grid_h: int = 0
    grid_w: int = 0
    sink_size: int = 0  # Number of sink frames (matches LongLive's sink_size); first sink_size*frame_seq_length slots are pinned to absolute positions 0..sink_tokens-1
    protect_sink: bool = True  # When True, compressor never evicts sink slots


class LongLiveKVCompressor:
    """
    KV-level compressor in pre-RoPE space.

    Supported pruning modes:
    - perhead: each KV head chooses keep indices independently, synchronized across layers.
    - layer_perhead: each (layer, KV head) chooses keep indices independently.
    """

    def __init__(self, config: KVCompressionConfig) -> None:
        self.config = config
        self.last_compressed_frame = 0
        self._loaded = False

        self.q_mean: torch.Tensor | None = None  # [L, H, F]
        self.q_abs_mean: torch.Tensor | None = None  # [L, H, F]

        self.omega: torch.Tensor | None = None  # [F]
        self.freq_scale_sq: torch.Tensor | None = None  # [F]
        self.temporal_mask: torch.Tensor | None = None  # [F], 1 for temporal freqs else 0
        self.offsets: torch.Tensor | None = None  # [O], frame offsets

        # Per-head absolute token positions after each compression.
        # Shape [H, S], where S matches current cache sequence length.
        self.cache_positions_per_head: torch.Tensor | None = None
        # Per-(layer, head) absolute token positions after each compression.
        # Shape [L, H, S], where S matches current cache sequence length.
        self.cache_positions_per_layer_head: torch.Tensor | None = None

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(self.config.random_seed))

    def _init_rotary_metadata(self, device: torch.device, freq_dim: int) -> None:
        # Wan uses 3-axis RoPE split over complex frequency bins:
        # temporal + height + width, with temporal first.
        temporal_count = freq_dim - 2 * (freq_dim // 3)
        spatial_count = freq_dim // 3
        if temporal_count <= 0 or spatial_count < 0:
            raise ValueError(
                f"Invalid frequency split: freq_dim={freq_dim}, temporal={temporal_count}, spatial={spatial_count}"
            )

        # Wan's causal_model.py builds freqs by calling rope_params *independently*
        # per axis, each with its own dimension:
        #   rope_params(1024, d - 4*(d//6))  -> temporal (d_t = 44 for d=128)
        #   rope_params(1024, 2*(d//6))      -> height   (d_s = 42 for d=128)
        #   rope_params(1024, 2*(d//6))      -> width    (d_s = 42 for d=128)
        # Each rope_params(max_len, dim) produces inv_freq = 1/10000^(arange(0,dim,2)/dim),
        # giving dim//2 frequency bins with denominator = axis-local dim.
        # This means omega_t[j] = 1/10000^(j/temporal_count) and
        # omega_s[j] = 1/10000^(j/spatial_count), NOT a single global 1/10000^(2i/head_dim).
        def _axis_omega(count: int) -> torch.Tensor:
            if count <= 0:
                return torch.empty(0, device=device, dtype=torch.float32)
            idx = torch.arange(count, device=device, dtype=torch.float32)
            return 1.0 / torch.pow(10000.0, idx / float(count))

        omega_t = _axis_omega(temporal_count)
        omega_h = _axis_omega(spatial_count)
        omega_w = _axis_omega(spatial_count)
        omega = torch.cat([omega_t, omega_h, omega_w], dim=0)
        if omega.numel() != freq_dim:
            raise ValueError(
                f"Omega dim mismatch: expected={freq_dim}, got={omega.numel()} "
                f"(temporal={temporal_count}, spatial={spatial_count})"
            )

        temporal_mask = torch.zeros(freq_dim, device=device, dtype=torch.float32)
        temporal_mask[:temporal_count] = 1.0

        self.omega = omega
        # Wan 2.1 uses pure unit-modulus RoPE (torch.polar(ones, freqs)),
        # so the per-frequency squared magnitude is identically 1.0.
        # WARNING: If RoPE scaling (YaRN, NTK, etc.) is ever added to the model,
        # this must be updated to read the actual per-frequency scaling factors.
        self.freq_scale_sq = torch.ones(freq_dim, device=device, dtype=torch.float32)
        self.temporal_mask = temporal_mask
        self.offsets = _build_geometric_offsets(
            self.config.offset_max_frames, device=device
        )

        # Pre-split omega counts for _invert_rope_wan.
        self._temporal_count = temporal_count
        self._spatial_count = spatial_count

    def _pos_to_grid(
        self, flat_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose absolute token positions into (frame, h, w) coordinates.

        Uses the same row-major layout as Wan's ``flatten(2)`` on ``[C, F, H, W]``.
        """
        H, W = self.config.grid_h, self.config.grid_w
        hw = H * W
        frame = torch.div(flat_pos, hw, rounding_mode="floor")
        rem = flat_pos % hw
        h = torch.div(rem, W, rounding_mode="floor")
        w = rem % W
        return frame.long(), h.long(), w.long()

    def _invert_rope_wan(
        self, k_post_complex: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        """Recover pre-RoPE complex key from post-RoPE complex key.

        Wan's RoPE is a complex multiplication by unit-modulus rotors:
            ``k_post[f] = k_pre[f] * exp(i * theta[f])``
        so the inverse is:
            ``k_pre[f] = k_post[f] * exp(-i * theta[f])``

        Args:
            k_post_complex: ``[T, F]`` complex64, post-RoPE key in complex-pair form.
            pos: ``[T]`` long, absolute token positions.
        Returns:
            ``[T, F]`` complex64, pre-RoPE key.
        """
        frame, h, w = self._pos_to_grid(pos)

        # omega is the angular frequency vector: omega[f] = 1/theta^(idx/count).
        # The rotation angle for each token is omega[f] * coordinate.
        omega = self.omega  # [F]
        tc = self._temporal_count
        sc = self._spatial_count

        # Build per-token angle vector [T, F], matching Wan's freq construction:
        #   bins 0..tc-1        : omega_t[i] * frame
        #   bins tc..tc+sc-1    : omega_h[j] * h
        #   bins tc+sc..tc+2*sc-1: omega_w[k] * w
        angle_t = frame.unsqueeze(1).float() * omega[:tc].unsqueeze(0)  # [T, tc]
        angle_h = h.unsqueeze(1).float() * omega[tc : tc + sc].unsqueeze(0)  # [T, sc]
        angle_w = w.unsqueeze(1).float() * omega[tc + sc :].unsqueeze(0)  # [T, sc]
        angle = torch.cat([angle_t, angle_h, angle_w], dim=1)  # [T, F]

        # Conjugate rotation: multiply by exp(-i * angle)
        inv_freqs = torch.polar(
            torch.ones_like(angle), -angle
        )  # [T, F] complex
        return k_post_complex * inv_freqs

    def _load_stats(self, device: torch.device) -> None:
        if self._loaded:
            return
        if self.config.grid_h <= 0 or self.config.grid_w <= 0:
            raise ValueError(
                f"grid_h and grid_w must be positive for RoPE inversion, "
                f"got grid_h={self.config.grid_h}, grid_w={self.config.grid_w}"
            )
        payload = torch.load(self.config.stats_path, map_location="cpu")
        q_mean = torch.complex(payload["q_mean_real"], payload["q_mean_imag"]).to(
            device=device, dtype=torch.complex64
        )
        q_abs = payload["q_abs_mean"].to(device=device, dtype=torch.float32)
        self.q_mean = q_mean
        self.q_abs_mean = q_abs
        self._init_rotary_metadata(device=device, freq_dim=int(q_mean.shape[-1]))
        self._loaded = True
        if _is_main_process():
            print(
                f"[KV-Compress] Loaded stats from {self.config.stats_path} "
                f"(layers={q_mean.shape[0]}, heads={q_mean.shape[1]}, freq={q_mean.shape[2]})"
            )

    def _sync_position_state(
        self, kv_cache: List[dict], num_layers: int, num_heads: int, device: torch.device
    ) -> None:
        # Tail-anchored position derivation: read current cache state from kv_cache[0]
        # (local_end_index, global_end_index) and rebuild absolute positions from scratch.
        # Sink slots [0, sink_tokens) map to absolute positions [0, sink_tokens); the
        # remaining slots [sink_tokens, seq_len) are tail-anchored to global_end such that
        # the last slot corresponds to absolute position global_end-1. This makes the
        # method idempotent and robust to sliding-window eviction, since no incremental
        # state is maintained between calls.
        seq_len = int(kv_cache[0]["local_end_index"].item())
        global_end = int(kv_cache[0]["global_end_index"].item())
        sink_tokens = int(self.config.sink_size) * int(self.config.frame_seq_length)

        positions = torch.empty(seq_len, device=device, dtype=torch.long)
        sink_in_cache = min(sink_tokens, seq_len)
        if sink_in_cache > 0:
            positions[:sink_in_cache] = torch.arange(
                0, sink_in_cache, device=device, dtype=torch.long
            )
        dynamic_in_cache = seq_len - sink_in_cache
        if dynamic_in_cache > 0:
            tail_start = global_end - dynamic_in_cache
            positions[sink_in_cache:] = torch.arange(
                tail_start, global_end, device=device, dtype=torch.long
            )

        self.cache_positions_per_head = (
            positions.unsqueeze(0).expand(num_heads, -1).clone()
        )
        self.cache_positions_per_layer_head = (
            positions.view(1, 1, -1).expand(num_layers, num_heads, -1).clone()
        )

    def _compute_layer_head_scores(
        self,
        kv_cache: List[dict],
        dynamic_len: int,
        current_end_frame: int,
        dynamic_offset: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            dynamic_offset: number of leading cache slots to skip (sink region).
                Scoring reads positions/keys starting at this offset for
                ``dynamic_len`` slots.
        Returns:
            scores: [L, H, T], layer/head token scores before per-head layer aggregation.
        """
        if (
            self.q_mean is None
            or self.q_abs_mean is None
            or self.omega is None
            or self.freq_scale_sq is None
            or self.temporal_mask is None
            or self.offsets is None
        ):
            raise RuntimeError("Compressor state is not initialized.")

        if self.config.pruning_mode == "layer_perhead":
            if self.cache_positions_per_layer_head is None:
                raise RuntimeError("Layer-perhead position state is not initialized.")
            pos_head_dim = int(self.cache_positions_per_layer_head.shape[1])
            pos_layer_dim = int(self.cache_positions_per_layer_head.shape[0])
        else:
            if self.cache_positions_per_head is None:
                raise RuntimeError("Perhead position state is not initialized.")
            pos_head_dim = int(self.cache_positions_per_head.shape[0])
            pos_layer_dim = len(kv_cache)

        layers = min(len(kv_cache), int(self.q_mean.shape[0]))
        kv_heads = min(
            int(kv_cache[0]["k"].shape[2]),
            int(self.q_mean.shape[1]),
            pos_head_dim,
        )
        layers = min(layers, pos_layer_dim)

        # Current implementation assumes MHA (num_q_heads == num_kv_heads).
        # For GQA models, calibration must store per-Q-head stats and
        # _aggregate_perhead must do max-over-Q-heads-per-group first.
        if self.q_mean is not None:
            q_head_count = int(self.q_mean.shape[1])
            kv_head_count = int(kv_cache[0]["k"].shape[2])
            if q_head_count != kv_head_count:
                raise RuntimeError(
                    f"GQA detected: q_mean has {q_head_count} heads but KV cache has "
                    f"{kv_head_count} heads. GQA is not yet supported by this compressor. "
                    f"Calibration and per-head aggregation need GQA-aware logic."
                )

        freq_dim = min(
            int(kv_cache[0]["k"].shape[3] // 2),
            int(self.q_mean.shape[2]),
            int(self.omega.shape[0]),
        )
        if layers <= 0 or kv_heads <= 0 or dynamic_len <= 0 or freq_dim <= 0:
            return torch.empty(0, device=kv_cache[0]["k"].device, dtype=torch.float32)

        device = kv_cache[0]["k"].device
        offsets = self.offsets
        omega = self.omega[:freq_dim]
        freq_scale_sq = self.freq_scale_sq[:freq_dim]
        temporal_mask = self.temporal_mask[:freq_dim]

        scores = torch.empty(
            layers, kv_heads, dynamic_len, device=device, dtype=torch.float32
        )

        dyn_start = int(dynamic_offset)
        dyn_end = dyn_start + dynamic_len
        for layer_idx in range(layers):
            for head_idx in range(kv_heads):
                if self.config.pruning_mode == "layer_perhead":
                    pos = self.cache_positions_per_layer_head[
                        layer_idx, head_idx, dyn_start:dyn_end
                    ].to(device=device, dtype=torch.long)
                else:
                    pos = self.cache_positions_per_head[head_idx, dyn_start:dyn_end].to(
                        device=device, dtype=torch.long
                    )
                key_frame = torch.div(
                    pos, self.config.frame_seq_length, rounding_mode="floor"
                ).to(torch.float32)
                # delta_t shape [T, O]
                delta_t = (
                    float(current_end_frame) - key_frame
                ).unsqueeze(1) + offsets.unsqueeze(0)

                # Read post-RoPE key and analytically invert RoPE to get pre-RoPE complex key.
                k_post = kv_cache[layer_idx]["k"][0, dyn_start:dyn_end, head_idx].to(
                    torch.float32
                )  # [T, D]
                k_post_complex = _to_complex_pairs(k_post)[:, :freq_dim]  # [T, F]
                k_complex = self._invert_rope_wan(k_post_complex, pos)  # [T, F]
                k_abs = torch.abs(k_complex)

                q_mean = self.q_mean[layer_idx, head_idx, :freq_dim]
                q_abs = self.q_abs_mean[layer_idx, head_idx, :freq_dim]
                q_mean_abs = torch.abs(q_mean)

                relative = q_mean.unsqueeze(0) * torch.conj(k_complex)
                phi = torch.atan2(relative.imag, relative.real)  # [T, F]
                amp = q_mean_abs.unsqueeze(0) * k_abs  # [T, F]
                extra = (q_abs - q_mean_abs).unsqueeze(0) * k_abs  # [T, F]

                # Keep all 3 axis frequency bands, but enforce Δh=Δw=0 by temporal_mask.
                phase = (
                    delta_t.unsqueeze(-1)
                    * temporal_mask.view(1, 1, -1)
                    * omega.view(1, 1, -1)
                    + phi.unsqueeze(1)
                )  # [T, O, F]
                base_scores = (
                    amp.unsqueeze(1)
                    * freq_scale_sq.view(1, 1, -1)
                    * torch.cos(phase)
                ).sum(dim=2)  # [T, O]
                additive = (
                    extra * freq_scale_sq.view(1, -1)
                ).sum(dim=1, keepdim=True)  # [T, 1]
                combined = base_scores + additive  # [T, O]

                if self.config.score_aggregation == "max":
                    head_scores = combined.max(dim=1).values
                else:
                    head_scores = combined.mean(dim=1)
                scores[layer_idx, head_idx] = head_scores

        return scores

    def _normalize_and_noise(self, layer_head_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_head_scores: [L, H, T]
        Returns:
            normalized/noisy scores with same shape.
        """
        scores = layer_head_scores
        if self.config.normalize_scores and scores.numel() > 0:
            mean = scores.mean(dim=2, keepdim=True)
            std = scores.std(dim=2, unbiased=False, keepdim=True).clamp_min(1e-6)
            scores = (scores - mean) / std

        if self.config.tie_break_noise and scores.numel() > 0:
            noise = (
                torch.rand(
                    scores.shape,
                    generator=self._generator,
                    device="cpu",
                    dtype=torch.float32,
                )
                * float(self.config.tie_break_noise_scale)
            ).to(device=scores.device, dtype=scores.dtype)
            scores = scores + noise
        return scores

    def _aggregate_perhead(self, layer_head_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_head_scores: [L, H, T]
        Returns:
            perhead token scores: [H, T]
        """
        if layer_head_scores.numel() == 0:
            return torch.empty(0, device=layer_head_scores.device, dtype=torch.float32)

        # In this model, each (layer, kv_head) has a single score row.
        # "max_then_mean" reduces to mean across layers.
        if self.config.perhead_layer_aggregation == "max":
            return layer_head_scores.max(dim=0).values
        return layer_head_scores.mean(dim=0)

    @staticmethod
    def _gather_perhead(data: torch.Tensor, keep_idx_per_head: torch.Tensor) -> torch.Tensor:
        # data: [B, S, H, D], keep_idx_per_head: [H, K]
        bsz, _, num_heads, head_dim = data.shape
        if keep_idx_per_head.shape[0] != num_heads:
            if keep_idx_per_head.shape[0] > num_heads:
                keep_idx = keep_idx_per_head[:num_heads]
            else:
                pad = keep_idx_per_head[0:1].expand(num_heads - keep_idx_per_head.shape[0], -1)
                keep_idx = torch.cat([keep_idx_per_head, pad], dim=0)
        else:
            keep_idx = keep_idx_per_head

        x = data.permute(0, 2, 1, 3)  # [B, H, S, D]
        k = int(keep_idx.shape[1])
        gather_idx = keep_idx.unsqueeze(0).unsqueeze(-1).expand(bsz, num_heads, k, head_dim)
        y = x.gather(dim=2, index=gather_idx)  # [B, H, K, D]
        return y.permute(0, 2, 1, 3).contiguous()  # [B, K, H, D]

    def _compress_perhead(
        self,
        kv_cache: List[dict],
        seq_len: int,
        current_end_frame: int,
    ) -> tuple[bool, int]:
        keep_count = min(int(self.config.budget_tokens), seq_len)
        keep_last_tokens = min(
            keep_count, max(0, int(self.config.keep_last_frames)) * self.config.frame_seq_length
        )
        # Sink protection: reserve the first sink_tokens slots as survivors.
        sink_tokens_cfg = int(self.config.sink_size) * int(self.config.frame_seq_length)
        if self.config.protect_sink and sink_tokens_cfg > 0:
            sink_tokens = min(sink_tokens_cfg, seq_len)
        else:
            sink_tokens = 0
        # If sink_tokens + keep_last_tokens exceeds keep_count, shrink keep_last_tokens
        # so the budget can still hold the sink. Sink protection has priority.
        if sink_tokens + keep_last_tokens > keep_count:
            keep_last_tokens = max(0, keep_count - sink_tokens)
        dynamic_len = max(0, seq_len - keep_last_tokens - sink_tokens)
        dynamic_keep = max(0, keep_count - keep_last_tokens - sink_tokens)
        num_heads = int(kv_cache[0]["k"].shape[2])

        if dynamic_keep > 0 and dynamic_len > 0:
            layer_head_scores = self._compute_layer_head_scores(
                kv_cache=kv_cache,
                dynamic_len=dynamic_len,
                current_end_frame=current_end_frame,
                dynamic_offset=sink_tokens,
            )  # [L, H, T]
            layer_head_scores = self._normalize_and_noise(layer_head_scores)
            perhead_scores = self._aggregate_perhead(layer_head_scores)  # [H, T]

            keep_rows: List[torch.Tensor] = []
            for head_idx in range(num_heads):
                if head_idx < perhead_scores.shape[0]:
                    scores_h = perhead_scores[head_idx]
                elif perhead_scores.shape[0] > 0:
                    scores_h = perhead_scores.mean(dim=0)
                else:
                    scores_h = torch.zeros(dynamic_len, device=kv_cache[0]["k"].device, dtype=torch.float32)
                top_idx = torch.topk(scores_h, k=dynamic_keep, largest=True).indices
                # Offset dynamic-region local indices back into absolute cache indices.
                top_idx = top_idx + sink_tokens
                keep_rows.append(torch.sort(top_idx).values)
            top_idx_per_head = torch.stack(keep_rows, dim=0)
        else:
            top_idx_per_head = torch.empty(
                num_heads, 0, device=kv_cache[0]["k"].device, dtype=torch.long
            )

        keep_segments: List[torch.Tensor] = []
        if sink_tokens > 0:
            sink_idx = torch.arange(
                0, sink_tokens, device=kv_cache[0]["k"].device, dtype=torch.long
            ).unsqueeze(0).expand(num_heads, -1)
            keep_segments.append(sink_idx)
        keep_segments.append(top_idx_per_head)
        if keep_last_tokens > 0:
            tail_idx = torch.arange(
                seq_len - keep_last_tokens,
                seq_len,
                device=kv_cache[0]["k"].device,
                dtype=torch.long,
            )
            tail_idx = tail_idx.unsqueeze(0).expand(num_heads, -1)
            keep_segments.append(tail_idx)
        keep_idx_per_head = torch.cat(keep_segments, dim=1)
        keep_idx_per_head = torch.sort(keep_idx_per_head, dim=1).values

        new_len = int(keep_idx_per_head.shape[1])
        if self.cache_positions_per_head is not None:
            head_count = min(
                int(self.cache_positions_per_head.shape[0]),
                int(keep_idx_per_head.shape[0]),
            )
            pos = self.cache_positions_per_head[:head_count]
            selected_pos = pos.gather(dim=1, index=keep_idx_per_head[:head_count])
            if head_count < self.cache_positions_per_head.shape[0]:
                pad_rows = self.cache_positions_per_head[head_count:, :new_len]
                self.cache_positions_per_head = torch.cat([selected_pos, pad_rows], dim=0)
            else:
                self.cache_positions_per_head = selected_pos

        for cache in kv_cache:
            for key in ("k", "v"):
                if key not in cache:
                    continue
                data = cache[key]
                selected = self._gather_perhead(data, keep_idx_per_head)
                data[:, :new_len].copy_(selected)
                if new_len < data.shape[1]:
                    data[:, new_len:].zero_()
            cache["local_end_index"].fill_(new_len)
        return True, new_len

    def _compress_layer_perhead(
        self,
        kv_cache: List[dict],
        seq_len: int,
        current_end_frame: int,
    ) -> tuple[bool, int]:
        keep_count = min(int(self.config.budget_tokens), seq_len)
        keep_last_tokens = min(
            keep_count, max(0, int(self.config.keep_last_frames)) * self.config.frame_seq_length
        )
        # Sink protection: reserve the first sink_tokens slots as survivors.
        sink_tokens_cfg = int(self.config.sink_size) * int(self.config.frame_seq_length)
        if self.config.protect_sink and sink_tokens_cfg > 0:
            sink_tokens = min(sink_tokens_cfg, seq_len)
        else:
            sink_tokens = 0
        if sink_tokens + keep_last_tokens > keep_count:
            keep_last_tokens = max(0, keep_count - sink_tokens)
        dynamic_len = max(0, seq_len - keep_last_tokens - sink_tokens)
        dynamic_keep = max(0, keep_count - keep_last_tokens - sink_tokens)
        num_layers = len(kv_cache)
        num_heads = int(kv_cache[0]["k"].shape[2])

        if dynamic_keep > 0 and dynamic_len > 0:
            layer_head_scores = self._compute_layer_head_scores(
                kv_cache=kv_cache,
                dynamic_len=dynamic_len,
                current_end_frame=current_end_frame,
                dynamic_offset=sink_tokens,
            )  # [L, H, T]
            layer_head_scores = self._normalize_and_noise(layer_head_scores)

            if layer_head_scores.numel() > 0:
                global_fallback = layer_head_scores.mean(dim=(0, 1))
            else:
                global_fallback = torch.zeros(
                    dynamic_len, device=kv_cache[0]["k"].device, dtype=torch.float32
                )

            keep_layers: List[torch.Tensor] = []
            for layer_idx in range(num_layers):
                keep_rows: List[torch.Tensor] = []
                for head_idx in range(num_heads):
                    if (
                        layer_idx < layer_head_scores.shape[0]
                        and head_idx < layer_head_scores.shape[1]
                    ):
                        scores = layer_head_scores[layer_idx, head_idx]
                    elif layer_idx < layer_head_scores.shape[0] and layer_head_scores.shape[1] > 0:
                        scores = layer_head_scores[layer_idx].mean(dim=0)
                    elif layer_head_scores.shape[0] > 0 and head_idx < layer_head_scores.shape[1]:
                        scores = layer_head_scores[:, head_idx].mean(dim=0)
                    else:
                        scores = global_fallback
                    top_idx = torch.topk(scores, k=dynamic_keep, largest=True).indices
                    # Offset dynamic-region local indices back into absolute cache indices.
                    top_idx = top_idx + sink_tokens
                    keep_rows.append(torch.sort(top_idx).values)
                keep_layers.append(torch.stack(keep_rows, dim=0))
            top_idx_per_layer_head = torch.stack(keep_layers, dim=0)  # [L, H, K]
        else:
            top_idx_per_layer_head = torch.empty(
                num_layers, num_heads, 0, device=kv_cache[0]["k"].device, dtype=torch.long
            )

        keep_segments: List[torch.Tensor] = []
        if sink_tokens > 0:
            sink_idx = torch.arange(
                0, sink_tokens, device=kv_cache[0]["k"].device, dtype=torch.long
            ).view(1, 1, -1).expand(num_layers, num_heads, -1)
            keep_segments.append(sink_idx)
        keep_segments.append(top_idx_per_layer_head)
        if keep_last_tokens > 0:
            tail_idx = torch.arange(
                seq_len - keep_last_tokens,
                seq_len,
                device=kv_cache[0]["k"].device,
                dtype=torch.long,
            )
            tail_idx = tail_idx.view(1, 1, -1).expand(num_layers, num_heads, -1)
            keep_segments.append(tail_idx)
        keep_idx_per_layer_head = torch.cat(keep_segments, dim=2)
        keep_idx_per_layer_head = torch.sort(keep_idx_per_layer_head, dim=2).values

        new_len = int(keep_idx_per_layer_head.shape[2])
        if self.cache_positions_per_layer_head is not None:
            state = self.cache_positions_per_layer_head
            if state.shape[0] == keep_idx_per_layer_head.shape[0] and state.shape[1] == keep_idx_per_layer_head.shape[1]:
                self.cache_positions_per_layer_head = state.gather(
                    dim=2, index=keep_idx_per_layer_head
                )
            else:
                layer_count = min(int(state.shape[0]), int(keep_idx_per_layer_head.shape[0]))
                head_count = min(int(state.shape[1]), int(keep_idx_per_layer_head.shape[1]))
                selected_pos = state[:layer_count, :head_count].gather(
                    dim=2, index=keep_idx_per_layer_head[:layer_count, :head_count]
                )
                new_state = state[:, :, :new_len].contiguous()
                new_state[:layer_count, :head_count] = selected_pos
                self.cache_positions_per_layer_head = new_state

        for layer_idx, cache in enumerate(kv_cache):
            layer_keep_idx = keep_idx_per_layer_head[layer_idx]
            for key in ("k", "v"):
                if key not in cache:
                    continue
                data = cache[key]
                selected = self._gather_perhead(data, layer_keep_idx)
                data[:, :new_len].copy_(selected)
                if new_len < data.shape[1]:
                    data[:, new_len:].zero_()
            cache["local_end_index"].fill_(new_len)
        return True, new_len

    def maybe_compress(self, kv_cache: List[dict], current_end_frame: int, force: bool = False) -> bool:
        if self.config.mode != "compress":
            return False
        if not force and (current_end_frame - self.last_compressed_frame < self.config.compress_every_n_frames):
            return False
        seq_len = int(kv_cache[0]["local_end_index"].item())
        if seq_len <= int(self.config.budget_tokens):
            return False
        if kv_cache[0]["k"].shape[0] != 1:
            raise ValueError("Current KV compressor supports batch_size=1 only.")

        self._load_stats(kv_cache[0]["k"].device)
        num_layers = len(kv_cache)
        num_heads = int(kv_cache[0]["k"].shape[2])
        device = kv_cache[0]["k"].device
        self._sync_position_state(
            kv_cache, num_layers=num_layers, num_heads=num_heads, device=device
        )

        if self.config.pruning_mode == "perhead":
            compressed, new_len = self._compress_perhead(
                kv_cache=kv_cache,
                seq_len=seq_len,
                current_end_frame=current_end_frame,
            )
        elif self.config.pruning_mode == "layer_perhead":
            compressed, new_len = self._compress_layer_perhead(
                kv_cache=kv_cache,
                seq_len=seq_len,
                current_end_frame=current_end_frame,
            )
        else:
            raise ValueError(
                f"Unsupported pruning_mode={self.config.pruning_mode}. "
                "Supported: perhead, layer_perhead"
            )
        if not compressed:
            return False

        self.last_compressed_frame = current_end_frame
        if _is_main_process():
            keep_ratio = float(new_len) / float(max(1, seq_len))
            print(
                f"[KV-Compress] Compressed cache tokens: {seq_len} -> {new_len} "
                f"(frame={current_end_frame}, ratio={keep_ratio:.4f}, mode={self.config.pruning_mode})"
            )
        return True
