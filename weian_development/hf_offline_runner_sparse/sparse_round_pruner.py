"""Runtime sparse KV cache pruner for HuggingFace decoding."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from weian_development.attention_qk_analysis.freq_magnitude_plots import invert_rope
from weian_development.hf_offline_runner_sparse.round_pruning_utils import (
    HeadFrequencyStats,
    build_geometric_offsets,
    build_rotary,
    compute_frequency_statistics_from_means,
    compute_frequency_scaling,
    load_head_frequency_stats,
    score_keys_for_round,
)


@dataclass
class SparsePruningConfig:
    stats_path: Path
    model_path: Path
    device: torch.device
    dtype: torch.dtype
    max_keys: int
    round_window: int
    offset_max_length: int
    score_aggregation: str
    seed: int | None = None
    head_limit: int | None = None


class SparseRoundPruner:
    """Maintains a round-based sparse attention cache by dropping KV entries."""

    def __init__(self, config: SparsePruningConfig) -> None:
        if config.max_keys < config.round_window:
            raise ValueError("max_keys must be >= round_window for round-based pruning")
        self.config = config
        metadata, stats_map = load_head_frequency_stats(config.stats_path, config.device)
        sampled_heads = [tuple(item) for item in metadata.get("sampled_heads", [])]
        if not sampled_heads:
            raise ValueError("Stats file does not contain any sampled heads")
        if config.head_limit is not None and config.head_limit > 0:
            sampled_heads = sampled_heads[: config.head_limit]
        self.sampled_heads: List[Tuple[int, int]] = sampled_heads
        self.head_stats: Dict[Tuple[int, int], HeadFrequencyStats] = {
            key: stats_map[key]
            for key in sampled_heads
            if key in stats_map
        }
        if len(self.head_stats) != len(self.sampled_heads):
            missing = set(self.sampled_heads) - set(self.head_stats.keys())
            raise ValueError(f"Missing stats for heads: {sorted(missing)}")

        self.rotary = build_rotary(config.device, config.model_path, config.dtype)
        self.attention_scale = float(getattr(self.rotary, "attention_scaling", 1.0))
        inv_freq = self.rotary.inv_freq.to(device=config.device, dtype=torch.float32)
        self.head_dim = int(metadata.get("head_dim", inv_freq.numel() * 2))
        freq_count = max(1, self.head_dim // 2)
        self.omega = inv_freq[:freq_count]
        self.offsets = build_geometric_offsets(config.offset_max_length, config.device)
        freq_scale = compute_frequency_scaling(
            self.rotary, self.head_dim, config.dtype, config.device
        )
        self.freq_scale_sq = freq_scale.pow(2)

        rope_config = getattr(self.rotary, "config", None)
        self.num_attention_heads = getattr(rope_config, "num_attention_heads", None)
        self.num_key_value_heads = getattr(rope_config, "num_key_value_heads", self.num_attention_heads)
        if self.num_attention_heads and self.num_key_value_heads:
            self.num_key_value_groups = max(1, self.num_attention_heads // self.num_key_value_heads)
        else:
            self.num_key_value_heads = None
            self.num_key_value_groups = None

        self.cache_positions: List[int] = []
        self.absolute_position: int = 0
        self.tokens_in_round: int = 0
        self.round_window = config.round_window
        self.max_keys = config.max_keys
        self.score_aggregation = config.score_aggregation
        self.generator: torch.Generator | None = None
        if config.seed is not None:
            if config.device.type == "cuda":
                self.generator = torch.Generator(device=config.device)
            else:
                self.generator = torch.Generator()
            self.generator.manual_seed(int(config.seed))

    def attach_initial_cache(self, past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        if not past_key_values:
            self.cache_positions = []
            self.absolute_position = 0
            return
        seq_len = past_key_values[0][0].shape[2]
        self.cache_positions = list(range(seq_len))
        self.absolute_position = seq_len
        self.tokens_in_round = 0

    def enforce_max_limit(
        self, past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ):
        if len(self.cache_positions) <= self.max_keys:
            return past_key_values
        pruned = self._prune_to_size(past_key_values, self.max_keys)
        self.tokens_in_round = 0
        return pruned

    def ensure_capacity(self, past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        keep_capacity = max(0, self.max_keys - self.round_window)
        if len(self.cache_positions) <= keep_capacity:
            return past_key_values
        pruned = self._prune_to_size(past_key_values, keep_capacity)
        self.tokens_in_round = 0
        return pruned

    def on_token_appended(self) -> None:
        self.cache_positions.append(self.absolute_position)
        self.absolute_position += 1
        self.tokens_in_round += 1

    def should_start_next_round(self) -> bool:
        return self.tokens_in_round >= self.round_window

    def start_next_round(self, past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        keep_capacity = max(0, self.max_keys - self.round_window)
        pruned = self._prune_to_size(past_key_values, keep_capacity)
        self.tokens_in_round = 0
        return pruned

    def _prune_to_size(self, past_key_values, keep_count: int):
        if keep_count <= 0:
            self.cache_positions = []
            return self._slice_cache(past_key_values, torch.empty(0, dtype=torch.long))
        if not self.cache_positions:
            return past_key_values
        candidate_count = len(self.cache_positions)
        candidate_indices = torch.arange(candidate_count, device=self.config.device, dtype=torch.long)
        key_positions = torch.tensor(self.cache_positions, device=self.config.device, dtype=torch.long)
        if candidate_count <= keep_count:
            keep_tensor = candidate_indices
        else:
            keep_tensor = self._select_keep_indices(
                past_key_values, key_positions, keep_count
            )
        self.cache_positions = [self.cache_positions[idx] for idx in keep_tensor.tolist()]
        return self._slice_cache(past_key_values, keep_tensor)

    def _select_keep_indices(
        self,
        past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        key_positions: torch.Tensor,
        keep_count: int,
    ) -> torch.Tensor:
        per_head_scores = self._compute_head_scores(past_key_values, key_positions)
        combined = per_head_scores.max(dim=0).values

        candidate_count = combined.shape[0]
        if candidate_count <= keep_count:
            return torch.arange(candidate_count, device=combined.device, dtype=torch.long)

        per_head_quota = min(keep_count, candidate_count)
        union_mask = torch.zeros(candidate_count, device=combined.device, dtype=torch.bool)
        for head_scores in per_head_scores:
            head_k = min(per_head_quota, head_scores.numel())
            if head_k == 0:
                continue
            top_idx = torch.topk(head_scores, k=head_k, largest=True).indices
            union_mask[top_idx] = True

        union_indices = torch.nonzero(union_mask, as_tuple=False).view(-1)
        if union_indices.numel() == 0:
            union_indices = torch.arange(0, 0, device=combined.device, dtype=torch.long)

        if union_indices.numel() >= keep_count:
            subset_scores = combined.index_select(0, union_indices)
            top_subset = torch.topk(subset_scores, k=keep_count, largest=True).indices
            return union_indices.index_select(0, torch.sort(top_subset).values)

        remaining = keep_count - union_indices.numel()
        available = candidate_count - union_indices.numel()
        if remaining > 0 and available > 0:
            residual_scores = combined.clone()
            residual_scores[union_mask] = float("-inf")
            extra_k = min(remaining, available)
            extra_indices = torch.topk(residual_scores, k=extra_k, largest=True).indices
            union_indices = torch.cat([union_indices, extra_indices])

        return torch.sort(union_indices).values

    def _compute_head_scores(
        self,
        past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        if key_positions.numel() == 0:
            return torch.empty(0, device=self.config.device, dtype=torch.float32)

        seq_indices_base = torch.arange(key_positions.shape[0], device=self.config.device, dtype=torch.long)
        cos_table, sin_table = self._rotary_for_positions(key_positions)

        per_head_scores: List[torch.Tensor] = []
        for layer, head in self.sampled_heads:
            stats = self.head_stats[(layer, head)]
            key_tensor = past_key_values[layer][0]
            if key_tensor.shape[0] != 1:
                raise RuntimeError("Sparse pruner currently supports batch_size=1")
            kv_head = head
            if self.num_key_value_heads and self.num_attention_heads:
                kv_head = min(
                    key_tensor.shape[1] - 1,
                    head // max(1, self.num_key_value_groups),
                )
            elif head >= key_tensor.shape[1]:
                raise RuntimeError(
                    f"Head index {head} exceeds kv heads ({key_tensor.shape[1]}) for layer {layer}"
                )
            local_indices = seq_indices_base.to(key_tensor.device)
            k_values = key_tensor[0, kv_head].index_select(0, local_indices)
            k_values = k_values.to(device=self.config.device, dtype=self.config.dtype)
            k_unrot = invert_rope(k_values, cos_table, sin_table, self.attention_scale)
            amp, phi, extra = compute_frequency_statistics_from_means(
                stats.q_mean_complex,
                stats.q_abs_mean,
                k_unrot,
            )
            head_scores = score_keys_for_round(
                key_indices=key_positions,
                round_start=self.absolute_position,
                amp=amp,
                phi=phi,
                omega=self.omega,
                extra=extra,
                offsets=self.offsets,
                aggregation=self.score_aggregation,
                freq_scale_sq=self.freq_scale_sq,
            )
            per_head_scores.append(head_scores)

        head_matrix = torch.stack(per_head_scores, dim=0)
        if self.generator is not None and head_matrix.numel() > 0:
            noise = torch.rand(
                head_matrix.shape,
                device=head_matrix.device,
                generator=self.generator,
            ) * 1e-6
            head_matrix = head_matrix + noise
        return head_matrix

    def _rotary_for_positions(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base = torch.zeros(
            1,
            positions.shape[0],
            self.head_dim,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        cos, sin = self.rotary(base, positions.unsqueeze(0))
        return cos[0], sin[0]

    def _slice_cache(
        self,
        past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        keep_indices: torch.Tensor,
    ):
        new_cache = []
        for key_tensor, value_tensor in past_key_values:
            local_keep = keep_indices.to(device=key_tensor.device, dtype=torch.long)
            key_slice = key_tensor.index_select(2, local_keep)
            value_slice = value_tensor.index_select(2, local_keep)
            new_cache.append((key_slice.contiguous(), value_slice.contiguous()))

        if hasattr(past_key_values, "layers"):
            for layer_obj, (key_slice, value_slice) in zip(past_key_values.layers, new_cache):
                layer_obj.keys = key_slice
                layer_obj.values = value_slice
            return past_key_values

        return tuple(new_cache)
