"""Pinned-prefix variant of the SparseRound KV pruner."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoConfig

from rkv.utils import cal_similarity
from weian_development.speckv.round_pruning_utils import (
    HeadFrequencyStats,
    build_geometric_offsets,
    build_rotary,
    compute_frequency_statistics_from_means,
    compute_frequency_scaling,
    determine_rope_style,
    invert_rope,
    load_head_frequency_stats,
    score_keys_for_round,
)
from weian_development.speckv.stats_utils import validate_stats_metadata


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
    metadata_expectations: Dict[str, object] | None = None
    normalize_scores: bool = False
    use_rank_aggregation: bool = False  # If True, use ranking + min-pooling instead of z-score + max-pooling
    # Similarity Deduplication parameters (SpecKV + R-KV integration)
    sparse_use_similarity: bool = False
    sparse_similarity_mix_lambda: float = 0.1
    # Rank + Similarity combination: normalize and invert rank direction for proper combination
    use_rank_similarity_combination: bool = False
    # Alignment: include prefill tokens in budget calculation (aligns with R-KV behavior)
    include_prefill_in_budget: bool = False
    # Alignment: compress to exact budget instead of budget - round_window (aligns with R-KV behavior)
    rkv_aligned_budget: bool = False
    # Alignment: divide_length for R-KV style compression interval (cache fluctuates in [budget, budget + divide_length])
    divide_length: int = 128
    # Per-head pruning: enable independent per-head pruning mode
    use_per_head_pruning: bool = False
    # R-KV alignment: allow prefill tokens to be compressed (like R-KV behavior)
    # When False (default): prefill tokens are always preserved
    # When True: prefill tokens compete with decode tokens for budget (R-KV style)
    allow_prefill_compression: bool = False
    # High-frequency ablation: disable top-n high-frequency components in position-dependent scoring
    # When 0 (default): all frequency components are used
    # When > 0: omega[0:n] (highest frequencies) are masked in base_scores, but additive term is unaffected
    disable_top_n_high_freq: int = 0
    # Bug 896cbca6 phase offset simulation: simulate the phase offset caused by the bug
    # When 0 (default): no phase offset applied
    # When > 0: subtract N×ω from phase to simulate bug behavior (Δ ≈ 156 tokens average)
    simulate_bug_phase_offset: int = 0


class SparseRoundPruner:
    """
    Maintains a round-based sparse attention cache while pinning the prompt KV entries.

    The pinned-prefix behavior keeps the full prefill context intact and only applies
    pruning / head scoring to decode-time tokens, allowing a fair comparison with
    methods that retain the entire question prefix.
    """

    def __init__(self, config: SparsePruningConfig) -> None:
        if config.max_keys < config.round_window:
            raise ValueError("max_keys must be >= round_window for round-based pruning")
        self.config = config
        model_config = AutoConfig.from_pretrained(
            str(config.model_path), trust_remote_code=True
        )
        rope_scaling = getattr(model_config, "rope_scaling", {}) or {}
        default_expectations: Dict[str, object] = {
            "rope_style": determine_rope_style(model_config),
        }
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or getattr(model_config, "rope_type", None)
        if rope_type:
            default_expectations["rope_type"] = rope_type

        metadata, stats_map = load_head_frequency_stats(config.stats_path, config.device)
        merged_expectations: Dict[str, object] = {}
        if config.metadata_expectations:
            merged_expectations.update(config.metadata_expectations)
        merged_expectations.update({k: v for k, v in default_expectations.items() if v is not None})
        if merged_expectations:
            validate_stats_metadata(metadata, merged_expectations, stats_path=config.stats_path)

        sampled_heads = [tuple(item) for item in metadata.get("sampled_heads", [])]
        if not sampled_heads:
            raise ValueError("Stats file does not contain any sampled heads")
        layer_count = int(getattr(model_config, "num_hidden_layers", len(sampled_heads)))
        filtered_heads = [
            head for head in sampled_heads if 0 <= head[0] < layer_count
        ]
        if config.head_limit is not None and config.head_limit > 0:
            filtered_heads = filtered_heads[: config.head_limit]
        if not filtered_heads:
            raise ValueError(
                f"No valid heads remain after filtering with layer_count={layer_count}"
            )
        self.sampled_heads: List[Tuple[int, int]] = filtered_heads
        self.head_stats: Dict[Tuple[int, int], HeadFrequencyStats] = {
            key: stats_map[key]
            for key in filtered_heads
            if key in stats_map
        }
        if len(self.head_stats) != len(self.sampled_heads):
            missing = set(self.sampled_heads) - set(self.head_stats.keys())
            raise ValueError(f"Missing stats for heads: {sorted(missing)}")

        self.rotary = build_rotary(
            config.device, config.model_path, config.dtype, config=model_config
        )
        self.rope_style = getattr(self.rotary, "_rope_style", "half")
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
        self.num_key_value_heads = getattr(
            rope_config, "num_key_value_heads", self.num_attention_heads
        )
        if self.num_attention_heads and self.num_key_value_heads:
            self.num_key_value_groups = max(
                1, self.num_attention_heads // self.num_key_value_heads
            )
        else:
            self.num_key_value_heads = None
            self.num_key_value_groups = None

        self.cache_positions: List[int] = []
        self.absolute_position: int = 0
        self.tokens_in_round: int = 0
        self.round_window = config.round_window
        self.max_keys = config.max_keys
        self.score_aggregation = config.score_aggregation
        self.normalize_scores = bool(getattr(config, "normalize_scores", False))
        self.use_rank_aggregation = bool(getattr(config, "use_rank_aggregation", False))
        # Similarity Deduplication parameters (SpecKV + R-KV integration)
        self.use_similarity = bool(getattr(config, "sparse_use_similarity", False))
        self.similarity_mix_lambda = float(getattr(config, "sparse_similarity_mix_lambda", 0.1))
        # Rank + Similarity combination mode
        self.use_rank_similarity_combination = bool(getattr(config, "use_rank_similarity_combination", False))
        # Per-head pruning mode
        self.use_per_head_pruning = bool(getattr(config, "use_per_head_pruning", False))
        # R-KV aligned budget: compress to exact budget instead of budget - round_window
        self.rkv_aligned_budget = bool(getattr(config, "rkv_aligned_budget", False))
        # R-KV divide_length: compression interval (cache fluctuates in [budget, budget + divide_length])
        raw_divide_length = getattr(config, "divide_length", 128)
        validated_divide_length = 128 if raw_divide_length is None else int(raw_divide_length)
        if validated_divide_length <= 0:
            raise ValueError(f"divide_length must be positive; got {validated_divide_length}")
        self.divide_length = validated_divide_length
        # R-KV alignment: allow prefill tokens to be compressed
        self.allow_prefill_compression = bool(getattr(config, "allow_prefill_compression", False))
        # High-frequency ablation: disable top-n high-frequency components in position-dependent scoring
        self.disable_top_n_high_freq = int(getattr(config, "disable_top_n_high_freq", 0))
        # Bug 896cbca6 phase offset simulation
        self.simulate_bug_phase_offset = int(getattr(config, "simulate_bug_phase_offset", 0))
        self.generator: torch.Generator | None = None
        self.prefix_length: int = 0
        if config.seed is not None:
            if config.device.type == "cuda":
                self.generator = torch.Generator(device=config.device)
            else:
                self.generator = torch.Generator()
            self.generator.manual_seed(int(config.seed))

    @property
    def _dynamic_cache_size(self) -> int:
        if self.config.include_prefill_in_budget:
            # Align with R-KV: include prefill tokens in cache size calculation
            return len(self.cache_positions)
        # Original behavior: exclude prefill tokens from cache size
        return max(0, len(self.cache_positions) - self.prefix_length)

    def attach_initial_cache(
        self, past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        if not past_key_values:
            self.cache_positions = []
            self.absolute_position = 0
            self.tokens_in_round = 0
            self.prefix_length = 0
            return
        seq_len = past_key_values[0][0].shape[2]
        self.cache_positions = list(range(seq_len))
        self.absolute_position = seq_len
        self.tokens_in_round = 0
        self.prefix_length = seq_len

    def enforce_max_limit(
        self, past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ):
        if self._dynamic_cache_size <= self.max_keys:
            return past_key_values
        pruned = self._prune_to_size(
            past_key_values, self.max_keys, dynamic_only=True
        )
        self.tokens_in_round = 0
        return pruned

    def ensure_capacity(self, past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        keep_capacity = self.max_keys if self.rkv_aligned_budget else max(0, self.max_keys - self.round_window)
        if self._dynamic_cache_size <= keep_capacity:
            return past_key_values
        # When allow_prefill_compression=True: use keep_capacity directly (all tokens compete)
        # Otherwise: subtract prefix from total target (prefill is preserved)
        if self.allow_prefill_compression:
            prune_target = keep_capacity
        elif self.config.include_prefill_in_budget:
            prune_target = max(0, keep_capacity - self.prefix_length)
        else:
            prune_target = keep_capacity
        pruned = self._prune_to_size(
            past_key_values, prune_target, dynamic_only=True
        )
        self.tokens_in_round = 0
        return pruned

    def on_token_appended(self) -> None:
        self.cache_positions.append(self.absolute_position)
        self.absolute_position += 1
        self.tokens_in_round += 1

    def should_start_next_round(self) -> bool:
        return self.tokens_in_round >= self.round_window

    def start_next_round(
        self, past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ):
        keep_capacity = self.max_keys if self.rkv_aligned_budget else max(0, self.max_keys - self.round_window)
        # When allow_prefill_compression=True: use keep_capacity directly (all tokens compete)
        # Otherwise: subtract prefix from total target (prefill is preserved)
        if self.allow_prefill_compression:
            prune_target = keep_capacity
        elif self.config.include_prefill_in_budget:
            prune_target = max(0, keep_capacity - self.prefix_length)
        else:
            prune_target = keep_capacity
        pruned = self._prune_to_size(
            past_key_values, prune_target, dynamic_only=True
        )
        self.tokens_in_round = 0
        return pruned

    def _prune_to_size(
        self,
        past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        keep_count: int,
        *,
        dynamic_only: bool = False,
    ):
        if not self.cache_positions:
            return past_key_values

        candidate_count = len(self.cache_positions)
        candidate_indices = torch.arange(
            candidate_count, device=self.config.device, dtype=torch.long
        )
        key_positions = torch.tensor(
            self.cache_positions, device=self.config.device, dtype=torch.long
        )

        if not dynamic_only and keep_count <= 0:
            self.cache_positions = []
            return self._slice_cache(
                past_key_values, torch.empty(0, device=self.config.device, dtype=torch.long)
            )

        if not dynamic_only and candidate_count <= keep_count:
            keep_tensor = candidate_indices
        elif not dynamic_only:
            keep_tensor = self._select_keep_indices(
                past_key_values, key_positions, keep_count
            )
        else:
            # R-KV alignment mode: allow prefill tokens to be compressed
            if self.allow_prefill_compression:
                # All tokens (including prefill) compete for budget
                if candidate_count <= keep_count:
                    keep_tensor = candidate_indices
                else:
                    keep_tensor = self._select_keep_indices(
                        past_key_values, key_positions, keep_count
                    )
            else:
                # Original behavior: prefill tokens are always preserved
                prefix_count = min(self.prefix_length, candidate_count)
                dynamic_count = max(0, candidate_count - prefix_count)
                keep_count = max(0, min(keep_count, dynamic_count))
                prefix_indices = candidate_indices[:prefix_count]
                if dynamic_count == 0:
                    keep_tensor = prefix_indices
                elif dynamic_count <= keep_count:
                    keep_tensor = candidate_indices
                else:
                    dynamic_positions = key_positions[prefix_count:]
                    dynamic_keep_rel = self._select_keep_indices(
                        past_key_values,
                        dynamic_positions,
                        keep_count,
                        start_index=prefix_count,
                    )
                    dynamic_keep = dynamic_keep_rel + prefix_count
                    if dynamic_keep_rel.dim() == 2:
                        # Per-head mode: broadcast prefix to all heads, then concat along dim=1
                        num_kv_heads = dynamic_keep_rel.size(0)
                        prefix_broadcast = prefix_indices.unsqueeze(0).expand(num_kv_heads, -1)
                        keep_tensor = torch.cat([prefix_broadcast, dynamic_keep], dim=1)
                    else:
                        keep_tensor = torch.cat([prefix_indices, dynamic_keep])

        if keep_tensor.dim() == 2:
            # Per-head mode: use head 0's indices as representative for cache_positions
            # This ensures correct length for _dynamic_cache_size calculations
            head0_indices = keep_tensor[0].tolist()
            self.cache_positions = [self.cache_positions[idx] for idx in head0_indices]
        else:
            self.cache_positions = [self.cache_positions[idx] for idx in keep_tensor.tolist()]
        return self._slice_cache(past_key_values, keep_tensor)

    def _select_keep_indices(
        self,
        past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        key_positions: torch.Tensor,
        keep_count: int,
        start_index: int = 0,
    ) -> torch.Tensor:
        # Block rank aggregation when per-head pruning enabled
        assert not (self.use_per_head_pruning and self.use_rank_aggregation), \
            "Per-head pruning only supports norm aggregation, not rank aggregation"

        per_head_scores = self._compute_head_scores(
            past_key_values, key_positions, start_index=start_index
        )
        if per_head_scores.numel() == 0:
            return torch.empty(0, device=self.config.device, dtype=torch.long)

        # Similarity Deduplication: combine frequency scores with similarity scores per-head, then aggregate
        # Formula: per_head_final = freq_score * mix_lambda - similarity * (1 - mix_lambda)
        # This aligns with R-KV's per-head combination approach
        if self.use_similarity:
            seq_len = key_positions.shape[0]
            device = per_head_scores.device

            # Compute similarity for each unique layer, then align with sampled_heads
            # This ensures each (layer, head) pair uses similarity from its own layer
            unique_layers = sorted(set(layer for layer, head in self.sampled_heads))
            layer_similarity_cache: dict[int, torch.Tensor] = {}

            for layer_idx in unique_layers:
                key_tensor = past_key_values[layer_idx][0]  # [batch=1, num_heads, seq_len, head_dim]
                gather_indices = torch.arange(start_index, start_index + seq_len, device=key_tensor.device)
                key_states = key_tensor[:, :, gather_indices, :]  # [1, num_heads, seq_len, head_dim]
                # cal_similarity returns [num_heads, seq_len] after mean(dim=1).softmax(dim=-1)
                layer_similarity = cal_similarity(
                    key_states,
                    threshold=0.5,
                    retain_ratio=0.1,
                    retain_direction="last",
                )
                layer_similarity_cache[layer_idx] = layer_similarity

            # Build aligned similarity scores matching per_head_scores shape [num_sampled_heads, seq_len]
            # Use same GQA head mapping as _compute_head_scores: attention_head -> kv_head
            similarity_aligned_list = []
            for layer, head in self.sampled_heads:
                layer_sim = layer_similarity_cache[layer]  # [num_kv_heads, seq_len]
                # Map attention head index to KV head index (for GQA models like DeepSeek/LLaMA)
                kv_head = head
                if self.num_key_value_heads and self.num_attention_heads:
                    kv_head = min(
                        layer_sim.shape[0] - 1,
                        head // max(1, self.num_key_value_groups),
                    )
                head_sim = layer_sim[kv_head]  # [seq_len]
                similarity_aligned_list.append(head_sim)
            similarity_scores_aligned = torch.stack(similarity_aligned_list, dim=0)  # [num_sampled_heads, seq_len]

            # Per-head combination: mix frequency and similarity scores at head level BEFORE aggregation
            # Higher similarity = more redundant = should be removed = lower final score
            if self.use_rank_similarity_combination and self.use_rank_aggregation:
                # Rank + Similarity combination: normalize and invert rank direction
                # Rank values are [0, N-1] where 0 = best. Convert to [0, 1] with 1 = best.
                max_rank = float(per_head_scores.shape[1] - 1)
                if max_rank > 0:
                    normalized_inverted_rank = (max_rank - per_head_scores) / max_rank
                else:
                    normalized_inverted_rank = torch.ones_like(per_head_scores)
                # Now normalized_inverted_rank: 1 = best (was rank 0), 0 = worst (was rank N-1)
                per_head_final = normalized_inverted_rank * self.similarity_mix_lambda - similarity_scores_aligned * (1 - self.similarity_mix_lambda)
                # Use max-pooling since higher is now better after inversion
                combined = per_head_final.max(dim=0).values
            else:
                per_head_final = per_head_scores * self.similarity_mix_lambda - similarity_scores_aligned * (1 - self.similarity_mix_lambda)
                # Aggregate across heads: min for ranks (best=lowest), max for scores (best=highest)
                if self.use_rank_aggregation:
                    combined = per_head_final.min(dim=0).values
                    combined = -combined  # Negate so topk(largest=True) selects lowest ranks
                else:
                    combined = per_head_final.max(dim=0).values
        else:
            # Original behavior: frequency-only scoring with head aggregation
            if self.use_rank_aggregation:
                combined = per_head_scores.min(dim=0).values
                combined = -combined  # Negate so topk(largest=True) selects lowest ranks
            else:
                combined = per_head_scores.max(dim=0).values

        # Per-head independent pruning mode: select tokens independently for each KV head
        if self.use_per_head_pruning:
            # Group sampled attention heads by KV head
            kv_head_groups = {}  # kv_head_idx -> [sampled_head_indices]
            for i, (layer, attn_head) in enumerate(self.sampled_heads):
                kv_head = attn_head // max(1, self.num_key_value_groups)
                if kv_head not in kv_head_groups:
                    kv_head_groups[kv_head] = []
                kv_head_groups[kv_head].append(i)

            # For each KV head, aggregate scores and perform independent top-k selection
            keep_indices_list = []
            for kv_head_idx in range(self.num_key_value_heads):
                if kv_head_idx in kv_head_groups:
                    indices = kv_head_groups[kv_head_idx]
                    group_scores = per_head_scores[indices]  # [num_heads_in_group, seq_len]
                    aggregated = group_scores.max(dim=0).values  # [seq_len] - same as existing norm aggregation
                else:
                    # Fallback for KV heads without sampled heads: use mean of all scores
                    aggregated = per_head_scores.mean(dim=0)

                # Independent top-k selection for this KV head
                keep_indices_for_head = aggregated.topk(keep_count, largest=True).indices
                keep_indices_list.append(keep_indices_for_head)

            # Stack into 2D tensor: [num_kv_heads, budget]
            keep_indices = torch.stack(keep_indices_list, dim=0)
            return keep_indices

        # Global unified pruning mode (existing behavior)
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
        *,
        start_index: int = 0,
    ) -> torch.Tensor:
        if key_positions.numel() == 0:
            return torch.empty(0, device=self.config.device, dtype=torch.float32)

        seq_indices_base = torch.arange(
            key_positions.shape[0], device=self.config.device, dtype=torch.long
        )
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
            gather_indices = local_indices + start_index
            k_values = key_tensor[0, kv_head].index_select(0, gather_indices)
            k_values = k_values.to(device=self.config.device, dtype=self.config.dtype)
            k_unrot = invert_rope(
                k_values,
                cos_table,
                sin_table,
                self.attention_scale,
                style=self.rope_style,
            )
            amp, phi, extra = compute_frequency_statistics_from_means(
                stats.q_mean_complex,
                stats.q_abs_mean,
                k_unrot,
                style=self.rope_style,
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
                disable_top_n_high_freq=self.disable_top_n_high_freq,
                simulate_bug_phase_offset=self.simulate_bug_phase_offset,
            )
            per_head_scores.append(head_scores)

        head_matrix = torch.stack(per_head_scores, dim=0)
        if self.use_rank_aggregation and head_matrix.numel() > 0:
            # Convert scores to ranks: lower rank = more important (rank 0 = best)
            # First argsort (descending=True) orders by score (highest first)
            # Second argsort converts positions to ranks
            ranks = torch.argsort(torch.argsort(head_matrix, dim=1, descending=True), dim=1)
            head_matrix = ranks.float()
        elif self.normalize_scores and head_matrix.numel() > 0:
            # Existing z-score normalization
            mean = head_matrix.mean(dim=1, keepdim=True)
            std = head_matrix.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
            head_matrix = (head_matrix - mean) / std
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

            if local_keep.dim() == 2:
                # Per-head mode: keep_indices shape [num_kv_heads, budget]
                # key_tensor/value_tensor shape: [batch, num_kv_heads, seq_len, head_dim]
                batch_size = key_tensor.size(0)
                num_kv_heads = key_tensor.size(1)
                budget = local_keep.size(1)
                head_dim = key_tensor.size(3)

                # Expand indices for gather: [batch, num_kv_heads, budget, head_dim]
                expanded_indices = local_keep.unsqueeze(0).unsqueeze(-1).expand(
                    batch_size, num_kv_heads, budget, head_dim
                )

                # Gather along sequence dimension (dim=2)
                key_slice = key_tensor.gather(dim=2, index=expanded_indices)
                value_slice = value_tensor.gather(dim=2, index=expanded_indices)
            else:
                # Global mode: keep_indices shape [budget]
                key_slice = key_tensor.index_select(2, local_keep)
                value_slice = value_tensor.index_select(2, local_keep)

            new_cache.append((key_slice.contiguous(), value_slice.contiguous()))

        if hasattr(past_key_values, "layers"):
            for layer_obj, (key_slice, value_slice) in zip(
                past_key_values.layers, new_cache
            ):
                layer_obj.keys = key_slice
                layer_obj.values = value_slice
            return past_key_values

        return tuple(new_cache)
