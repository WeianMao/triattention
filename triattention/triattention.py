"""TriAttention implementation using attention-layer compression.

This module provides a TriAttention implementation that triggers compression
inside the attention forward pass instead of in the generate() wrapper.
The frequency-based scoring logic implements the core pruning algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .pruning_utils import (
    HeadFrequencyStats,
    build_geometric_offsets,
    build_rotary,
    compute_frequency_statistics_from_means,
    compute_frequency_scaling,
    determine_rope_style,
    invert_rope,
    load_head_frequency_stats,
    score_keys_for_round,
    verify_rotary_alignment,
)
from .stats_utils import validate_stats_metadata


@dataclass
class TriAttentionConfig:
    """Configuration for TriAttention attention-layer compression."""
    stats_path: Path
    model_path: Path
    device: torch.device
    dtype: torch.dtype
    budget: int
    offset_max_length: int = 65536
    score_aggregation: str = "mean"
    seed: int | None = None
    metadata_expectations: Dict[str, object] | None = None
    normalize_scores: bool = False
    count_prompt_tokens: bool = False
    allow_prefill_compression: bool = False
    divide_length: int = 128  # Compress every N steps (like R-KV's divide_length)
    use_slack_trigger: bool = False  # If True, trigger at budget + divide_length (like generate wrapper)
    per_head_pruning: bool = False  # If True, each KV head selects tokens independently
    per_layer_perhead_pruning: bool = False  # If True, each (layer, KV head) selects tokens independently
    layer_perhead_aggregation: str = "max"  # Aggregation method for per_layer_perhead_pruning: "max" or "mean"
    disable_mlr: bool = False  # If True, use q_abs_mean directly for extra term
    disable_trig: bool = False  # If True, drop position-dependent term (base_scores)


class TriAttention:
    """
    TriAttention compression using attention-layer triggering.

    This class implements compression in the attention layer:
    - Compression is triggered during attention forward when cache >= budget
    - After compression, cache size returns to budget
    - Uses frequency-based scoring for token importance
    """

    def __init__(self, config: TriAttentionConfig) -> None:
        self.config = config
        self.budget = config.budget
        if config.allow_prefill_compression and not config.count_prompt_tokens:
            print(
                "[warn] allow_prefill_compression=True with count_prompt_tokens=False "
                "can delay compression when prefill dominates the cache.",
                file=sys.stderr,
            )

        # Load model config
        model_config = AutoConfig.from_pretrained(
            str(config.model_path), trust_remote_code=True
        )

        # Build default expectations
        rope_scaling = getattr(model_config, "rope_scaling", {}) or {}
        default_expectations: Dict[str, object] = {
            "rope_style": determine_rope_style(model_config),
        }
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or getattr(model_config, "rope_type", None)
        if rope_type:
            default_expectations["rope_type"] = rope_type

        # Load and validate stats
        metadata, stats_map = load_head_frequency_stats(config.stats_path, config.device)
        merged_expectations: Dict[str, object] = {}
        if config.metadata_expectations:
            merged_expectations.update(config.metadata_expectations)
        merged_expectations.update({k: v for k, v in default_expectations.items() if v is not None})
        if merged_expectations:
            validate_stats_metadata(metadata, merged_expectations, stats_path=config.stats_path)

        # Setup sampled heads
        sampled_heads = [tuple(item) for item in metadata.get("sampled_heads", [])]
        if not sampled_heads:
            raise ValueError("Stats file does not contain any sampled heads")
        layer_count = int(getattr(model_config, "num_hidden_layers", len(sampled_heads)))
        filtered_heads = [head for head in sampled_heads if 0 <= head[0] < layer_count]
        if not filtered_heads:
            raise ValueError(f"No valid heads remain after filtering with layer_count={layer_count}")
        self.sampled_heads: List[Tuple[int, int]] = filtered_heads
        self.head_stats: Dict[Tuple[int, int], HeadFrequencyStats] = {
            key: stats_map[key] for key in filtered_heads if key in stats_map
        }

        # Setup rotary embeddings
        self.rotary = build_rotary(config.device, config.model_path, config.dtype, config=model_config)
        self.rope_style = getattr(self.rotary, "_rope_style", "half")
        self.attention_scale = float(getattr(self.rotary, "attention_scaling", 1.0))
        inv_freq = self.rotary.inv_freq.to(device=config.device, dtype=torch.float32)
        self.head_dim = int(metadata.get("head_dim", inv_freq.numel() * 2))
        freq_count = max(1, self.head_dim // 2)
        self.omega = inv_freq[:freq_count]
        self.offsets = build_geometric_offsets(config.offset_max_length, config.device)
        freq_scale = compute_frequency_scaling(self.rotary, self.head_dim, config.dtype, config.device)
        self.freq_scale_sq = freq_scale.pow(2)

        # GQA support
        rope_config = getattr(self.rotary, "config", None)
        self.num_attention_heads = getattr(rope_config, "num_attention_heads", None)
        self.num_key_value_heads = getattr(rope_config, "num_key_value_heads", self.num_attention_heads)
        if self.num_attention_heads and self.num_key_value_heads:
            self.num_key_value_groups = max(1, self.num_attention_heads // self.num_key_value_heads)
        else:
            self.num_key_value_heads = None
            self.num_key_value_groups = None

        # State tracking
        self.cache_positions: List[int] = []
        # Per-head positions: when per_head_pruning is active and compression happens,
        # each KV head may have different token positions at the same cache index.
        # Shape: [num_kv_heads][seq_len] - None until first per-head compression
        self.cache_positions_per_head: Optional[List[List[int]]] = None
        # Per-layer-per-head positions: when per_layer_perhead_pruning is active,
        # each (layer, KV head) has independent token positions.
        # Dict[(layer_idx, kv_head_idx)] -> List[int] - None until first compression
        self.cache_positions_per_layer_perhead: Optional[Dict[Tuple[int, int], List[int]]] = None
        self.absolute_position: int = 0
        self.prefix_length: int = 0
        self.divide_length = config.divide_length
        self.score_aggregation = config.score_aggregation
        self.normalize_scores = config.normalize_scores
        self.use_slack_trigger = config.use_slack_trigger
        self.per_head_pruning = config.per_head_pruning
        self.per_layer_perhead_pruning = config.per_layer_perhead_pruning
        self.disable_mlr = config.disable_mlr
        self.disable_trig = config.disable_trig
        self.allow_prefill_compression = config.allow_prefill_compression

        # Random generator
        self.generator: torch.Generator | None = None
        if config.seed is not None:
            if config.device.type == "cuda":
                self.generator = torch.Generator(device=config.device)
            else:
                self.generator = torch.Generator()
            self.generator.manual_seed(int(config.seed))

    def compute_keep_indices(
        self,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        prefix_length: int = 0,
    ) -> torch.Tensor:
        """
        Compute keep_indices using union-based selection.

        Algorithm:
        1. Compute scores only for decode tokens (not prefill)
        2. Apply normalization only on decode tokens
        3. Add noise for tie-breaking
        4. Union-based selection: each head selects top-k, then select from union

        Prefill tokens (first prefix_length) are always preserved unless
        allow_prefill_compression is enabled.
        Only decode tokens compete for the remaining budget when prefill is pinned.

        Args:
            pkv_tuple: Tuple of (key, value) for each layer
            prefix_length: Number of prefill tokens to always preserve

        Returns:
            keep_indices: Indices of tokens to keep (sorted)
        """
        if not pkv_tuple:
            return torch.arange(0, device=self.config.device)

        kv_cache_len = pkv_tuple[0][0].shape[-2]

        if self.allow_prefill_compression:
            prefix_length = 0

        # Nothing to compress
        if kv_cache_len <= self.budget:
            return torch.arange(kv_cache_len, device=self.config.device)

        # Determine decode range
        decode_start = min(prefix_length, kv_cache_len)
        decode_count = max(0, kv_cache_len - decode_start)

        if decode_count == 0:
            # Only prefill tokens, keep as many as budget allows
            return torch.arange(min(self.budget, kv_cache_len), device=self.config.device)

        # Budget for decode tokens
        decode_budget = max(0, self.budget - decode_start)
        if decode_budget == 0:
            # Budget exhausted by prefill
            return torch.arange(min(self.budget, decode_start), device=self.config.device)

        # Get positions for decode tokens only
        decode_positions = torch.tensor(
            self.cache_positions[decode_start:kv_cache_len],
            device=self.config.device,
            dtype=torch.long
        )

        # Build per-KV-head positions if available (after per-head compression)
        # This is critical for correct RoPE inversion when different KV heads have different tokens
        positions_per_kv_head: Optional[List[torch.Tensor]] = None
        if self.cache_positions_per_head is not None:
            positions_per_kv_head = [
                torch.tensor(
                    head_positions[decode_start:kv_cache_len],
                    device=self.config.device,
                    dtype=torch.long
                )
                for head_positions in self.cache_positions_per_head
            ]

        # Collect scores from all layers' sampled heads (decode tokens only)
        all_head_scores: List[torch.Tensor] = []
        for layer_idx, (key_states, _) in enumerate(pkv_tuple):
            layer_scores = self._compute_layer_head_scores(
                key_states, decode_positions, layer_idx, start_index=decode_start,
                positions_per_kv_head=positions_per_kv_head
            )
            if layer_scores is not None:
                all_head_scores.append(layer_scores)

        if not all_head_scores:
            # No sampled heads, return first budget indices
            prefill_indices = torch.arange(decode_start, device=self.config.device)
            decode_indices = torch.arange(decode_start, min(decode_start + decode_budget, kv_cache_len), device=self.config.device)
            return torch.cat([prefill_indices, decode_indices])

        # Stack all head scores: [total_sampled_heads, decode_seq_len]
        head_matrix = torch.cat(all_head_scores, dim=0)

        # Apply normalization (only on decode tokens)
        if self.normalize_scores and head_matrix.numel() > 0:
            mean = head_matrix.mean(dim=1, keepdim=True)
            std = head_matrix.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
            head_matrix = (head_matrix - mean) / std

        # Add noise for tie-breaking
        if self.generator is not None and head_matrix.numel() > 0:
            noise = torch.rand(
                head_matrix.shape,
                device=head_matrix.device,
                generator=self.generator,
            ) * 1e-6
            head_matrix = head_matrix + noise

        # Per-layer-per-head independent pruning mode: each (layer, KV head) selects independently
        # Returns 3D tensor [num_layers, num_kv_heads, budget]
        # This mode does NOT use the global head_matrix - it computes scores per-layer inside the method
        if self.per_layer_perhead_pruning:
            keep_count = min(decode_budget, decode_count)
            return self._select_per_layer_perhead_independent(
                pkv_tuple, keep_count, decode_start, decode_positions
            )

        # Per-head independent pruning mode: each KV head selects tokens independently
        # Returns 2D tensor [num_kv_heads, budget] instead of 1D [budget]
        # Per-head independent pruning: each KV head selects tokens independently
        if self.per_head_pruning:
            keep_count = min(decode_budget, decode_count)
            return self._select_per_head_independent(
                head_matrix, keep_count, decode_start
            )

        # Compute combined scores for union-based selection
        combined = head_matrix.max(dim=0).values

        # Union-based selection
        keep_count = min(decode_budget, decode_count)
        decode_keep_indices = self._select_union_based(head_matrix, combined, keep_count)

        # Combine prefill (always kept) + selected decode tokens
        prefill_indices = torch.arange(decode_start, device=self.config.device)
        decode_keep_absolute = decode_keep_indices + decode_start
        keep_indices = torch.cat([prefill_indices, decode_keep_absolute])
        keep_indices = torch.sort(keep_indices).values

        return keep_indices

    def _select_union_based(
        self,
        per_head_scores: torch.Tensor,
        combined: torch.Tensor,
        keep_count: int,
    ) -> torch.Tensor:
        """
        Union-based token selection.

        Algorithm:
        1. Each head independently selects top-k tokens
        2. Take union of all heads' selections
        3. From union, select top-k by combined score
        4. If union < k, fill from remaining tokens

        Args:
            per_head_scores: [num_heads, seq_len] scores per head
            combined: [seq_len] aggregated scores
            keep_count: number of tokens to keep

        Returns:
            keep_indices: indices of tokens to keep (relative to decode start)
        """
        candidate_count = combined.shape[0]

        if candidate_count <= keep_count:
            return torch.arange(candidate_count, device=combined.device, dtype=torch.long)

        # Step 1: Each head selects top-k
        per_head_quota = min(keep_count, candidate_count)
        union_mask = torch.zeros(candidate_count, device=combined.device, dtype=torch.bool)

        for head_scores in per_head_scores:
            head_k = min(per_head_quota, head_scores.numel())
            if head_k == 0:
                continue
            top_idx = torch.topk(head_scores, k=head_k, largest=True).indices
            union_mask[top_idx] = True

        # Step 2: Get union indices
        union_indices = torch.nonzero(union_mask, as_tuple=False).view(-1)
        if union_indices.numel() == 0:
            union_indices = torch.arange(0, 0, device=combined.device, dtype=torch.long)

        # Step 3: Select from union by combined score
        if union_indices.numel() >= keep_count:
            subset_scores = combined.index_select(0, union_indices)
            top_subset = torch.topk(subset_scores, k=keep_count, largest=True).indices
            return union_indices.index_select(0, torch.sort(top_subset).values)

        # Step 4: Fill from remaining if union is too small
        remaining = keep_count - union_indices.numel()
        available = candidate_count - union_indices.numel()
        if remaining > 0 and available > 0:
            residual_scores = combined.clone()
            residual_scores[union_mask] = float("-inf")
            extra_k = min(remaining, available)
            extra_indices = torch.topk(residual_scores, k=extra_k, largest=True).indices
            union_indices = torch.cat([union_indices, extra_indices])

        return torch.sort(union_indices).values

    def _select_per_head_independent(
        self,
        head_matrix: torch.Tensor,
        keep_count: int,
        decode_start: int,
    ) -> torch.Tensor:
        """
        Per-head independent token selection with per-layer grouping.

        Each KV head independently selects top-k tokens based on aggregated scores
        from sampled attention heads that map to that KV head.

        Bug fix: Changed aggregation from max(all 196 heads) to mean(max(7 heads per layer)).
        - Before: max over 196 heads (28 layers × 7 heads/layer mixed) ≈ global max
        - After: mean of per-layer max (each layer contributes equally)

        This preserves per-head variance by:
        1. Grouping heads by (layer, kv_head) - 7 heads per group
        2. Computing max within each layer's group - layer-specific importance
        3. Averaging across layers - balanced contribution from all layers

        Args:
            head_matrix: [num_sampled_heads, decode_seq_len] scores per sampled head
            keep_count: Number of decode tokens to keep per KV head
            decode_start: Index where decode tokens start (for prefill expansion)

        Returns:
            keep_indices: 2D tensor [num_kv_heads, decode_start + keep_count] (absolute indices)
        """
        # Group sampled attention heads by (layer, kv_head) - each group has ~7 heads
        kv_head_groups: Dict[Tuple[int, int], List[int]] = {}
        for i, (layer, attn_head) in enumerate(self.sampled_heads):
            kv_head = attn_head // max(1, self.num_key_value_groups)
            group_key = (layer, kv_head)
            if group_key not in kv_head_groups:
                kv_head_groups[group_key] = []
            kv_head_groups[group_key].append(i)

        # Get unique layers from sampled_heads
        unique_layers = sorted(set(l for l, _ in self.sampled_heads))
        num_layers = len(unique_layers)

        # For each KV head, compute mean of per-layer max scores
        decode_keep_list: List[torch.Tensor] = []
        for kv_head_idx in range(self.num_key_value_heads):
            # Collect per-layer max scores for this KV head
            layer_max_scores: List[torch.Tensor] = []
            for layer_idx in unique_layers:
                group_key = (layer_idx, kv_head_idx)
                if group_key in kv_head_groups:
                    indices = kv_head_groups[group_key]
                    group_scores = head_matrix[indices]  # [~7, seq_len]
                    layer_max = group_scores.max(dim=0).values  # [seq_len]
                    layer_max_scores.append(layer_max)

            if layer_max_scores:
                # Stack and compute mean across layers
                stacked = torch.stack(layer_max_scores, dim=0)  # [num_layers, seq_len]
                aggregated = stacked.mean(dim=0)  # [seq_len] - mean of per-layer max
            else:
                # Fallback for KV heads without sampled heads: use mean of all scores
                aggregated = head_matrix.mean(dim=0)

            # Independent top-k selection for this KV head
            actual_keep = min(keep_count, aggregated.numel())
            if actual_keep > 0:
                keep_indices_for_head = aggregated.topk(actual_keep, largest=True).indices
            else:
                keep_indices_for_head = torch.empty(0, device=head_matrix.device, dtype=torch.long)
            decode_keep_list.append(keep_indices_for_head)

        # Stack into 2D tensor: [num_kv_heads, keep_count] (relative to decode_start)
        decode_keep_indices = torch.stack(decode_keep_list, dim=0)

        # Expand prefill indices to 2D: [num_kv_heads, decode_start]
        prefill_indices = torch.arange(decode_start, device=self.config.device, dtype=torch.long)
        prefill_broadcast = prefill_indices.unsqueeze(0).expand(self.num_key_value_heads, -1)

        # Convert decode indices to absolute and concatenate with prefill
        decode_keep_absolute = decode_keep_indices + decode_start
        keep_indices = torch.cat([prefill_broadcast, decode_keep_absolute], dim=1)

        return keep_indices

    def _select_per_layer_perhead_independent(
        self,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        keep_count: int,
        decode_start: int,
        decode_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-layer-per-head independent token selection.

        Each (layer, KV head) independently selects top-k tokens based on that layer's
        sampled attention heads only. This provides maximum independence - no cross-layer
        aggregation of scores.

        Args:
            pkv_tuple: Tuple of (key, value) for each layer
            keep_count: Number of decode tokens to keep per (layer, KV head)
            decode_start: Index where decode tokens start (for prefill expansion)
            decode_positions: Absolute positions of decode tokens

        Returns:
            keep_indices: 3D tensor [num_layers, num_kv_heads, decode_start + keep_count]
        """
        num_layers = len(pkv_tuple)
        all_layer_indices: List[torch.Tensor] = []

        for layer_idx, (key_states, _) in enumerate(pkv_tuple):
            # Build per-KV-head positions for this layer (if available after first compression)
            layer_positions_per_kv_head: Optional[List[torch.Tensor]] = None
            if self.cache_positions_per_layer_perhead is not None:
                layer_positions_per_kv_head = [
                    torch.tensor(
                        self.cache_positions_per_layer_perhead[(layer_idx, kv)],
                        device=self.config.device,
                        dtype=torch.long
                    )[decode_start:]
                    for kv in range(self.num_key_value_heads)
                ]

            # Compute scores for this layer only (using only this layer's sampled heads)
            layer_scores = self._compute_layer_head_scores(
                key_states, decode_positions, layer_idx, start_index=decode_start,
                positions_per_kv_head=layer_positions_per_kv_head
            )

            if layer_scores is None or layer_scores.numel() == 0:
                # Fallback: no sampled heads for this layer, use uniform selection
                layer_keep = torch.arange(keep_count, device=self.config.device, dtype=torch.long)
                layer_keep = layer_keep.unsqueeze(0).expand(self.num_key_value_heads, -1)
            else:
                # Normalize scores for this layer independently
                if self.normalize_scores and layer_scores.numel() > 0:
                    mean = layer_scores.mean(dim=1, keepdim=True)
                    std = layer_scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
                    layer_scores = (layer_scores - mean) / std

                # Add noise for tie-breaking
                if self.generator is not None and layer_scores.numel() > 0:
                    noise = torch.rand(
                        layer_scores.shape,
                        device=layer_scores.device,
                        generator=self.generator,
                    ) * 1e-6
                    layer_scores = layer_scores + noise

                # Group this layer's sampled heads by KV head and aggregate (max)
                layer_heads = [(l, h) for l, h in self.sampled_heads if l == layer_idx]
                kv_head_groups: Dict[int, List[int]] = {}
                for i, (_, attn_head) in enumerate(layer_heads):
                    kv_head = attn_head // max(1, self.num_key_value_groups)
                    if kv_head not in kv_head_groups:
                        kv_head_groups[kv_head] = []
                    kv_head_groups[kv_head].append(i)

                # Per-KV-head selection for this layer
                layer_keep_list: List[torch.Tensor] = []
                for kv_head_idx in range(self.num_key_value_heads):
                    if kv_head_idx in kv_head_groups:
                        indices = kv_head_groups[kv_head_idx]
                        group_scores = layer_scores[indices]  # [~7 heads, seq_len]
                        if self.config.layer_perhead_aggregation == "mean":
                            aggregated = group_scores.mean(dim=0)  # [seq_len]
                        else:
                            aggregated = group_scores.max(dim=0).values  # [seq_len]
                    else:
                        # Fallback: use mean of all layer scores
                        aggregated = layer_scores.mean(dim=0)

                    actual_keep = min(keep_count, aggregated.numel())
                    if actual_keep > 0:
                        keep_indices_for_head = aggregated.topk(actual_keep, largest=True).indices
                    else:
                        keep_indices_for_head = torch.empty(0, device=self.config.device, dtype=torch.long)
                    layer_keep_list.append(keep_indices_for_head)

                layer_keep = torch.stack(layer_keep_list, dim=0)  # [num_kv_heads, keep_count]

            all_layer_indices.append(layer_keep)

        # Stack all layers: [num_layers, num_kv_heads, keep_count]
        decode_keep_indices = torch.stack(all_layer_indices, dim=0)

        # Expand prefill indices to 3D: [num_layers, num_kv_heads, decode_start]
        prefill_indices = torch.arange(decode_start, device=self.config.device, dtype=torch.long)
        prefill_broadcast = prefill_indices.unsqueeze(0).unsqueeze(0).expand(
            num_layers, self.num_key_value_heads, -1
        )

        # Convert decode indices to absolute and concatenate with prefill
        decode_keep_absolute = decode_keep_indices + decode_start
        keep_indices = torch.cat([prefill_broadcast, decode_keep_absolute], dim=2)

        return keep_indices

    def _compute_layer_head_scores(
        self,
        key_states: torch.Tensor,
        key_positions: torch.Tensor,
        layer_idx: int,
        start_index: int = 0,
        positions_per_kv_head: Optional[List[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute per-head frequency scores for a single layer.

        Args:
            key_states: Key tensor from the cache [batch, num_heads, seq_len, head_dim]
            key_positions: Absolute positions of the tokens to score (used when positions_per_kv_head is None)
            layer_idx: Which layer this is
            start_index: Starting index in key_states to gather from (for decode-only scoring)
            positions_per_kv_head: Optional per-KV-head positions [num_kv_heads][seq_len].
                                   When provided, each KV head uses its own position array for RoPE inversion.
                                   This is critical after per-head compression where different KV heads
                                   have different tokens at the same cache index.

        Returns:
            Tensor of shape [num_heads_in_layer, seq_len] or None if no sampled heads
        """
        # Collect scores from sampled heads in this layer
        layer_heads = [(l, h) for l, h in self.sampled_heads if l == layer_idx]
        if not layer_heads:
            return None

        # Build gather indices for extracting keys from cache
        seq_len = key_positions.shape[0]
        gather_indices = torch.arange(seq_len, device=self.config.device, dtype=torch.long) + start_index

        # Pre-compute RoPE tables per KV head if using per-head positions
        # Otherwise use shared positions for all heads
        if positions_per_kv_head is not None:
            # Per-head mode: compute cos/sin tables for each KV head's positions
            cos_sin_per_kv_head: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            base = torch.zeros(1, seq_len, self.head_dim,
                              device=self.config.device, dtype=self.config.dtype)
            for kv_head_idx, kv_positions in enumerate(positions_per_kv_head):
                cos, sin = self.rotary(base, kv_positions.unsqueeze(0))
                cos_sin_per_kv_head[kv_head_idx] = (cos[0], sin[0])
        else:
            # Shared mode: single cos/sin table for all heads
            base = torch.zeros(1, key_positions.shape[0], self.head_dim,
                              device=self.config.device, dtype=self.config.dtype)
            cos, sin = self.rotary(base, key_positions.unsqueeze(0))
            shared_cos_table, shared_sin_table = cos[0], sin[0]

        per_head_scores: List[torch.Tensor] = []
        for layer, head in layer_heads:
            stats = self.head_stats[(layer, head)]

            # Get key values for this head (handle GQA)
            kv_head = head
            if self.num_key_value_heads and self.num_attention_heads:
                kv_head = min(key_states.shape[1] - 1, head // max(1, self.num_key_value_groups))

            # Gather keys at specified indices
            k_values = key_states[0, kv_head].index_select(0, gather_indices)
            k_values = k_values.to(device=self.config.device, dtype=self.config.dtype)

            # Get appropriate RoPE tables and positions for this KV head
            if positions_per_kv_head is not None:
                cos_table, sin_table = cos_sin_per_kv_head[kv_head]
                head_key_positions = positions_per_kv_head[kv_head]
            else:
                cos_table, sin_table = shared_cos_table, shared_sin_table
                head_key_positions = key_positions

            # Invert RoPE
            k_unrot = invert_rope(k_values, cos_table, sin_table, self.attention_scale, style=self.rope_style)

            # Compute frequency statistics
            amp, phi, extra = compute_frequency_statistics_from_means(
                stats.q_mean_complex,
                stats.q_abs_mean,
                k_unrot,
                style=self.rope_style,
                disable_mlr=self.disable_mlr,
            )

            # Score keys
            head_scores = score_keys_for_round(
                key_indices=head_key_positions,
                round_start=self.absolute_position,
                amp=amp,
                phi=phi,
                omega=self.omega,
                extra=extra,
                offsets=self.offsets,
                aggregation=self.score_aggregation,
                freq_scale_sq=self.freq_scale_sq,
                disable_trig=self.disable_trig,
            )
            per_head_scores.append(head_scores)

        if not per_head_scores:
            return None

        return torch.stack(per_head_scores, dim=0)  # [num_heads, seq_len]

    def reset_compression_state(self) -> None:
        """Reset state for new generation."""
        self.cache_positions = []
        self.cache_positions_per_head = None
        self.cache_positions_per_layer_perhead = None
        self.absolute_position = 0
        self.prefix_length = 0
        # Reset generator to initial seed (aligned with original generate wrapper
        # which recreates the compression module for each generation)
        if self.config.seed is not None:
            if self.generator is None:
                if self.config.device.type == "cuda":
                    self.generator = torch.Generator(device=self.config.device)
                else:
                    self.generator = torch.Generator()
            self.generator.manual_seed(int(self.config.seed))


def apply_triattention_patch(
    model,
    *,
    stats_path: Path,
    model_path: Path,
    kv_budget: int,
    offset_max_length: int = 65536,
    score_aggregation: str = "mean",
    pruning_seed: int = 0,
    metadata_expectations: Dict[str, object] | None = None,
    normalize_scores: bool = False,
    count_prompt_tokens: bool = False,
    allow_prefill_compression: bool = False,
    divide_length: int = 128,
    use_slack_trigger: bool = False,
    per_head_pruning: bool = False,
    per_layer_perhead_pruning: bool = False,
    layer_perhead_aggregation: str = "max",
    disable_mlr: bool = False,
    disable_trig: bool = False,
) -> None:
    """
    Apply TriAttention compression patch.

    This patches the model to use attention-layer compression instead of
    generate() wrapper compression. The scoring logic remains frequency-based.
    """
    device = next(model.parameters()).device
    dtype = torch.float32

    config = TriAttentionConfig(
        stats_path=stats_path,
        model_path=model_path,
        device=device,
        dtype=dtype,
        budget=kv_budget,
        offset_max_length=offset_max_length,
        score_aggregation=score_aggregation,
        seed=pruning_seed,
        metadata_expectations=metadata_expectations,
        normalize_scores=normalize_scores,
        count_prompt_tokens=count_prompt_tokens,
        allow_prefill_compression=allow_prefill_compression,
        divide_length=divide_length,
        use_slack_trigger=use_slack_trigger,
        per_head_pruning=per_head_pruning,
        per_layer_perhead_pruning=per_layer_perhead_pruning,
        layer_perhead_aggregation=layer_perhead_aggregation,
        disable_mlr=disable_mlr,
        disable_trig=disable_trig,
    )

    compressor = TriAttention(config)

    # Verify rotary alignment
    model_rotary_emb = None
    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            if len(layers) > 0 and hasattr(layers[0], "self_attn"):
                attn = layers[0].self_attn
                if hasattr(attn, "rotary_emb"):
                    model_rotary_emb = attn.rotary_emb
    except Exception:
        pass

    if model_rotary_emb is not None:
        verify_rotary_alignment(compressor.rotary, model_rotary_emb)
    else:
        print("[TriAttention] WARNING: Could not locate model rotary_emb for alignment verification.")

    # Store compressor on model for access during forward
    model._triattention_compressor = compressor

    # Patch model.forward to apply compression after each forward pass
    orig_forward = model.forward

    def triattention_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        comp = self._triattention_compressor
        cache_position_override = cache_position
        position_ids_override = position_ids
        attention_mask_override = attention_mask

        # Check if this is a new generation (empty cache) BEFORE computing positions
        # This fixes a bug where comp.absolute_position was stale from previous generation
        is_empty_cache = True
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                if past_key_values.get_seq_length() > 0:
                    is_empty_cache = False
            elif isinstance(past_key_values, (tuple, list)):
                if len(past_key_values) > 0 and past_key_values[0][0].shape[2] > 0:
                    is_empty_cache = False

        if is_empty_cache:
            comp.reset_compression_state()

        if past_key_values is not None and input_ids is not None and not is_empty_cache:
            bsz, step = input_ids.shape

            # Absolute positions for rotary
            start_pos = comp.absolute_position
            abs_positions = torch.arange(
                start_pos, start_pos + step,
                device=input_ids.device, dtype=torch.long,
            ).unsqueeze(0)
            if bsz > 1:
                abs_positions = abs_positions.expand(bsz, -1)
            position_ids_override = abs_positions

            # Relative positions for cache placement
            current_cache_len = None
            if isinstance(past_key_values, Cache) and hasattr(past_key_values, "get_seq_length"):
                current_cache_len = int(past_key_values.get_seq_length())
            elif isinstance(past_key_values, (tuple, list)) and past_key_values:
                current_cache_len = int(past_key_values[0][0].shape[2])

            if current_cache_len is not None:
                # cache_position should be 1D tensor [step], not 2D
                cache_position_override = torch.arange(
                    current_cache_len, current_cache_len + step,
                    device=input_ids.device, dtype=torch.long,
                )

            attention_mask_override = None
        else:
            cache_position_override = None

        # print current sequence for debugging
        if input_ids is not None:
            current_seq = input_ids[0].cpu().numpy().tolist()
            import sys
            # sys.stderr.write(f"[TriAttention] Current sequence: {current_seq}\n")    

        outputs = orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask_override,
            position_ids=position_ids_override,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position_override,
            **kwargs,
        )

        # # print top5 logits for debugging
        # if hasattr(outputs, "logits") and outputs.logits is not None:
        #     top5_logits = outputs.logits.topk(5, dim=-1).values
        #     import sys
        #     sys.stderr.write(f"[TriAttention] Top 5 logits: {top5_logits}\n")
        

        if getattr(outputs, "past_key_values", None) is None:
            return outputs

        # Note: reset_compression_state() is now called BEFORE forward (at the top)
        # to ensure correct position_ids computation

        # Convert cache to tuple for manipulation
        pkv = outputs.past_key_values
        if isinstance(pkv, Cache):
            pkv_tuple = pkv.to_legacy_cache()
        else:
            pkv_tuple = tuple(pkv) if pkv else ()

        if not pkv_tuple:
            return outputs

        # Track positions and apply compression
        seq_len = pkv_tuple[0][0].shape[2]
        cached_len = len(comp.cache_positions)

        is_decode_step = False
        if cached_len == 0:
            # First forward (prefill)
            comp.cache_positions = list(range(seq_len))
            comp.absolute_position = seq_len
            comp.prefix_length = seq_len
        elif cached_len < seq_len:
            # Decode step: add new positions
            is_decode_step = True
            added = seq_len - cached_len
            new_positions = list(range(comp.absolute_position, comp.absolute_position + added))
            comp.cache_positions.extend(new_positions)
            # Also extend per-head positions if active (all heads get same new tokens)
            if comp.cache_positions_per_head is not None:
                for head_positions in comp.cache_positions_per_head:
                    head_positions.extend(new_positions)
            # Also extend per-layer-per-head positions if active
            if comp.cache_positions_per_layer_perhead is not None:
                for key in comp.cache_positions_per_layer_perhead:
                    comp.cache_positions_per_layer_perhead[key].extend(new_positions)
            comp.absolute_position += added

        # Apply compression based on trigger mode
        effective_size = seq_len
        if not comp.config.count_prompt_tokens:
            effective_size = max(0, seq_len - comp.prefix_length)

        if comp.use_slack_trigger:
            # Mimic generate-wrapper behavior: allow cache to grow to budget + divide_length, then prune
            trigger_threshold = comp.budget + comp.divide_length
            should_compress = is_decode_step and effective_size >= trigger_threshold
        else:
            # Original R-KV style: compress when cache hits budget, gated by divide_length interval
            trigger_threshold = comp.budget
            should_compress = (
                is_decode_step
                and effective_size >= trigger_threshold
                and (comp.absolute_position % comp.divide_length == 0)
            )

        # import sys
        # sys.stderr.write(f"[TriAttention] Effective size: {effective_size}, Should compress: {should_compress}\n")

        if should_compress:
            # Compute keep_indices using scores from ALL layers' sampled heads
            # Prefill tokens are always preserved, only decode tokens are compressed
            keep_indices = comp.compute_keep_indices(pkv_tuple, prefix_length=comp.prefix_length)

            # Handle 3D per-layer-per-head, 2D per-head, or 1D global indices
            if keep_indices.dim() == 3:
                # Per-layer-per-head mode: keep_indices shape [num_layers, num_kv_heads, budget]
                # sys.stderr.write(f"[TriAttention] Per-layer-per-head compressed size: {keep_indices.shape}\n")
                num_layers = keep_indices.size(0)
                num_kv_heads = keep_indices.size(1)
                budget = keep_indices.size(2)

                new_pkv = []
                for layer_idx, (k, v) in enumerate(pkv_tuple):
                    batch_size = k.size(0)
                    head_dim = k.size(3)
                    layer_indices = keep_indices[layer_idx]  # [num_kv_heads, budget]

                    # Expand indices for gather: [batch, num_kv_heads, budget, head_dim]
                    expanded_indices = layer_indices.unsqueeze(0).unsqueeze(-1).expand(
                        batch_size, num_kv_heads, budget, head_dim
                    )

                    # Gather along sequence dimension (dim=2)
                    k_new = k.gather(dim=2, index=expanded_indices)
                    v_new = v.gather(dim=2, index=expanded_indices)
                    new_pkv.append((k_new.contiguous(), v_new.contiguous()))
                pkv_tuple = tuple(new_pkv)

                # Update cache_positions_per_layer_perhead: each (layer, KV head) has its own position list
                if comp.cache_positions_per_layer_perhead is None:
                    # First compression: initialize from shared cache_positions
                    comp.cache_positions_per_layer_perhead = {
                        (layer_idx, kv_head): [comp.cache_positions[idx] for idx in keep_indices[layer_idx, kv_head].tolist()]
                        for layer_idx in range(num_layers)
                        for kv_head in range(num_kv_heads)
                    }
                else:
                    # Subsequent compression: update each (layer, kv_head)'s positions
                    comp.cache_positions_per_layer_perhead = {
                        (layer_idx, kv_head): [
                            comp.cache_positions_per_layer_perhead[(layer_idx, kv_head)][idx]
                            for idx in keep_indices[layer_idx, kv_head].tolist()
                        ]
                        for layer_idx in range(num_layers)
                        for kv_head in range(num_kv_heads)
                    }
                # Keep cache_positions as (layer 0, head 0)'s for compatibility (used for length tracking)
                comp.cache_positions = comp.cache_positions_per_layer_perhead[(0, 0)].copy()

            elif keep_indices.dim() == 2:
                # Per-head mode: keep_indices shape [num_kv_heads, budget]
                # Use gather-based slicing for per-head independent compression
                # sys.stderr.write(f"[TriAttention] Per-head compressed size: {keep_indices.shape}\n")
                new_pkv = []
                for k, v in pkv_tuple:
                    batch_size = k.size(0)
                    num_kv_heads = k.size(1)
                    budget = keep_indices.size(1)
                    head_dim = k.size(3)

                    # Expand indices for gather: [batch, num_kv_heads, budget, head_dim]
                    expanded_indices = keep_indices.unsqueeze(0).unsqueeze(-1).expand(
                        batch_size, num_kv_heads, budget, head_dim
                    )

                    # Gather along sequence dimension (dim=2)
                    k_new = k.gather(dim=2, index=expanded_indices)
                    v_new = v.gather(dim=2, index=expanded_indices)
                    new_pkv.append((k_new.contiguous(), v_new.contiguous()))
                pkv_tuple = tuple(new_pkv)

                # Update cache_positions_per_head: each KV head has its own position list
                # This is critical for correct RoPE inversion in subsequent scoring rounds
                if comp.cache_positions_per_head is None:
                    # First per-head compression: initialize from shared cache_positions
                    comp.cache_positions_per_head = [
                        [comp.cache_positions[idx] for idx in keep_indices[kv_head].tolist()]
                        for kv_head in range(num_kv_heads)
                    ]
                else:
                    # Subsequent compression: update each head's positions
                    comp.cache_positions_per_head = [
                        [comp.cache_positions_per_head[kv_head][idx] for idx in keep_indices[kv_head].tolist()]
                        for kv_head in range(num_kv_heads)
                    ]
                # Keep cache_positions as head 0's for compatibility (used for length tracking)
                comp.cache_positions = comp.cache_positions_per_head[0].copy()
            else:
                # Global mode: 1D keep_indices shape [budget]
                # Use index_select (existing behavior)
                # sys.stderr.write(f"[TriAttention] Global compressed size: {len(keep_indices)}\n")
                new_pkv = []
                for k, v in pkv_tuple:
                    k_new = k.index_select(2, keep_indices)
                    v_new = v.index_select(2, keep_indices)
                    new_pkv.append((k_new, v_new))
                pkv_tuple = tuple(new_pkv)

                # Update cache_positions
                comp.cache_positions = [comp.cache_positions[i] for i in keep_indices.tolist()]

        # Convert back to original cache type
        if isinstance(outputs.past_key_values, Cache):
            new_cache = DynamicCache.from_legacy_cache(pkv_tuple)
        else:
            new_cache = pkv_tuple

        outputs = CausalLMOutputWithPast(
            loss=getattr(outputs, "loss", None),
            logits=outputs.logits,
            past_key_values=new_cache,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
        return outputs

    model.forward = MethodType(triattention_forward, model)

    print(f"[TriAttention] Applied compression (budget={kv_budget}, "
          f"divide_length={divide_length}, normalize_scores={normalize_scores}, "
          f"per_head_pruning={per_head_pruning}, "
          f"per_layer_perhead_pruning={per_layer_perhead_pruning})")
