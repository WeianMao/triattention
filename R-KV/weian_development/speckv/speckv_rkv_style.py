"""SpeckV implementation using R-KV style attention-layer compression.

This module provides an alternative SpeckV implementation that triggers compression
inside the attention forward pass (like R-KV) instead of in the generate() wrapper.
The frequency-based scoring logic is identical to SparseRoundPruner.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

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
    verify_rotary_alignment,
)
from weian_development.speckv.stats_utils import validate_stats_metadata


@dataclass
class SpeckVRKVStyleConfig:
    """Configuration for R-KV style SpeckV compression."""
    stats_path: Path
    model_path: Path
    device: torch.device
    dtype: torch.dtype
    budget: int
    window_size: int = 8  # For R-KV compatibility: last N tokens always retained
    offset_max_length: int = 65536
    score_aggregation: str = "mean"
    seed: int | None = None
    head_limit: int | None = None
    metadata_expectations: Dict[str, object] | None = None
    normalize_scores: bool = False
    use_rank_aggregation: bool = False
    include_prefill_in_budget: bool = False


class SpeckVRKVStyle:
    """
    SpeckV compression using R-KV style attention-layer triggering.

    This class mimics R-KV's compression pattern:
    - Compression is triggered during attention forward when cache >= budget
    - After compression, cache size returns to budget
    - Uses SpeckV's frequency-based scoring instead of R-KV's attention+similarity
    """

    def __init__(self, config: SpeckVRKVStyleConfig) -> None:
        self.config = config
        self.budget = config.budget
        self.window_size = config.window_size

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
        if config.head_limit is not None and config.head_limit > 0:
            filtered_heads = filtered_heads[:config.head_limit]
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
        self.absolute_position: int = 0
        self.prefix_length: int = 0
        self.score_aggregation = config.score_aggregation
        self.normalize_scores = config.normalize_scores
        self.use_rank_aggregation = config.use_rank_aggregation

        # Random generator
        self.generator: torch.Generator | None = None
        if config.seed is not None:
            if config.device.type == "cuda":
                self.generator = torch.Generator(device=config.device)
            else:
                self.generator = torch.Generator()
            self.generator.manual_seed(int(config.seed))

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        R-KV style update: compress KV cache based on frequency scores.

        Note: Position tracking is handled by the forward patch, not here.

        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            query_states: [batch, num_heads, seq_len, head_dim] (unused, for API compat)
            value_states: [batch, num_heads, seq_len, head_dim]
            layer_idx: Current layer index

        Returns:
            Compressed (key_states, value_states)
        """
        kv_cache_len = key_states.shape[-2]

        # Ensure cache_positions is synced with key_states length
        if len(self.cache_positions) < kv_cache_len:
            # Add missing positions
            missing = kv_cache_len - len(self.cache_positions)
            new_positions = list(range(self.absolute_position, self.absolute_position + missing))
            self.cache_positions.extend(new_positions)
            self.absolute_position += missing

        # Compute frequency-based scores for all positions
        key_positions = torch.tensor(self.cache_positions[:kv_cache_len], device=self.config.device, dtype=torch.long)
        scores = self._compute_scores(key_states, key_positions, layer_idx)

        # Compress to (budget - 2*window_size) so next compression triggers after ~2*window_size steps
        # Clamp window_size to not exceed cache length
        effective_window = min(self.window_size, kv_cache_len)
        compress_interval = effective_window * 2  # Every 2*window_size tokens (64 if window=32)
        target_size = self.budget - compress_interval
        keep_count = target_size - effective_window

        # Handle edge case where window_size >= budget
        if keep_count <= 0:
            # Only keep the recent window
            recent_start = kv_cache_len - effective_window
            recent_indices = torch.arange(recent_start, kv_cache_len, device=self.config.device)
            key_states = key_states.index_select(2, recent_indices)
            value_states = value_states.index_select(2, recent_indices)
            if layer_idx == 0:
                self.cache_positions = [self.cache_positions[i] for i in recent_indices.tolist()]
            return key_states, value_states

        # Split into past (can be pruned) and recent (always kept)
        past_len = kv_cache_len - effective_window
        past_scores = scores[:past_len] if past_len > 0 else scores[:0]

        if past_scores.numel() <= keep_count:
            return key_states, value_states

        # Select top-k indices from past tokens
        topk_indices = torch.topk(past_scores, k=keep_count, largest=True).indices
        topk_indices_sorted = torch.sort(topk_indices).values

        # Add recent window indices
        recent_start = kv_cache_len - effective_window
        recent_indices = torch.arange(recent_start, kv_cache_len, device=self.config.device)
        keep_indices = torch.cat([topk_indices_sorted, recent_indices])

        # Slice cache
        key_states = key_states.index_select(2, keep_indices)
        value_states = value_states.index_select(2, keep_indices)

        # Update position tracking (only on first layer to avoid redundant updates)
        if layer_idx == 0:
            self.cache_positions = [self.cache_positions[i] for i in keep_indices.tolist()]

        return key_states, value_states

    def _compute_scores(
        self,
        key_states: torch.Tensor,
        key_positions: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute frequency-based scores for all positions."""
        # Get rotary cos/sin for positions
        base = torch.zeros(1, key_positions.shape[0], self.head_dim,
                          device=self.config.device, dtype=self.config.dtype)
        cos, sin = self.rotary(base, key_positions.unsqueeze(0))
        cos_table, sin_table = cos[0], sin[0]

        # Collect scores from sampled heads in this layer
        layer_heads = [(l, h) for l, h in self.sampled_heads if l == layer_idx]
        if not layer_heads:
            # No sampled heads for this layer, return uniform scores
            return torch.ones(key_positions.shape[0], device=self.config.device)

        per_head_scores: List[torch.Tensor] = []
        for layer, head in layer_heads:
            stats = self.head_stats[(layer, head)]

            # Get key values for this head
            kv_head = head
            if self.num_key_value_heads and self.num_attention_heads:
                kv_head = min(key_states.shape[1] - 1, head // max(1, self.num_key_value_groups))

            k_values = key_states[0, kv_head].to(device=self.config.device, dtype=self.config.dtype)

            # Invert RoPE
            k_unrot = invert_rope(k_values, cos_table, sin_table, self.attention_scale, style=self.rope_style)

            # Compute frequency statistics
            amp, phi, extra = compute_frequency_statistics_from_means(
                stats.q_mean_complex, stats.q_abs_mean, k_unrot, style=self.rope_style
            )

            # Score keys
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

        # Aggregate across heads
        head_matrix = torch.stack(per_head_scores, dim=0)

        if self.use_rank_aggregation:
            ranks = torch.argsort(torch.argsort(head_matrix, dim=1, descending=True), dim=1)
            head_matrix = ranks.float()
            combined = -head_matrix.min(dim=0).values  # Negate for topk
        elif self.normalize_scores:
            mean = head_matrix.mean(dim=1, keepdim=True)
            std = head_matrix.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
            head_matrix = (head_matrix - mean) / std
            combined = head_matrix.max(dim=0).values
        else:
            combined = head_matrix.max(dim=0).values

        return combined

    def reset_compression_state(self) -> None:
        """Reset state for new generation."""
        self.cache_positions = []
        self.absolute_position = 0
        self.prefix_length = 0


def apply_speckv_rkv_style_patch(
    model,
    *,
    stats_path: Path,
    model_path: Path,
    kv_budget: int,
    window_size: int = 8,
    offset_max_length: int = 65536,
    score_aggregation: str = "mean",
    sparse_seed: int = 0,
    head_limit: Optional[int] = None,
    metadata_expectations: Dict[str, object] | None = None,
    normalize_scores: bool = False,
    use_rank_aggregation: bool = False,
    include_prefill_in_budget: bool = False,
) -> None:
    """
    Apply SpeckV with R-KV style compression triggering.

    This patches the model to use attention-layer compression instead of
    generate() wrapper compression. The scoring logic remains frequency-based.
    """
    device = next(model.parameters()).device
    dtype = torch.float32

    config = SpeckVRKVStyleConfig(
        stats_path=stats_path,
        model_path=model_path,
        device=device,
        dtype=dtype,
        budget=kv_budget,
        window_size=window_size,
        offset_max_length=offset_max_length,
        score_aggregation=score_aggregation,
        seed=sparse_seed,
        head_limit=head_limit,
        metadata_expectations=metadata_expectations,
        normalize_scores=normalize_scores,
        use_rank_aggregation=use_rank_aggregation,
        include_prefill_in_budget=include_prefill_in_budget,
    )

    compressor = SpeckVRKVStyle(config)

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
        print("[SpeckV-RKV] WARNING: Could not locate model rotary_emb for alignment verification.")

    # Store compressor on model for access during forward
    model._speckv_rkv_compressor = compressor

    # Patch model.forward to apply compression after each forward pass
    orig_forward = model.forward

    def speckv_rkv_forward(
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
        comp = self._speckv_rkv_compressor
        cache_position_override = cache_position
        position_ids_override = position_ids
        attention_mask_override = attention_mask

        if past_key_values is not None and input_ids is not None:
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
                rel_positions = torch.arange(
                    current_cache_len, current_cache_len + step,
                    device=input_ids.device, dtype=torch.long,
                ).unsqueeze(0)
                if bsz > 1:
                    rel_positions = rel_positions.expand(bsz, -1)
                cache_position_override = rel_positions

            attention_mask_override = None
        else:
            cache_position_override = None

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

        if getattr(outputs, "past_key_values", None) is None:
            return outputs

        # Reset compressor state if starting a new generation
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

        if cached_len == 0:
            # First forward (prefill)
            comp.cache_positions = list(range(seq_len))
            comp.absolute_position = seq_len
            comp.prefix_length = seq_len
        elif cached_len < seq_len:
            # Decode step: add new positions
            added = seq_len - cached_len
            new_positions = list(range(comp.absolute_position, comp.absolute_position + added))
            comp.cache_positions.extend(new_positions)
            comp.absolute_position += added

        # Apply compression when cache exceeds budget
        effective_size = seq_len
        if not comp.config.include_prefill_in_budget:
            effective_size = max(0, seq_len - comp.prefix_length)

        if effective_size > comp.budget:
            # Compress each layer
            new_pkv = []
            for layer_idx, (k, v) in enumerate(pkv_tuple):
                k_new, v_new = comp.update_kv(k, None, v, layer_idx)
                new_pkv.append((k_new, v_new))
            pkv_tuple = tuple(new_pkv)

            # Sync cache_positions with new length
            new_len = pkv_tuple[0][0].shape[2]
            if len(comp.cache_positions) > new_len:
                comp.cache_positions = comp.cache_positions[:new_len]

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

    model.forward = MethodType(speckv_rkv_forward, model)

    print(f"[SpeckV-RKV] Applied R-KV style compression (budget={kv_budget}, window={window_size}, "
          f"normalize_scores={normalize_scores}, use_rank_aggregation={use_rank_aggregation})")
