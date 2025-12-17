"""SpeckV integration that runs inside HF generate by pruning past_key_values per step."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Optional

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from weian_development.speckv.sparse_round_pruner_prefill_keep import (
    SparsePruningConfig,
    SparseRoundPruner,
)
from weian_development.speckv.round_pruning_utils import verify_rotary_alignment


@dataclass
class _SpeckVState:
    pruner: SparseRoundPruner
    attached: bool = False
    config: SparsePruningConfig | None = None
    initial_prefix_length: int | None = None


def _cache_to_tuple(past_key_values) -> tuple:
    if isinstance(past_key_values, Cache):
        return past_key_values.to_legacy_cache()
    if isinstance(past_key_values, tuple):
        return past_key_values
    if isinstance(past_key_values, list):
        return tuple(past_key_values)
    raise TypeError(f"Unsupported cache type: {type(past_key_values)}")


def _cache_from_tuple(pkv_tuple: tuple, template):
    if isinstance(template, Cache):
        return DynamicCache.from_legacy_cache(pkv_tuple)
    if isinstance(template, tuple):
        return pkv_tuple
    if template is None:
        return pkv_tuple
    return pkv_tuple


def apply_speckv_generate_patch(
    model,
    *,
    stats_path: Path,
    model_path: Path,
    kv_budget: int,
    round_window: int,
    offset_max_length: int,
    score_aggregation: str,
    sparse_seed: int,
    head_limit: Optional[int],
    metadata_expectations: dict[str, object] | None = None,
    normalize_scores: bool = False,
    use_rank_aggregation: bool = False,
    sparse_use_similarity: bool = False,
    sparse_similarity_mix_lambda: float = 0.1,
    use_rank_similarity_combination: bool = False,
) -> None:
    """Attach SparseRoundPruner to a CausalLM model so HF generate can be used."""
    device = next(model.parameters()).device
    dtype = torch.float32

    pruner_cfg = SparsePruningConfig(
        stats_path=stats_path,
        model_path=model_path,
        device=device,
        dtype=dtype,
        max_keys=kv_budget,
        round_window=round_window,
        offset_max_length=offset_max_length,
        score_aggregation=score_aggregation,
        seed=sparse_seed,
        head_limit=head_limit,
        metadata_expectations=metadata_expectations,
        normalize_scores=normalize_scores,
        use_rank_aggregation=use_rank_aggregation,
        sparse_use_similarity=sparse_use_similarity,
        sparse_similarity_mix_lambda=sparse_similarity_mix_lambda,
        use_rank_similarity_combination=use_rank_similarity_combination,
    )
    state = _SpeckVState(pruner=SparseRoundPruner(pruner_cfg), config=pruner_cfg)

    # CRITICAL SAFETY CHECK: Verify that the Pruner's internal Rotary Embedding matches the Model's live one.
    #
    # WHY THIS IS MANDATORY:
    # If these do not match (e.g., due to "attn_factor" vs "factor" config mismatches in Llama/Qwen),
    # the pruner will use WRONG frequencies to invert keys. This results in garbage scores (GIGO),
    # causing the algorithm to randomly drop important tokens while thinking it's doing a good job.
    # This leads to silent performance degradation ("landmines").
    #
    # DO NOT REMOVE OR SUPPRESS THIS CHECK.
    model_rotary_emb = None
    try:
        # Robustly attempt to locate the rotary embedding in standard HF models (Llama, Qwen, etc.)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            if len(layers) > 0 and hasattr(layers[0], "self_attn"):
                attn = layers[0].self_attn
                if hasattr(attn, "rotary_emb"):
                    model_rotary_emb = attn.rotary_emb
    except Exception:
        pass

    if model_rotary_emb is not None:
        # This function raises ValueError on mismatch. We MUST let it crash the process.
        verify_rotary_alignment(state.pruner.rotary, model_rotary_emb)
    else:
        print("[SpeckV] WARNING: Could not locate model.model.layers[0].self_attn.rotary_emb to verify alignment.")
        print("[SpeckV] BEWARE: Silent config mismatches are possible if architecture is non-standard.")

    orig_forward = model.forward

    def speckv_forward(
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
        cache_position_override = cache_position
        position_ids_override = position_ids
        attention_mask_override = attention_mask

        if past_key_values is not None and input_ids is not None:
            # Keep RoPE absolute (matches stats), but write new keys/values into a compact cache
            # so pruned tokens are not reintroduced as zero-filled holes.
            bsz, step = input_ids.shape

            # Absolute positions for rotary.
            start_pos = state.pruner.absolute_position
            abs_positions = torch.arange(
                start_pos,
                start_pos + step,
                device=input_ids.device,
                dtype=torch.long,
            )
            if bsz > 1:
                abs_positions = abs_positions.unsqueeze(0).expand(bsz, -1)
            else:
                abs_positions = abs_positions.unsqueeze(0)
            position_ids_override = abs_positions

            # Relative positions for cache placement (compact, contiguous).
            current_cache_len = None
            if isinstance(past_key_values, Cache) and hasattr(past_key_values, "get_seq_length"):
                current_cache_len = int(past_key_values.get_seq_length())
            elif isinstance(past_key_values, (tuple, list)) and past_key_values:
                current_cache_len = int(past_key_values[0][0].shape[2])

            if current_cache_len is not None:
                rel_positions = torch.arange(
                    current_cache_len,
                    current_cache_len + step,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                if bsz > 1:
                    rel_positions = rel_positions.unsqueeze(0).expand(bsz, -1)
                else:
                    rel_positions = rel_positions.unsqueeze(0)
                cache_position_override = rel_positions

            # After pruning, the cached KV length no longer matches the original attention_mask.
            # Rely on cache_position + causal mask instead of a stale full-length mask.
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

        # Reset pruner state if starting a new generation (empty cache).
        # transformers.generate() may pass None or an empty DynamicCache.
        is_empty_cache = True
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                if past_key_values.get_seq_length() > 0:
                    is_empty_cache = False
            elif isinstance(past_key_values, (tuple, list)):
                if len(past_key_values) > 0 and past_key_values[0][0].shape[2] > 0:
                    is_empty_cache = False
        
        if is_empty_cache and state.attached:
            state.pruner = SparseRoundPruner(state.config)
            state.attached = False
            state.initial_prefix_length = None

        pkv_tuple = _cache_to_tuple(outputs.past_key_values)

        if not state.attached:
            state.pruner.attach_initial_cache(pkv_tuple)
            state.initial_prefix_length = state.pruner.prefix_length
            pkv_tuple = state.pruner.enforce_max_limit(pkv_tuple)
            state.attached = True
        else:
            seq_len = pkv_tuple[0][0].shape[2]
            cached_len = len(state.pruner.cache_positions)
            if cached_len < seq_len:
                added = seq_len - cached_len
                start_pos = state.pruner.absolute_position
                new_positions = list(range(start_pos, start_pos + added))
                state.pruner.cache_positions.extend(new_positions)
                state.pruner.absolute_position += added
                state.pruner.tokens_in_round += added
            elif cached_len > seq_len:
                state.pruner.cache_positions = state.pruner.cache_positions[-seq_len:]

            if state.pruner.prefix_length == 0 and state.initial_prefix_length:
                state.pruner.prefix_length = state.initial_prefix_length

            while state.pruner.should_start_next_round():
                pkv_tuple = state.pruner.start_next_round(pkv_tuple)

        # Align with R-KV budget-based compression: only prune when exceeding budget.
        if state.pruner._dynamic_cache_size > state.pruner.max_keys:
            pkv_tuple = state.pruner.ensure_capacity(pkv_tuple)

        outputs = CausalLMOutputWithPast(
            loss=getattr(outputs, "loss", None),
            logits=outputs.logits,
            past_key_values=_cache_from_tuple(pkv_tuple, outputs.past_key_values),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
        return outputs

    model.forward = MethodType(speckv_forward, model)
