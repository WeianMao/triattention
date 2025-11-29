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
    )
    state = _SpeckVState(pruner=SparseRoundPruner(pruner_cfg), config=pruner_cfg)

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

        if past_key_values is not None and input_ids is not None:
            # Keep absolute RoPE positions aligned with pruner state during decode.
            bsz, step = input_ids.shape
            start_pos = state.pruner.absolute_position
            cache_positions = torch.arange(
                start_pos,
                start_pos + step,
                device=input_ids.device,
                dtype=torch.long,
            )
            if bsz > 1:
                cache_positions = cache_positions.unsqueeze(0).expand(bsz, -1)
            else:
                cache_positions = cache_positions.unsqueeze(0)
            cache_position_override = cache_positions
            if position_ids is None:
                position_ids_override = cache_positions

        outputs = orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

        if past_key_values is None and state.attached:
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
