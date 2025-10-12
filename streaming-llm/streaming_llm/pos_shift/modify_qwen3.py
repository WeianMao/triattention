from typing import Optional, Tuple

import torch
import types

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    eager_attention_forward,
    repeat_kv,
    rotate_half,
)

__all__ = ["enable_qwen3_pos_shift_attention"]


def _ensure_2d(position_ids: torch.LongTensor, batch_size: int) -> torch.LongTensor:
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.size(0) == 1 and batch_size > 1:
        position_ids = position_ids.expand(batch_size, -1)
    return position_ids


def _gather_rotary_components(
    cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    position_ids = position_ids.to(cos.device)
    gather_idx = position_ids.unsqueeze(-1).expand(-1, -1, cos.size(-1))
    gathered_cos = torch.gather(cos, 1, gather_idx)
    gathered_sin = torch.gather(sin, 1, gather_idx)
    return gathered_cos.unsqueeze(1), gathered_sin.unsqueeze(1)


def apply_rotary_pos_emb_single(
    states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
) -> torch.Tensor:
    gathered_cos, gathered_sin = _gather_rotary_components(cos, sin, position_ids)
    if gathered_cos.device != states.device:
        gathered_cos = gathered_cos.to(states.device)
        gathered_sin = gathered_sin.to(states.device)
    return (states * gathered_cos) + (rotate_half(states) * gathered_sin)


def qwen3_pos_shift_attention_forward(
    self: Qwen3Attention,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[object] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    position_ids = kwargs.get("position_ids")
    rotary_emb = getattr(self, "_streaming_rotary_emb", None)
    original_forward = getattr(self, "_orig_forward", None)

    if position_ids is None or rotary_emb is None or original_forward is None:
        return original_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    bsz = hidden_states.size(0)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    position_ids = _ensure_2d(position_ids, bsz).to(hidden_states.device)

    past_kv_len = 0
    if past_key_value is not None:
        if hasattr(past_key_value, "get_seq_length"):
            past_kv_len = past_key_value.get_seq_length(self.layer_idx)
        else:
            past_kv_len = past_key_value[0][0].size(-2)

    kv_seq_len = past_kv_len + key_states.shape[-2]
    key_position_ids = torch.arange(
        kv_seq_len, device=hidden_states.device, dtype=position_ids.dtype
    ).unsqueeze(0)
    key_position_ids = _ensure_2d(key_position_ids, bsz)

    full_cos, full_sin = rotary_emb(hidden_states, key_position_ids)
    query_states = apply_rotary_pos_emb_single(query_states, full_cos, full_sin, position_ids)

    sin_slice, cos_slice = position_embeddings
    cache_kwargs = {"sin": sin_slice, "cos": cos_slice, "cache_position": cache_position}

    if past_key_value is not None:
        if hasattr(past_key_value, "update"):
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        else:
            past_keys, past_values = past_key_value
            key_states = torch.cat([past_keys, key_states], dim=-2)
            value_states = torch.cat([past_values, value_states], dim=-2)
            past_key_value = (key_states, value_states)
    else:
        cache_kwargs = None

    key_states = apply_rotary_pos_emb_single(key_states, full_cos, full_sin, key_position_ids)

    if past_key_value is not None and hasattr(past_key_value, "layers"):
        past_key_value.layers[self.layer_idx].keys = key_states
        past_key_value.layers[self.layer_idx].values = value_states

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _attach_rotary_reference(module, rotary_emb):
    if isinstance(module, Qwen3Attention):
        if not hasattr(module, "_orig_forward"):
            module._orig_forward = module.forward
        module._streaming_rotary_emb = rotary_emb
        module.forward = types.MethodType(qwen3_pos_shift_attention_forward, module)
        return
    for child in module.children():
        _attach_rotary_reference(child, rotary_emb)


def enable_qwen3_pos_shift_attention(model):
    rotary_emb = None
    modules_to_visit = [model]
    while modules_to_visit:
        current = modules_to_visit.pop()
        if hasattr(current, "rotary_emb") and rotary_emb is None:
            rotary_emb = current.rotary_emb
        modules_to_visit.extend(list(current.children()))
    if rotary_emb is None:
        raise ValueError("Failed to locate rotary embedding module for Qwen3")
    _attach_rotary_reference(model, rotary_emb)
