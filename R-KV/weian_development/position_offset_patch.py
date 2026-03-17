"""Position offset patch for simulating Bug 896cbca6 attention position effect."""
from __future__ import annotations

from types import MethodType
from typing import Optional

import torch
from transformers.cache_utils import Cache


def apply_position_offset_patch(model) -> None:
    """Patch model.forward to simulate Bug 896cbca6 attention position offset.

    Effect: Decode phase position_ids -= prefill_len
    Makes decode "see" prefill tokens as closer (relative position = -k instead of P-k)
    """
    orig_forward = model.forward
    # Store state on model instance
    model._pos_offset_prefill_len = None
    model._pos_offset_absolute_position = 0

    def patched_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        position_ids_override = position_ids

        # Detect prefill vs decode
        is_empty_cache = True
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                if past_key_values.get_seq_length() > 0:
                    is_empty_cache = False
            elif isinstance(past_key_values, (tuple, list)):
                if len(past_key_values) > 0 and past_key_values[0][0].shape[2] > 0:
                    is_empty_cache = False

        if is_empty_cache:
            # Reset state for new generation
            self._pos_offset_prefill_len = None
            self._pos_offset_absolute_position = 0

        if input_ids is not None:
            bsz, seq_len = input_ids.shape

            if is_empty_cache:
                # Prefill: store prefill_len, use normal positions
                self._pos_offset_prefill_len = seq_len
                self._pos_offset_absolute_position = seq_len
            else:
                # Decode: subtract prefill_len to simulate bug
                position_offset = -self._pos_offset_prefill_len if self._pos_offset_prefill_len else 0
                start_pos = self._pos_offset_absolute_position + position_offset
                abs_positions = torch.arange(
                    start_pos,
                    start_pos + seq_len,
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                if bsz > 1:
                    abs_positions = abs_positions.expand(bsz, -1)
                position_ids_override = abs_positions
                self._pos_offset_absolute_position += seq_len

        return orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids_override,
            past_key_values=past_key_values,
            **kwargs,
        )

    model.forward = MethodType(patched_forward, model)
