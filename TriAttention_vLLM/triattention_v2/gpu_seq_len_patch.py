"""Patch vLLM GPU input prep to decouple seq_lens from absolute positions.

TriAttention compression needs two different notions of "length":
- absolute decode progress (for positions / RoPE),
- effective KV cache length after compaction (for attention seq_lens).

vLLM v1 GPU input prep derives both from a single `num_computed_tokens` tensor.
This patch preserves positions behavior and overwrites seq_lens using a
TriAttention-provided effective-length tensor.
"""

from __future__ import annotations

from .effective_overrides import (
    build_effective_sparse_overrides as _build_effective_sparse_overrides_impl,
    build_effective_positions_override_tensor as _build_effective_positions_override_tensor_impl,
    build_effective_seq_len_override_tensor as _build_effective_seq_len_override_tensor_impl,
)
from .input_patch_ops import (
    overwrite_seq_lens_from_effective_base_map as _overwrite_seq_lens_from_effective_base_map_impl,
    overwrite_seq_lens_from_effective_lengths as _overwrite_seq_lens_from_effective_lengths_impl,
    shift_positions_from_sparse_deltas as _shift_positions_from_sparse_deltas_impl,
    validate_slot_mapping_capacity as _validate_slot_mapping_capacity_impl,
    validate_slot_mapping_values as _validate_slot_mapping_values_impl,
)
from . import input_patch_state as _patch_state
from .input_patch_installer import install_runtime_input_patch_hooks as _install_patch_hooks


set_active_effective_num_computed_tokens = _patch_state.set_active_effective_num_computed_tokens
set_active_effective_positions = _patch_state.set_active_effective_positions
set_active_effective_sparse_overrides = _patch_state.set_active_effective_sparse_overrides


def install_seq_len_override_patch() -> bool:
    """Compatibility wrapper for the runtime input patch installer."""
    return _install_patch_hooks()


_overwrite_seq_lens_from_effective_lengths = _overwrite_seq_lens_from_effective_lengths_impl
_overwrite_seq_lens_from_effective_base_map = _overwrite_seq_lens_from_effective_base_map_impl
build_effective_seq_len_override_tensor = _build_effective_seq_len_override_tensor_impl
build_effective_sparse_overrides = _build_effective_sparse_overrides_impl
build_effective_positions_override_tensor = _build_effective_positions_override_tensor_impl
_shift_positions_from_sparse_deltas = _shift_positions_from_sparse_deltas_impl
_validate_slot_mapping_capacity = _validate_slot_mapping_capacity_impl
_validate_slot_mapping_values = _validate_slot_mapping_values_impl
