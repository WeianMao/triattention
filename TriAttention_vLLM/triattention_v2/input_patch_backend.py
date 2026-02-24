"""Backend facade for TriAttention V2 runtime input patch integration.

This gives runner/worker/input_adapter a stable interface while the underlying
implementation transitions away from patch-heavy logic.
"""

from __future__ import annotations

import torch

from .input_patch_installer import install_runtime_input_patch_hooks
from .input_patch_state import (
    active_effective_overrides_consumed,
    mark_active_effective_overrides_consumed,
    set_active_effective_num_computed_tokens,
    set_active_effective_overrides_enabled,
    set_active_effective_positions,
    set_active_effective_sparse_overrides,
)


def install_runtime_input_patch() -> bool:
    return install_runtime_input_patch_hooks()


def activate_effective_sparse_overrides(
    *,
    seq_base_map: dict[int, int] | None,
    pos_delta_map: dict[int, int] | None,
    single_seq_base: int | None,
    single_pos_delta: int,
    expected_req_row_indices: tuple[int, ...] | None = None,
    expected_query_lens: tuple[int, ...] | None = None,
) -> None:
    set_active_effective_overrides_enabled(True)
    # Dense overrides intentionally disabled in normal path.
    set_active_effective_num_computed_tokens(None)
    set_active_effective_positions(None)
    set_active_effective_sparse_overrides(
        effective_base_by_req_idx=seq_base_map,
        effective_pos_delta_by_req_idx=pos_delta_map,
        single_effective_seq_base=single_seq_base,
        single_effective_pos_delta=single_pos_delta,
        expected_req_row_indices=expected_req_row_indices,
        expected_query_lens=expected_query_lens,
    )


def clear_effective_overrides() -> None:
    set_active_effective_overrides_enabled(False)
    set_active_effective_sparse_overrides(
        effective_base_by_req_idx=None,
        effective_pos_delta_by_req_idx=None,
        single_effective_seq_base=None,
        single_effective_pos_delta=0,
        expected_req_row_indices=None,
        expected_query_lens=None,
    )
    set_active_effective_positions(None)
    set_active_effective_num_computed_tokens(None)


def mark_effective_overrides_consumed() -> None:
    mark_active_effective_overrides_consumed()


def assert_effective_overrides_consumed() -> None:
    if not active_effective_overrides_consumed():
        raise RuntimeError(
            "TRIATTN_EFFECTIVE_OVERRIDES_NOT_CONSUMED:"
            "runtime_overrides_activated_but_patched_input_prep_not_observed"
        )
