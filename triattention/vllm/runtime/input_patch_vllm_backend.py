"""vLLM patch backend closures for TriAttention runtime input overrides."""

from __future__ import annotations

import os
from typing import Any, Callable

import torch

from . import input_patch_state as _patch_state
from .input_patch_ops import (
    overwrite_seq_lens_from_effective_base_map,
    overwrite_seq_lens_from_effective_lengths,
    shift_positions_from_sparse_deltas,
    validate_slot_mapping_capacity,
    validate_slot_mapping_values,
)


def _debug_disable_seq_override() -> bool:
    return os.environ.get("TRIATTN_DEBUG_DISABLE_SEQ_OVERRIDE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_disable_slot_shift() -> bool:
    return os.environ.get("TRIATTN_DEBUG_DISABLE_SLOT_SHIFT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_enable_slot_shift() -> bool:
    return os.environ.get("TRIATTN_DEBUG_ENABLE_SLOT_SHIFT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _validate_idx_mapping_matches_expected_rows(idx_mapping: torch.Tensor) -> None:
    expected = _patch_state.get_active_expected_req_row_indices(idx_mapping.device)
    if expected is None:
        return
    actual = idx_mapping.to(device=idx_mapping.device, dtype=torch.long)
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise RuntimeError(
            "TRIATTN_IDX_MAPPING_MISMATCH:"
            f"actual={actual.detach().to('cpu', dtype=torch.long).tolist()}:"
            f"expected={expected.detach().to('cpu', dtype=torch.long).tolist()}"
        )


def _validate_query_start_loc_matches_expected_q_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    expected_q_lens = _patch_state.get_active_expected_query_lens(query_start_loc.device)
    if expected_q_lens is None:
        return
    num_reqs = int(idx_mapping.shape[0])
    if expected_q_lens.numel() != num_reqs:
        raise RuntimeError(
            "TRIATTN_QUERY_LENS_COUNT_MISMATCH:"
            f"expected_num_reqs={int(expected_q_lens.numel())}:actual_num_reqs={num_reqs}"
        )
    qsl = query_start_loc[: num_reqs + 1].to(device=query_start_loc.device, dtype=torch.long)
    if qsl.numel() != (num_reqs + 1):
        raise RuntimeError(
            "TRIATTN_QUERY_START_LOC_SHAPE_INVALID:"
            f"expected_numel={num_reqs + 1}:actual_numel={int(qsl.numel())}"
        )
    if bool(torch.any(qsl[1:] < qsl[:-1])):
        raise RuntimeError("TRIATTN_QUERY_START_LOC_NON_MONOTONIC")
    actual_q_lens = qsl[1:] - qsl[:-1]
    if not torch.equal(actual_q_lens, expected_q_lens):
        raise RuntimeError(
            "TRIATTN_QUERY_LENS_MISMATCH:"
            f"actual={actual_q_lens.detach().to('cpu', dtype=torch.long).tolist()}:"
            f"expected={expected_q_lens.detach().to('cpu', dtype=torch.long).tolist()}"
        )


def _validate_mapping_once(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    if os.environ.get("TRIATTN_RUNTIME_VALIDATE_MAPPING", "0") != "1":
        return
    # Both patched vLLM hooks observe the same per-step request packing. Running
    # the validation once per active override window preserves fail-fast
    # guarantees while avoiding duplicate host/device sync in the decode hot
    # path.
    if _patch_state.active_effective_mapping_validated():
        return
    _validate_idx_mapping_matches_expected_rows(idx_mapping)
    _validate_query_start_loc_matches_expected_q_lens(idx_mapping, query_start_loc)
    _patch_state.mark_active_effective_mapping_validated()


def make_patched_prepare_pos_seq_lens(
    original_prepare_pos_seq_lens: Callable[..., None],
) -> Callable[..., None]:
    def _patched_prepare_pos_seq_lens(
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        pos: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        # Hard no-op fast path after monkey patch is installed but the current
        # step does not require any effective-length overrides.
        if not _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED:
            original_prepare_pos_seq_lens(
                idx_mapping,
                query_start_loc,
                num_computed_tokens,
                pos,
                seq_lens,
            )
            return
        _patch_state.mark_active_effective_overrides_consumed()
        _validate_mapping_once(idx_mapping, query_start_loc)
        original_prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            pos,
            seq_lens,
        )
        if _debug_disable_seq_override():
            return
        eff = _patch_state.ACTIVE_EFFECTIVE_NUM_COMPUTED_TOKENS
        if eff is None:
            if (
                int(idx_mapping.shape[0]) == 1
                and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None
            ):
                qlen_t = (query_start_loc[1] - query_start_loc[0]).to(
                    device=seq_lens.device, dtype=seq_lens.dtype
                )
                base_t = torch.as_tensor(
                    _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE,
                    device=seq_lens.device,
                    dtype=seq_lens.dtype,
                )
                seq_lens[0].copy_(base_t + qlen_t)
                return
            sparse_bases = _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX
            if sparse_bases:
                sparse_base_lookup = _patch_state.get_active_effective_base_lookup_tensors(
                    seq_lens.device
                )
                applied = overwrite_seq_lens_from_effective_base_map(
                    idx_mapping=idx_mapping,
                    query_start_loc=query_start_loc,
                    effective_base_by_req_idx=sparse_bases,
                    seq_lens=seq_lens,
                    effective_base_lookup_tensors=sparse_base_lookup,
                )
                if not applied and int(idx_mapping.shape[0]) > 0:
                    raise RuntimeError(
                        "TRIATTN_SEQ_LENS_SPARSE_BASE_APPLY_FAILED:"
                        "sparse_base_present_but_no_rows_were_overwritten"
                    )
            return
        overwrite_seq_lens_from_effective_lengths(
            idx_mapping=idx_mapping,
            query_start_loc=query_start_loc,
            effective_num_computed_tokens=eff,
            seq_lens=seq_lens,
        )

    return _patched_prepare_pos_seq_lens


def make_patched_compute_slot_mappings(
    original_compute_slot_mappings: Callable[..., Any],
) -> Callable[..., Any]:
    def _patched_compute_slot_mappings(self, idx_mapping, query_start_loc, positions):
        # Hard no-op fast path when overrides are not active for this step.
        if not _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED:
            return original_compute_slot_mappings(self, idx_mapping, query_start_loc, positions)
        _patch_state.mark_active_effective_overrides_consumed()
        _validate_mapping_once(idx_mapping, query_start_loc)
        # Keep decode positions in the original absolute RoPE space. After
        # compaction we still need seq_len overrides so attention/masking sees
        # the effective compressed history length, but shifting token positions
        # here corrupts continuation semantics for real serve/chat payloads.
        #
        # The old shift path is retained behind a debug-only opt-in so we can
        # bisect regressions without putting extra work back on the hot path.
        if _debug_disable_slot_shift() or not _debug_enable_slot_shift():
            return original_compute_slot_mappings(self, idx_mapping, query_start_loc, positions)
        eff_positions = _patch_state.ACTIVE_EFFECTIVE_POSITIONS
        if (
            isinstance(eff_positions, torch.Tensor)
            and eff_positions.ndim == 1
            and eff_positions.numel() == positions.numel()
        ):
            if os.environ.get("TRIATTN_DEBUG_VALIDATE_SLOT_MAPPING", "0") == "1":
                validate_slot_mapping_capacity(
                    block_tables=self,
                    idx_mapping=idx_mapping,
                    query_start_loc=query_start_loc,
                    effective_positions=eff_positions,
                )
            out = original_compute_slot_mappings(self, idx_mapping, query_start_loc, eff_positions)
            if os.environ.get("TRIATTN_DEBUG_VALIDATE_SLOT_MAPPING_VALUES", "0") == "1":
                validate_slot_mapping_values(
                    block_tables=self,
                    idx_mapping=idx_mapping,
                    query_start_loc=query_start_loc,
                    effective_positions=eff_positions,
                    slot_mappings=out,
                )
            return out
        if isinstance(eff_positions, torch.Tensor):
            raise RuntimeError(
                "TRIATTN_EFFECTIVE_POSITIONS_SHAPE_MISMATCH:"
                f"expected_numel={int(positions.numel())}:"
                f"actual_numel={int(eff_positions.numel())}:"
                f"actual_ndim={int(eff_positions.ndim)}"
            )
        sparse_pos_deltas = _patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX
        if (
            int(idx_mapping.shape[0]) == 1
            and _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0
        ):
            out = positions.clone()
            out.add_(_patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA)
            return original_compute_slot_mappings(self, idx_mapping, query_start_loc, out)
        if sparse_pos_deltas:
            sparse_pos_delta_lookup = _patch_state.get_active_effective_pos_delta_lookup_tensors(
                positions.device
            )
            shifted_positions = shift_positions_from_sparse_deltas(
                idx_mapping=idx_mapping,
                query_start_loc=query_start_loc,
                positions=positions,
                pos_delta_by_req_idx=sparse_pos_deltas,
                pos_delta_lookup_tensors=sparse_pos_delta_lookup,
            )
            if shifted_positions is not None:
                return original_compute_slot_mappings(
                    self, idx_mapping, query_start_loc, shifted_positions
                )
            if int(idx_mapping.shape[0]) > 0:
                raise RuntimeError(
                    "TRIATTN_SLOT_MAPPING_SPARSE_SHIFT_FAILED:"
                    "sparse_pos_delta_present_but_shift_positions_failed"
                )
        return original_compute_slot_mappings(self, idx_mapping, query_start_loc, positions)

    return _patched_compute_slot_mappings
