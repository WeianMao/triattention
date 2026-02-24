from __future__ import annotations

import torch

from triattention_v2 import input_patch_state as patch_state
from triattention_v2.input_patch_vllm_backend import (
    make_patched_compute_slot_mappings,
    make_patched_prepare_pos_seq_lens,
)


def _reset_patch_state() -> None:
    patch_state.set_active_effective_overrides_enabled(False)
    patch_state.set_active_effective_num_computed_tokens(None)
    patch_state.set_active_effective_positions(None)
    patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx=None,
        effective_pos_delta_by_req_idx=None,
        single_effective_seq_base=None,
        single_effective_pos_delta=0,
        expected_req_row_indices=None,
    )


def test_prepare_pos_seq_lens_fails_when_sparse_seq_base_present_but_no_rows_match():
    _reset_patch_state()
    patch_state.set_active_effective_overrides_enabled(True)
    patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={99: 100},
        effective_pos_delta_by_req_idx=None,
        single_effective_seq_base=None,
        single_effective_pos_delta=0,
        expected_query_lens=(1,),
    )

    def _orig(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens):
        del pos
        qlen = query_start_loc[1:] - query_start_loc[:-1]
        seq_lens[: idx_mapping.shape[0]] = (
            num_computed_tokens.index_select(0, idx_mapping.to(num_computed_tokens.device))
            + qlen.to(num_computed_tokens.device)
        ).to(seq_lens.dtype)

    patched = make_patched_prepare_pos_seq_lens(_orig)
    idx_mapping = torch.tensor([1], dtype=torch.long)
    query_start_loc = torch.tensor([0, 1], dtype=torch.long)
    num_computed_tokens = torch.tensor([10, 20], dtype=torch.long)
    pos = torch.tensor([20], dtype=torch.long)
    seq_lens = torch.zeros(1, dtype=torch.long)

    try:
        patched(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
    except RuntimeError as exc:
        assert "TRIATTN_SEQ_LENS_SPARSE_BASE_APPLY_FAILED" in str(exc)
    else:
        raise AssertionError("expected sparse seq-base apply failure")
    finally:
        _reset_patch_state()


def test_compute_slot_mappings_fails_when_sparse_pos_delta_present_but_shift_fails():
    _reset_patch_state()
    patch_state.set_active_effective_overrides_enabled(True)
    patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx=None,
        effective_pos_delta_by_req_idx={1: 7},
        single_effective_seq_base=None,
        single_effective_pos_delta=0,
        expected_query_lens=(1,),
    )

    def _orig(_self, idx_mapping, query_start_loc, positions):
        del _self, idx_mapping, query_start_loc
        return positions.clone()

    patched = make_patched_compute_slot_mappings(_orig)
    idx_mapping = torch.tensor([1], dtype=torch.long)
    # Non-monotonic qsl will make shift_positions_from_sparse_deltas return None.
    query_start_loc = torch.tensor([2, 1], dtype=torch.long)
    positions = torch.tensor([5, 6, 7], dtype=torch.long)

    try:
        patched(object(), idx_mapping, query_start_loc, positions)
    except RuntimeError as exc:
        assert (
            "TRIATTN_QUERY_START_LOC_NON_MONOTONIC" in str(exc)
            or "TRIATTN_SLOT_MAPPING_SPARSE_SHIFT_FAILED" in str(exc)
        )
    else:
        raise AssertionError("expected sparse slot-mapping shift failure")
    finally:
        _reset_patch_state()


def test_prepare_pos_seq_lens_fails_when_idx_mapping_does_not_match_expected_rows():
    _reset_patch_state()
    patch_state.set_active_effective_overrides_enabled(True)
    patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={1: 10},
        effective_pos_delta_by_req_idx=None,
        single_effective_seq_base=None,
        single_effective_pos_delta=0,
        expected_req_row_indices=(5,),  # mismatches actual idx_mapping=(1,)
        expected_query_lens=(1,),
    )

    def _orig(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens):
        del idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens

    patched = make_patched_prepare_pos_seq_lens(_orig)
    try:
        patched(
            torch.tensor([1], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([10, 20], dtype=torch.long),
            torch.tensor([20], dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
    except RuntimeError as exc:
        assert "TRIATTN_IDX_MAPPING_MISMATCH" in str(exc)
    else:
        raise AssertionError("expected idx_mapping mismatch failure")
    finally:
        _reset_patch_state()


def test_prepare_pos_seq_lens_fails_when_query_start_loc_does_not_match_expected_query_lens():
    _reset_patch_state()
    patch_state.set_active_effective_overrides_enabled(True)
    patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={1: 10},
        effective_pos_delta_by_req_idx=None,
        single_effective_seq_base=None,
        single_effective_pos_delta=0,
        expected_req_row_indices=(1,),
        expected_query_lens=(2,),  # expect qlen=2 but actual qlen below is 1
    )

    def _orig(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens):
        del idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens

    patched = make_patched_prepare_pos_seq_lens(_orig)
    try:
        patched(
            torch.tensor([1], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([10, 20], dtype=torch.long),
            torch.tensor([20], dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
    except RuntimeError as exc:
        assert "TRIATTN_QUERY_LENS_MISMATCH" in str(exc)
    else:
        raise AssertionError("expected query lens mismatch failure")
    finally:
        _reset_patch_state()


def test_compute_slot_mappings_fails_on_dense_effective_positions_shape_mismatch():
    _reset_patch_state()
    patch_state.set_active_effective_overrides_enabled(True)
    patch_state.set_active_effective_positions(torch.tensor([1, 2], dtype=torch.long))
    patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx=None,
        effective_pos_delta_by_req_idx=None,
        single_effective_seq_base=None,
        single_effective_pos_delta=0,
        expected_req_row_indices=(1,),
        expected_query_lens=(1,),
    )

    def _orig(_self, idx_mapping, query_start_loc, positions):
        del _self, idx_mapping, query_start_loc
        return positions.clone()

    patched = make_patched_compute_slot_mappings(_orig)
    try:
        patched(
            object(),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([10, 11, 12], dtype=torch.long),
        )
    except RuntimeError as exc:
        assert "TRIATTN_EFFECTIVE_POSITIONS_SHAPE_MISMATCH" in str(exc)
    else:
        raise AssertionError("expected dense effective positions shape mismatch failure")
    finally:
        _reset_patch_state()
