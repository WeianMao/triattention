import torch

from triattention_v2.input_patch_ops import (
    overwrite_seq_lens_from_effective_base_map,
    shift_positions_from_sparse_deltas,
)


def test_shift_positions_from_sparse_deltas_contiguous_fast_path():
    idx_mapping = torch.tensor([3, 7], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.long)
    positions = torch.tensor([100, 101, 200, 201, 202], dtype=torch.long)

    out = shift_positions_from_sparse_deltas(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        positions=positions,
        pos_delta_by_req_idx={3: -10, 7: 5},
    )

    assert out is not None
    assert out.tolist() == [90, 91, 205, 206, 207]


def test_shift_positions_from_sparse_deltas_handles_non_contiguous_query_ranges():
    # Simulate a layout where query tokens occupy disjoint slices rather than a
    # simple prefix. This should not shift unrelated token positions.
    idx_mapping = torch.tensor([11, 22], dtype=torch.long)
    query_start_loc = torch.tensor([2, 4, 7], dtype=torch.long)
    positions = torch.tensor([0, 1, 10, 11, 20, 21, 22, 30, 31, 32], dtype=torch.long)

    out = shift_positions_from_sparse_deltas(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        positions=positions,
        pos_delta_by_req_idx={11: 100, 22: -5},
    )

    assert out is not None
    # Tokens [2:4] shifted by +100, [4:7] shifted by -5, others untouched.
    assert out.tolist() == [0, 1, 110, 111, 15, 16, 17, 30, 31, 32]


def test_shift_positions_from_sparse_deltas_returns_none_when_no_rows_match():
    idx_mapping = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.long)
    positions = torch.tensor([9, 10], dtype=torch.long)

    out = shift_positions_from_sparse_deltas(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        positions=positions,
        pos_delta_by_req_idx={99: 7},
    )
    assert out is None


def test_shift_positions_from_sparse_deltas_returns_none_on_partial_row_match():
    idx_mapping = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.long)
    positions = torch.tensor([9, 10], dtype=torch.long)

    out = shift_positions_from_sparse_deltas(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        positions=positions,
        pos_delta_by_req_idx={1: 7, 99: 3},
    )
    assert out is None


def test_shift_positions_from_sparse_deltas_returns_none_on_non_monotonic_query_start_loc():
    idx_mapping = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([3, 2, 5], dtype=torch.long)
    positions = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

    out = shift_positions_from_sparse_deltas(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        positions=positions,
        pos_delta_by_req_idx={1: 10, 2: -3},
    )
    assert out is None


def test_overwrite_seq_lens_from_effective_base_map_returns_true_on_match():
    idx_mapping = torch.tensor([3, 7], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.long)
    seq_lens = torch.tensor([2, 3], dtype=torch.long)

    applied = overwrite_seq_lens_from_effective_base_map(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        effective_base_by_req_idx={3: 100, 7: 200},
        seq_lens=seq_lens,
    )

    assert applied is True
    assert seq_lens.tolist() == [102, 203]


def test_overwrite_seq_lens_from_effective_base_map_returns_false_when_no_rows_match():
    idx_mapping = torch.tensor([3, 7], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.long)
    seq_lens = torch.tensor([2, 3], dtype=torch.long)

    applied = overwrite_seq_lens_from_effective_base_map(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        effective_base_by_req_idx={99: 100},
        seq_lens=seq_lens,
    )

    assert applied is False
    assert seq_lens.tolist() == [2, 3]


def test_overwrite_seq_lens_from_effective_base_map_returns_false_on_partial_row_match():
    idx_mapping = torch.tensor([3, 7], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.long)
    seq_lens = torch.tensor([2, 3], dtype=torch.long)

    applied = overwrite_seq_lens_from_effective_base_map(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        effective_base_by_req_idx={3: 100, 99: 200},
        seq_lens=seq_lens,
    )

    assert applied is False
    assert seq_lens.tolist() == [2, 3]
