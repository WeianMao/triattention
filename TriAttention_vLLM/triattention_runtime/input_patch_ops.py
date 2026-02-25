"""Low-level patch operations for vLLM GPU input override backend.

This module hosts tensor patch/validation helpers used by `gpu_seq_len_patch.py`
so the patch module can remain focused on installation and routing.
"""

from __future__ import annotations

from typing import Any

import torch


def _lookup_sparse_values_for_req_rows(
    *,
    req_rows: torch.Tensor,
    sparse_values_by_req_idx: dict[int, int] | None,
    device: torch.device,
    sparse_lookup_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Vectorized sparse lookup for req_state indices.

    Returns `(has_match, matched_values)` aligned with `req_rows`.
    """
    if req_rows.numel() == 0:
        return None
    keys: torch.Tensor
    values: torch.Tensor
    if sparse_lookup_tensors is not None:
        keys, values = sparse_lookup_tensors
        if keys.numel() == 0:
            return None
        if keys.device != device:
            keys = keys.to(device=device, dtype=torch.long)
        elif keys.dtype != torch.long:
            keys = keys.to(dtype=torch.long)
        if values.device != device:
            values = values.to(device=device, dtype=torch.long)
        elif values.dtype != torch.long:
            values = values.to(dtype=torch.long)
    else:
        if not sparse_values_by_req_idx:
            return None
        keys_list = [int(k) for k in sparse_values_by_req_idx.keys()]
        vals_list = [int(v) for v in sparse_values_by_req_idx.values()]
        keys = torch.as_tensor(keys_list, device=device, dtype=torch.long)
        if keys.numel() == 0:
            return None
        values = torch.as_tensor(vals_list, device=device, dtype=torch.long)
        if keys.numel() > 1:
            order = torch.argsort(keys)
            keys = keys.index_select(0, order)
            values = values.index_select(0, order)
    rows = req_rows.to(device=device, dtype=torch.long)
    if keys.numel() == 0:
        return None
    insert_pos = torch.searchsorted(keys, rows)
    in_range = insert_pos < int(keys.numel())
    safe_pos = insert_pos.clamp(max=int(keys.numel()) - 1)
    matched_keys = keys.index_select(0, safe_pos)
    has_match = in_range & (matched_keys == rows)
    matched_values = values.index_select(0, safe_pos)
    return has_match, matched_values


def overwrite_seq_lens_from_effective_lengths(
    *,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    effective_num_computed_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    num_reqs = int(idx_mapping.shape[0])
    if num_reqs <= 0:
        return
    req_state_indices = idx_mapping.to(device=effective_num_computed_tokens.device, dtype=torch.long)
    base_lens = torch.index_select(effective_num_computed_tokens, 0, req_state_indices)
    q_starts = query_start_loc[:num_reqs].to(device=base_lens.device, dtype=base_lens.dtype)
    q_ends = query_start_loc[1 : num_reqs + 1].to(device=base_lens.device, dtype=base_lens.dtype)
    query_lens = q_ends - q_starts
    out = (base_lens + query_lens).to(device=seq_lens.device, dtype=seq_lens.dtype)
    seq_lens[:num_reqs].copy_(out)


def overwrite_seq_lens_from_effective_base_map(
    *,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    effective_base_by_req_idx: dict[int, int],
    seq_lens: torch.Tensor,
    effective_base_lookup_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> bool:
    num_reqs = int(idx_mapping.shape[0])
    if num_reqs <= 0 or not effective_base_by_req_idx:
        return False
    device = seq_lens.device
    req_rows = idx_mapping[:num_reqs].to(device=device, dtype=torch.long)
    lookup = _lookup_sparse_values_for_req_rows(
        req_rows=req_rows,
        sparse_values_by_req_idx=effective_base_by_req_idx,
        device=device,
        sparse_lookup_tensors=effective_base_lookup_tensors,
    )
    if lookup is None:
        return False
    has_match, base_vals = lookup
    row_ids = has_match.nonzero(as_tuple=False).flatten()
    expected_rows = len(effective_base_by_req_idx)
    if row_ids.numel() == 0:
        return False
    if int(row_ids.numel()) != int(expected_rows):
        return False

    qsl = query_start_loc[: num_reqs + 1].to(device=device, dtype=torch.long)
    q_lens = qsl[1:] - qsl[:-1]
    vals = base_vals.index_select(0, row_ids) + q_lens.index_select(0, row_ids)
    seq_lens.index_copy_(
        0,
        row_ids,
        vals.to(device=device, dtype=seq_lens.dtype),
    )
    return True


def shift_positions_from_sparse_deltas(
    *,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    pos_delta_by_req_idx: dict[int, int],
    pos_delta_lookup_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    num_reqs = int(idx_mapping.shape[0])
    if num_reqs <= 0 or not pos_delta_by_req_idx:
        return None
    device = positions.device
    req_rows = idx_mapping[:num_reqs].to(device=device, dtype=torch.long)
    lookup = _lookup_sparse_values_for_req_rows(
        req_rows=req_rows,
        sparse_values_by_req_idx=pos_delta_by_req_idx,
        device=device,
        sparse_lookup_tensors=pos_delta_lookup_tensors,
    )
    if lookup is None:
        return None
    has_match, deltas = lookup
    expected_rows = len(pos_delta_by_req_idx)
    matched_rows = int(has_match.sum().item()) if has_match.numel() > 0 else 0
    if matched_rows != int(expected_rows):
        return None

    qsl = query_start_loc[: num_reqs + 1].to(device=device, dtype=torch.long)
    if bool(torch.any(qsl[1:] < qsl[:-1])):
        return None
    q_lens = (qsl[1:] - qsl[:-1]).clamp_min_(0)
    if q_lens.numel() == 0:
        return None

    row_deltas = torch.where(
        has_match,
        deltas,
        torch.zeros_like(deltas),
    )
    changed_rows = (row_deltas != 0).nonzero(as_tuple=False).flatten()
    if changed_rows.numel() == 0:
        return None

    token_deltas = torch.repeat_interleave(row_deltas, q_lens)
    if token_deltas.numel() == 0:
        return None

    out = positions.clone()
    token_deltas = token_deltas.to(device=device, dtype=out.dtype)
    total_query_tokens = int(token_deltas.numel())
    # Fast path: vLLM decode packing is typically contiguous [0, total_query_tokens).
    # When this assumption does not hold (e.g. future input layouts / padding),
    # falling back to explicit token indices avoids silently shifting the wrong
    # positions ("should modify A, actually modifies B").
    if (
        qsl.numel() >= 1
        and int(qsl[0].item()) == 0
        and int(qsl[-1].item()) == total_query_tokens
        and total_query_tokens <= int(out.numel())
    ):
        out[:total_query_tokens].add_(token_deltas)
        return out

    changed_q_lens = q_lens.index_select(0, changed_rows)
    if changed_q_lens.numel() == 0:
        return None
    changed_row_deltas = row_deltas.index_select(0, changed_rows)
    token_deltas = torch.repeat_interleave(changed_row_deltas, changed_q_lens).to(
        device=device,
        dtype=out.dtype,
    )
    changed_starts = qsl[:-1].index_select(0, changed_rows)
    segment_offsets = torch.repeat_interleave(changed_starts, changed_q_lens)
    intra_offsets = torch.cat(
        [
            torch.arange(int(seg_len.item()), device=device, dtype=torch.long)
            for seg_len in changed_q_lens
        ]
    )
    token_indices = segment_offsets + intra_offsets
    if token_indices.numel() != token_deltas.numel():
        # Defensive fallback: shape mismatch means input packing assumptions are
        # broken; avoid corrupting unrelated rows.
        return None
    valid_mask = (token_indices >= 0) & (token_indices < int(out.numel()))
    if not bool(torch.all(valid_mask)):
        return None
    if token_indices.numel() == 0:
        return None
    out.index_add_(0, token_indices, token_deltas)
    return out


def validate_slot_mapping_capacity(
    *,
    block_tables: Any,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    effective_positions: torch.Tensor,
) -> None:
    num_blocks_buf = getattr(block_tables, "num_blocks", None)
    block_sizes = getattr(block_tables, "block_sizes", None)
    if num_blocks_buf is None or not isinstance(block_sizes, list):
        return
    if not hasattr(num_blocks_buf, "np"):
        return

    idx_cpu = idx_mapping.detach().to("cpu", dtype=torch.int64)
    qsl_cpu = query_start_loc.detach().to("cpu", dtype=torch.int64)
    pos_cpu = effective_positions.detach().to("cpu", dtype=torch.int64)
    num_reqs = int(idx_cpu.numel())

    for batch_idx in range(num_reqs):
        req_idx = int(idx_cpu[batch_idx].item())
        start = int(qsl_cpu[batch_idx].item())
        end = int(qsl_cpu[batch_idx + 1].item())
        if end <= start:
            continue
        req_positions = pos_cpu[start:end]
        max_pos = int(req_positions.max().item())
        for gid, block_size in enumerate(block_sizes):
            if block_size <= 0:
                continue
            needed_blocks = (max_pos // int(block_size)) + 1
            current_blocks = int(num_blocks_buf.np[gid, req_idx])
            if needed_blocks > current_blocks:
                raise RuntimeError(
                    "TRIATTN_SLOT_MAPPING_CAPACITY_MISMATCH:"
                    f"batch_idx={batch_idx}:req_idx={req_idx}:gid={gid}:"
                    f"needed_blocks={needed_blocks}:current_blocks={current_blocks}:"
                    f"max_pos={max_pos}:block_size={int(block_size)}"
                )


def validate_slot_mapping_values(
    *,
    block_tables: Any,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    effective_positions: torch.Tensor,
    slot_mappings: torch.Tensor,
) -> None:
    num_blocks_buf = getattr(block_tables, "num_blocks", None)
    block_sizes = getattr(block_tables, "block_sizes", None)
    block_table_rows = getattr(block_tables, "block_tables", None)
    if (
        num_blocks_buf is None
        or not hasattr(num_blocks_buf, "np")
        or not isinstance(block_sizes, list)
        or not isinstance(block_table_rows, list)
    ):
        return

    idx_cpu = idx_mapping.detach().to("cpu", dtype=torch.int64)
    qsl_cpu = query_start_loc.detach().to("cpu", dtype=torch.int64)
    pos_cpu = effective_positions.detach().to("cpu", dtype=torch.int64)
    slot_cpu = slot_mappings.detach().to("cpu", dtype=torch.int64)
    num_reqs = int(idx_cpu.numel())

    for gid, block_size in enumerate(block_sizes):
        if gid >= slot_cpu.shape[0] or gid >= len(block_table_rows):
            break
        if int(block_size) <= 0:
            continue
        for batch_idx in range(num_reqs):
            req_idx = int(idx_cpu[batch_idx].item())
            start = int(qsl_cpu[batch_idx].item())
            end = int(qsl_cpu[batch_idx + 1].item())
            if end <= start:
                continue
            pos_seg = pos_cpu[start:end]
            if pos_seg.numel() == 0:
                continue
            max_pos = int(pos_seg.max().item())
            needed_blocks = (max_pos // int(block_size)) + 1
            current_blocks = int(num_blocks_buf.np[gid, req_idx])
            if needed_blocks > current_blocks:
                continue
            block_row_prefix = (
                block_table_rows[gid].gpu[req_idx, :needed_blocks]
                .detach()
                .to("cpu", dtype=torch.int64)
            )
            block_indices = torch.div(pos_seg, int(block_size), rounding_mode="floor")
            block_numbers = block_row_prefix[block_indices]
            expected = block_numbers * int(block_size) + (pos_seg % int(block_size))
            actual = slot_cpu[gid, start:end]
            if expected.shape != actual.shape or not torch.equal(expected, actual):
                mismatch = (expected != actual).nonzero(as_tuple=False)
                first = int(mismatch[0].item()) if mismatch.numel() > 0 else -1
                exp_first = int(expected[first].item()) if first >= 0 else -1
                act_first = int(actual[first].item()) if first >= 0 else -1
                pos_first = int(pos_seg[first].item()) if first >= 0 else -1
                raise RuntimeError(
                    "TRIATTN_SLOT_MAPPING_VALUE_MISMATCH:"
                    f"gid={gid}:batch_idx={batch_idx}:req_idx={req_idx}:"
                    f"local_idx={first}:pos={pos_first}:"
                    f"expected={exp_first}:actual={act_first}:"
                    f"needed_blocks={needed_blocks}:current_blocks={current_blocks}"
                )
