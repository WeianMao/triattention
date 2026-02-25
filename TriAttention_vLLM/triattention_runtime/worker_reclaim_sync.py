"""Worker-side block-table reclaim synchronization helpers for TriAttention V2."""

from __future__ import annotations

import os
from typing import Any

import torch


def apply_worker_block_reclaim_events(
    *,
    base_runner: Any,
    events: list[dict[str, Any]] | None,
) -> None:
    """Apply reclaim shrink to worker-side BlockTables before input preparation."""
    if not isinstance(events, list) or not events:
        return

    block_tables = getattr(base_runner, "block_tables", None)
    req_states = getattr(base_runner, "req_states", None)
    req_id_to_index = getattr(req_states, "req_id_to_index", None)
    if block_tables is None or not isinstance(req_id_to_index, dict):
        return
    block_table_rows = getattr(block_tables, "block_tables", None)
    num_blocks = getattr(block_tables, "num_blocks", None)
    if not isinstance(block_table_rows, list) or num_blocks is None:
        return

    staged_any = False
    for event in events:
        if not isinstance(event, dict) or event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        if not isinstance(req_id, str):
            continue
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            continue
        reclaim = event.get("block_reclaim")
        if not isinstance(reclaim, dict):
            continue
        groups = reclaim.get("groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            gid = group.get("gid")
            block_ids_after = group.get("block_ids_after")
            if (
                not isinstance(gid, int)
                or gid < 0
                or gid >= len(block_table_rows)
                or not isinstance(block_ids_after, list)
            ):
                continue
            normalized = [int(x) for x in block_ids_after]
            if normalized:
                block_table_rows[gid].stage_write(req_index, 0, normalized)
                staged_any = True
            num_blocks.np[gid, req_index] = len(normalized)

    if staged_any:
        block_tables.apply_staged_writes()
    else:
        try:
            num_blocks.copy_to_uva()
        except Exception:
            pass

    if os.environ.get("TRIATTN_DEBUG_VALIDATE_WORKER_RECLAIM", "0") == "1":
        _debug_validate_worker_reclaim_views(
            base_runner=base_runner,
            events=events,
        )


def _debug_validate_worker_reclaim_views(
    *,
    base_runner: Any,
    events: list[dict[str, Any]],
) -> None:
    """Debug-only assertion: worker BlockTables matches reclaim payload."""
    block_tables = getattr(base_runner, "block_tables", None)
    req_states = getattr(base_runner, "req_states", None)
    req_id_to_index = getattr(req_states, "req_id_to_index", None)
    if block_tables is None or not isinstance(req_id_to_index, dict):
        return
    block_table_rows = getattr(block_tables, "block_tables", None)
    num_blocks = getattr(block_tables, "num_blocks", None)
    if (
        not isinstance(block_table_rows, list)
        or num_blocks is None
        or not hasattr(num_blocks, "np")
    ):
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    for event in events:
        if not isinstance(event, dict) or event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        if not isinstance(req_id, str):
            continue
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            continue
        reclaim = event.get("block_reclaim")
        if not isinstance(reclaim, dict):
            continue
        groups = reclaim.get("groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            gid = group.get("gid")
            block_ids_after = group.get("block_ids_after")
            if (
                not isinstance(gid, int)
                or gid < 0
                or gid >= len(block_table_rows)
                or not isinstance(block_ids_after, list)
            ):
                continue
            expected = [int(x) for x in block_ids_after]
            actual_num_blocks = int(num_blocks.np[gid, req_index])
            if actual_num_blocks != len(expected):
                raise RuntimeError(
                    "TRIATTN_WORKER_RECLAIM_NUM_BLOCKS_MISMATCH:"
                    f"req={req_id}:req_index={req_index}:gid={gid}:"
                    f"expected={len(expected)}:actual={actual_num_blocks}"
                )
            if not expected:
                continue
            actual_prefix = (
                block_table_rows[gid].gpu[req_index, : len(expected)]
                .detach()
                .to("cpu", dtype=torch.int64)
                .tolist()
            )
            if actual_prefix != expected:
                raise RuntimeError(
                    "TRIATTN_WORKER_RECLAIM_PREFIX_MISMATCH:"
                    f"req={req_id}:req_index={req_index}:gid={gid}:"
                    f"expected={expected}:actual={actual_prefix}"
                )
