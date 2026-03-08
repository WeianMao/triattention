"""Worker-side block-table reclaim synchronization helpers for TriAttention runtime."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def apply_worker_block_reclaim_events(
    *,
    base_runner: Any,
    events: list[dict[str, Any]] | None,
) -> None:
    """Apply reclaim shrink to worker-side block tables after compression.

    In vLLM V1, the block table lives at ``base_runner.input_batch.block_table``
    and tracks per-request block counts in ``num_blocks_per_row``.  After
    compression compacts KV cache data into fewer blocks, we must update these
    counters so that subsequent ``append_row()`` calls start from the correct
    offset and don't overflow the max-blocks-per-request limit.
    """
    if not isinstance(events, list) or not events:
        return

    # Resolve the vLLM V1 block table.
    input_batch = getattr(base_runner, "input_batch", None)
    block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None
    if block_table_obj is None:
        logger.warning(
            "TriAttention worker reclaim: block table not found. "
            "input_batch=%s block_table=%s",
            type(input_batch).__name__ if input_batch else None,
            type(block_table_obj).__name__ if block_table_obj else None,
        )
        return

    # Resolve request-id → row-index mapping.
    # In vLLM V1, req_id_to_index lives on input_batch, and request states
    # (with block_ids) live in base_runner.requests.
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(req_id_to_index, dict):
        logger.warning(
            "TriAttention worker reclaim: req_id_to_index not found on input_batch. "
            "input_batch=%s",
            type(input_batch).__name__ if input_batch else None,
        )
        return

    # The block table may be a single BlockTable (with num_blocks_per_row) or
    # a MultiGroupBlockTable (with .block_tables list of per-group BlockTables).
    inner_tables = getattr(block_table_obj, "block_tables", None)
    if isinstance(inner_tables, list):
        # MultiGroupBlockTable
        tables = inner_tables
    else:
        # Single BlockTable
        tables = [block_table_obj]

    cache_config = getattr(base_runner, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 16))
    if block_size <= 0:
        block_size = 16

    for event in events:
        if not isinstance(event, dict) or event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        if not isinstance(req_id, str):
            continue
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            continue
        cache_len_after = event.get("cache_len_after")
        if not isinstance(cache_len_after, int) or cache_len_after <= 0:
            continue

        required_blocks = (cache_len_after + block_size - 1) // block_size

        for table in tables:
            num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
            if num_blocks_per_row is None:
                continue
            if not isinstance(num_blocks_per_row, np.ndarray):
                continue
            current = int(num_blocks_per_row[req_index])
            if current > required_blocks:
                num_blocks_per_row[req_index] = required_blocks
                logger.info(
                    "TriAttention worker reclaim: req=%s num_blocks %d -> %d "
                    "(cache_len_after=%d block_size=%d)",
                    req_id, current, required_blocks, cache_len_after, block_size,
                )

        # Also truncate req_state.block_ids (CPU-side block tracking).
        # In vLLM V1, per-request state lives in base_runner.requests dict.
        requests_dict = getattr(base_runner, "requests", None)
        if isinstance(requests_dict, dict):
            req_state = requests_dict.get(req_id)
            if req_state is not None:
                block_ids_attr = getattr(req_state, "block_ids", None)
                if isinstance(block_ids_attr, (list, tuple)):
                    for group_blocks in block_ids_attr:
                        if isinstance(group_blocks, list) and len(group_blocks) > required_blocks:
                            del group_blocks[required_blocks:]
