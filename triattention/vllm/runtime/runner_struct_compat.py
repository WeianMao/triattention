"""Compatibility helpers for differing vLLM runner state layouts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


def debug_v1_override_path_enabled() -> bool:
    return os.environ.get("TRIATTN_DEBUG_ENABLE_V1_OVERRIDE_PATH", "0") == "1"


def debug_input_batch_req_state_enabled() -> bool:
    return os.environ.get("TRIATTN_DEBUG_ALLOW_INPUT_BATCH_REQ_STATE", "0") == "1"


def _build_req_id_map_from_input_batch(input_batch: Any) -> dict[Any, int] | None:
    if input_batch is None:
        return None
    req_ids_attr = getattr(input_batch, "req_ids", None)
    try:
        req_ids = req_ids_attr() if callable(req_ids_attr) else req_ids_attr
    except Exception:
        req_ids = None
    if not isinstance(req_ids, list):
        req_ids = getattr(input_batch, "_req_ids", None)
    if not isinstance(req_ids, list):
        return None
    rebuilt = {
        req_id: idx
        for idx, req_id in enumerate(req_ids)
        if req_id is not None
    }
    return rebuilt or None


def resolve_req_id_to_index(base_runner: Any) -> tuple[dict[Any, int] | None, str]:
    req_states = getattr(base_runner, "req_states", None)
    req_id_to_index = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    if isinstance(req_id_to_index, dict) and len(req_id_to_index) > 0:
        return req_id_to_index, "req_states"

    input_batch = getattr(base_runner, "input_batch", None)
    req_id_to_index = getattr(input_batch, "req_id_to_index", None) if input_batch is not None else None
    if isinstance(req_id_to_index, dict) and len(req_id_to_index) > 0:
        return req_id_to_index, "input_batch"

    rebuilt = _build_req_id_map_from_input_batch(input_batch)
    if isinstance(rebuilt, dict) and len(rebuilt) > 0:
        return rebuilt, "input_batch_req_ids"

    return None, "none"



@dataclass
class CompatRequestStateView:
    """Minimal per-request view spanning vLLM's old/default and formal paths."""

    base_runner: Any
    req_id: str
    req_index: int
    requests_state: Any | None = None

    @property
    def num_computed_tokens(self) -> int:
        req_states = getattr(self.base_runner, "req_states", None)
        num_computed = getattr(getattr(req_states, "num_computed_tokens", None), "gpu", None)
        if torch.is_tensor(num_computed):
            return int(num_computed[self.req_index].item())
        input_batch = getattr(self.base_runner, "input_batch", None)
        num_computed_cpu = getattr(input_batch, "num_computed_tokens_cpu", None)
        if num_computed_cpu is not None:
            try:
                return int(num_computed_cpu[self.req_index])
            except Exception:
                pass
        return 0

    @property
    def block_ids(self) -> tuple[list[int], ...] | None:
        block_tables = getattr(self.base_runner, "block_tables", None)
        table_list = getattr(block_tables, "block_tables", None) if block_tables is not None else None
        num_blocks = getattr(block_tables, "num_blocks", None) if block_tables is not None else None
        num_blocks_np = getattr(num_blocks, "np", None) if num_blocks is not None else None
        if isinstance(table_list, list) and num_blocks_np is not None:
            block_ids_by_group: list[list[int]] = []
            valid = True
            for gid, table in enumerate(table_list):
                try:
                    count = int(num_blocks_np[gid, self.req_index])
                except Exception:
                    valid = False
                    break
                if count <= 0:
                    block_ids_by_group.append([])
                    continue
                row = getattr(table, "gpu", None)
                if not torch.is_tensor(row):
                    valid = False
                    break
                block_ids_by_group.append([int(x) for x in row[self.req_index, :count].tolist()])
            if valid and any(group_block_ids for group_block_ids in block_ids_by_group):
                return tuple(block_ids_by_group)

        fallback_block_ids = getattr(self.requests_state, "block_ids", None)
        if isinstance(fallback_block_ids, (list, tuple)):
            normalized: list[list[int]] = []
            for group_block_ids in fallback_block_ids:
                if not isinstance(group_block_ids, (list, tuple)):
                    normalized.append([])
                    continue
                normalized.append([int(block_id) for block_id in group_block_ids])
            return tuple(normalized)
        return None

    @block_ids.setter
    def block_ids(self, new_block_ids: Any) -> None:
        block_tables = getattr(self.base_runner, "block_tables", None)
        if not isinstance(new_block_ids, (list, tuple)):
            raise TypeError("block_ids must be a list/tuple by group")
        normalized = tuple(
            [int(block_id) for block_id in group_block_ids]
            if isinstance(group_block_ids, (list, tuple))
            else []
            for group_block_ids in new_block_ids
        )
        if block_tables is not None:
            block_tables.append_block_ids(self.req_index, normalized, overwrite=True)
            block_tables.apply_staged_writes()
        if self.requests_state is not None:
            setattr(self.requests_state, "block_ids", normalized)


def resolve_request_state_view(base_runner: Any, req_id: str) -> tuple[Any | None, str]:
    requests = getattr(base_runner, "requests", None)
    requests_state = requests.get(req_id) if isinstance(requests, dict) else None
    req_states = getattr(base_runner, "req_states", None)
    req_id_to_index = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    if isinstance(req_id_to_index, dict):
        req_index = req_id_to_index.get(req_id)
        if isinstance(req_index, int):
            return CompatRequestStateView(
                base_runner=base_runner,
                req_id=req_id,
                req_index=req_index,
                requests_state=requests_state,
            ), "req_states_proxy"

    input_batch = getattr(base_runner, "input_batch", None)
    req_id_to_index = getattr(input_batch, "req_id_to_index", None) if input_batch is not None else None
    if isinstance(req_id_to_index, dict):
        req_index = req_id_to_index.get(req_id)
        if isinstance(req_index, int):
            return CompatRequestStateView(
                base_runner=base_runner,
                req_id=req_id,
                req_index=req_index,
                requests_state=requests_state,
            ), "input_batch_proxy"

    rebuilt = _build_req_id_map_from_input_batch(input_batch)
    if isinstance(rebuilt, dict):
        req_index = rebuilt.get(req_id)
        if isinstance(req_index, int):
            return CompatRequestStateView(
                base_runner=base_runner,
                req_id=req_id,
                req_index=req_index,
                requests_state=requests_state,
            ), "input_batch_req_ids_proxy"

    if requests_state is not None:
        return requests_state, "requests"

    return None, "none"
