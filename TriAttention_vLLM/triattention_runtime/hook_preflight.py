"""Preflight helpers for TriAttention runtime compression hook."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hook_group_pipeline import normalize_mutable_block_ids_by_group
from .runner_struct_compat import resolve_request_state_view


@dataclass(frozen=True)
class HookRequestContext:
    req_state: Any
    req_runtime_state: Any


@dataclass(frozen=True)
class HookCompactionInputs:
    block_size: int
    mutable_block_ids_by_group: list[list[int]]


def resolve_hook_request_context(*, base_runner: Any, req_id: str) -> HookRequestContext | dict[str, Any]:
    req_state, _source = resolve_request_state_view(base_runner, req_id)
    if req_state is None:
        return {"applied": False, "reason": "req_state_not_found"}
    state_store = getattr(base_runner, "_triattention_state_store", None)
    req_runtime_state = (
        state_store.get(req_id)
        if state_store is not None and hasattr(state_store, "get")
        else None
    )
    return HookRequestContext(req_state=req_state, req_runtime_state=req_runtime_state)


def resolve_hook_compaction_inputs(
    *,
    base_runner: Any,
    original_block_ids_by_group: Any,
) -> HookCompactionInputs | dict[str, Any]:
    kv_caches = getattr(base_runner, "kv_caches", None)
    cache_config = getattr(base_runner, "cache_config", None)
    if not isinstance(kv_caches, list) or cache_config is None:
        return {"applied": False, "reason": "kv_cache_unavailable"}

    block_size = int(getattr(cache_config, "block_size", 0))
    if block_size <= 0:
        return {"applied": False, "reason": "invalid_block_size"}

    if not original_block_ids_by_group:
        return {"applied": False, "reason": "missing_block_ids"}
    if not isinstance(original_block_ids_by_group, (list, tuple)):
        return {"applied": False, "reason": "invalid_block_ids_container"}

    mutable_block_ids_by_group = normalize_mutable_block_ids_by_group(original_block_ids_by_group)
    if mutable_block_ids_by_group is None:
        return {"applied": False, "reason": "invalid_block_ids_container"}

    return HookCompactionInputs(
        block_size=block_size,
        mutable_block_ids_by_group=mutable_block_ids_by_group,
    )
