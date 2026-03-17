"""Preflight helpers for TriAttention runtime compression hook."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .debug_trace import trace_event
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


@dataclass
class SchedulerOutputRequestStateView:
    req_id: str
    num_computed_tokens: int
    _block_ids: Any

    @property
    def block_ids(self) -> Any:
        return self._block_ids

    @block_ids.setter
    def block_ids(self, value: Any) -> None:
        self._block_ids = value


def _resolve_scheduler_output_request_view(*, scheduler_output: Any, req_id: str) -> Any | None:
    scheduled_new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if isinstance(scheduled_new_reqs, list):
        for new_req in scheduled_new_reqs:
            if getattr(new_req, "req_id", None) == req_id:
                block_ids = getattr(new_req, "block_ids", None)
                if block_ids is None:
                    continue
                prefill_token_ids = getattr(new_req, "prefill_token_ids", None)
                prompt_token_ids = getattr(new_req, "prompt_token_ids", None)
                if isinstance(prefill_token_ids, list):
                    effective_num_computed = len(prefill_token_ids)
                elif isinstance(prompt_token_ids, list):
                    effective_num_computed = len(prompt_token_ids)
                else:
                    effective_num_computed = int(getattr(new_req, "num_computed_tokens", 0) or 0)
                return SchedulerOutputRequestStateView(
                    req_id=req_id,
                    num_computed_tokens=int(effective_num_computed),
                    _block_ids=block_ids,
                )
    return None


def resolve_hook_request_context(
    *,
    base_runner: Any,
    req_id: str,
    scheduler_output: Any | None = None,
) -> HookRequestContext | dict[str, Any]:
    req_state, source = resolve_request_state_view(base_runner, req_id)
    if req_state is None:
        if scheduler_output is not None:
            req_state = _resolve_scheduler_output_request_view(
                scheduler_output=scheduler_output,
                req_id=req_id,
            )
            if req_state is not None:
                source = "scheduler_output_new_req"
        if req_state is None:
            req_states = getattr(base_runner, "req_states", None)
            req_states_req_id_to_index = getattr(req_states, "req_id_to_index", None)
            input_batch = getattr(base_runner, "input_batch", None)
            input_batch_req_id_to_index = getattr(input_batch, "req_id_to_index", None)
            trace_event(
                "hook_req_state_missing",
                req_id=repr(req_id),
                source=source,
                has_requests=isinstance(getattr(base_runner, "requests", None), dict),
                has_req_states=(req_states is not None),
                has_req_states_req_id_to_index=isinstance(req_states_req_id_to_index, dict),
                req_states_contains_req_id=(
                    bool(isinstance(req_states_req_id_to_index, dict) and req_id in req_states_req_id_to_index)
                ),
                req_states_keys_sample=(
                    [repr(k) for k in list(req_states_req_id_to_index.keys())[:4]]
                    if isinstance(req_states_req_id_to_index, dict)
                    else None
                ),
                has_input_batch=(input_batch is not None),
                has_input_batch_req_id_to_index=isinstance(input_batch_req_id_to_index, dict),
                input_batch_contains_req_id=(
                    bool(isinstance(input_batch_req_id_to_index, dict) and req_id in input_batch_req_id_to_index)
                ),
                input_batch_keys_sample=(
                    [repr(k) for k in list(input_batch_req_id_to_index.keys())[:4]]
                    if isinstance(input_batch_req_id_to_index, dict)
                    else None
                ),
                has_block_tables=(getattr(base_runner, "block_tables", None) is not None),
                has_scheduler_output=(scheduler_output is not None),
                scheduled_new_req_ids=(
                    [
                        repr(getattr(new_req, "req_id", None))
                        for new_req in (getattr(scheduler_output, "scheduled_new_reqs", None) or [])[:4]
                    ]
                    if scheduler_output is not None
                    else None
                ),
            )
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
