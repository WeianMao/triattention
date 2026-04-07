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


@dataclass
class MergedRequestStateView:
    base_req_state: Any
    overridden_num_computed_tokens: int
    fallback_block_ids: Any = None

    @property
    def num_computed_tokens(self) -> int:
        return int(self.overridden_num_computed_tokens)

    @property
    def block_ids(self) -> Any:
        block_ids = getattr(self.base_req_state, "block_ids", None)
        if block_ids is not None:
            return block_ids
        return self.fallback_block_ids

    @block_ids.setter
    def block_ids(self, value: Any) -> None:
        setattr(self.base_req_state, "block_ids", value)


def _resolve_scheduler_output_request_view(*, scheduler_output: Any, req_id: str) -> Any | None:
    scheduled_new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if isinstance(scheduled_new_reqs, list):
        for new_req in scheduled_new_reqs:
            if getattr(new_req, "req_id", None) == req_id:
                block_ids = getattr(new_req, "block_ids", None)
                if block_ids is None:
                    continue
                candidates: list[int] = []
                prompt_token_ids = getattr(new_req, "prompt_token_ids", None)
                if prompt_token_ids is not None:
                    try:
                        candidates.append(len(prompt_token_ids))
                    except Exception:
                        pass
                for attr_name in ("prompt_token_ids_len", "num_prompt_tokens"):
                    raw_value = getattr(new_req, attr_name, None)
                    if raw_value is None:
                        continue
                    try:
                        candidates.append(int(raw_value))
                    except (TypeError, ValueError):
                        continue
                prefill_token_ids = getattr(new_req, "prefill_token_ids", None)
                if prefill_token_ids is not None:
                    try:
                        candidates.append(len(prefill_token_ids))
                    except Exception:
                        pass
                effective_num_computed = max(candidates, default=0)
                return SchedulerOutputRequestStateView(
                    req_id=req_id,
                    num_computed_tokens=int(effective_num_computed),
                    _block_ids=block_ids,
                )
    scheduled_cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    cached_req_ids = getattr(scheduled_cached_reqs, "req_ids", None)
    cached_num_computed_tokens = getattr(
        scheduled_cached_reqs, "num_computed_tokens", None
    )
    if isinstance(cached_req_ids, list) and isinstance(cached_num_computed_tokens, list):
        for idx, cached_req_id in enumerate(cached_req_ids):
            if cached_req_id != req_id:
                continue
            if idx >= len(cached_num_computed_tokens):
                break
            try:
                effective_num_computed = int(cached_num_computed_tokens[idx])
            except Exception:
                continue
            return SchedulerOutputRequestStateView(
                req_id=req_id,
                num_computed_tokens=int(effective_num_computed),
                _block_ids=None,
            )
    return None


def resolve_hook_request_context(
    *,
    base_runner: Any,
    req_id: str,
    scheduler_output: Any | None = None,
) -> HookRequestContext | dict[str, Any]:
    requests = getattr(base_runner, "requests", None)
    requests_state = requests.get(req_id) if isinstance(requests, dict) else None
    requests_num_computed = None
    if requests_state is not None:
        try:
            requests_num_computed = int(getattr(requests_state, "num_computed_tokens", 0) or 0)
        except Exception:
            requests_num_computed = None

    req_states = getattr(base_runner, "req_states", None)
    req_states_req_id_to_index = getattr(req_states, "req_id_to_index", None)
    req_states_num_computed = None
    try:
        req_index = (
            req_states_req_id_to_index.get(req_id)
            if isinstance(req_states_req_id_to_index, dict)
            else None
        )
        req_states_num_computed_gpu = getattr(getattr(req_states, "num_computed_tokens", None), "gpu", None)
        if isinstance(req_index, int) and req_states_num_computed_gpu is not None:
            req_states_num_computed = int(req_states_num_computed_gpu[req_index].item())
    except Exception:
        req_states_num_computed = None

    input_batch = getattr(base_runner, "input_batch", None)
    input_batch_req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    input_batch_num_computed = None
    try:
        req_index = (
            input_batch_req_id_to_index.get(req_id)
            if isinstance(input_batch_req_id_to_index, dict)
            else None
        )
        num_computed_cpu = getattr(input_batch, "num_computed_tokens_cpu", None)
        if isinstance(req_index, int) and num_computed_cpu is not None:
            input_batch_num_computed = int(num_computed_cpu[req_index])
    except Exception:
        input_batch_num_computed = None

    req_state, source = resolve_request_state_view(base_runner, req_id)
    scheduler_output_req_state = None
    if scheduler_output is not None:
        scheduler_output_req_state = _resolve_scheduler_output_request_view(
            scheduler_output=scheduler_output,
            req_id=req_id,
        )
    if req_state is not None:
        try:
            resolved_num_computed = int(getattr(req_state, "num_computed_tokens", 0) or 0)
        except Exception:
            resolved_num_computed = None
        scheduler_num_computed = None
        if scheduler_output_req_state is not None:
            try:
                scheduler_num_computed = int(
                    getattr(scheduler_output_req_state, "num_computed_tokens", 0) or 0
                )
            except Exception:
                scheduler_num_computed = None
        if (
            isinstance(resolved_num_computed, int)
            and isinstance(scheduler_num_computed, int)
            and scheduler_num_computed > resolved_num_computed
        ):
            req_state = MergedRequestStateView(
                base_req_state=req_state,
                overridden_num_computed_tokens=scheduler_num_computed,
                fallback_block_ids=getattr(
                    scheduler_output_req_state, "block_ids", None
                ),
            )
            source = f"{source}+scheduler_output_num_computed"
            resolved_num_computed = scheduler_num_computed
        trace_event(
            "hook_req_state_resolved",
            req_id=repr(req_id),
            source=source,
            resolved_num_computed=resolved_num_computed,
            requests_num_computed=requests_num_computed,
            req_states_num_computed=req_states_num_computed,
            input_batch_num_computed=input_batch_num_computed,
            scheduler_output_num_computed=scheduler_num_computed,
            has_scheduler_output=(scheduler_output is not None),
        )
    if req_state is None:
        if scheduler_output_req_state is not None:
            req_state = scheduler_output_req_state
            if req_state is not None:
                source = "scheduler_output_new_req"
        if req_state is None:
            trace_event(
                "hook_req_state_missing",
                req_id=repr(req_id),
                source=source,
                has_requests=isinstance(requests, dict),
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
