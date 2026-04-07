"""Base-runner compression hook implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import torch

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .kv_compaction import (
    build_keep_token_indices,
    compact_request_kv_in_place,  # compatibility for tests monkeypatching hook_impl symbol
    compact_request_kv_in_place_per_head,  # compatibility for tests monkeypatching
    gather_request_k_dense,  # compatibility for tests importing hook_impl symbol
)
from .hook_runtime_context import build_hook_runtime_context
from .hook_group_pipeline import (
    finalize_hook_placement_result,
    run_group_compaction_pipeline,
)
from .hook_preflight import resolve_hook_compaction_inputs, resolve_hook_request_context
from .kv_group_resolver import resolve_group_tensors as _resolve_group_tensors
from .selector_hf import build_triattention_selector as _build_triattention_selector_impl
from .signals import CompressionSignal

# Selector implementation moved to triattention_runtime/selector_hf.py (D-017).

_DEBUG_DUMP_REQUEST_TOKEN_STATE_DIR = os.environ.get(
    "TRIATTN_DEBUG_DUMP_REQUEST_TOKEN_STATE_DIR", ""
).strip()
_DEBUG_DUMP_REQUEST_TOKEN_STATE_MIN_EST_CACHE_LEN = int(
    os.environ.get("TRIATTN_DEBUG_DUMP_REQUEST_TOKEN_STATE_MIN_EST_CACHE_LEN", "0") or 0
)
_DEBUG_DUMP_REQUEST_TOKEN_STATE_MIN_NUM_COMPUTED = int(
    os.environ.get("TRIATTN_DEBUG_DUMP_REQUEST_TOKEN_STATE_MIN_NUM_COMPUTED", "0") or 0
)
_DEBUG_DUMP_REQUEST_TOKEN_STATE_MIN_SIGNAL_STEP = int(
    os.environ.get("TRIATTN_DEBUG_DUMP_REQUEST_TOKEN_STATE_MIN_SIGNAL_STEP", "0") or 0
)
_SEEN_REQUEST_TOKEN_STATE_DUMPS: set[tuple[str, int]] = set()


def _summarize_token_like_value(value: Any) -> dict[str, Any] | None:
    if value is None:
        return {"kind": "none"}
    if torch.is_tensor(value):
        flat = value.detach().to(device="cpu").reshape(-1)
        sample_head = flat[:8].tolist()
        sample_tail = flat[-8:].tolist() if flat.numel() > 8 else sample_head
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "head": [int(x) if isinstance(x, (int, bool)) else float(x) for x in sample_head],
            "tail": [int(x) if isinstance(x, (int, bool)) else float(x) for x in sample_tail],
        }

    if isinstance(value, (list, tuple)):
        if value and all(isinstance(x, int) for x in value[: min(16, len(value))]):
            head = [int(x) for x in value[:8]]
            tail = [int(x) for x in value[-8:]] if len(value) > 8 else head
            return {
                "kind": type(value).__name__,
                "len": int(len(value)),
                "head": head,
                "tail": tail,
            }
        return {
            "kind": type(value).__name__,
            "len": int(len(value)),
        }

    if isinstance(value, dict):
        return {
            "kind": "dict",
            "keys_sample": [str(k) for k in list(value.keys())[:8]],
            "len": int(len(value)),
        }

    if isinstance(value, (int, float, bool, str)):
        return {"kind": type(value).__name__, "value": value}
    return {"kind": type(value).__name__}


def _maybe_dump_request_token_state(
    *,
    req_id: str,
    signal: CompressionSignal,
    req_state: Any,
) -> None:
    if not _DEBUG_DUMP_REQUEST_TOKEN_STATE_DIR:
        return
    est_cache_len = int(getattr(signal, "estimated_cache_len", 0) or 0)
    if est_cache_len < _DEBUG_DUMP_REQUEST_TOKEN_STATE_MIN_EST_CACHE_LEN:
        return
    signal_step = int(getattr(signal, "step", -1) or -1)
    if signal_step < _DEBUG_DUMP_REQUEST_TOKEN_STATE_MIN_SIGNAL_STEP:
        return
    try:
        num_computed_tokens = int(getattr(req_state, "num_computed_tokens", 0) or 0)
    except Exception:
        num_computed_tokens = 0
    if num_computed_tokens < _DEBUG_DUMP_REQUEST_TOKEN_STATE_MIN_NUM_COMPUTED:
        return
    dump_key = (str(req_id), est_cache_len)
    if dump_key in _SEEN_REQUEST_TOKEN_STATE_DUMPS:
        return
    _SEEN_REQUEST_TOKEN_STATE_DUMPS.add(dump_key)

    payload: dict[str, Any] = {
        "req_id": str(req_id),
        "signal_step": signal_step,
        "estimated_cache_len": est_cache_len,
        "req_state_type": type(req_state).__name__,
        "num_computed_tokens": num_computed_tokens,
    }
    requests_state = getattr(req_state, "requests_state", None)
    payload["requests_state_type"] = (
        None if requests_state is None else type(requests_state).__name__
    )

    def _capture_attrs(obj: Any, prefix: str) -> None:
        if obj is None:
            return
        tokenish: dict[str, Any] = {}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            lname = name.lower()
            if not any(
                key in lname for key in ("token", "ids", "prompt", "output", "generated")
            ):
                continue
            try:
                value = getattr(obj, name)
            except Exception as exc:
                tokenish[name] = {"error": type(exc).__name__}
                continue
            if callable(value):
                continue
            tokenish[name] = _summarize_token_like_value(value)
        payload[prefix] = tokenish

    _capture_attrs(req_state, "req_state_tokenish_attrs")
    _capture_attrs(requests_state, "requests_state_tokenish_attrs")

    base_runner = getattr(req_state, "base_runner", None)
    req_index = getattr(req_state, "req_index", None)
    if base_runner is not None:
        payload["base_runner_type"] = type(base_runner).__name__
        payload["req_index"] = None if req_index is None else int(req_index)
        input_batch = getattr(base_runner, "input_batch", None)
        req_states = getattr(base_runner, "req_states", None)
        payload["input_batch_type"] = None if input_batch is None else type(input_batch).__name__
        payload["req_states_type"] = None if req_states is None else type(req_states).__name__

        def _capture_container_rows(obj: Any, prefix: str) -> None:
            if obj is None:
                return
            tokenish: dict[str, Any] = {}
            for name in dir(obj):
                if name.startswith("_"):
                    continue
                lname = name.lower()
                if not any(
                    key in lname for key in ("token", "ids", "prompt", "output", "generated")
                ):
                    continue
                try:
                    value = getattr(obj, name)
                except Exception as exc:
                    tokenish[name] = {"error": type(exc).__name__}
                    continue
                if callable(value):
                    continue
                if req_index is not None:
                    try:
                        row = value[req_index]
                    except Exception:
                        row = None
                    if row is not None:
                        tokenish[f"{name}[req_index]"] = _summarize_token_like_value(row)
                tokenish[name] = _summarize_token_like_value(value)
            payload[prefix] = tokenish

        _capture_container_rows(input_batch, "input_batch_tokenish_attrs")
        _capture_container_rows(req_states, "req_states_tokenish_attrs")
        if input_batch is not None and isinstance(req_index, int):
            try:
                active_len = int(getattr(input_batch, "num_tokens_no_spec")[req_index])
            except Exception:
                active_len = 0
            payload["input_batch_active_len"] = active_len
            if active_len > 0:
                try:
                    token_row = getattr(input_batch, "token_ids_cpu")[req_index][:active_len]
                    payload["input_batch_active_token_ids"] = [int(x) for x in token_row.tolist()]
                except Exception as exc:
                    payload["input_batch_active_token_ids_error"] = type(exc).__name__
                try:
                    is_token_row = getattr(input_batch, "is_token_ids")[req_index][:active_len]
                    payload["input_batch_active_is_token_ids"] = [bool(x) for x in is_token_row.tolist()]
                except Exception as exc:
                    payload["input_batch_active_is_token_ids_error"] = type(exc).__name__

    dump_dir = Path(_DEBUG_DUMP_REQUEST_TOKEN_STATE_DIR)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / (
        f"req_{str(req_id).replace('/', '_')}_cache_{est_cache_len}.json"
    )
    dump_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def make_runner_compression_hook(
    base_runner: Any,
    config: TriAttentionRuntimeConfig,
) -> Callable[..., dict[str, Any]]:
    """Create a hook function bound to a concrete base runner."""
    # Route through extracted selector module. `_build_triattention_selector` symbol is
    # kept below for unit tests that monkeypatch the hook_impl-local name.
    try:
        (
            select_keep_indices,
            select_keep_indices_for_group,
            selector_status,
        ) = _build_triattention_selector(config, base_runner=base_runner)
    except TypeError:
        # Backward-compatible path for unit tests that monkeypatch the selector
        # builder with the legacy single-arg signature.
        (
            select_keep_indices,
            select_keep_indices_for_group,
            selector_status,
        ) = _build_triattention_selector(config)
    group_tensors_cache: dict[int, list[tuple[int, torch.Tensor]]] | None = None
    compressed_once: set[str] = set()

    def _get_group_tensors() -> dict[int, list[tuple[int, torch.Tensor]]]:
        nonlocal group_tensors_cache
        if group_tensors_cache is None:
            group_tensors_cache = _resolve_group_tensors(base_runner)
        return group_tensors_cache

    def _hook(req_id: str, signal: CompressionSignal, scheduler_output: Any) -> dict[str, Any]:
        setattr(base_runner, "_triattention_active_req_id", req_id)
        strict_triton_required = bool(
            config.enable_experimental_kv_compaction and config.require_triton_scoring
        )
        req_ctx = resolve_hook_request_context(
            base_runner=base_runner,
            req_id=req_id,
            scheduler_output=scheduler_output,
        )
        if isinstance(req_ctx, dict):
            return req_ctx
        req_state = req_ctx.req_state
        req_runtime_state = req_ctx.req_runtime_state
        _maybe_dump_request_token_state(
            req_id=req_id,
            signal=signal,
            req_state=req_state,
        )
        recent_unabsorbed_tokens: int | None = None
        cache_config_hint = getattr(base_runner, "cache_config", None)
        block_size_hint = int(getattr(cache_config_hint, "block_size", 0))
        if block_size_hint <= 0:
            block_size_hint = 1
        original_block_ids_by_group = getattr(req_state, "block_ids", None)
        runtime_ctx = build_hook_runtime_context(
            base_runner=base_runner,
            config=config,
            req_id=req_id,
            req_state=req_state,
            req_runtime_state=req_runtime_state,
            signal=signal,
            scheduler_output=scheduler_output,
            compressed_once=compressed_once,
            original_block_ids_by_group=original_block_ids_by_group,
            block_size_hint=block_size_hint,
        )
        num_computed_tokens = runtime_ctx.num_computed_tokens
        effective_tokens = runtime_ctx.effective_tokens
        budget_total = runtime_ctx.budget_total
        recent_unabsorbed_tokens = runtime_ctx.recent_unabsorbed_tokens
        should_defer_recompress = runtime_ctx.should_defer_recompress
        if effective_tokens <= budget_total or should_defer_recompress:
            return {
                "applied": False,
                "reason": "under_budget",
                "cache_len_after": effective_tokens,
            }

        if not config.enable_experimental_kv_compaction:
            keep_indices_plan = build_keep_token_indices(
                total_tokens=effective_tokens,
                kv_budget=config.kv_budget,
                prefill_len=signal.prefill_len,
                protect_prefill=signal.protect_prefill,
                include_prefill_in_budget=config.include_prefill_in_budget,
            )
            if keep_indices_plan is None:
                return {"applied": False, "reason": "prefill_exceeds_budget"}
            return {
                "applied": False,
                "reason": "plan_only",
                "cache_len_after": len(keep_indices_plan),
            }
        if strict_triton_required and select_keep_indices is None:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:selector_unavailable:{selector_status}"
            )

        compaction_inputs = resolve_hook_compaction_inputs(
            base_runner=base_runner,
            original_block_ids_by_group=original_block_ids_by_group,
        )
        if isinstance(compaction_inputs, dict):
            return compaction_inputs
        block_size = compaction_inputs.block_size
        mutable_block_ids_by_group = compaction_inputs.mutable_block_ids_by_group

        group_tensors = _get_group_tensors()
        pipeline_out = run_group_compaction_pipeline(
            req_id=req_id,
            signal=signal,
            config=config,
            strict_triton_required=strict_triton_required,
            num_computed_tokens=num_computed_tokens,
            effective_tokens=effective_tokens,
            budget_total=budget_total,
            block_size=block_size,
            mutable_block_ids_by_group=mutable_block_ids_by_group,
            group_tensors=group_tensors,
            select_keep_indices=select_keep_indices,
            select_keep_indices_for_group=select_keep_indices_for_group,
            shared_compact_fn=compact_request_kv_in_place,
            per_head_compact_fn=compact_request_kv_in_place_per_head,
            gather_dense_fn=gather_request_k_dense,
        )
        if isinstance(pipeline_out, dict):
            return pipeline_out

        compressed_once.add(req_id)
        return finalize_hook_placement_result(
            req_state=req_state,
            original_block_ids_by_group=original_block_ids_by_group,
            config=config,
            selector_status=str(selector_status),
            outcome=pipeline_out,
            effective_tokens=effective_tokens,
            budget_total=budget_total,
            recent_unabsorbed_tokens=recent_unabsorbed_tokens,
        )

    return _hook


# Keep hook_impl-local symbol for backward compatibility with existing tests
# and monkeypatch call sites, while routing runtime behavior to the extracted
# selector implementation module.
_build_triattention_selector = _build_triattention_selector_impl


def install_runner_compression_hook(
    base_runner: Any,
    config: TriAttentionRuntimeConfig,
) -> None:
    """Install default hook on the underlying base runner if missing."""
    if hasattr(base_runner, "triattention_apply_compression"):
        return
    setattr(
        base_runner,
        "triattention_apply_compression",
        make_runner_compression_hook(base_runner=base_runner, config=config),
    )
