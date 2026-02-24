"""Base-runner compression hook implementation."""

from __future__ import annotations

from typing import Any, Callable

import torch

from .config import TriAttentionV2Config
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
from .selector_hf import build_speckv_selector as _build_speckv_selector_impl
from .signals import CompressionSignal

# Selector implementation moved to triattention_v2/selector_hf.py (D-017).

def make_runner_compression_hook(
    base_runner: Any,
    config: TriAttentionV2Config,
) -> Callable[..., dict[str, Any]]:
    """Create a hook function bound to a concrete base runner."""
    # Route through extracted selector module. `_build_speckv_selector` symbol is
    # kept below for unit tests that monkeypatch the hook_impl-local name.
    try:
        (
            select_keep_indices,
            select_keep_indices_for_group,
            selector_status,
        ) = _build_speckv_selector(config, base_runner=base_runner)
    except TypeError:
        # Backward-compatible path for unit tests that monkeypatch the selector
        # builder with the legacy single-arg signature.
        (
            select_keep_indices,
            select_keep_indices_for_group,
            selector_status,
        ) = _build_speckv_selector(config)
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
        req_ctx = resolve_hook_request_context(base_runner=base_runner, req_id=req_id)
        if isinstance(req_ctx, dict):
            return req_ctx
        req_state = req_ctx.req_state
        req_runtime_state = req_ctx.req_runtime_state
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
_build_speckv_selector = _build_speckv_selector_impl


def install_runner_compression_hook(
    base_runner: Any,
    config: TriAttentionV2Config,
) -> None:
    """Install default hook on the underlying base runner if missing."""
    if hasattr(base_runner, "triattention_apply_compression"):
        return
    setattr(
        base_runner,
        "triattention_apply_compression",
        make_runner_compression_hook(base_runner=base_runner, config=config),
    )
