"""Group-level compaction pipeline orchestration for TriAttention V2 hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .layout_engine import execute_group_compaction
from .plan_models import PlacementPlan, ReclaimEvent, ReclaimGroup
from .selection_planner import prepare_group_layer_compactions
from .signals import CompressionSignal


@dataclass(frozen=True)
class GroupPipelineOutcome:
    cache_len_after: int
    selection_mode: str
    block_reclaim_groups: list[ReclaimGroup]
    mutable_block_ids_by_group: list[list[int] | None]


def normalize_mutable_block_ids_by_group(
    original_block_ids_by_group: Any,
) -> list[list[int] | None] | None:
    if not original_block_ids_by_group:
        return None
    if not isinstance(original_block_ids_by_group, (list, tuple)):
        return None
    mutable_block_ids_by_group: list[list[int] | None] = []
    for group_block_ids in original_block_ids_by_group:
        if not isinstance(group_block_ids, (list, tuple)):
            mutable_block_ids_by_group.append(None)
            continue
        mutable_block_ids_by_group.append([int(block_id) for block_id in group_block_ids])
    return mutable_block_ids_by_group


def run_group_compaction_pipeline(
    *,
    req_id: str,
    signal: CompressionSignal,
    config: TriAttentionRuntimeConfig,
    strict_triton_required: bool,
    num_computed_tokens: int,
    effective_tokens: int,
    budget_total: int,
    block_size: int,
    mutable_block_ids_by_group: list[list[int] | None],
    group_tensors: dict[int, list[tuple[int, torch.Tensor]]],
    select_keep_indices: Callable[..., dict[str, Any] | None] | None,
    select_keep_indices_for_group: Callable[..., dict[str, Any] | None] | None,
    shared_compact_fn: Callable[..., Any],
    per_head_compact_fn: Callable[..., Any],
    gather_dense_fn: Callable[..., torch.Tensor] | None = None,
) -> GroupPipelineOutcome | dict[str, Any]:
    compacted_any_group = False
    cache_len_after: int | None = None
    expected_cache_len_after: int | None = None
    selection_mode = "fallback"
    block_reclaim_groups: list[ReclaimGroup] = []

    for gid, normalized_block_ids in enumerate(mutable_block_ids_by_group):
        if not normalized_block_ids:
            continue
        layer_tensors = group_tensors.get(gid)
        if not layer_tensors:
            continue
        group_capacity_tokens = len(normalized_block_ids) * block_size
        group_total_tokens = min(effective_tokens, group_capacity_tokens)
        if group_total_tokens <= 0:
            continue
        group_prefill_len = min(int(signal.prefill_len), group_total_tokens)
        group_budget_total = min(budget_total, group_total_tokens)
        round_start = int(max(0, num_computed_tokens))
        group_cache_len_after: int | None = None
        try:
            group_selection = prepare_group_layer_compactions(
                req_id=req_id,
                gid=gid,
                layer_tensors=layer_tensors,
                normalized_block_ids=normalized_block_ids,
                block_size=block_size,
                group_total_tokens=group_total_tokens,
                group_prefill_len=group_prefill_len,
                protect_prefill=signal.protect_prefill,
                round_start=round_start,
                group_budget_total=group_budget_total,
                config=config,
                strict_triton_required=strict_triton_required,
                select_keep_indices=select_keep_indices,
                select_keep_indices_for_group=select_keep_indices_for_group,
                gather_dense_fn=gather_dense_fn,
            )
        except ValueError as exc:
            if str(exc) == "prefill_exceeds_budget":
                return {"applied": False, "reason": "prefill_exceeds_budget"}
            raise
        prepared_layer_compactions = group_selection.tasks
        selection_mode = group_selection.selection_mode

        try:
            group_outcome = execute_group_compaction(
                req_id=req_id,
                gid=gid,
                normalized_block_ids=normalized_block_ids,
                tasks=prepared_layer_compactions,
                block_size=block_size,
                total_tokens=group_total_tokens,
                enable_experimental_block_reclaim=config.enable_experimental_block_reclaim,
                require_physical_reclaim=config.require_physical_reclaim,
                shared_compact_fn=shared_compact_fn,
                per_head_compact_fn=per_head_compact_fn,
            )
        except Exception as exc:
            first_layer_idx = (
                prepared_layer_compactions[0].layer_idx if prepared_layer_compactions else -1
            )
            return {
                "applied": False,
                "reason": f"compaction_failed:g{gid}:l{first_layer_idx}:{type(exc).__name__}",
            }

        for layer_idx, layer_compaction in group_outcome.layer_results:
            layer_cache_len_after = int(layer_compaction.cache_len_after)
            if expected_cache_len_after is None:
                expected_cache_len_after = layer_cache_len_after
            elif layer_cache_len_after != expected_cache_len_after:
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:inconsistent_cache_len_after:"
                    f"req={req_id}:gid={gid}:layer={layer_idx}:"
                    f"expected={expected_cache_len_after}:actual={layer_cache_len_after}"
                )
            cache_len_after = layer_cache_len_after
            if group_cache_len_after is None:
                group_cache_len_after = layer_cache_len_after
            compacted_any_group = True

        if config.enable_experimental_block_reclaim and group_cache_len_after is not None:
            mutable_block_ids_by_group[gid] = list(group_outcome.kept_block_ids)
            if group_outcome.reclaim_group is not None:
                block_reclaim_groups.append(group_outcome.reclaim_group)

    if not compacted_any_group or cache_len_after is None:
        return {"applied": False, "reason": "no_compactable_groups"}

    return GroupPipelineOutcome(
        cache_len_after=int(cache_len_after),
        selection_mode=str(selection_mode),
        block_reclaim_groups=block_reclaim_groups,
        mutable_block_ids_by_group=mutable_block_ids_by_group,
    )


def finalize_hook_placement_result(
    *,
    req_state: Any,
    original_block_ids_by_group: Any,
    config: TriAttentionRuntimeConfig,
    selector_status: str,
    outcome: GroupPipelineOutcome,
    effective_tokens: int,
    budget_total: int,
    recent_unabsorbed_tokens: int | None,
) -> dict[str, Any]:
    block_reclaim_payload: ReclaimEvent | None = None
    if config.enable_experimental_block_reclaim and outcome.block_reclaim_groups:
        reassigned_block_ids = []
        for idx, group_block_ids in enumerate(outcome.mutable_block_ids_by_group):
            if group_block_ids is None:
                reassigned_block_ids.append(original_block_ids_by_group[idx])
            else:
                reassigned_block_ids.append(group_block_ids)
        req_state.block_ids = (
            tuple(reassigned_block_ids)
            if isinstance(original_block_ids_by_group, tuple)
            else reassigned_block_ids
        )
        block_reclaim_payload = ReclaimEvent(
            mode="truncate_tail",
            groups=outcome.block_reclaim_groups,
        )

    placement_plan = PlacementPlan(
        cache_len_after=int(outcome.cache_len_after),
        selector_status=str(selector_status),
        selection_mode=str(outcome.selection_mode),
        effective_tokens_before=int(effective_tokens),
        budget_total=int(budget_total),
        recent_unabsorbed_tokens=(
            int(recent_unabsorbed_tokens)
            if isinstance(recent_unabsorbed_tokens, int)
            else None
        ),
        block_reclaim=block_reclaim_payload,
    )
    return placement_plan.to_hook_result_dict()
