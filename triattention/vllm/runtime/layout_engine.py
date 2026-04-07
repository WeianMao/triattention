"""Layout/compaction execution helpers for TriAttention runtime.

This module is the first step of splitting layout/reclaim logic out of
`hook_impl.py`. It intentionally keeps APIs narrow and compatibility-focused.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch

from .constants import TRITON_SCORING_REQUIRED_MARKER
from .kv_compaction import compact_request_kv_in_place, compact_request_kv_in_place_per_head
from .plan_models import KeepPlan, ReclaimGroup

_DEBUG_FORCE_PRESERVE_DROPPED = (
    os.environ.get("TRIATTN_DEBUG_FORCE_PRESERVE_DROPPED", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


def num_required_blocks(total_tokens: int, block_size: int) -> int:
    if total_tokens <= 0:
        return 0
    return (total_tokens + block_size - 1) // block_size


@dataclass(frozen=True)
class LayerCompactionResult:
    cache_len_after: int
    keep_count: int
    before_required_blocks: int
    after_required_blocks: int
    preserve_dropped_tokens: bool


@dataclass(frozen=True)
class PreparedLayerCompaction:
    layer_idx: int
    kv_cache: torch.Tensor
    block_ids: torch.Tensor
    keep_plan: KeepPlan


@dataclass(frozen=True)
class GroupCompactionOutcome:
    layer_results: list[tuple[int, LayerCompactionResult]]
    cache_len_after: int | None
    kept_block_ids: list[int]
    removed_block_ids: list[int]
    reclaim_group: ReclaimGroup | None


def compact_layer_with_keep_plan(
    *,
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    block_size: int,
    keep_plan: KeepPlan,
    total_tokens: int,
    enable_experimental_block_reclaim: bool,
    shared_compact_fn: Any | None = None,
    per_head_compact_fn: Any | None = None,
) -> LayerCompactionResult:
    keep_count = keep_plan.keep_count()
    before_required = num_required_blocks(total_tokens, block_size)
    after_required = num_required_blocks(keep_count, block_size)
    preserve_dropped_tokens = True
    if enable_experimental_block_reclaim and after_required < before_required:
        # Physical reclaim can still happen by truncating tail blocks after
        # compaction. We intentionally keep the stable, order-preserving
        # permutation here because the fill-hole path can preserve the kept
        # token set while silently scrambling prefix order across repeated
        # compressions.
        preserve_dropped_tokens = True
    if _DEBUG_FORCE_PRESERVE_DROPPED:
        preserve_dropped_tokens = True

    if keep_plan.mode == "per_head":
        per_head_fn = per_head_compact_fn or compact_request_kv_in_place_per_head
        cache_len_after = per_head_fn(
            kv_cache=kv_cache,
            block_ids=block_ids,
            block_size=block_size,
            keep_token_indices_per_head=keep_plan.indices,
            total_tokens=total_tokens,
            preserve_dropped_tokens=preserve_dropped_tokens,
        )
    else:
        shared_fn = shared_compact_fn or compact_request_kv_in_place
        cache_len_after = shared_fn(
            kv_cache=kv_cache,
            block_ids=block_ids,
            block_size=block_size,
            keep_token_indices=keep_plan.indices,
            total_tokens=total_tokens,
            preserve_dropped_tokens=preserve_dropped_tokens,
        )

    return LayerCompactionResult(
        cache_len_after=int(cache_len_after),
        keep_count=int(keep_count),
        before_required_blocks=int(before_required),
        after_required_blocks=int(after_required),
        preserve_dropped_tokens=bool(preserve_dropped_tokens),
    )


def compact_prepared_group_layers(
    *,
    tasks: list[PreparedLayerCompaction],
    block_size: int,
    total_tokens: int,
    enable_experimental_block_reclaim: bool,
    shared_compact_fn: Any | None = None,
    per_head_compact_fn: Any | None = None,
) -> list[tuple[int, LayerCompactionResult]]:
    """Apply compaction for all prepared layers in one group.

    Selection is assumed to be finished already (`keep_plan` attached per layer).
    """
    results: list[tuple[int, LayerCompactionResult]] = []
    for task in tasks:
        result = compact_layer_with_keep_plan(
            kv_cache=task.kv_cache,
            block_ids=task.block_ids,
            block_size=block_size,
            keep_plan=task.keep_plan,
            total_tokens=total_tokens,
            enable_experimental_block_reclaim=enable_experimental_block_reclaim,
            shared_compact_fn=shared_compact_fn,
            per_head_compact_fn=per_head_compact_fn,
        )
        results.append((task.layer_idx, result))
    return results


def execute_group_compaction(
    *,
    req_id: str,
    gid: int,
    normalized_block_ids: list[int],
    tasks: list[PreparedLayerCompaction],
    block_size: int,
    total_tokens: int,
    enable_experimental_block_reclaim: bool,
    require_physical_reclaim: bool,
    shared_compact_fn: Any | None = None,
    per_head_compact_fn: Any | None = None,
) -> GroupCompactionOutcome:
    """Execute compaction + reclaim derivation for one KV-cache group."""
    layer_results = compact_prepared_group_layers(
        tasks=tasks,
        block_size=block_size,
        total_tokens=total_tokens,
        enable_experimental_block_reclaim=enable_experimental_block_reclaim,
        shared_compact_fn=shared_compact_fn,
        per_head_compact_fn=per_head_compact_fn,
    )

    cache_len_after: int | None = None
    expected_cache_len_after: int | None = None
    for layer_idx, layer_compaction in layer_results:
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

    kept_block_ids = list(normalized_block_ids)
    removed_block_ids: list[int] = []
    reclaim_group: ReclaimGroup | None = None
    if enable_experimental_block_reclaim and cache_len_after is not None:
        kept_block_ids, removed_block_ids, reclaim_group = truncate_tail_reclaim_group(
            gid=gid,
            normalized_block_ids=normalized_block_ids,
            cache_len_after=cache_len_after,
            block_size=block_size,
        )
        if require_physical_reclaim:
            before_required = num_required_blocks(total_tokens, block_size)
            required_blocks = num_required_blocks(cache_len_after, block_size)
            expected_removed_min = max(0, before_required - required_blocks)
            if expected_removed_min > 0 and len(removed_block_ids) < expected_removed_min:
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:physical_reclaim_missing:"
                    f"req={req_id}:gid={gid}:expected_removed>={expected_removed_min}:"
                    f"actual_removed={len(removed_block_ids)}:"
                    f"effective_tokens={total_tokens}:cache_len_after={cache_len_after}"
                )

    return GroupCompactionOutcome(
        layer_results=layer_results,
        cache_len_after=cache_len_after,
        kept_block_ids=kept_block_ids,
        removed_block_ids=removed_block_ids,
        reclaim_group=reclaim_group,
    )


def truncate_tail_reclaim_group(
    *,
    gid: int,
    normalized_block_ids: list[int],
    cache_len_after: int,
    block_size: int,
) -> tuple[list[int], list[int], ReclaimGroup | None]:
    required_blocks = num_required_blocks(cache_len_after, block_size)
    kept_block_ids = list(normalized_block_ids[:required_blocks])
    removed_block_ids = list(normalized_block_ids[required_blocks:])
    group = None
    if removed_block_ids:
        group = ReclaimGroup(
            gid=gid,
            block_ids_before=list(normalized_block_ids),
            block_ids_after=kept_block_ids,
            block_ids_removed=removed_block_ids,
        )
    return kept_block_ids, removed_block_ids, group


def count_reclaimed_blocks(groups: list[ReclaimGroup]) -> int:
    return sum(len(group.block_ids_removed) for group in groups)
