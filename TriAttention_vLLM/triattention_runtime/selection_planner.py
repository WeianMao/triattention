"""Bridge layer: turn HF selector outputs into prepared layout compaction tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .debug_trace import trace_event
from .kv_compaction import build_keep_token_indices, gather_request_k_dense
from .layout_engine import PreparedLayerCompaction
from .plan_models import KeepPlan

_SELECTOR_METRICS_DEBUG = (
    os.environ.get("TRIATTN_RUNTIME_SELECTOR_METRICS_DEBUG", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_DEBUG_DISABLE_GROUP_SELECTOR = (
    os.environ.get("TRIATTN_RUNTIME_DEBUG_DISABLE_GROUP_SELECTOR", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


@dataclass(frozen=True)
class PreparedGroupSelection:
    tasks: list[PreparedLayerCompaction]
    selection_mode: str


def _per_head_debug_metrics(indices: Any) -> dict[str, Any]:
    if isinstance(indices, torch.Tensor):
        keep = indices.detach().to(dtype=torch.long)
    else:
        keep = torch.as_tensor(indices, dtype=torch.long)
    if keep.ndim != 2:
        return {"valid": False, "reason": f"ndim_{int(keep.ndim)}"}
    num_heads = int(keep.shape[0])
    keep_count = int(keep.shape[1])
    if num_heads <= 0 or keep_count <= 0:
        return {
            "valid": True,
            "num_heads": num_heads,
            "keep_count": keep_count,
            "same_slot_all_heads_ratio": 0.0,
            "pair_jaccard_min": 0.0,
            "pair_jaccard_mean": 0.0,
            "pair_jaccard_max": 0.0,
        }

    same_slot_all_heads = (keep == keep[:1]).all(dim=0)
    same_slot_all_heads_ratio = float(same_slot_all_heads.float().mean().item())

    if num_heads == 1:
        return {
            "valid": True,
            "num_heads": num_heads,
            "keep_count": keep_count,
            "same_slot_all_heads_ratio": same_slot_all_heads_ratio,
            "pair_jaccard_min": 1.0,
            "pair_jaccard_mean": 1.0,
            "pair_jaccard_max": 1.0,
        }

    pair_vals: list[float] = []
    head_sets = [set(int(x) for x in keep[h].tolist()) for h in range(num_heads)]
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            inter = len(head_sets[i] & head_sets[j])
            union = max(1, len(head_sets[i] | head_sets[j]))
            pair_vals.append(float(inter) / float(union))
    if not pair_vals:
        pair_vals = [0.0]
    return {
        "valid": True,
        "num_heads": num_heads,
        "keep_count": keep_count,
        "same_slot_all_heads_ratio": same_slot_all_heads_ratio,
        "pair_jaccard_min": float(min(pair_vals)),
        "pair_jaccard_mean": float(sum(pair_vals) / len(pair_vals)),
        "pair_jaccard_max": float(max(pair_vals)),
    }


def prepare_group_layer_compactions(
    *,
    req_id: str,
    gid: int,
    layer_tensors: list[tuple[int, torch.Tensor]],
    normalized_block_ids: list[int],
    block_size: int,
    group_total_tokens: int,
    group_prefill_len: int,
    protect_prefill: bool,
    round_start: int,
    group_budget_total: int,
    config: TriAttentionRuntimeConfig,
    strict_triton_required: bool,
    select_keep_indices: Callable[..., dict[str, Any] | None] | None,
    select_keep_indices_for_group: Callable[..., dict[str, Any] | None] | None,
    gather_dense_fn: Callable[..., torch.Tensor] | None = None,
) -> PreparedGroupSelection:
    gather_dense = gather_dense_fn or gather_request_k_dense
    block_ids_tensor_cache: dict[torch.device, torch.Tensor] = {}
    selected_for_group: dict[str, Any] | None = None
    prepared_layer_compactions: list[PreparedLayerCompaction] = []
    selection_mode = "fallback"

    if (
        select_keep_indices_for_group is not None
        and config.pruning_mode == "per_head"
        and config.per_head_selection_semantics == "hf_aligned_global_per_head"
        and not _DEBUG_DISABLE_GROUP_SELECTOR
    ):
        try:
            if strict_triton_required:
                if not getattr(select_keep_indices_for_group, "_supports_paged_group", False):
                    raise RuntimeError("paged_group_selector_required")

            if getattr(select_keep_indices_for_group, "_supports_paged_group", False):

                def _iter_layer_kv() -> Iterable[
                    tuple[int, torch.Tensor, list[int] | torch.Tensor, int]
                ]:
                    for layer_idx, kv_cache in layer_tensors:
                        yield layer_idx, kv_cache, normalized_block_ids, block_size

                selected_for_group = select_keep_indices_for_group(
                    layer_inputs=None,
                    layer_input_iter=None,
                    layer_kv_iter=_iter_layer_kv,
                    total_tokens=group_total_tokens,
                    prefill_len=group_prefill_len,
                    protect_prefill=protect_prefill,
                    round_start=round_start,
                    budget_total=group_budget_total,
                )
            else:

                def _iter_layer_inputs() -> Iterable[tuple[int, torch.Tensor]]:
                    for layer_idx, kv_cache in layer_tensors:
                        block_ids_tensor = block_ids_tensor_cache.get(kv_cache.device)
                        if block_ids_tensor is None:
                            block_ids_tensor = torch.as_tensor(
                                normalized_block_ids,
                                device=kv_cache.device,
                                dtype=torch.long,
                            )
                            block_ids_tensor_cache[kv_cache.device] = block_ids_tensor
                        keys_dense = gather_dense(
                            kv_cache=kv_cache,
                            block_ids=block_ids_tensor,
                            block_size=block_size,
                            total_tokens=group_total_tokens,
                        )
                        yield layer_idx, keys_dense

                selected_for_group = select_keep_indices_for_group(
                    layer_inputs=None,
                    layer_input_iter=_iter_layer_inputs,
                    layer_kv_iter=None,
                    total_tokens=group_total_tokens,
                    prefill_len=group_prefill_len,
                    protect_prefill=protect_prefill,
                    round_start=round_start,
                    budget_total=group_budget_total,
                )
        except Exception as exc:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:"
                f"req={req_id}:gid={gid}:global_per_head:{type(exc).__name__}"
            ) from exc

    for layer_idx, kv_cache in layer_tensors:
        block_ids_tensor = block_ids_tensor_cache.get(kv_cache.device)
        if block_ids_tensor is None:
            block_ids_tensor = torch.as_tensor(
                normalized_block_ids,
                device=kv_cache.device,
                dtype=torch.long,
            )
            block_ids_tensor_cache[kv_cache.device] = block_ids_tensor

        selected: dict[str, Any] | None = selected_for_group
        if selected is None and select_keep_indices is not None:
            try:
                if strict_triton_required:
                    if not getattr(select_keep_indices, "_supports_paged", False):
                        raise RuntimeError("paged_selector_required")
                if getattr(select_keep_indices, "_supports_paged", False):
                    selected = select_keep_indices(
                        keys_dense=None,
                        kv_cache=kv_cache,
                        block_ids=block_ids_tensor,
                        block_size=block_size,
                        total_tokens=group_total_tokens,
                        prefill_len=group_prefill_len,
                        protect_prefill=protect_prefill,
                        layer_idx=layer_idx,
                        round_start=round_start,
                        budget_total=group_budget_total,
                    )
                else:
                    keys_dense = gather_dense(
                        kv_cache=kv_cache,
                        block_ids=block_ids_tensor,
                        block_size=block_size,
                        total_tokens=group_total_tokens,
                    )
                    selected = select_keep_indices(
                        keys_dense=keys_dense,
                        total_tokens=group_total_tokens,
                        prefill_len=group_prefill_len,
                        protect_prefill=protect_prefill,
                        layer_idx=layer_idx,
                        round_start=round_start,
                        budget_total=group_budget_total,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:"
                    f"req={req_id}:gid={gid}:layer={layer_idx}:"
                    f"{type(exc).__name__}"
                ) from exc

        selected_from_fallback = False
        if selected is None:
            keep_indices = build_keep_token_indices(
                total_tokens=group_total_tokens,
                kv_budget=config.kv_budget,
                prefill_len=group_prefill_len,
                protect_prefill=protect_prefill,
                include_prefill_in_budget=config.include_prefill_in_budget,
            )
            if keep_indices is None:
                raise ValueError("prefill_exceeds_budget")
            if strict_triton_required:
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:selector_returned_none:"
                    f"req={req_id}:gid={gid}:layer={layer_idx}"
                )
            selected = {"mode": "shared", "indices": keep_indices}
            selected_from_fallback = True

        keep_plan = KeepPlan.from_selector_result(selected)
        if _SELECTOR_METRICS_DEBUG and keep_plan.mode == "per_head":
            metrics = _per_head_debug_metrics(keep_plan.indices)
            trace_event(
                "selector_per_head_metrics",
                req_id=repr(req_id),
                gid=int(gid),
                layer_idx=int(layer_idx),
                selection_mode=keep_plan.selection_mode_label,
                **metrics,
            )
        selection_mode = "fallback" if selected_from_fallback else keep_plan.selection_mode_label
        prepared_layer_compactions.append(
            PreparedLayerCompaction(
                layer_idx=layer_idx,
                kv_cache=kv_cache,
                block_ids=block_ids_tensor,
                keep_plan=keep_plan,
            )
        )

    return PreparedGroupSelection(
        tasks=prepared_layer_compactions,
        selection_mode=str(selection_mode),
    )
