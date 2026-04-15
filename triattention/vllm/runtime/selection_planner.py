"""Bridge layer: turn HF selector outputs into prepared layout compaction tasks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .kv_compaction import build_keep_token_indices, gather_request_k_dense
from .layout_engine import PreparedLayerCompaction
from .plan_models import KeepPlan

_DEBUG_DISABLE_GROUP_SELECTOR = (
    os.environ.get("TRIATTN_RUNTIME_DEBUG_DISABLE_GROUP_SELECTOR", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_DEBUG_OVERRIDE_FIRST_KEEP_JSON = os.environ.get(
    "TRIATTN_DEBUG_OVERRIDE_FIRST_KEEP_JSON", ""
).strip()
_FIRST_KEEP_OVERRIDE_USED = False
_OVERRIDE_FIRST_KEEP_CACHE: torch.Tensor | None = None


@dataclass(frozen=True)
class PreparedGroupSelection:
    tasks: list[PreparedLayerCompaction]
    selection_mode: str


def _load_override_first_keep_tensor() -> torch.Tensor:
    global _OVERRIDE_FIRST_KEEP_CACHE
    if _OVERRIDE_FIRST_KEEP_CACHE is not None:
        return _OVERRIDE_FIRST_KEEP_CACHE
    if not _DEBUG_OVERRIDE_FIRST_KEEP_JSON:
        raise RuntimeError("override_keep_json_not_set")
    payload = json.loads(
        Path(_DEBUG_OVERRIDE_FIRST_KEEP_JSON).read_text(encoding="utf-8")
    )
    indices = payload.get("indices") if isinstance(payload, dict) else payload
    tensor = torch.as_tensor(indices, dtype=torch.long).contiguous()
    if tensor.ndim != 2:
        raise RuntimeError(f"override_keep_ndim_{int(tensor.ndim)}")
    _OVERRIDE_FIRST_KEEP_CACHE = tensor
    return tensor


def _maybe_override_first_keep_plan(
    *,
    keep_plan: KeepPlan,
    req_id: str,
    gid: int,
    round_start: int,
    group_total_tokens: int,
) -> KeepPlan:
    global _FIRST_KEEP_OVERRIDE_USED
    if not _DEBUG_OVERRIDE_FIRST_KEEP_JSON or _FIRST_KEEP_OVERRIDE_USED:
        return keep_plan
    if keep_plan.mode != "per_head":
        return keep_plan
    if int(round_start) != int(group_total_tokens):
        return keep_plan

    override_cpu = _load_override_first_keep_tensor()
    current = keep_plan.indices
    current_tensor = (
        current.detach().to(dtype=torch.long)
        if isinstance(current, torch.Tensor)
        else torch.as_tensor(current, dtype=torch.long)
    )
    if tuple(current_tensor.shape) != tuple(override_cpu.shape):
        raise RuntimeError(
            "override_keep_shape_mismatch:"
            f"current={tuple(current_tensor.shape)}:"
            f"override={tuple(override_cpu.shape)}"
        )
    if isinstance(current, torch.Tensor):
        override_indices = override_cpu.to(device=current.device)
    else:
        override_indices = override_cpu.tolist()
    _FIRST_KEEP_OVERRIDE_USED = True
    return KeepPlan(
        mode=keep_plan.mode,
        indices=override_indices,
        semantic=keep_plan.semantic,
    )


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
        keep_plan = _maybe_override_first_keep_plan(
            keep_plan=keep_plan,
            req_id=req_id,
            gid=gid,
            round_start=round_start,
            group_total_tokens=group_total_tokens,
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
