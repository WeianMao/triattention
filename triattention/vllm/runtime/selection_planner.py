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
from .debug_trace import trace_event
from .kv_compaction import build_keep_token_indices, gather_request_k_dense
from .layout_engine import PreparedLayerCompaction
from .plan_models import KeepPlan

_SELECTOR_METRICS_DEBUG = (
    os.environ.get("TRIATTN_RUNTIME_SELECTOR_METRICS_DEBUG", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_DUMP_FIRST_KEEP_INDICES = (
    os.environ.get("TRIATTN_DEBUG_DUMP_FIRST_KEEP_INDICES", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_DEBUG_DISABLE_GROUP_SELECTOR = (
    os.environ.get("TRIATTN_RUNTIME_DEBUG_DISABLE_GROUP_SELECTOR", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_DEBUG_COMPARE_GROUP_PAGED_DENSE_KEEP = (
    os.environ.get("TRIATTN_DEBUG_COMPARE_GROUP_PAGED_DENSE_KEEP", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_DEBUG_COMPARE_GROUP_FRESH_DENSE_KEEP = (
    os.environ.get("TRIATTN_DEBUG_COMPARE_GROUP_FRESH_DENSE_KEEP", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_DEBUG_COMPARE_GROUP_CLONE_DENSE_KEEP = (
    os.environ.get("TRIATTN_DEBUG_COMPARE_GROUP_CLONE_DENSE_KEEP", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_DEBUG_COMPARE_GROUP_CPU_ROUNDTRIP_DENSE_KEEP = (
    os.environ.get("TRIATTN_DEBUG_COMPARE_GROUP_CPU_ROUNDTRIP_DENSE_KEEP", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_FIRST_KEEP_DUMP_DONE = False
_DUMP_KEEP_DIR = os.environ.get("TRIATTN_DEBUG_DUMP_KEEP_DIR", "").strip()
_DUMP_GROUP_COMPARE_DIR = os.environ.get(
    "TRIATTN_DEBUG_DUMP_GROUP_COMPARE_DIR", ""
).strip()
_DEBUG_OVERRIDE_FIRST_KEEP_JSON = os.environ.get(
    "TRIATTN_DEBUG_OVERRIDE_FIRST_KEEP_JSON", ""
).strip()
_SEEN_KEEP_DUMP_KEYS: set[tuple[str, int, int]] = set()
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
    trace_event(
        "selector_first_keep_overridden",
        req_id=repr(req_id),
        gid=int(gid),
        round_start=int(round_start),
        total_tokens=int(group_total_tokens),
        override_path=_DEBUG_OVERRIDE_FIRST_KEEP_JSON,
    )
    return KeepPlan(
        mode=keep_plan.mode,
        indices=override_indices,
        semantic=keep_plan.semantic,
    )


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
    global _FIRST_KEEP_DUMP_DONE
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

        if (
            _DEBUG_COMPARE_GROUP_PAGED_DENSE_KEEP
            and selected_for_group is not None
            and getattr(select_keep_indices_for_group, "_supports_paged_group", False)
        ):
            try:
                def _iter_layer_inputs_dbg() -> Iterable[tuple[int, torch.Tensor]]:
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

                dense_layer_inputs_dbg = list(_iter_layer_inputs_dbg())

                dense_selected_for_group = select_keep_indices_for_group(
                    layer_inputs=dense_layer_inputs_dbg,
                    layer_input_iter=None,
                    layer_kv_iter=None,
                    total_tokens=group_total_tokens,
                    prefill_len=group_prefill_len,
                    protect_prefill=protect_prefill,
                    round_start=round_start,
                    budget_total=group_budget_total,
                )
                fresh_dense_selected_for_group: dict[str, Any] | None = None
                clone_dense_selected_for_group: dict[str, Any] | None = None
                cpu_roundtrip_dense_selected_for_group: dict[str, Any] | None = None
                if _DEBUG_COMPARE_GROUP_FRESH_DENSE_KEEP:
                    from .selector_hf import build_triattention_selector

                    _fresh_select_keep_indices, fresh_group_selector, _fresh_status = (
                        build_triattention_selector(config=config, base_runner=None)
                    )
                    if fresh_group_selector is not None:
                        fresh_dense_selected_for_group = fresh_group_selector(
                            layer_inputs=dense_layer_inputs_dbg,
                            layer_input_iter=None,
                            layer_kv_iter=None,
                            total_tokens=group_total_tokens,
                            prefill_len=group_prefill_len,
                            protect_prefill=protect_prefill,
                            round_start=round_start,
                            budget_total=group_budget_total,
                        )
                if _DEBUG_COMPARE_GROUP_CLONE_DENSE_KEEP:
                    from .selector_hf import build_triattention_selector

                    _clone_select_keep_indices, clone_group_selector, _clone_status = (
                        build_triattention_selector(config=config, base_runner=None)
                    )
                    if clone_group_selector is not None:
                        clone_layer_inputs_dbg = [
                            (layer_idx, keys_dense.clone())
                            for layer_idx, keys_dense in dense_layer_inputs_dbg
                        ]
                        clone_dense_selected_for_group = clone_group_selector(
                            layer_inputs=clone_layer_inputs_dbg,
                            layer_input_iter=None,
                            layer_kv_iter=None,
                            total_tokens=group_total_tokens,
                            prefill_len=group_prefill_len,
                            protect_prefill=protect_prefill,
                            round_start=round_start,
                            budget_total=group_budget_total,
                        )
                if _DEBUG_COMPARE_GROUP_CPU_ROUNDTRIP_DENSE_KEEP:
                    from .selector_hf import build_triattention_selector

                    _rt_select_keep_indices, rt_group_selector, _rt_status = (
                        build_triattention_selector(config=config, base_runner=None)
                    )
                    if rt_group_selector is not None:
                        rt_layer_inputs_dbg = [
                            (
                                layer_idx,
                                keys_dense.detach().to(device="cpu").to(device=keys_dense.device),
                            )
                            for layer_idx, keys_dense in dense_layer_inputs_dbg
                        ]
                        cpu_roundtrip_dense_selected_for_group = rt_group_selector(
                            layer_inputs=rt_layer_inputs_dbg,
                            layer_input_iter=None,
                            layer_kv_iter=None,
                            total_tokens=group_total_tokens,
                            prefill_len=group_prefill_len,
                            protect_prefill=protect_prefill,
                            round_start=round_start,
                            budget_total=group_budget_total,
                        )

                def _to_cpu_tensor(obj: Any) -> torch.Tensor | None:
                    if obj is None:
                        return None
                    indices = obj.get("indices") if isinstance(obj, dict) else None
                    if indices is None:
                        return None
                    if isinstance(indices, torch.Tensor):
                        return indices.detach().to(dtype=torch.long, device="cpu").contiguous()
                    return torch.as_tensor(indices, dtype=torch.long).contiguous()

                paged_idx = _to_cpu_tensor(selected_for_group)
                dense_idx = _to_cpu_tensor(dense_selected_for_group)
                fresh_dense_idx = _to_cpu_tensor(fresh_dense_selected_for_group)
                clone_dense_idx = _to_cpu_tensor(clone_dense_selected_for_group)
                cpu_roundtrip_dense_idx = _to_cpu_tensor(cpu_roundtrip_dense_selected_for_group)
                if paged_idx is None or dense_idx is None:
                    raise RuntimeError(
                        f"missing_group_indices:paged={type(selected_for_group).__name__}:"
                        f"dense={type(dense_selected_for_group).__name__}"
                    )
                if paged_idx.shape != dense_idx.shape:
                    raise RuntimeError(
                        f"group_shape_mismatch:paged={tuple(paged_idx.shape)}:"
                        f"dense={tuple(dense_idx.shape)}"
                    )
                if not torch.equal(paged_idx, dense_idx):
                    if paged_idx.ndim == 2:
                        head_jaccards: list[float] = []
                        for h in range(paged_idx.shape[0]):
                            p_set = set(int(x) for x in paged_idx[h].tolist())
                            d_set = set(int(x) for x in dense_idx[h].tolist())
                            inter = len(p_set & d_set)
                            union = max(1, len(p_set | d_set))
                            head_jaccards.append(inter / union)
                        min_jaccard = min(head_jaccards) if head_jaccards else 0.0
                        mean_jaccard = (
                            sum(head_jaccards) / len(head_jaccards)
                            if head_jaccards else 0.0
                        )
                        if min_jaccard < 0.98:
                            raise RuntimeError(
                                "TRIATTN_GROUP_PAGED_DENSE_COMPARE_FAILED:"
                                f"req={req_id}:gid={gid}:round_start={round_start}:"
                                f"total_tokens={group_total_tokens}:"
                                f"min_jaccard={min_jaccard:.4f}:"
                                f"mean_jaccard={mean_jaccard:.4f}"
                            )
                    else:
                        raise RuntimeError(
                            "TRIATTN_GROUP_PAGED_DENSE_COMPARE_FAILED:"
                            f"req={req_id}:gid={gid}:round_start={round_start}:"
                            f"total_tokens={group_total_tokens}:non_equal_indices"
                        )
                if _DUMP_GROUP_COMPARE_DIR:
                    dense_layer_samples: dict[str, Any] = {}
                    dense_layer_checksums: dict[str, Any] = {}
                    dense_layer_layouts: dict[str, Any] = {}
                    for layer_idx, keys_dense in dense_layer_inputs_dbg:
                        if int(layer_idx) not in {0, 1, 16, 32, 48, 63}:
                            continue
                        total = int(keys_dense.shape[2]) if keys_dense.ndim >= 3 else 0
                        sample_positions = sorted(
                            {
                                0,
                                1,
                                2,
                                max(0, total // 2),
                                max(0, total - 3),
                                max(0, total - 2),
                                max(0, total - 1),
                            }
                        )
                        head0 = []
                        for pos in sample_positions:
                            head0.append(
                                keys_dense[0, 0, pos]
                                .detach()
                                .to(device="cpu", dtype=torch.float32)
                                .tolist()
                            )
                        dense_layer_samples[str(int(layer_idx))] = {
                            "sample_positions": sample_positions,
                            "head0_vectors": head0,
                        }
                        dense_fp32 = keys_dense.detach().to(device="cpu", dtype=torch.float32)
                        dense_layer_checksums[str(int(layer_idx))] = {
                            "sum": float(dense_fp32.sum().item()),
                            "sumsq": float((dense_fp32 * dense_fp32).sum().item()),
                            "head0_sum": float(dense_fp32[:, 0].sum().item()),
                            "head0_sumsq": float((dense_fp32[:, 0] * dense_fp32[:, 0]).sum().item()),
                        }
                        dense_layer_layouts[str(int(layer_idx))] = {
                            "shape": list(keys_dense.shape),
                            "stride": list(keys_dense.stride()),
                            "is_contiguous": bool(keys_dense.is_contiguous()),
                        }
                    dump_dir = Path(_DUMP_GROUP_COMPARE_DIR)
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    dump_path = dump_dir / (
                        f"group_req_{str(req_id).replace('/', '_')}_"
                        f"gid_{int(gid)}_round_{int(round_start)}_tokens_{int(group_total_tokens)}.json"
                    )
                    payload = {
                        "req_id": str(req_id),
                        "gid": int(gid),
                        "round_start": int(round_start),
                        "total_tokens": int(group_total_tokens),
                        "prefill_len": int(group_prefill_len),
                        "budget_total": int(group_budget_total),
                        "paged_shape": list(paged_idx.shape),
                        "dense_shape": list(dense_idx.shape),
                        "paged_indices": paged_idx.tolist(),
                        "dense_indices": dense_idx.tolist(),
                        "paged_group_layer_indices": (
                            list(selected_for_group.get("debug_group_layer_indices", []))
                            if isinstance(selected_for_group, dict)
                            else []
                        ),
                        "dense_group_layer_indices": (
                            list(dense_selected_for_group.get("debug_group_layer_indices", []))
                            if isinstance(dense_selected_for_group, dict)
                            else []
                        ),
                        "paged_recent_count": (
                            selected_for_group.get("debug_recent_count")
                            if isinstance(selected_for_group, dict)
                            else None
                        ),
                        "dense_recent_count": (
                            dense_selected_for_group.get("debug_recent_count")
                            if isinstance(dense_selected_for_group, dict)
                            else None
                        ),
                        "fresh_dense_recent_count": (
                            fresh_dense_selected_for_group.get("debug_recent_count")
                            if isinstance(fresh_dense_selected_for_group, dict)
                            else None
                        ),
                        "clone_dense_recent_count": (
                            clone_dense_selected_for_group.get("debug_recent_count")
                            if isinstance(clone_dense_selected_for_group, dict)
                            else None
                        ),
                        "cpu_roundtrip_dense_recent_count": (
                            cpu_roundtrip_dense_selected_for_group.get("debug_recent_count")
                            if isinstance(cpu_roundtrip_dense_selected_for_group, dict)
                            else None
                        ),
                        "fresh_dense_shape": (
                            list(fresh_dense_idx.shape) if fresh_dense_idx is not None else []
                        ),
                        "fresh_dense_indices": (
                            fresh_dense_idx.tolist() if fresh_dense_idx is not None else []
                        ),
                        "clone_dense_shape": (
                            list(clone_dense_idx.shape) if clone_dense_idx is not None else []
                        ),
                        "clone_dense_indices": (
                            clone_dense_idx.tolist() if clone_dense_idx is not None else []
                        ),
                        "cpu_roundtrip_dense_shape": (
                            list(cpu_roundtrip_dense_idx.shape)
                            if cpu_roundtrip_dense_idx is not None else []
                        ),
                        "cpu_roundtrip_dense_indices": (
                            cpu_roundtrip_dense_idx.tolist()
                            if cpu_roundtrip_dense_idx is not None else []
                        ),
                        "dense_layer_samples": dense_layer_samples,
                        "dense_layer_checksums": dense_layer_checksums,
                        "dense_layer_layouts": dense_layer_layouts,
                        "dense_score_sample_positions": (
                            list(dense_selected_for_group.get("debug_score_sample_positions", []))
                            if isinstance(dense_selected_for_group, dict)
                            else []
                        ),
                        "dense_layer_score_samples": (
                            dense_selected_for_group.get("debug_layer_score_samples", {})
                            if isinstance(dense_selected_for_group, dict)
                            else {}
                        ),
                        "dense_agg_head0_scores": (
                            list(dense_selected_for_group.get("debug_agg_head0_scores", []))
                            if isinstance(dense_selected_for_group, dict)
                            else []
                        ),
                        "dense_effective_model_path": (
                            dense_selected_for_group.get("debug_effective_model_path")
                            if isinstance(dense_selected_for_group, dict)
                            else None
                        ),
                    }
                    dump_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"TRIATTN_GROUP_PAGED_DENSE_COMPARE_FAILED:"
                    f"req={req_id}:gid={gid}:round_start={round_start}:"
                    f"total_tokens={group_total_tokens}:{type(exc).__name__}:{exc}"
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
        if (
            _DUMP_FIRST_KEEP_INDICES
            and not _FIRST_KEEP_DUMP_DONE
            and keep_plan.mode == "per_head"
            and hasattr(keep_plan.indices, "detach")
        ):
            keep_indices_cpu = keep_plan.indices.detach().to(dtype=torch.long, device="cpu")
            trace_event(
                "first_keep_indices",
                req_id=repr(req_id),
                gid=int(gid),
                layer_idx=int(layer_idx),
                total_tokens=int(group_total_tokens),
                prefill_len=int(group_prefill_len),
                budget_total=int(group_budget_total),
                selection_mode=keep_plan.selection_mode_label,
                group_agg_mode=(
                    selected.get("group_agg_mode")
                    if isinstance(selected, dict)
                    else None
                ),
                debug_group_layer_indices=(
                    selected.get("debug_group_layer_indices")
                    if isinstance(selected, dict)
                    else None
                ),
                indices=keep_indices_cpu.tolist(),
            )
            _FIRST_KEEP_DUMP_DONE = True
        if (
            _DUMP_KEEP_DIR
            and keep_plan.mode == "per_head"
            and hasattr(keep_plan.indices, "detach")
        ):
            dump_key = (str(req_id), int(round_start), int(group_total_tokens))
            if dump_key not in _SEEN_KEEP_DUMP_KEYS:
                keep_indices_cpu = keep_plan.indices.detach().to(dtype=torch.long, device="cpu")
                payload = {
                    "req_id": str(req_id),
                    "gid": int(gid),
                    "layer_idx": int(layer_idx),
                    "round_start": int(round_start),
                    "total_tokens": int(group_total_tokens),
                    "prefill_len": int(group_prefill_len),
                    "budget_total": int(group_budget_total),
                    "selection_mode": keep_plan.selection_mode_label,
                    "group_agg_mode": (
                        selected.get("group_agg_mode")
                        if isinstance(selected, dict)
                        else None
                    ),
                    "debug_group_layer_indices": (
                        selected.get("debug_group_layer_indices")
                        if isinstance(selected, dict)
                        else None
                    ),
                    "shape": list(keep_indices_cpu.shape),
                    "indices": keep_indices_cpu.tolist(),
                }
                dump_dir = Path(_DUMP_KEEP_DIR)
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / (
                    f"keep_req_{str(req_id).replace('/', '_')}_"
                    f"round_{int(round_start)}_tokens_{int(group_total_tokens)}.json"
                )
                dump_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                _SEEN_KEEP_DUMP_KEYS.add(dump_key)
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
