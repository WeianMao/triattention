"""Base-runner compression hook implementation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from .config import TriAttentionV2Config
from .kv_compaction import (
    build_keep_token_indices,
    compact_request_kv_in_place,
    compact_request_kv_in_place_per_head,
    gather_request_k_dense,
    gather_request_k_dense_range,
)
from .signals import CompressionSignal

logger = logging.getLogger(__name__)
TRITON_SCORING_REQUIRED_MARKER = "TRIATTN_FATAL_TRITON_SCORING_REQUIRED"


def _infer_layer_idx(layer_name: str, layer_obj: Any, fallback_idx: int) -> int:
    for attr in ("layer_idx", "layer_id", "idx"):
        value = getattr(layer_obj, attr, None)
        if isinstance(value, int):
            return value
    matches = re.findall(r"\d+", layer_name)
    if matches:
        return int(matches[-1])
    return fallback_idx


def _resolve_group_tensors(base_runner: Any) -> dict[int, list[tuple[int, torch.Tensor]]]:
    """Resolve kv cache tensors for each kv cache group.

    Returns:
        gid -> list of (layer_idx, kv_cache_tensor)
    """
    group_tensors: dict[int, list[tuple[int, torch.Tensor]]] = {}

    kv_cache_config = getattr(base_runner, "kv_cache_config", None)
    compilation_config = getattr(base_runner, "compilation_config", None)
    static_forward_context = (
        getattr(compilation_config, "static_forward_context", None)
        if compilation_config is not None
        else None
    )

    if kv_cache_config is None or not isinstance(static_forward_context, dict):
        fallback = getattr(base_runner, "kv_caches", None)
        if isinstance(fallback, list):
            tensors = [
                (idx, t)
                for idx, t in enumerate(fallback)
                if isinstance(t, torch.Tensor)
            ]
            if tensors:
                group_tensors[0] = tensors
        return group_tensors

    kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not isinstance(kv_cache_groups, (list, tuple)):
        return group_tensors

    for gid, group in enumerate(kv_cache_groups):
        layer_names = getattr(group, "layer_names", None)
        if not isinstance(layer_names, (list, tuple)):
            continue
        tensors: list[tuple[int, torch.Tensor]] = []
        seen_ptrs: set[int] = set()
        for local_idx, layer_name in enumerate(layer_names):
            layer = static_forward_context.get(layer_name)
            if layer is None:
                continue
            kv_cache_list = getattr(layer, "kv_cache", None)
            if not isinstance(kv_cache_list, list) or not kv_cache_list:
                continue
            tensor = kv_cache_list[0]
            if not isinstance(tensor, torch.Tensor):
                continue
            ptr = tensor.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            tensors.append(
                (
                    _infer_layer_idx(
                        layer_name=layer_name,
                        layer_obj=layer,
                        fallback_idx=local_idx,
                    ),
                    tensor,
                )
            )
        if tensors:
            group_tensors[gid] = tensors
    return group_tensors


def _effective_budget_for_signal(
    config: TriAttentionV2Config,
    signal: CompressionSignal,
    total_tokens: int,
) -> int:
    budget = config.kv_budget
    if signal.protect_prefill and not config.include_prefill_in_budget:
        budget += max(signal.prefill_len, 0)
    return min(total_tokens, budget)


def _effective_len_guard_upper(
    config: TriAttentionV2Config,
    signal: CompressionSignal,
) -> int:
    budget = config.kv_budget
    if signal.protect_prefill and not config.include_prefill_in_budget:
        budget += max(signal.prefill_len, 0)
    return budget + max(1, config.effective_len_guard_divide_multiples) * max(
        1,
        config.divide_length,
    )


def _num_required_blocks(total_tokens: int, block_size: int) -> int:
    if total_tokens <= 0:
        return 0
    return (total_tokens + block_size - 1) // block_size


def _scheduled_tokens_for_req(scheduler_output: Any, req_id: str) -> int:
    """Best-effort extraction of scheduled token count for one request."""
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict):
        return 1
    for key, value in scheduled.items():
        key_req_id = None
        if isinstance(key, str):
            key_req_id = key
        else:
            key_req_id = getattr(key, "request_id", None)
            if not isinstance(key_req_id, str):
                key_req_id = getattr(key, "req_id", None)
            if not isinstance(key_req_id, str):
                key_req_id = str(key)
        if key_req_id == req_id:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                return 1
    return 1


def _min_block_capacity_tokens(
    block_ids_by_group: Any,
    block_size: int,
) -> int | None:
    """Best-effort physical token capacity inferred from request block ids."""
    if block_size <= 0:
        return None
    if not isinstance(block_ids_by_group, (list, tuple)):
        return None
    capacities: list[int] = []
    for group_block_ids in block_ids_by_group:
        if not isinstance(group_block_ids, (list, tuple)):
            continue
        capacities.append(len(group_block_ids) * block_size)
    if not capacities:
        return None
    return min(capacities)


def _selected_keep_count(selected: dict[str, Any]) -> int:
    mode = str(selected.get("mode", "shared"))
    indices = selected.get("indices")
    if mode == "per_head":
        if isinstance(indices, torch.Tensor):
            if indices.ndim != 2:
                raise ValueError(
                    f"per_head indices tensor must be 2D, got {tuple(indices.shape)}"
                )
            return int(indices.shape[1])
        if isinstance(indices, list):
            if not indices:
                return 0
            first_row = indices[0]
            return len(first_row) if isinstance(first_row, list) else 0
        raise ValueError(f"unsupported per_head indices type: {type(indices).__name__}")
    if isinstance(indices, torch.Tensor):
        return int(indices.numel())
    if isinstance(indices, list):
        return len(indices)
    raise ValueError(f"unsupported shared indices type: {type(indices).__name__}")


def _build_speckv_selector(
    config: TriAttentionV2Config,
) -> tuple[
    Callable[..., dict[str, Any] | None] | None,
    Callable[..., dict[str, Any] | None] | None,
    str,
]:
    """Build SpeckV selector callable.

    The returned selector emits either:
    - {"mode": "shared", "indices": Tensor|list[int]}
    - {"mode": "per_head", "indices": Tensor|list[list[int]]}
    """
    strict_triton_required = bool(
        config.enable_experimental_kv_compaction and config.require_triton_scoring
    )
    if config.sparse_stats_path is None:
        if strict_triton_required:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:stats_path_not_set"
            )
        return None, None, "stats_path_not_set"

    stats_path = Path(config.sparse_stats_path).expanduser()
    if not stats_path.exists():
        if strict_triton_required:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:stats_path_not_found"
            )
        return None, None, "stats_path_not_found"

    try:
        from triattention import TriAttentionConfig
        from triattention.compressor import TriAttentionCompressor
        from triattention.scoring import compute_scores_triton
        from triattention.utils import normalize_scores
    except Exception as exc:  # pragma: no cover - import safety
        raise RuntimeError(
            f"{TRITON_SCORING_REQUIRED_MARKER}:import_failed:{type(exc).__name__}"
        ) from exc

    requested_pruning_mode = config.pruning_mode
    if requested_pruning_mode not in {"per_layer", "per_head", "per_layer_per_head"}:
        if strict_triton_required:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:unsupported_pruning_mode:{requested_pruning_mode}"
            )
        return None, None, f"unsupported_pruning_mode:{requested_pruning_mode}"
    # Keep per-head score tensor and decide aggregation in selector;
    # this matches HF path better than forcing mean aggregation inside scoring.
    pruning_mode = "per_head"
    per_head_semantics = config.per_head_selection_semantics

    tri_cfg = TriAttentionConfig(
        stats_path=stats_path,
        kv_budget=config.kv_budget,
        divide_length=config.divide_length,
        pruning_mode=pruning_mode,
        score_aggregation=config.sparse_score_aggregation,
        sparse_normalize_scores=config.sparse_normalize_scores,
        window_size=min(config.window_size, max(config.kv_budget - 1, 0)),
        include_prefill_in_budget=config.include_prefill_in_budget,
        protect_prefill=config.protect_prefill,
        disable_mlr=config.disable_mlr,
        disable_trig=config.disable_trig,
        disable_top_n_high_freq=config.disable_top_n_high_freq,
        use_triton_scoring=True,
    )
    compressor = TriAttentionCompressor(tri_cfg)
    available_layers_sorted: tuple[int, ...] | None = None
    available_layers_set: set[int] | None = None

    def _resolve_layer_idx_for_stats(layer_idx: int) -> int:
        nonlocal available_layers_sorted
        nonlocal available_layers_set
        compressor._lazy_init()
        if available_layers_sorted is None or available_layers_set is None:
            available_layers_sorted = tuple(sorted(compressor.head_stats.keys()))
            available_layers_set = set(available_layers_sorted)
        if not available_layers_sorted:
            raise RuntimeError("empty_head_stats")
        if layer_idx in available_layers_set:
            return layer_idx
        return available_layers_sorted[layer_idx % len(available_layers_sorted)]

    reduced_head_stats_cache: dict[tuple[int, int], tuple[dict[str, torch.Tensor], torch.Tensor]] = {}

    def _build_reduced_layer_stats(
        *,
        resolved_layer_idx: int,
        target_heads: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        cache_key = (resolved_layer_idx, target_heads)
        cached = reduced_head_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        layer_stats = compressor.head_stats[resolved_layer_idx]
        layer_freq_scale_sq = compressor.freq_scale_sq[resolved_layer_idx]
        source_heads = int(layer_freq_scale_sq.shape[0])
        if source_heads == target_heads:
            reduced = (layer_stats, layer_freq_scale_sq)
            reduced_head_stats_cache[cache_key] = reduced
            return reduced
        if target_heads <= 0 or source_heads % target_heads != 0:
            raise RuntimeError(
                f"incompatible_head_mapping:source={source_heads},target={target_heads}"
            )
        group_size = source_heads // target_heads

        reduced_stats: dict[str, torch.Tensor] = {}
        q_abs_mean = layer_stats.get("q_abs_mean")
        if isinstance(q_abs_mean, torch.Tensor):
            reduced_stats["q_abs_mean"] = (
                q_abs_mean.reshape(target_heads, group_size, q_abs_mean.shape[1])
                .mean(dim=1)
                .contiguous()
            )

        q_mean_complex = layer_stats.get("q_mean_complex")
        if isinstance(q_mean_complex, torch.Tensor):
            reduced_stats["q_mean_complex"] = (
                q_mean_complex.reshape(
                    target_heads,
                    group_size,
                    q_mean_complex.shape[1],
                    q_mean_complex.shape[2],
                )
                .mean(dim=1)
                .contiguous()
            )

        reduced_freq_scale_sq = (
            layer_freq_scale_sq.reshape(
                target_heads,
                group_size,
                layer_freq_scale_sq.shape[1],
            )
            .mean(dim=1)
            .contiguous()
        )
        reduced = (reduced_stats, reduced_freq_scale_sq)
        reduced_head_stats_cache[cache_key] = reduced
        return reduced

    def _compute_layer_scores(
        keys_dense: torch.Tensor,
        *,
        layer_idx: int,
        round_start: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:
        runtime_heads = int(keys_dense.shape[1])
        (
            score_head_stats,
            score_freq_scale_sq,
            use_hf_group_max,
            group_size,
        ) = _resolve_layer_score_inputs(
            layer_idx=layer_idx,
            runtime_heads=runtime_heads,
        )
        scores = _compute_layer_scores_raw(
            keys_dense=keys_dense,
            score_head_stats=score_head_stats,
            score_freq_scale_sq=score_freq_scale_sq,
            use_hf_group_max=use_hf_group_max,
            group_size=group_size,
            round_start=round_start,
        )

        return _finalize_layer_scores(
            scores=scores,
            runtime_heads=runtime_heads,
            use_hf_group_max=use_hf_group_max,
            group_size=group_size,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
        )

    def _resolve_layer_score_inputs(
        *,
        layer_idx: int,
        runtime_heads: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, int]:
        resolved_layer_idx = _resolve_layer_idx_for_stats(layer_idx)
        layer_head_stats = compressor.head_stats[resolved_layer_idx]
        layer_freq_scale_sq = compressor.freq_scale_sq[resolved_layer_idx]
        stats_heads = int(layer_freq_scale_sq.shape[0])
        use_hf_group_max = (
            requested_pruning_mode == "per_head"
            and per_head_semantics == "hf_aligned_global_per_head"
            and stats_heads != runtime_heads
        )
        score_head_stats = layer_head_stats
        score_freq_scale_sq = layer_freq_scale_sq
        group_size = 1
        if use_hf_group_max:
            if runtime_heads <= 0 or stats_heads % runtime_heads != 0:
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:incompatible_head_mapping:source={stats_heads},target={runtime_heads}"
                )
            group_size = stats_heads // runtime_heads
        elif stats_heads != runtime_heads:
            score_head_stats, score_freq_scale_sq = _build_reduced_layer_stats(
                resolved_layer_idx=resolved_layer_idx,
                target_heads=runtime_heads,
            )
        return score_head_stats, score_freq_scale_sq, use_hf_group_max, group_size

    def _compute_layer_scores_raw(
        *,
        keys_dense: torch.Tensor,
        score_head_stats: dict[str, torch.Tensor],
        score_freq_scale_sq: torch.Tensor,
        use_hf_group_max: bool,
        group_size: int,
        round_start: int,
    ) -> torch.Tensor:
        score_inputs = (
            keys_dense.repeat_interleave(group_size, dim=1).contiguous()
            if use_hf_group_max and group_size > 1
            else keys_dense
        )
        try:
            return compute_scores_triton(
                key_states=score_inputs,
                cache_positions=None,
                head_stats=score_head_stats,
                omega=compressor.omega,
                offsets=compressor.offsets,
                freq_scale_sq=score_freq_scale_sq,
                config=tri_cfg,
                round_start=round_start,
                trig_cache=getattr(compressor, "trig_cache", None),
            )
        except Exception as exc:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:score_failed:{type(exc).__name__}"
            ) from exc

    def _finalize_layer_scores(
        *,
        scores: torch.Tensor,
        runtime_heads: int,
        use_hf_group_max: bool,
        group_size: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:

        if config.sparse_normalize_scores:
            scores = normalize_scores(scores)
        mutate_scores = (
            config.window_size > 0
            or (protect_prefill and prefill_len > 0)
        )
        if mutate_scores:
            scores = scores.clone()
        if config.window_size > 0:
            scores[..., -config.window_size:] = float("inf")
        if protect_prefill and prefill_len > 0:
            scores[..., :prefill_len] = float("inf")
        if use_hf_group_max:
            scores = scores.view(
                scores.shape[0],
                runtime_heads,
                group_size,
                scores.shape[-1],
            ).max(dim=2).values
        return scores

    def _compute_layer_scores_paged(
        *,
        kv_cache: torch.Tensor,
        block_ids: list[int] | torch.Tensor,
        block_size: int,
        total_tokens: int,
        layer_idx: int,
        round_start: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:
        runtime_heads = int(kv_cache.shape[3])
        (
            score_head_stats,
            score_freq_scale_sq,
            use_hf_group_max,
            group_size,
        ) = _resolve_layer_score_inputs(
            layer_idx=layer_idx,
            runtime_heads=runtime_heads,
        )
        chunk_tokens = max(block_size, min(1024, max(config.divide_length, block_size)))
        chunks: list[torch.Tensor] = []
        start = 0
        while start < total_tokens:
            curr_tokens = min(chunk_tokens, total_tokens - start)
            keys_chunk = gather_request_k_dense_range(
                kv_cache=kv_cache,
                block_ids=block_ids,
                block_size=block_size,
                start_token=start,
                num_tokens=curr_tokens,
            )
            chunk_scores = _compute_layer_scores_raw(
                keys_dense=keys_chunk,
                score_head_stats=score_head_stats,
                score_freq_scale_sq=score_freq_scale_sq,
                use_hf_group_max=use_hf_group_max,
                group_size=group_size,
                round_start=round_start,
            )
            chunks.append(chunk_scores)
            start += curr_tokens
        scores = torch.cat(chunks, dim=-1)
        return _finalize_layer_scores(
            scores=scores,
            runtime_heads=runtime_heads,
            use_hf_group_max=use_hf_group_max,
            group_size=group_size,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
        )

    def _build_token_guard_mask(
        *,
        start_token: int,
        num_tokens: int,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        if config.window_size <= 0 and not (protect_prefill and prefill_len > 0):
            return None
        token_positions = torch.arange(
            start_token,
            start_token + num_tokens,
            device=device,
            dtype=torch.long,
        )
        guard_mask = torch.zeros_like(token_positions, dtype=torch.bool)
        if config.window_size > 0:
            window_start = max(0, total_tokens - config.window_size)
            guard_mask |= token_positions >= window_start
        if protect_prefill and prefill_len > 0:
            guard_mask |= token_positions < prefill_len
        return guard_mask

    def _apply_token_guards(
        *,
        scores: torch.Tensor,
        start_token: int,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:
        guard_mask = _build_token_guard_mask(
            start_token=start_token,
            num_tokens=int(scores.shape[-1]),
            total_tokens=total_tokens,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
            device=scores.device,
        )
        if guard_mask is None:
            return scores
        # Avoid host sync on guard_mask.any().item() in hot path.
        # masked_fill is a no-op when guard_mask has no true elements.
        return scores.masked_fill(guard_mask.view(1, 1, -1), float("inf"))

    def _score_chunk_tokens(block_size: int) -> int:
        upper = max(1, int(config.score_chunk_max_tokens))
        return max(block_size, upper)

    def _select_keep_indices_paged_streaming(
        *,
        kv_cache: torch.Tensor,
        block_ids: list[int] | torch.Tensor,
        block_size: int,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        layer_idx: int,
        round_start: int,
        budget_total: int,
    ) -> dict[str, Any]:
        runtime_heads = int(kv_cache.shape[3])
        (
            score_head_stats,
            score_freq_scale_sq,
            use_hf_group_max,
            group_size,
        ) = _resolve_layer_score_inputs(
            layer_idx=layer_idx,
            runtime_heads=runtime_heads,
        )
        chunk_tokens = _score_chunk_tokens(block_size)
        k = min(budget_total, total_tokens)
        if k <= 0:
            return {"mode": "shared", "indices": []}

        # normalize_scores is z-score along token axis (affine monotonic per head/layer),
        # so it does not change top-k ranking. We skip it here to avoid materializing full
        # sequence scores in paged path.
        wants_per_head = requested_pruning_mode in {"per_head", "per_layer_per_head"}
        if wants_per_head:
            best_scores: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None
        else:
            best_scores = None
            best_indices = None

        start = 0
        while start < total_tokens:
            curr_tokens = min(chunk_tokens, total_tokens - start)
            keys_chunk = gather_request_k_dense_range(
                kv_cache=kv_cache,
                block_ids=block_ids,
                block_size=block_size,
                start_token=start,
                num_tokens=curr_tokens,
            )
            chunk_scores = _compute_layer_scores_raw(
                keys_dense=keys_chunk,
                score_head_stats=score_head_stats,
                score_freq_scale_sq=score_freq_scale_sq,
                use_hf_group_max=use_hf_group_max,
                group_size=group_size,
                round_start=round_start,
            )
            if use_hf_group_max:
                chunk_scores = chunk_scores.view(
                    chunk_scores.shape[0],
                    runtime_heads,
                    group_size,
                    chunk_scores.shape[-1],
                ).max(dim=2).values
            chunk_scores = _apply_token_guards(
                scores=chunk_scores,
                start_token=start,
                total_tokens=total_tokens,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
            )

            if wants_per_head and chunk_scores.ndim == 3:
                cand_k = min(k, int(chunk_scores.shape[-1]))
                cand = torch.topk(
                    chunk_scores[0],
                    k=cand_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                cand_scores = cand.values
                cand_indices = cand.indices + start
                if best_scores is None or best_indices is None:
                    best_scores = cand_scores
                    best_indices = cand_indices
                else:
                    merged_scores = torch.cat([best_scores, cand_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, cand_indices], dim=-1)
                    merge_k = min(k, int(merged_scores.shape[-1]))
                    picked = torch.topk(
                        merged_scores,
                        k=merge_k,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )
                    best_scores = picked.values
                    best_indices = torch.gather(
                        merged_indices,
                        dim=-1,
                        index=picked.indices,
                    )
            else:
                if chunk_scores.ndim == 3:
                    chunk_scores = chunk_scores.max(dim=1).values
                cand_k = min(k, int(chunk_scores.shape[-1]))
                cand = torch.topk(
                    chunk_scores[0],
                    k=cand_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                cand_scores = cand.values
                cand_indices = cand.indices + start
                if best_scores is None or best_indices is None:
                    best_scores = cand_scores
                    best_indices = cand_indices
                else:
                    merged_scores = torch.cat([best_scores, cand_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, cand_indices], dim=-1)
                    merge_k = min(k, int(merged_scores.shape[-1]))
                    picked = torch.topk(
                        merged_scores,
                        k=merge_k,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )
                    best_scores = picked.values
                    best_indices = torch.gather(
                        merged_indices,
                        dim=-1,
                        index=picked.indices,
                    )
            start += curr_tokens

        if best_indices is None:
            return {"mode": "shared", "indices": []}
        if wants_per_head and best_indices.ndim == 2:
            keep_per_head = torch.sort(best_indices, dim=-1).values.contiguous()
            return {"mode": "per_head", "indices": keep_per_head}
        keep = torch.sort(best_indices, dim=-1).values.contiguous()
        return {"mode": "shared", "indices": keep}

    def _select_keep_indices(
        *,
        keys_dense: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        block_ids: list[int] | torch.Tensor | None = None,
        block_size: int | None = None,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        layer_idx: int,
        round_start: int,
        budget_total: int,
    ) -> dict[str, Any] | None:
        if total_tokens <= budget_total:
            return {"mode": "shared", "indices": list(range(total_tokens))}
        if protect_prefill and config.include_prefill_in_budget and prefill_len > budget_total:
            return None

        if keys_dense is not None:
            scores = _compute_layer_scores(
                keys_dense=keys_dense,
                layer_idx=layer_idx,
                round_start=round_start,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
            )
        elif kv_cache is not None and block_ids is not None and block_size is not None:
            return _select_keep_indices_paged_streaming(
                kv_cache=kv_cache,
                block_ids=block_ids,
                block_size=block_size,
                total_tokens=total_tokens,
                layer_idx=layer_idx,
                round_start=round_start,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
                budget_total=budget_total,
            )
        else:
            raise RuntimeError("missing scoring inputs for selector")

        k = min(budget_total, scores.shape[-1])
        if k <= 0:
            return {"mode": "shared", "indices": []}

        if requested_pruning_mode in {"per_head", "per_layer_per_head"} and scores.ndim == 3:
            topk = torch.topk(
                scores,
                k=k,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices[0]
            keep_per_head = torch.sort(topk, dim=-1).values.contiguous()
            return {"mode": "per_head", "indices": keep_per_head}

        if scores.ndim == 3:
            # HF SpeckV global mode uses max-style head aggregation by default.
            scores = scores.max(dim=1).values
        selected = torch.topk(
            scores,
            k=k,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices[0]
        keep = torch.sort(selected).values.contiguous()
        return {"mode": "shared", "indices": keep}

    def _select_keep_indices_for_group_per_head(
        *,
        layer_inputs: list[tuple[int, torch.Tensor]] | None = None,
        layer_input_iter: Callable[[], Iterable[tuple[int, torch.Tensor]]] | None = None,
        layer_kv_iter: Callable[
            [],
            Iterable[tuple[int, torch.Tensor, list[int] | torch.Tensor, int]],
        ]
        | None = None,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        round_start: int,
        budget_total: int,
    ) -> dict[str, Any] | None:
        if requested_pruning_mode != "per_head":
            return None
        if per_head_semantics != "hf_aligned_global_per_head":
            return None
        if total_tokens <= budget_total:
            head_count = 0
            if layer_inputs:
                head_count = int(layer_inputs[0][1].shape[1])
            elif layer_input_iter is not None:
                first_item = next(iter(layer_input_iter()), None)
                if first_item is not None:
                    head_count = int(first_item[1].shape[1])
            elif layer_kv_iter is not None:
                first_item = next(iter(layer_kv_iter()), None)
                if first_item is not None:
                    head_count = int(first_item[1].shape[3])
            if head_count <= 0:
                return {"mode": "per_head", "indices": []}
            all_indices = torch.arange(
                total_tokens,
                dtype=torch.long,
                device=(
                    layer_inputs[0][1].device
                    if layer_inputs
                    else (
                        first_item[1].device
                        if first_item is not None
                        else torch.device("cpu")
                    )
                ),
            )
            return {
                "mode": "per_head",
                "indices": all_indices.unsqueeze(0).expand(head_count, -1).contiguous(),
            }
        if protect_prefill and config.include_prefill_in_budget and prefill_len > budget_total:
            return None
        if layer_kv_iter is not None:
            iter_inputs = layer_kv_iter()
            iter_mode = "paged"
        elif layer_input_iter is not None:
            iter_inputs = layer_input_iter()
            iter_mode = "dense_iter"
        else:
            iter_inputs = layer_inputs or []
            iter_mode = "dense_list"
        if not iter_inputs:
            return None

        if iter_mode == "paged":
            layer_entries = list(iter_inputs)
            if not layer_entries:
                return None
            k = min(budget_total, total_tokens)
            if k <= 0:
                return {"mode": "per_head", "indices": []}
            prepared_layers: list[dict[str, Any]] = []
            for layer_idx, kv_cache, block_ids, layer_block_size in layer_entries:
                runtime_heads = int(kv_cache.shape[3])
                (
                    score_head_stats,
                    score_freq_scale_sq,
                    use_hf_group_max,
                    group_size,
                ) = _resolve_layer_score_inputs(
                    layer_idx=layer_idx,
                    runtime_heads=runtime_heads,
                )
                prepared_layers.append(
                    {
                        "layer_idx": layer_idx,
                        "kv_cache": kv_cache,
                        "block_ids": block_ids,
                        "block_size": layer_block_size,
                        "runtime_heads": runtime_heads,
                        "score_head_stats": score_head_stats,
                        "score_freq_scale_sq": score_freq_scale_sq,
                        "use_hf_group_max": use_hf_group_max,
                        "group_size": group_size,
                    }
                )

            min_block_size = min(entry["block_size"] for entry in prepared_layers)
            chunk_tokens = _score_chunk_tokens(min_block_size)
            norm_stats: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(prepared_layers)
            if config.sparse_normalize_scores:
                eps = 1e-8
                for layer_pos, entry in enumerate(prepared_layers):
                    sum_vec: torch.Tensor | None = None
                    sumsq_vec: torch.Tensor | None = None
                    count = 0
                    start = 0
                    while start < total_tokens:
                        curr_tokens = min(chunk_tokens, total_tokens - start)
                        keys_chunk = gather_request_k_dense_range(
                            kv_cache=entry["kv_cache"],
                            block_ids=entry["block_ids"],
                            block_size=entry["block_size"],
                            start_token=start,
                            num_tokens=curr_tokens,
                        )
                        raw_scores = _compute_layer_scores_raw(
                            keys_dense=keys_chunk,
                            score_head_stats=entry["score_head_stats"],
                            score_freq_scale_sq=entry["score_freq_scale_sq"],
                            use_hf_group_max=entry["use_hf_group_max"],
                            group_size=entry["group_size"],
                            round_start=round_start,
                        )[0]
                        raw_fp32 = raw_scores.to(dtype=torch.float32)
                        chunk_sum = raw_fp32.sum(dim=-1)
                        chunk_sumsq = (raw_fp32 * raw_fp32).sum(dim=-1)
                        if sum_vec is None:
                            sum_vec = chunk_sum
                            sumsq_vec = chunk_sumsq
                        else:
                            sum_vec = sum_vec + chunk_sum
                            sumsq_vec = sumsq_vec + chunk_sumsq
                        count += curr_tokens
                        start += curr_tokens

                    if (
                        sum_vec is None
                        or sumsq_vec is None
                        or count <= 0
                    ):
                        return None
                    mean = sum_vec / float(count)
                    if count > 1:
                        var = (sumsq_vec - float(count) * (mean * mean)) / float(count - 1)
                    else:
                        var = torch.zeros_like(mean)
                    var = torch.clamp(var, min=0.0)
                    std = torch.sqrt(var)
                    std_safe = torch.where(std < eps, torch.ones_like(std), std)
                    norm_stats[layer_pos] = (
                        mean,
                        std_safe,
                    )

            best_scores: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None
            start = 0
            while start < total_tokens:
                curr_tokens = min(chunk_tokens, total_tokens - start)
                chunk_guard_mask = _build_token_guard_mask(
                    start_token=start,
                    num_tokens=curr_tokens,
                    total_tokens=total_tokens,
                    prefill_len=prefill_len,
                    protect_prefill=protect_prefill,
                    device=prepared_layers[0]["kv_cache"].device,
                )
                chunk_sum: torch.Tensor | None = None
                layer_count = 0
                for layer_pos, entry in enumerate(prepared_layers):
                    keys_chunk = gather_request_k_dense_range(
                        kv_cache=entry["kv_cache"],
                        block_ids=entry["block_ids"],
                        block_size=entry["block_size"],
                        start_token=start,
                        num_tokens=curr_tokens,
                    )
                    chunk_scores = _compute_layer_scores_raw(
                        keys_dense=keys_chunk,
                        score_head_stats=entry["score_head_stats"],
                        score_freq_scale_sq=entry["score_freq_scale_sq"],
                        use_hf_group_max=entry["use_hf_group_max"],
                        group_size=entry["group_size"],
                        round_start=round_start,
                    )
                    if config.sparse_normalize_scores:
                        mean, std_safe = norm_stats[layer_pos] or (None, None)
                        if mean is None or std_safe is None:
                            return None
                        chunk_scores = (chunk_scores - mean.view(1, -1, 1)) / std_safe.view(1, -1, 1)
                    if chunk_guard_mask is not None:
                        chunk_scores = chunk_scores.masked_fill(
                            chunk_guard_mask.view(1, 1, -1),
                            float("inf"),
                        )
                    if entry["use_hf_group_max"]:
                        chunk_scores = chunk_scores.view(
                            chunk_scores.shape[0],
                            entry["runtime_heads"],
                            entry["group_size"],
                            chunk_scores.shape[-1],
                        ).max(dim=2).values
                    if chunk_scores.ndim != 3:
                        raise RuntimeError(
                            f"unexpected_score_rank_for_per_head:{chunk_scores.ndim}"
                        )
                    layer_scores = chunk_scores[0]
                    if chunk_sum is None:
                        chunk_sum = layer_scores.clone()
                    else:
                        chunk_sum.add_(layer_scores)
                    layer_count += 1

                if chunk_sum is None or layer_count <= 0:
                    return None
                chunk_avg = chunk_sum.div(float(layer_count))
                cand_k = min(k, int(chunk_avg.shape[-1]))
                cand = torch.topk(
                    chunk_avg,
                    k=cand_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                cand_scores = cand.values
                cand_indices = cand.indices + start
                if best_scores is None or best_indices is None:
                    best_scores = cand_scores
                    best_indices = cand_indices
                else:
                    merged_scores = torch.cat([best_scores, cand_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, cand_indices], dim=-1)
                    merge_k = min(k, int(merged_scores.shape[-1]))
                    picked = torch.topk(
                        merged_scores,
                        k=merge_k,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )
                    best_scores = picked.values
                    best_indices = torch.gather(
                        merged_indices,
                        dim=-1,
                        index=picked.indices,
                    )
                start += curr_tokens

            if best_indices is None:
                return None
            keep_per_head = torch.sort(best_indices, dim=-1).values.contiguous()
            return {
                "mode": "per_head",
                "indices": keep_per_head,
                "semantic": "hf_aligned_global_per_head",
            }
        else:
            aggregated_scores: torch.Tensor | None = None
            layer_count = 0
            for layer_idx, keys_dense in iter_inputs:
                scores = _compute_layer_scores(
                    keys_dense=keys_dense,
                    layer_idx=layer_idx,
                    round_start=round_start,
                    prefill_len=prefill_len,
                    protect_prefill=protect_prefill,
                )
                if scores.ndim != 3:
                    raise RuntimeError(
                        f"unexpected_score_rank_for_per_head:{scores.ndim}"
                    )
                layer_scores = scores[0]
                if aggregated_scores is None:
                    aggregated_scores = layer_scores.clone()
                else:
                    aggregated_scores.add_(layer_scores)
                layer_count += 1
            if aggregated_scores is None or layer_count <= 0:
                return None
            aggregated_scores.div_(layer_count)
            k = min(budget_total, aggregated_scores.shape[-1])
            if k <= 0:
                return {"mode": "per_head", "indices": []}

            topk = torch.topk(
                aggregated_scores,
                k=k,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            keep_per_head = torch.sort(topk, dim=-1).values.contiguous()
            return {
                "mode": "per_head",
                "indices": keep_per_head,
                "semantic": "hf_aligned_global_per_head",
            }

    setattr(_select_keep_indices, "_supports_paged", True)
    setattr(_select_keep_indices_for_group_per_head, "_supports_paged_group", True)
    return _select_keep_indices, _select_keep_indices_for_group_per_head, "enabled"


def make_runner_compression_hook(
    base_runner: Any,
    config: TriAttentionV2Config,
) -> Callable[..., dict[str, Any]]:
    """Create a hook function bound to a concrete base runner."""
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
        strict_triton_required = bool(
            config.enable_experimental_kv_compaction and config.require_triton_scoring
        )
        requests = getattr(base_runner, "requests", None)
        if not isinstance(requests, dict):
            return {"applied": False, "reason": "missing_requests"}

        req_state = requests.get(req_id)
        if req_state is None:
            return {"applied": False, "reason": "req_state_not_found"}
        cache_config_hint = getattr(base_runner, "cache_config", None)
        block_size_hint = int(getattr(cache_config_hint, "block_size", 0))
        if block_size_hint <= 0:
            block_size_hint = 1
        original_block_ids_by_group = getattr(req_state, "block_ids", None)
        block_capacity_hint = _min_block_capacity_tokens(
            block_ids_by_group=original_block_ids_by_group,
            block_size=block_size_hint,
        )

        # Use scheduler-estimated effective cache length (tracker-aware) as the
        # logical KV length for gather/selection/compaction work. num_computed_tokens
        # is monotonic and may be far larger than effective compressed cache length.
        num_computed_tokens = int(getattr(req_state, "num_computed_tokens", 0))
        effective_tokens = int(getattr(signal, "estimated_cache_len", num_computed_tokens))
        if effective_tokens < 0:
            effective_tokens = 0
        if effective_tokens > num_computed_tokens:
            effective_tokens = num_computed_tokens
        # Runner-side block ids are the nearest physical source of truth when
        # scheduler-side effective length tracking is delayed.
        if isinstance(block_capacity_hint, int):
            # Allow one extra block of scheduler/runner skew before clamping.
            physical_upper = block_capacity_hint + block_size_hint
            if effective_tokens > physical_upper:
                effective_tokens = physical_upper

        if (
            config.fail_on_effective_len_regression
            and req_id in compressed_once
        ):
            guard_upper = _effective_len_guard_upper(config, signal)
            # Scheduler signal carries an estimate produced before runner-side
            # compression/execution; allow a small slack based on that estimate
            # to avoid false positives from async step skew.
            estimated_slack = max(
                1,
                int(getattr(signal, "estimated_cache_len", 0)) - num_computed_tokens,
            )
            scheduled_tokens = _scheduled_tokens_for_req(
                scheduler_output=scheduler_output,
                req_id=req_id,
            )
            # Keep a minimal block-granularity slack. Even in strict mode, a few
            # tokens above guard can appear due to scheduler/runner skew and
            # block-level accounting lag; this should not abort the engine.
            regression_slack = (
                block_size_hint
                + estimated_slack
                + max(1, scheduled_tokens)
            )
            if (
                effective_tokens > (guard_upper + regression_slack)
                and num_computed_tokens > (guard_upper + regression_slack)
                and effective_tokens
                >= int(config.effective_len_regression_ratio * num_computed_tokens)
            ):
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:effective_len_regressed:"
                    f"req={req_id}:effective_tokens={effective_tokens}:"
                    f"num_computed_tokens={num_computed_tokens}:guard_upper={guard_upper}"
                )

        budget_total = _effective_budget_for_signal(config, signal, effective_tokens)
        # Local threshold gate: avoid step-by-step recompression once scheduler-side
        # effective length tracking lags behind. Keep kv-usage trigger as override.
        local_length_threshold = budget_total + max(1, config.divide_length)
        length_gate_hit = effective_tokens >= local_length_threshold
        kv_override = str(getattr(signal, "reason", "")) == "kv_usage_threshold"
        should_defer_recompress = (
            config.enable_experimental_kv_compaction
            and req_id in compressed_once
            and not kv_override
            and not length_gate_hit
        )
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

        mutable_block_ids_by_group: list[list[int] | None] = []
        for group_block_ids in original_block_ids_by_group:
            if not isinstance(group_block_ids, (list, tuple)):
                mutable_block_ids_by_group.append(None)
                continue
            mutable_block_ids_by_group.append([int(block_id) for block_id in group_block_ids])

        group_tensors = _get_group_tensors()
        compacted_any_group = False
        cache_len_after = None
        selection_mode = "fallback"
        block_reclaim_groups: list[dict[str, Any]] = []

        for gid, normalized_block_ids in enumerate(mutable_block_ids_by_group):
            if not normalized_block_ids:
                continue
            layer_tensors = group_tensors.get(gid)
            if not layer_tensors:
                continue
            # Physical page capacity can lag scheduler-side token counters by ~1 step.
            # Clamp compaction/scoring to block_ids capacity to avoid gather OOB.
            group_capacity_tokens = len(normalized_block_ids) * block_size
            group_total_tokens = min(effective_tokens, group_capacity_tokens)
            if group_total_tokens <= 0:
                continue
            group_prefill_len = min(int(signal.prefill_len), group_total_tokens)
            group_budget_total = min(budget_total, group_total_tokens)
            block_ids_tensor_cache: dict[torch.device, torch.Tensor] = {}
            round_start = int(max(effective_tokens, num_computed_tokens))
            selected_for_group: dict[str, Any] | None = None
            group_cache_len_after: int | None = None
            if (
                select_keep_indices_for_group is not None
                and config.pruning_mode == "per_head"
                and config.per_head_selection_semantics == "hf_aligned_global_per_head"
            ):
                try:
                    if strict_triton_required:
                        if not getattr(select_keep_indices_for_group, "_supports_paged_group", False):
                            raise RuntimeError("paged_group_selector_required")
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
                            protect_prefill=signal.protect_prefill,
                            round_start=round_start,
                            budget_total=group_budget_total,
                        )
                    elif getattr(select_keep_indices_for_group, "_supports_paged_group", False):
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
                            protect_prefill=signal.protect_prefill,
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
                                keys_dense = gather_request_k_dense(
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
                            protect_prefill=signal.protect_prefill,
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
                            selected = select_keep_indices(
                                keys_dense=None,
                                kv_cache=kv_cache,
                                block_ids=block_ids_tensor,
                                block_size=block_size,
                                total_tokens=group_total_tokens,
                                prefill_len=group_prefill_len,
                                protect_prefill=signal.protect_prefill,
                                layer_idx=layer_idx,
                                round_start=round_start,
                                budget_total=group_budget_total,
                            )
                        elif getattr(select_keep_indices, "_supports_paged", False):
                            selected = select_keep_indices(
                                keys_dense=None,
                                kv_cache=kv_cache,
                                block_ids=block_ids_tensor,
                                block_size=block_size,
                                total_tokens=group_total_tokens,
                                prefill_len=group_prefill_len,
                                protect_prefill=signal.protect_prefill,
                                layer_idx=layer_idx,
                                round_start=round_start,
                                budget_total=group_budget_total,
                            )
                        else:
                            keys_dense = gather_request_k_dense(
                                kv_cache=kv_cache,
                                block_ids=block_ids_tensor,
                                block_size=block_size,
                                total_tokens=group_total_tokens,
                            )
                            selected = select_keep_indices(
                                keys_dense=keys_dense,
                                total_tokens=group_total_tokens,
                                prefill_len=group_prefill_len,
                                protect_prefill=signal.protect_prefill,
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

                if selected is None:
                    keep_indices = build_keep_token_indices(
                        total_tokens=group_total_tokens,
                        kv_budget=config.kv_budget,
                        prefill_len=group_prefill_len,
                        protect_prefill=signal.protect_prefill,
                        include_prefill_in_budget=config.include_prefill_in_budget,
                    )
                    if keep_indices is None:
                        return {"applied": False, "reason": "prefill_exceeds_budget"}
                    if strict_triton_required:
                        raise RuntimeError(
                            f"{TRITON_SCORING_REQUIRED_MARKER}:selector_returned_none:"
                            f"req={req_id}:gid={gid}:layer={layer_idx}"
                        )
                    selected = {"mode": "shared", "indices": keep_indices}
                    selection_mode = "fallback"
                else:
                    selection_mode = str(selected.get("mode", "shared"))
                    semantic = selected.get("semantic")
                    if semantic:
                        selection_mode = f"{selection_mode}:{semantic}"

                try:
                    layer_cache_len_after: int
                    keep_count = _selected_keep_count(selected)
                    before_required = _num_required_blocks(group_total_tokens, block_size)
                    after_required = _num_required_blocks(keep_count, block_size)
                    preserve_dropped_tokens = True
                    if (
                        config.enable_experimental_block_reclaim
                        and after_required < before_required
                    ):
                        preserve_dropped_tokens = False
                    if selected["mode"] == "per_head":
                        layer_cache_len_after = compact_request_kv_in_place_per_head(
                            kv_cache=kv_cache,
                            block_ids=block_ids_tensor,
                            block_size=block_size,
                            keep_token_indices_per_head=selected["indices"],
                            total_tokens=group_total_tokens,
                            preserve_dropped_tokens=preserve_dropped_tokens,
                        )
                    else:
                        layer_cache_len_after = compact_request_kv_in_place(
                            kv_cache=kv_cache,
                            block_ids=block_ids_tensor,
                            block_size=block_size,
                            keep_token_indices=selected["indices"],
                            total_tokens=group_total_tokens,
                            preserve_dropped_tokens=preserve_dropped_tokens,
                        )
                    cache_len_after = layer_cache_len_after
                    if group_cache_len_after is None:
                        group_cache_len_after = layer_cache_len_after
                    compacted_any_group = True
                except Exception as exc:
                    return {
                        "applied": False,
                        "reason": f"compaction_failed:g{gid}:l{layer_idx}:{type(exc).__name__}",
                    }

            if (
                config.enable_experimental_block_reclaim
                and group_cache_len_after is not None
            ):
                required_blocks = _num_required_blocks(group_cache_len_after, block_size)
                kept_block_ids = normalized_block_ids[:required_blocks]
                removed_block_ids = normalized_block_ids[required_blocks:]
                mutable_block_ids_by_group[gid] = kept_block_ids
                if removed_block_ids:
                    block_reclaim_groups.append(
                        {
                            "gid": gid,
                            "block_ids_before": normalized_block_ids,
                            "block_ids_after": kept_block_ids,
                            "block_ids_removed": removed_block_ids,
                        }
                    )
                if config.require_physical_reclaim:
                    before_required = _num_required_blocks(group_total_tokens, block_size)
                    expected_removed_min = max(0, before_required - required_blocks)
                    if expected_removed_min > 0 and len(removed_block_ids) < expected_removed_min:
                        raise RuntimeError(
                            f"{TRITON_SCORING_REQUIRED_MARKER}:physical_reclaim_missing:"
                            f"req={req_id}:gid={gid}:expected_removed>={expected_removed_min}:"
                            f"actual_removed={len(removed_block_ids)}:"
                            f"effective_tokens={group_total_tokens}:cache_len_after={group_cache_len_after}"
                        )

        if not compacted_any_group:
            return {"applied": False, "reason": "no_compactable_groups"}

        compressed_once.add(req_id)

        block_reclaim_payload: dict[str, Any] | None = None
        reclaimed_block_count = 0
        if config.enable_experimental_block_reclaim and block_reclaim_groups:
            reclaimed_block_count = sum(
                len(group.get("block_ids_removed", []))
                for group in block_reclaim_groups
                if isinstance(group, dict)
            )
            reassigned_block_ids = []
            for idx, group_block_ids in enumerate(mutable_block_ids_by_group):
                if group_block_ids is None:
                    reassigned_block_ids.append(original_block_ids_by_group[idx])
                else:
                    reassigned_block_ids.append(group_block_ids)
            req_state.block_ids = (
                tuple(reassigned_block_ids)
                if isinstance(original_block_ids_by_group, tuple)
                else reassigned_block_ids
            )
            block_reclaim_payload = {
                "mode": "truncate_tail",
                "groups": block_reclaim_groups,
            }

        return {
            "applied": True,
            "reason": f"kv_compacted:{selection_mode}",
            "cache_len_after": cache_len_after,
            "selector_status": selector_status,
            "block_reclaim": block_reclaim_payload,
            "effective_tokens_before": effective_tokens,
            "budget_total": budget_total,
            "reclaimed_block_count": reclaimed_block_count,
        }

    return _hook


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
