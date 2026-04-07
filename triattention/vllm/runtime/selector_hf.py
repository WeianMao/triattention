"""HF-aligned TriAttention selector implementation for TriAttention runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .kv_compaction import gather_request_k_dense, gather_request_k_dense_range

def build_triattention_selector(
    config: TriAttentionRuntimeConfig,
    base_runner: Any | None = None,
) -> tuple[
    Callable[..., dict[str, Any] | None] | None,
    Callable[..., dict[str, Any] | None] | None,
    str,
]:
    """Build TriAttention selector callable.

    The returned selector emits either:
    - {"mode": "shared", "indices": Tensor|list[int]}
    - {"mode": "per_head", "indices": Tensor|list[list[int]]}
    """
    requested_pruning_mode = config.pruning_mode
    if requested_pruning_mode == "per_layer" and not bool(
        getattr(config, "allow_per_layer_mode", False)
    ):
        raise RuntimeError(
            f"{TRITON_SCORING_REQUIRED_MARKER}:per_layer_mode_disabled:"
            "set allow_per_layer_mode=True for explicit opt-in"
        )

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
        from triattention.vllm.core.config import TriAttentionConfig
        from triattention.vllm.core.compressor import TriAttentionCompressor
        from triattention.vllm.core.scoring import compute_scores_triton
        from triattention.vllm.core.utils import normalize_scores
    except Exception as exc:  # pragma: no cover - import safety
        raise RuntimeError(
            f"{TRITON_SCORING_REQUIRED_MARKER}:import_failed:{type(exc).__name__}"
        ) from exc

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

    def _resolve_effective_model_path() -> Path | None:
        if getattr(config, "model_path", None) is not None:
            return Path(config.model_path)
        if base_runner is None:
            return None
        candidates: list[Any] = []
        candidates.append(getattr(getattr(base_runner, "model_config", None), "model", None))
        candidates.append(
            getattr(
                getattr(getattr(base_runner, "vllm_config", None), "model_config", None),
                "model",
                None,
            )
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return Path(candidate)
            if isinstance(candidate, Path):
                return candidate
        return None

    effective_model_path = _resolve_effective_model_path()

    tri_cfg = TriAttentionConfig(
        stats_path=stats_path,
        model_path=effective_model_path,
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
        compute_dtype=torch.float32,
        topk_dtype=torch.float32,
    )
    compressor = TriAttentionCompressor(tri_cfg)
    available_layers_sorted: tuple[int, ...] | None = None
    available_layers_set: set[int] | None = None
    dump_k_sample_path = os.environ.get(
        "TRIATTN_DEBUG_DUMP_FIRST_K_SAMPLE_PATH",
        "",
    ).strip()
    dump_k_sample_done = False
    dump_dense_keys_path = os.environ.get(
        "TRIATTN_DEBUG_DUMP_FIRST_DENSE_KEYS_PATH",
        "",
    ).strip()
    dump_single_layer_dense_keys_path = os.environ.get(
        "TRIATTN_DEBUG_DUMP_FIRST_DENSE_LAYER_PATH",
        "",
    ).strip()
    dump_single_layer_target_raw = os.environ.get(
        "TRIATTN_DEBUG_DUMP_FIRST_DENSE_LAYER_INDEX",
        "",
    ).strip()
    dump_single_layer_target = (
        int(dump_single_layer_target_raw) if dump_single_layer_target_raw else None
    )
    dump_dense_keys_done = False
    dump_single_layer_dense_keys_done = False
    debug_dump_group_score_samples = (
        os.environ.get("TRIATTN_DEBUG_DUMP_GROUP_SCORE_SAMPLES", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    debug_dump_model_path_info_path = os.environ.get(
        "TRIATTN_DEBUG_DUMP_SELECTOR_MODEL_PATH_INFO_PATH",
        "",
    ).strip()
    debug_dump_model_path_info_done = False

    def _resolve_effective_recent_count(total_tokens: int) -> int:
        if total_tokens <= 0 or config.window_size <= 0:
            return 0
        # The runtime selector must preserve the same trailing protection window
        # regardless of request lifecycle details. Tying this to transient
        # "recent_unabsorbed" bookkeeping lets live serve requests under-protect
        # the tail (often collapsing to zero) even though fresh/offline
        # selection correctly preserves `window_size` tokens. That divergence
        # changes the keep set and cascades into output corruption.
        return min(config.window_size, total_tokens)

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

    if debug_dump_model_path_info_path and not debug_dump_model_path_info_done:
        try:
            compressor._lazy_init()
            dump_path = Path(debug_dump_model_path_info_path)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            omega_head = (
                compressor.omega[:8].detach().to(device="cpu", dtype=torch.float32).tolist()
                if getattr(compressor, "omega", None) is not None
                else None
            )
            inv_freq_head = (
                compressor.inv_freq[:8].detach().to(device="cpu", dtype=torch.float32).tolist()
                if getattr(compressor, "inv_freq", None) is not None
                else None
            )
            payload = {
                "config_model_path": (
                    None if getattr(config, "model_path", None) is None else str(config.model_path)
                ),
                "resolved_effective_model_path": (
                    None if effective_model_path is None else str(effective_model_path)
                ),
                "base_runner_type": None if base_runner is None else type(base_runner).__name__,
                "base_runner_model_config_model": None,
                "base_runner_vllm_config_model_config_model": None,
                "compressor_rope_style": getattr(compressor.config, "rope_style", None),
                "compressor_omega_head": omega_head,
                "compressor_inv_freq_head": inv_freq_head,
            }
            if base_runner is not None:
                model_config = getattr(base_runner, "model_config", None)
                model_value = getattr(model_config, "model", None)
                if model_value is not None:
                    payload["base_runner_model_config_model"] = str(model_value)
                vllm_config = getattr(base_runner, "vllm_config", None)
                inner_model_config = getattr(vllm_config, "model_config", None)
                inner_model_value = getattr(inner_model_config, "model", None)
                if inner_model_value is not None:
                    payload["base_runner_vllm_config_model_config_model"] = str(inner_model_value)
            dump_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            debug_dump_model_path_info_done = True
        except Exception:
            pass

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
        nonlocal dump_single_layer_dense_keys_done

        should_dump_single_layer = (
            dump_single_layer_dense_keys_path
            and not dump_single_layer_dense_keys_done
            and (
                dump_single_layer_target is None
                or int(layer_idx) == dump_single_layer_target
            )
        )
        if should_dump_single_layer:
            try:
                dump_path = Path(dump_single_layer_dense_keys_path)
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "layer_idx": int(layer_idx),
                    "round_start": int(round_start),
                    "total_tokens": int(keys_dense.shape[0]),
                    "num_heads": int(keys_dense.shape[1]),
                    "head_dim": int(keys_dense.shape[2]),
                    "dtype": str(keys_dense.dtype),
                    "keys_dense": keys_dense.detach().to(device="cpu"),
                }
                torch.save(payload, dump_path)
                dump_single_layer_dense_keys_done = True
            except Exception:
                pass

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
            stats_heads != runtime_heads
            and (
                (
                    requested_pruning_mode == "per_head"
                    and per_head_semantics == "hf_aligned_global_per_head"
                )
                or requested_pruning_mode == "per_layer_per_head"
            )
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

    def _reduce_grouped_head_scores(
        *,
        scores: torch.Tensor,
        runtime_heads: int,
        group_size: int,
        aggregate_mode: str,
    ) -> torch.Tensor:
        grouped = scores.view(
            scores.shape[0],
            runtime_heads,
            group_size,
            scores.shape[-1],
        )
        if aggregate_mode == "mean":
            return grouped.mean(dim=2)
        return grouped.max(dim=2).values

    def _layer_group_aggregation_mode() -> str:
        if requested_pruning_mode == "per_layer_per_head":
            return config.layer_perhead_aggregation
        return "max"

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
            total_tokens = int(scores.shape[-1])
            recent_count = _resolve_effective_recent_count(total_tokens)
            if recent_count > 0:
                scores[..., total_tokens - recent_count :] = float("inf")
        if protect_prefill and prefill_len > 0:
            scores[..., :prefill_len] = float("inf")
        if use_hf_group_max:
            scores = _reduce_grouped_head_scores(
                scores=scores,
                runtime_heads=runtime_heads,
                group_size=group_size,
                aggregate_mode=_layer_group_aggregation_mode(),
            )
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
        nonlocal dump_single_layer_dense_keys_done
        should_dump_single_layer = (
            dump_single_layer_dense_keys_path
            and not dump_single_layer_dense_keys_done
            and (
                dump_single_layer_target is None
                or int(layer_idx) == dump_single_layer_target
            )
        )
        if should_dump_single_layer:
            try:
                dump_path = Path(dump_single_layer_dense_keys_path)
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                dense_keys_full = gather_request_k_dense(
                    kv_cache=kv_cache,
                    block_ids=block_ids,
                    block_size=block_size,
                    total_tokens=total_tokens,
                )
                payload = {
                    "layer_idx": int(layer_idx),
                    "round_start": int(round_start),
                    "total_tokens": int(total_tokens),
                    "num_heads": int(dense_keys_full.shape[1]),
                    "head_dim": int(dense_keys_full.shape[2]),
                    "dtype": str(dense_keys_full.dtype),
                    "keys_dense": dense_keys_full.detach().to(device="cpu"),
                }
                torch.save(payload, dump_path)
                dump_single_layer_dense_keys_done = True
            except Exception:
                pass
        chunk_tokens = _score_chunk_tokens(block_size, total_tokens)
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
            recent_count = _resolve_effective_recent_count(total_tokens)
            window_start = max(0, total_tokens - recent_count)
            guard_mask |= token_positions >= window_start
        if protect_prefill and prefill_len > 0:
            guard_mask |= token_positions < prefill_len
        return guard_mask

    def _dump_first_k_sample(
        *,
        layer_entries: list[dict[str, Any]],
        total_tokens: int,
    ) -> None:
        nonlocal dump_k_sample_done
        if dump_k_sample_done or not dump_k_sample_path or not layer_entries:
            return
        sample_positions = sorted(
            set(
                [0, 1, 2, 3, 4]
                + [max(0, total_tokens // 4 - 1), total_tokens // 4, min(total_tokens - 1, total_tokens // 4 + 1)]
                + [max(0, total_tokens // 2 - 1), total_tokens // 2, min(total_tokens - 1, total_tokens // 2 + 1)]
                + [max(0, (3 * total_tokens) // 4 - 1), (3 * total_tokens) // 4, min(total_tokens - 1, (3 * total_tokens) // 4 + 1)]
                + [max(0, total_tokens - 5), max(0, total_tokens - 4), max(0, total_tokens - 3), max(0, total_tokens - 2), total_tokens - 1]
            )
        )
        payload = {
            "total_tokens": int(total_tokens),
            "sample_positions": [int(x) for x in sample_positions],
            "layers": {},
        }
        preferred_layers = {0, 1, 16, 32, 48, 63}
        selected_entries = [
            entry for entry in layer_entries if int(entry["layer_idx"]) in preferred_layers
        ]
        if not selected_entries:
            selected_entries = layer_entries[:2]
        for entry in selected_entries:
            keys_dense = gather_request_k_dense(
                kv_cache=entry["kv_cache"],
                block_ids=entry["block_ids"],
                block_size=entry["block_size"],
                total_tokens=total_tokens,
            )
            layer_payload = {}
            head_limit = min(2, int(keys_dense.shape[1]))
            for head_idx in range(head_limit):
                vectors = []
                for pos in sample_positions:
                    vectors.append(
                        keys_dense[0, head_idx, pos]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .tolist()
                    )
                layer_payload[str(head_idx)] = vectors
            payload["layers"][str(entry["layer_idx"])] = layer_payload
        out_path = Path(dump_k_sample_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        dump_k_sample_done = True

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

    def _score_chunk_tokens(block_size: int, total_tokens: int) -> int:
        upper = max(block_size, int(config.score_chunk_max_tokens))
        # Small/medium effective lengths do not need chunking; avoiding chunk splits
        # reduces Python loop overhead and kernel launches in the hot scoring path.
        if total_tokens <= upper:
            return max(block_size, total_tokens)
        return upper

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
        chunk_tokens = _score_chunk_tokens(block_size, total_tokens)
        k = min(budget_total, total_tokens)
        if k <= 0:
            return {"mode": "shared", "indices": []}

        norm_stats: tuple[torch.Tensor, torch.Tensor] | None = None
        raw_chunk_scores_cache: list[torch.Tensor] | None = None
        if config.sparse_normalize_scores:
            eps = 1e-8
            sum_vec: torch.Tensor | None = None
            sumsq_vec: torch.Tensor | None = None
            count = 0
            raw_chunk_scores_cache = []
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
                raw_scores = _compute_layer_scores_raw(
                    keys_dense=keys_chunk,
                    score_head_stats=score_head_stats,
                    score_freq_scale_sq=score_freq_scale_sq,
                    use_hf_group_max=use_hf_group_max,
                    group_size=group_size,
                    round_start=round_start,
                )[0]
                raw_chunk_scores_cache.append(raw_scores)
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
            if sum_vec is None or sumsq_vec is None or count <= 0:
                return None
            mean = sum_vec / float(count)
            if count > 1:
                var = (sumsq_vec - float(count) * (mean * mean)) / float(count - 1)
            else:
                var = torch.zeros_like(mean)
            var = torch.clamp(var, min=0.0)
            std = torch.sqrt(var)
            std_safe = torch.where(std < eps, torch.ones_like(std), std)
            norm_stats = (mean, std_safe)

        # normalize_scores is z-score along token axis (affine monotonic per head/layer),
        # but for paths that aggregate across heads (e.g. max), normalization must be
        # preserved for HF alignment semantics. We use a two-pass chunked statistics
        # accumulation above instead of materializing full sequence scores.
        wants_per_head = requested_pruning_mode in {"per_head", "per_layer_per_head"}
        if wants_per_head:
            best_scores: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None
        else:
            best_scores = None
            best_indices = None

        start = 0
        chunk_idx = 0
        while start < total_tokens:
            curr_tokens = min(chunk_tokens, total_tokens - start)
            if raw_chunk_scores_cache is not None and chunk_idx < len(raw_chunk_scores_cache):
                chunk_scores = raw_chunk_scores_cache[chunk_idx].unsqueeze(0)
            else:
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
            if norm_stats is not None:
                mean, std_safe = norm_stats
                chunk_scores = (
                    chunk_scores - mean.view(1, -1, 1)
                ) / std_safe.view(1, -1, 1)
            if use_hf_group_max:
                chunk_scores = _reduce_grouped_head_scores(
                    scores=chunk_scores,
                    runtime_heads=runtime_heads,
                    group_size=group_size,
                    aggregate_mode=_layer_group_aggregation_mode(),
                )
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
            chunk_idx += 1

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
            paged_result = _select_keep_indices_paged_streaming(
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
            if (
                os.environ.get("TRIATTN_DEBUG_COMPARE_PAGED_DENSE_KEEP", "0") == "1"
                and paged_result is not None
            ):
                try:
                    keys_dense_dbg = gather_request_k_dense(
                        kv_cache=kv_cache,
                        block_ids=block_ids,
                        block_size=block_size,
                        total_tokens=total_tokens,
                    )
                    dense_scores_dbg = _compute_layer_scores(
                        keys_dense=keys_dense_dbg,
                        layer_idx=layer_idx,
                        round_start=round_start,
                        prefill_len=prefill_len,
                        protect_prefill=protect_prefill,
                    )
                    k_dbg = min(int(budget_total), int(dense_scores_dbg.shape[-1]))
                    if k_dbg <= 0:
                        dense_result = {"mode": "shared", "indices": []}
                    elif (
                        requested_pruning_mode in {"per_head", "per_layer_per_head"}
                        and dense_scores_dbg.ndim == 3
                    ):
                        topk_dbg = torch.topk(
                            dense_scores_dbg,
                            k=k_dbg,
                            dim=-1,
                            largest=True,
                            sorted=False,
                        ).indices[0]
                        dense_result = {
                            "mode": "per_head",
                            "indices": torch.sort(topk_dbg, dim=-1).values.contiguous(),
                        }
                    else:
                        dense_scores_agg = dense_scores_dbg
                        if dense_scores_agg.ndim == 3:
                            dense_scores_agg = dense_scores_agg.max(dim=1).values
                        selected_dbg = torch.topk(
                            dense_scores_agg,
                            k=k_dbg,
                            dim=-1,
                            largest=True,
                            sorted=False,
                        ).indices[0]
                        dense_result = {
                            "mode": "shared",
                            "indices": torch.sort(selected_dbg).values.contiguous(),
                        }
                    _debug_compare_selector_results(
                        paged_result=paged_result,
                        dense_result=dense_result,
                        total_tokens=total_tokens,
                        layer_idx=layer_idx,
                        round_start=round_start,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "TRIATTN_PAGED_DENSE_COMPARE_FAILED:"
                        f"layer_idx={layer_idx}:total_tokens={total_tokens}:"
                        f"round_start={round_start}:{type(exc).__name__}:{exc}"
                    ) from exc
            return paged_result
        else:
            raise RuntimeError("missing scoring inputs for selector")

        k = min(int(budget_total), int(scores.shape[-1]))
        if k <= 0:
            return {"mode": "shared", "indices": []}
        wants_per_head = requested_pruning_mode in {"per_head", "per_layer_per_head"}
        if wants_per_head and scores.ndim == 3:
            topk = torch.topk(
                scores,
                k=k,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices[0]
            keep_per_head = torch.sort(topk, dim=-1).values.contiguous()
            return {"mode": "per_head", "indices": keep_per_head}

        scores_agg = scores
        if scores_agg.ndim == 3:
            scores_agg = scores_agg.max(dim=1).values
        selected = torch.topk(
            scores_agg,
            k=k,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices[0]
        keep = torch.sort(selected).values.contiguous()
        return {"mode": "shared", "indices": keep}

    def _debug_compare_selector_results(
        *,
        paged_result: dict[str, Any] | None,
        dense_result: dict[str, Any] | None,
        total_tokens: int,
        layer_idx: int,
        round_start: int,
    ) -> None:
        if paged_result is None or dense_result is None:
            if paged_result != dense_result:
                raise RuntimeError(
                    "TRIATTN_PAGED_DENSE_COMPARE_MISMATCH:"
                    f"layer_idx={layer_idx}:total_tokens={total_tokens}:"
                    f"round_start={round_start}:paged={paged_result}:dense={dense_result}"
                )
            return
        if paged_result.get("mode") != dense_result.get("mode"):
            raise RuntimeError(
                "TRIATTN_PAGED_DENSE_COMPARE_MODE_MISMATCH:"
                f"layer_idx={layer_idx}:total_tokens={total_tokens}:"
                f"round_start={round_start}:paged_mode={paged_result.get('mode')}:"
                f"dense_mode={dense_result.get('mode')}"
            )
        p_idx = paged_result.get("indices")
        d_idx = dense_result.get("indices")
        if isinstance(p_idx, list):
            p_idx = torch.as_tensor(p_idx)
        if isinstance(d_idx, list):
            d_idx = torch.as_tensor(d_idx)
        if not isinstance(p_idx, torch.Tensor) or not isinstance(d_idx, torch.Tensor):
            if p_idx != d_idx:
                raise RuntimeError(
                    "TRIATTN_PAGED_DENSE_COMPARE_INDEX_TYPE_MISMATCH:"
                    f"layer_idx={layer_idx}:paged_type={type(p_idx)}:dense_type={type(d_idx)}"
                )
            return
        p_cpu = p_idx.detach().to("cpu", dtype=torch.long).contiguous()
        d_cpu = d_idx.detach().to("cpu", dtype=torch.long).contiguous()
        if p_cpu.shape != d_cpu.shape or not torch.equal(p_cpu, d_cpu):
            # Ties can cause benign differences; fail only when overlap is clearly low.
            if p_cpu.ndim == 1 and d_cpu.ndim == 1:
                p_set = set(int(x) for x in p_cpu.tolist())
                d_set = set(int(x) for x in d_cpu.tolist())
                inter = len(p_set & d_set)
                union = max(1, len(p_set | d_set))
                jaccard = inter / union
                if jaccard >= 0.98:
                    return
                p_head = p_cpu[: min(16, p_cpu.numel())].tolist()
                d_head = d_cpu[: min(16, d_cpu.numel())].tolist()
                raise RuntimeError(
                    "TRIATTN_PAGED_DENSE_COMPARE_INDEX_MISMATCH:"
                    f"layer_idx={layer_idx}:total_tokens={total_tokens}:"
                    f"round_start={round_start}:jaccard={jaccard:.4f}:"
                    f"paged_head={p_head}:dense_head={d_head}"
                )
            if p_cpu.ndim == 2 and d_cpu.ndim == 2 and p_cpu.shape == d_cpu.shape:
                head_jaccards: list[float] = []
                for h in range(p_cpu.shape[0]):
                    p_set = set(int(x) for x in p_cpu[h].tolist())
                    d_set = set(int(x) for x in d_cpu[h].tolist())
                    inter = len(p_set & d_set)
                    union = max(1, len(p_set | d_set))
                    head_jaccards.append(inter / union)
                if head_jaccards and min(head_jaccards) >= 0.98:
                    return
                raise RuntimeError(
                    "TRIATTN_PAGED_DENSE_COMPARE_PERHEAD_MISMATCH:"
                    f"layer_idx={layer_idx}:total_tokens={total_tokens}:"
                    f"round_start={round_start}:min_jaccard={min(head_jaccards):.4f}:"
                    f"mean_jaccard={sum(head_jaccards)/len(head_jaccards):.4f}"
                )
            raise RuntimeError(
                "TRIATTN_PAGED_DENSE_COMPARE_SHAPE_MISMATCH:"
                f"layer_idx={layer_idx}:total_tokens={total_tokens}:"
                f"round_start={round_start}:paged_shape={tuple(p_cpu.shape)}:"
                f"dense_shape={tuple(d_cpu.shape)}"
            )

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
            nonlocal dump_dense_keys_done
            group_agg_mode = os.environ.get(
                "TRIATTN_RUNTIME_DEBUG_GROUP_PERHEAD_AGG_MODE",
                "mean",
            ).strip().lower()
            if group_agg_mode not in {"mean", "max"}:
                group_agg_mode = "mean"
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

            _dump_first_k_sample(
                layer_entries=prepared_layers,
                total_tokens=total_tokens,
            )
            if dump_dense_keys_path and not dump_dense_keys_done:
                try:
                    dump_path = Path(dump_dense_keys_path)
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    preferred_layers_env = os.environ.get(
                        "TRIATTN_DEBUG_DUMP_FIRST_DENSE_KEYS_LAYERS",
                        "",
                    ).strip()
                    if preferred_layers_env.lower() == "all":
                        selected_entries = prepared_layers
                    else:
                        preferred_layers = {0, 1, 16, 32, 48, 63}
                        if preferred_layers_env:
                            try:
                                preferred_layers = {
                                    int(part.strip())
                                    for part in preferred_layers_env.split(",")
                                    if part.strip()
                                }
                            except Exception:
                                preferred_layers = {0, 1, 16, 32, 48, 63}
                        selected_entries = [
                            entry
                            for entry in prepared_layers
                            if int(entry["layer_idx"]) in preferred_layers
                        ]
                        if not selected_entries:
                            selected_entries = prepared_layers[:1]
                    payload = {
                        "round_start": int(round_start),
                        "total_tokens": int(total_tokens),
                        "layers": {},
                    }
                    for entry in selected_entries:
                        dense_keys_full = gather_request_k_dense(
                            kv_cache=entry["kv_cache"],
                            block_ids=entry["block_ids"],
                            block_size=entry["block_size"],
                            total_tokens=total_tokens,
                        )
                        payload["layers"][str(int(entry["layer_idx"]))] = {
                            "num_heads": int(dense_keys_full.shape[1]),
                            "head_dim": int(dense_keys_full.shape[2]),
                            "dtype": str(dense_keys_full.dtype),
                            "keys_dense": dense_keys_full.detach().to(device="cpu"),
                        }
                    torch.save(payload, dump_path)
                    dump_dense_keys_done = True
                except Exception:
                    pass

            prepared_layer_indices = [int(entry["layer_idx"]) for entry in prepared_layers]

            min_block_size = min(entry["block_size"] for entry in prepared_layers)
            chunk_tokens = _score_chunk_tokens(min_block_size, total_tokens)
            norm_stats: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(prepared_layers)
            raw_scores_cache_by_layer: list[list[torch.Tensor] | None] = [None] * len(prepared_layers)
            if config.sparse_normalize_scores:
                eps = 1e-8
                for layer_pos, entry in enumerate(prepared_layers):
                    sum_vec: torch.Tensor | None = None
                    sumsq_vec: torch.Tensor | None = None
                    count = 0
                    layer_raw_scores: list[torch.Tensor] = []
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
                        layer_raw_scores.append(raw_scores)
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
                    raw_scores_cache_by_layer[layer_pos] = layer_raw_scores

            best_scores: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None
            start = 0
            chunk_idx = 0
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
                chunk_agg: torch.Tensor | None = None
                layer_count = 0
                for layer_pos, entry in enumerate(prepared_layers):
                    layer_raw_cache = raw_scores_cache_by_layer[layer_pos]
                    if layer_raw_cache is not None and chunk_idx < len(layer_raw_cache):
                        chunk_scores = layer_raw_cache[chunk_idx].unsqueeze(0)
                    else:
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
                        chunk_scores = _reduce_grouped_head_scores(
                            scores=chunk_scores,
                            runtime_heads=entry["runtime_heads"],
                            group_size=entry["group_size"],
                            aggregate_mode="max",
                        )
                    if chunk_scores.ndim != 3:
                        raise RuntimeError(
                            f"unexpected_score_rank_for_per_head:{chunk_scores.ndim}"
                        )
                    layer_scores = chunk_scores[0]
                    if chunk_agg is None:
                        chunk_agg = layer_scores.clone()
                    else:
                        if group_agg_mode == "max":
                            chunk_agg = torch.maximum(chunk_agg, layer_scores)
                        else:
                            chunk_agg.add_(layer_scores)
                    layer_count += 1

                if chunk_agg is None or layer_count <= 0:
                    return None
                chunk_final = (
                    chunk_agg
                    if group_agg_mode == "max"
                    else chunk_agg.div(float(layer_count))
                )
                cand_k = min(k, int(chunk_final.shape[-1]))
                cand = torch.topk(
                    chunk_final,
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
                chunk_idx += 1

            if best_indices is None:
                return None
            keep_per_head = torch.sort(best_indices, dim=-1).values.contiguous()
            return {
                "mode": "per_head",
                "indices": keep_per_head,
                "semantic": "hf_aligned_global_per_head",
                "group_agg_mode": group_agg_mode,
                "debug_group_layer_indices": prepared_layer_indices,
                "debug_recent_count": _resolve_effective_recent_count(total_tokens),
            }
        else:
            aggregated_scores: torch.Tensor | None = None
            layer_count = 0
            dense_layer_indices: list[int] = []
            sample_positions: list[int] | None = None
            layer_score_samples: dict[str, Any] = {}
            for layer_idx, keys_dense in iter_inputs:
                dense_layer_indices.append(int(layer_idx))
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
                if debug_dump_group_score_samples and int(layer_idx) in {0, 1, 16, 32, 48, 63}:
                    total = int(layer_scores.shape[-1])
                    if sample_positions is None:
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
                    layer_score_samples[str(int(layer_idx))] = {
                        "head0_scores": [
                            float(layer_scores[0, pos].detach().to(device="cpu", dtype=torch.float32).item())
                            for pos in sample_positions
                        ]
                    }
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
            payload = {
                "mode": "per_head",
                "indices": keep_per_head,
                "semantic": "hf_aligned_global_per_head",
                "group_agg_mode": "mean",
                "debug_group_layer_indices": dense_layer_indices,
                "debug_recent_count": _resolve_effective_recent_count(total_tokens),
            }
            if debug_dump_group_score_samples:
                total = int(aggregated_scores.shape[-1])
                if sample_positions is None:
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
                payload["debug_score_sample_positions"] = sample_positions
                payload["debug_layer_score_samples"] = layer_score_samples
                payload["debug_agg_head0_scores"] = [
                    float(aggregated_scores[0, pos].detach().to(device="cpu", dtype=torch.float32).item())
                    for pos in sample_positions
                ]
                payload["debug_effective_model_path"] = (
                    None if effective_model_path is None else str(effective_model_path)
                )
            return payload

    setattr(_select_keep_indices, "_supports_paged", True)
    setattr(_select_keep_indices_for_group_per_head, "_supports_paged_group", True)
    return _select_keep_indices, _select_keep_indices_for_group_per_head, "enabled"
