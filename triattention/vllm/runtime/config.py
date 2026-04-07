"""Configuration for TriAttention runtime integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


@dataclass
class TriAttentionRuntimeConfig:
    """Runtime config loaded by scheduler and worker.

    Phase 1B provides lifecycle + trigger signaling, with an optional
    experimental KV compaction hook for gather/score/select/scatter.
    """

    kv_budget: int = 2048
    divide_length: int = 128
    protect_prefill: bool = False
    disable_compression: bool = False

    enable_kv_usage_trigger: bool = False
    kv_usage_trigger: float = 0.98
    kv_usage_release: float = 0.90

    enable_experimental_kv_compaction: bool = True
    enable_experimental_block_reclaim: bool = True
    require_triton_scoring: bool = True
    require_physical_reclaim: bool = True
    log_decisions: bool = True
    fail_on_effective_len_regression: bool = True
    effective_len_regression_ratio: float = 0.9
    effective_len_guard_divide_multiples: int = 2
    score_chunk_max_tokens: int = 4096

    # Optional TriAttention-style scoring path (used by runtime hook when enabled).
    sparse_stats_path: Path | None = None
    model_path: Path | None = None
    pruning_mode: str = "per_head"
    sparse_score_aggregation: str = "mean"
    sparse_normalize_scores: bool = True
    window_size: int = 128
    include_prefill_in_budget: bool = True
    per_head_selection_semantics: str = "hf_aligned_global_per_head"
    layer_perhead_aggregation: str = "max"
    per_layer_aggregation: str = "max"
    allow_per_layer_mode: bool = False
    disable_mlr: bool = False
    disable_trig: bool = False
    disable_top_n_high_freq: int = 0

    # Debug compression log: when enabled, writes decoded text around each
    # compression event to a log file for quality diagnosis.
    debug_compression_log: bool = False
    debug_compression_log_path: str = ""
    debug_compression_log_context_tokens: int = 50

    @classmethod
    def from_env(cls, prefix: str = "TRIATTN_RUNTIME_") -> "TriAttentionRuntimeConfig":
        env = os.environ

        def _get_raw(name: str) -> str | None:
            return env.get(prefix + name)

        def maybe_int(name: str, default: int) -> int:
            raw = _get_raw(name)
            return default if raw is None else int(raw)

        def maybe_float(name: str, default: float) -> float:
            raw = _get_raw(name)
            return default if raw is None else float(raw)

        def maybe_bool(name: str, default: bool) -> bool:
            raw = _get_raw(name)
            return default if raw is None else _parse_bool(raw)

        def maybe_str(name: str, default: str | None) -> str | None:
            raw = _get_raw(name)
            if raw is None:
                return default
            raw = raw.strip()
            return raw if raw else default

        sparse_stats_path_raw = maybe_str("SPARSE_STATS_PATH", None)
        model_path_raw = maybe_str("MODEL_PATH", None)

        config = cls(
            kv_budget=maybe_int("KV_BUDGET", cls.kv_budget),
            divide_length=maybe_int("DIVIDE_LENGTH", cls.divide_length),
            protect_prefill=maybe_bool("PROTECT_PREFILL", cls.protect_prefill),
            disable_compression=maybe_bool(
                "DISABLE_COMPRESSION", cls.disable_compression
            ),
            enable_kv_usage_trigger=maybe_bool(
                "ENABLE_KV_USAGE_TRIGGER", cls.enable_kv_usage_trigger
            ),
            kv_usage_trigger=maybe_float("KV_USAGE_TRIGGER", cls.kv_usage_trigger),
            kv_usage_release=maybe_float("KV_USAGE_RELEASE", cls.kv_usage_release),
            enable_experimental_kv_compaction=maybe_bool(
                "ENABLE_EXPERIMENTAL_KV_COMPACTION",
                cls.enable_experimental_kv_compaction,
            ),
            enable_experimental_block_reclaim=maybe_bool(
                "ENABLE_EXPERIMENTAL_BLOCK_RECLAIM",
                cls.enable_experimental_block_reclaim,
            ),
            require_triton_scoring=maybe_bool(
                "REQUIRE_TRITON_SCORING",
                cls.require_triton_scoring,
            ),
            require_physical_reclaim=maybe_bool(
                "REQUIRE_PHYSICAL_RECLAIM",
                cls.require_physical_reclaim,
            ),
            log_decisions=maybe_bool("LOG_DECISIONS", cls.log_decisions),
            fail_on_effective_len_regression=maybe_bool(
                "FAIL_ON_EFFECTIVE_LEN_REGRESSION",
                cls.fail_on_effective_len_regression,
            ),
            effective_len_regression_ratio=maybe_float(
                "EFFECTIVE_LEN_REGRESSION_RATIO",
                cls.effective_len_regression_ratio,
            ),
            effective_len_guard_divide_multiples=maybe_int(
                "EFFECTIVE_LEN_GUARD_DIVIDE_MULTIPLES",
                cls.effective_len_guard_divide_multiples,
            ),
            score_chunk_max_tokens=maybe_int(
                "SCORE_CHUNK_MAX_TOKENS",
                cls.score_chunk_max_tokens,
            ),
            sparse_stats_path=Path(sparse_stats_path_raw) if sparse_stats_path_raw else None,
            model_path=Path(model_path_raw) if model_path_raw else None,
            pruning_mode=maybe_str("PRUNING_MODE", cls.pruning_mode) or cls.pruning_mode,
            sparse_score_aggregation=(
                maybe_str("SPARSE_SCORE_AGGREGATION", cls.sparse_score_aggregation)
                or cls.sparse_score_aggregation
            ),
            sparse_normalize_scores=maybe_bool(
                "SPARSE_NORMALIZE_SCORES", cls.sparse_normalize_scores
            ),
            window_size=maybe_int("WINDOW_SIZE", cls.window_size),
            include_prefill_in_budget=maybe_bool(
                "INCLUDE_PREFILL_IN_BUDGET", cls.include_prefill_in_budget
            ),
            per_head_selection_semantics=(
                maybe_str(
                    "PER_HEAD_SELECTION_SEMANTICS",
                    cls.per_head_selection_semantics,
                )
                or cls.per_head_selection_semantics
            ),
            layer_perhead_aggregation=(
                maybe_str(
                    "LAYER_PERHEAD_AGGREGATION",
                    cls.layer_perhead_aggregation,
                )
                or cls.layer_perhead_aggregation
            ),
            per_layer_aggregation=(
                maybe_str(
                    "PER_LAYER_AGGREGATION",
                    cls.per_layer_aggregation,
                )
                or cls.per_layer_aggregation
            ),
            allow_per_layer_mode=maybe_bool(
                "ALLOW_PER_LAYER_MODE", cls.allow_per_layer_mode
            ),
            disable_mlr=maybe_bool("DISABLE_MLR", cls.disable_mlr),
            disable_trig=maybe_bool("DISABLE_TRIG", cls.disable_trig),
            disable_top_n_high_freq=maybe_int(
                "DISABLE_TOP_N_HIGH_FREQ", cls.disable_top_n_high_freq
            ),
            debug_compression_log=maybe_bool(
                "DEBUG_COMPRESSION_LOG", cls.debug_compression_log
            ),
            debug_compression_log_path=(
                maybe_str("DEBUG_COMPRESSION_LOG_PATH", cls.debug_compression_log_path)
                or cls.debug_compression_log_path
            ),
            debug_compression_log_context_tokens=maybe_int(
                "DEBUG_COMPRESSION_LOG_CONTEXT_TOKENS",
                cls.debug_compression_log_context_tokens,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.kv_budget <= 0:
            raise ValueError(f"kv_budget must be > 0, got {self.kv_budget}")
        if self.divide_length <= 0:
            raise ValueError(
                f"divide_length must be > 0, got {self.divide_length}"
            )
        if not (0.0 < self.kv_usage_trigger <= 1.0):
            raise ValueError(
                "kv_usage_trigger must be in (0, 1], "
                f"got {self.kv_usage_trigger}"
            )
        if not (0.0 <= self.kv_usage_release <= 1.0):
            raise ValueError(
                "kv_usage_release must be in [0, 1], "
                f"got {self.kv_usage_release}"
            )
        if self.kv_usage_release > self.kv_usage_trigger:
            raise ValueError(
                "kv_usage_release should be <= kv_usage_trigger to avoid "
                "hysteresis inversion"
            )
        if self.pruning_mode not in {"per_layer", "per_head", "per_layer_per_head"}:
            raise ValueError(
                "pruning_mode must be one of {'per_layer','per_head','per_layer_per_head'}, "
                f"got {self.pruning_mode!r}"
            )
        if self.pruning_mode == "per_layer" and not self.allow_per_layer_mode:
            raise ValueError(
                "pruning_mode='per_layer' is disabled by default in the runtime to prevent "
                "accidental use. Set allow_per_layer_mode=True "
                "(env TRIATTN_RUNTIME_ALLOW_PER_LAYER_MODE=1) for explicit opt-in."
            )
        if self.sparse_score_aggregation not in {"mean", "max"}:
            raise ValueError(
                "sparse_score_aggregation must be 'mean' or 'max', "
                f"got {self.sparse_score_aggregation!r}"
            )
        if self.per_head_selection_semantics not in {
            "legacy_layer_local",
            "hf_aligned_global_per_head",
        }:
            raise ValueError(
                "per_head_selection_semantics must be one of "
                "{'legacy_layer_local','hf_aligned_global_per_head'}, "
                f"got {self.per_head_selection_semantics!r}"
            )
        if self.layer_perhead_aggregation not in {"max", "mean"}:
            raise ValueError(
                "layer_perhead_aggregation must be 'max' or 'mean', "
                f"got {self.layer_perhead_aggregation!r}"
            )
        if self.per_layer_aggregation not in {"max", "mean", "pure_mean"}:
            raise ValueError(
                "per_layer_aggregation must be one of {'max','mean','pure_mean'}, "
                f"got {self.per_layer_aggregation!r}"
            )
        if self.window_size < 0:
            raise ValueError(f"window_size must be >= 0, got {self.window_size}")
        if self.disable_top_n_high_freq < 0:
            raise ValueError(
                "disable_top_n_high_freq must be >= 0, "
                f"got {self.disable_top_n_high_freq}"
            )
        if not (0.0 < self.effective_len_regression_ratio <= 1.0):
            raise ValueError(
                "effective_len_regression_ratio must be in (0, 1], "
                f"got {self.effective_len_regression_ratio}"
            )
        if self.effective_len_guard_divide_multiples < 1:
            raise ValueError(
                "effective_len_guard_divide_multiples must be >= 1, "
                f"got {self.effective_len_guard_divide_multiples}"
            )
        if self.score_chunk_max_tokens < 1:
            raise ValueError(
                "score_chunk_max_tokens must be >= 1, "
                f"got {self.score_chunk_max_tokens}"
            )
        if self.enable_experimental_kv_compaction and not self.require_triton_scoring:
            raise ValueError(
                "enable_experimental_kv_compaction requires require_triton_scoring=True "
                "to prevent fallback downgrade"
            )
        if (
            self.enable_experimental_kv_compaction
            and self.require_physical_reclaim
            and not self.enable_experimental_block_reclaim
        ):
            raise ValueError(
                "enable_experimental_kv_compaction requires "
                "enable_experimental_block_reclaim=True when "
                "require_physical_reclaim=True"
            )
