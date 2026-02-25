"""Runtime length/guard context preparation for TriAttention V2 hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .request_key_compat import get_scheduled_token_items
from .signals import CompressionSignal


def effective_budget_for_signal(
    config: TriAttentionRuntimeConfig,
    signal: CompressionSignal,
    total_tokens: int,
) -> int:
    budget = config.kv_budget
    if signal.protect_prefill and not config.include_prefill_in_budget:
        budget += max(signal.prefill_len, 0)
    return min(total_tokens, budget)


def effective_len_guard_upper(
    config: TriAttentionRuntimeConfig,
    signal: CompressionSignal,
) -> int:
    budget = config.kv_budget
    if signal.protect_prefill and not config.include_prefill_in_budget:
        budget += max(signal.prefill_len, 0)
    return budget + max(1, config.effective_len_guard_divide_multiples) * max(
        1,
        config.divide_length,
    )


def scheduled_tokens_for_req(scheduler_output: Any, req_id: str) -> int:
    try:
        cached = getattr(scheduler_output, "_triattention_cached_scheduled_tokens_by_req_id", None)
    except Exception:
        cached = None
    if isinstance(cached, dict):
        value = cached.get(req_id)
        if isinstance(value, int):
            return max(1, value)
    by_req_id: dict[str, int] = {}
    for _raw_key, key_req_id, scheduled_tokens in get_scheduled_token_items(scheduler_output):
        by_req_id[key_req_id] = int(scheduled_tokens)
    try:
        setattr(scheduler_output, "_triattention_cached_scheduled_tokens_by_req_id", by_req_id)
    except Exception:
        pass
    if req_id in by_req_id:
        return max(1, int(by_req_id[req_id]))
    return 1


def min_block_capacity_tokens(
    block_ids_by_group: Any,
    block_size: int,
) -> int | None:
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


@dataclass(frozen=True)
class HookRuntimeContext:
    scheduled_tokens: int
    num_computed_tokens: int
    estimated_effective_tokens: int
    effective_tokens: int
    budget_total: int
    recent_unabsorbed_tokens: int | None
    should_defer_recompress: bool


def build_hook_runtime_context(
    *,
    base_runner: Any,
    config: TriAttentionRuntimeConfig,
    req_id: str,
    req_state: Any,
    req_runtime_state: Any,
    signal: CompressionSignal,
    scheduler_output: Any,
    compressed_once: set[str],
    original_block_ids_by_group: Any,
    block_size_hint: int,
) -> HookRuntimeContext:
    block_capacity_hint = min_block_capacity_tokens(
        block_ids_by_group=original_block_ids_by_group,
        block_size=block_size_hint,
    )

    scheduled_tokens = scheduled_tokens_for_req(
        scheduler_output=scheduler_output,
        req_id=req_id,
    )
    num_computed_tokens = int(getattr(req_state, "num_computed_tokens", 0))
    estimated_effective_tokens = int(
        getattr(signal, "estimated_cache_len", num_computed_tokens)
    )
    if estimated_effective_tokens < 0:
        estimated_effective_tokens = 0

    effective_tokens = max(0, estimated_effective_tokens - max(0, scheduled_tokens))
    if effective_tokens > num_computed_tokens:
        effective_tokens = num_computed_tokens

    if isinstance(block_capacity_hint, int):
        physical_upper = block_capacity_hint + block_size_hint
        if effective_tokens > physical_upper:
            effective_tokens = physical_upper

    recent_unabsorbed_tokens: int | None = None
    if req_runtime_state is not None:
        baseline = int(getattr(req_runtime_state, "last_absorbed_cache_len", 0))
        recent_unabsorbed_tokens = max(0, effective_tokens - baseline)
        setattr(
            base_runner,
            "_triattention_active_recent_unabsorbed_tokens",
            recent_unabsorbed_tokens,
        )
    else:
        setattr(base_runner, "_triattention_active_recent_unabsorbed_tokens", None)

    if (
        config.fail_on_effective_len_regression
        and config.enable_experimental_block_reclaim
        and req_id in compressed_once
    ):
        guard_upper = effective_len_guard_upper(config, signal)
        estimated_slack = max(1, int(getattr(signal, "estimated_cache_len", 0)) - num_computed_tokens)
        regression_slack = block_size_hint + estimated_slack + max(1, scheduled_tokens)
        if (
            effective_tokens > (guard_upper + regression_slack)
            and num_computed_tokens > (guard_upper + regression_slack)
            and effective_tokens >= int(config.effective_len_regression_ratio * num_computed_tokens)
        ):
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:effective_len_regressed:"
                f"req={req_id}:effective_tokens={effective_tokens}:"
                f"num_computed_tokens={num_computed_tokens}:guard_upper={guard_upper}"
            )

    budget_total = effective_budget_for_signal(config, signal, effective_tokens)
    local_length_threshold = budget_total + max(1, config.divide_length)
    length_gate_hit = estimated_effective_tokens >= local_length_threshold
    kv_override = str(getattr(signal, "reason", "")) == "kv_usage_threshold"
    should_defer_recompress = (
        config.enable_experimental_kv_compaction
        and req_id in compressed_once
        and not kv_override
        and not length_gate_hit
    )

    return HookRuntimeContext(
        scheduled_tokens=int(scheduled_tokens),
        num_computed_tokens=int(num_computed_tokens),
        estimated_effective_tokens=int(estimated_effective_tokens),
        effective_tokens=int(effective_tokens),
        budget_total=int(budget_total),
        recent_unabsorbed_tokens=(
            int(recent_unabsorbed_tokens)
            if isinstance(recent_unabsorbed_tokens, int)
            else None
        ),
        should_defer_recompress=bool(should_defer_recompress),
    )
