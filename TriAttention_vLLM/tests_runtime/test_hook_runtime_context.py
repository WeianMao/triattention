from __future__ import annotations

from types import SimpleNamespace

from triattention_runtime.config import TriAttentionRuntimeConfig
from triattention_runtime.hook_runtime_context import build_hook_runtime_context
from triattention_runtime.signals import CompressionSignal


def _signal(**kwargs) -> CompressionSignal:
    base = dict(
        req_id="r1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=21,
        step=1,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=0,
    )
    base.update(kwargs)
    return CompressionSignal(**base)


def test_build_hook_runtime_context_uses_pre_step_effective_len_and_recent_unabsorbed():
    req_state = SimpleNamespace(num_computed_tokens=20, block_ids=([0, 1],))
    runtime_state = SimpleNamespace(last_absorbed_cache_len=12)
    base_runner = SimpleNamespace()
    cfg = TriAttentionRuntimeConfig(
        kv_budget=8,
        divide_length=4,
        enable_experimental_kv_compaction=True,
    )

    ctx = build_hook_runtime_context(
        base_runner=base_runner,
        config=cfg,
        req_id="r1",
        req_state=req_state,
        req_runtime_state=runtime_state,
        signal=_signal(estimated_cache_len=21),
        scheduler_output=SimpleNamespace(num_scheduled_tokens={"r1": 1}),
        compressed_once=set(),
        original_block_ids_by_group=req_state.block_ids,
        block_size_hint=16,
    )
    assert ctx.num_computed_tokens == 20
    assert ctx.estimated_effective_tokens == 21
    assert ctx.effective_tokens == 20
    assert ctx.recent_unabsorbed_tokens == 8
    assert getattr(base_runner, "_triattention_active_recent_unabsorbed_tokens") == 8


def test_build_hook_runtime_context_sets_defer_recompress_after_first_compression():
    req_state = SimpleNamespace(num_computed_tokens=32, block_ids=([0, 1],))
    base_runner = SimpleNamespace()
    cfg = TriAttentionRuntimeConfig(
        kv_budget=16,
        divide_length=8,
        enable_experimental_kv_compaction=True,
    )
    # local threshold = 16 + 8 = 24; estimated 20 < 24 => defer after compressed_once
    ctx = build_hook_runtime_context(
        base_runner=base_runner,
        config=cfg,
        req_id="r1",
        req_state=req_state,
        req_runtime_state=None,
        signal=_signal(estimated_cache_len=20),
        scheduler_output=SimpleNamespace(num_scheduled_tokens={"r1": 1}),
        compressed_once={"r1"},
        original_block_ids_by_group=req_state.block_ids,
        block_size_hint=16,
    )
    assert ctx.should_defer_recompress is True


def test_build_hook_runtime_context_raises_on_effective_len_regression():
    req_state = SimpleNamespace(num_computed_tokens=118, block_ids=([0, 1, 2, 3, 4, 5, 6, 7],))
    base_runner = SimpleNamespace()
    cfg = TriAttentionRuntimeConfig(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        enable_experimental_block_reclaim=True,
        fail_on_effective_len_regression=True,
        effective_len_regression_ratio=0.9,
        effective_len_guard_divide_multiples=2,
    )
    try:
        build_hook_runtime_context(
            base_runner=base_runner,
            config=cfg,
            req_id="r1",
            req_state=req_state,
            req_runtime_state=None,
            signal=_signal(estimated_cache_len=118),
            scheduler_output=SimpleNamespace(),
            compressed_once={"r1"},
            original_block_ids_by_group=req_state.block_ids,
            block_size_hint=16,
        )
    except RuntimeError as exc:
        assert "effective_len_regressed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for effective_len_regressed")
