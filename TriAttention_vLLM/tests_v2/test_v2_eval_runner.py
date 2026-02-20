from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

from evaluation.runner.vllm_triattention_v2_runner import (
    _apply_v2_env,
    compute_local_runs,
    setup_vllm_engine,
)


@contextmanager
def _patched_environ():
    old = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


@contextmanager
def _fake_vllm_module():
    captured: dict[str, object] = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = ModuleType("vllm")
    fake.LLM = _FakeLLM
    old = sys.modules.get("vllm")
    sys.modules["vllm"] = fake
    try:
        yield captured
    finally:
        if old is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = old


def _args(disable_compression: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model_path="/tmp/model",
        load_dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        seed=123,
        max_length=1024,
        disable_compression=disable_compression,
        kv_budget=2048,
        divide_length=128,
        protect_prefill=True,
        enable_kv_usage_trigger=False,
        kv_usage_trigger=0.98,
        kv_usage_release=0.90,
        window_size=128,
        sparse_stats_path=None,
        sparse_score_aggregation="mean",
        pruning_mode="per_layer",
        sparse_normalize_scores=True,
        include_prefill_in_budget=True,
        per_head_selection_semantics="legacy_layer_local",
        disable_mlr=False,
        disable_trig=False,
        disable_top_n_high_freq=0,
        enable_experimental_kv_compaction=True,
        enable_experimental_block_reclaim=True,
        require_triton_scoring=True,
        require_physical_reclaim=True,
        fail_on_effective_len_regression=True,
        effective_len_regression_ratio=0.9,
        effective_len_guard_divide_multiples=2,
        log_decisions=True,
        enforce_eager=True,
    )


def test_apply_v2_env():
    args = _args(disable_compression=False)
    with _patched_environ():
        _apply_v2_env(args)
        assert os.environ["TRIATTN_V2_KV_BUDGET"] == "2048"
        assert os.environ["TRIATTN_V2_DIVIDE_LENGTH"] == "128"
        assert os.environ["TRIATTN_V2_PROTECT_PREFILL"] == "true"
        assert os.environ["TRIATTN_V2_ENABLE_EXPERIMENTAL_KV_COMPACTION"] == "true"
        assert os.environ["TRIATTN_V2_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM"] == "true"
        assert os.environ["TRIATTN_V2_REQUIRE_TRITON_SCORING"] == "true"
        assert os.environ["TRIATTN_V2_REQUIRE_PHYSICAL_RECLAIM"] == "true"
        assert os.environ["TRIATTN_V2_FAIL_ON_EFFECTIVE_LEN_REGRESSION"] == "true"
        assert os.environ["TRIATTN_V2_EFFECTIVE_LEN_REGRESSION_RATIO"] == "0.9"
        assert os.environ["TRIATTN_V2_EFFECTIVE_LEN_GUARD_DIVIDE_MULTIPLES"] == "2"
        assert os.environ["TRIATTN_V2_PRUNING_MODE"] == "per_layer"
        assert os.environ["TRIATTN_V2_PER_HEAD_SELECTION_SEMANTICS"] == "legacy_layer_local"


def test_setup_vllm_engine_sets_worker_scheduler_for_v2():
    args = _args(disable_compression=False)
    with _patched_environ(), _fake_vllm_module() as captured:
        setup_vllm_engine(args)
        assert captured["worker_cls"] == "triattention_v2.worker.TriAttentionWorker"
        assert captured["scheduler_cls"] == "triattention_v2.scheduler.TriAttentionScheduler"
        assert captured["enforce_eager"] is True
        assert os.environ["TRIATTN_V2_KV_BUDGET"] == "2048"


def test_setup_vllm_engine_without_compression():
    args = _args(disable_compression=True)
    with _patched_environ(), _fake_vllm_module() as captured:
        setup_vllm_engine(args)
        assert "worker_cls" not in captured
        assert "scheduler_cls" not in captured
        assert captured["enforce_eager"] is True
        assert "TRIATTN_V2_KV_BUDGET" not in os.environ


def test_setup_vllm_engine_respects_enforce_eager_flag():
    args = _args(disable_compression=True)
    args.enforce_eager = False
    with _patched_environ(), _fake_vllm_module() as captured:
        setup_vllm_engine(args)
        assert captured["enforce_eager"] is False


def test_setup_vllm_engine_strict_requires_compaction_enabled():
    args = _args(disable_compression=False)
    args.enable_experimental_kv_compaction = False
    with _patched_environ(), _fake_vllm_module():
        try:
            setup_vllm_engine(args)
        except RuntimeError as exc:
            assert "enable_experimental_kv_compaction=true" in str(exc)
        else:
            raise AssertionError("expected strict mode RuntimeError when compaction is disabled")


def test_setup_vllm_engine_strict_requires_block_reclaim_when_configured():
    args = _args(disable_compression=False)
    args.require_physical_reclaim = True
    args.enable_experimental_block_reclaim = False
    with _patched_environ(), _fake_vllm_module():
        try:
            setup_vllm_engine(args)
        except RuntimeError as exc:
            assert "enable_experimental_block_reclaim=true" in str(exc)
        else:
            raise AssertionError("expected strict mode RuntimeError when block reclaim is disabled")


def test_compute_local_runs_even_split():
    assert compute_local_runs(8, 8, 0) == (0, 1)
    assert compute_local_runs(8, 8, 7) == (7, 1)


def test_compute_local_runs_uneven_split():
    # 8 runs across 3 shards -> [3, 3, 2]
    assert compute_local_runs(8, 3, 0) == (0, 3)
    assert compute_local_runs(8, 3, 1) == (3, 3)
    assert compute_local_runs(8, 3, 2) == (6, 2)
