from triattention_runtime.config import TriAttentionRuntimeConfig
from triattention_runtime.planner import CompressionPlanner


def test_length_trigger():
    cfg = TriAttentionRuntimeConfig(kv_budget=16, divide_length=4)
    planner = CompressionPlanner(cfg)

    signal = planner.build_signal(
        req_id="r1",
        estimated_cache_len=20,
        prefill_len=8,
        step=1,
    )
    assert signal.should_compress is True
    assert signal.reason == "length_threshold"
    assert signal.scheduled_tokens == 1


def test_no_trigger_when_under_threshold():
    cfg = TriAttentionRuntimeConfig(kv_budget=16, divide_length=4)
    planner = CompressionPlanner(cfg)

    signal = planner.build_signal(
        req_id="r1",
        estimated_cache_len=19,
        prefill_len=8,
        step=1,
    )
    assert signal.should_compress is False
    assert signal.reason == "none"
    assert signal.scheduled_tokens == 1


def test_kv_usage_trigger_with_hysteresis():
    cfg = TriAttentionRuntimeConfig(
        enable_kv_usage_trigger=True,
        kv_usage_trigger=0.90,
        kv_usage_release=0.80,
    )
    planner = CompressionPlanner(cfg)

    hot_signal = planner.build_signal(
        req_id="r1",
        estimated_cache_len=10,
        prefill_len=3,
        step=1,
        kv_usage=0.92,
    )
    assert hot_signal.should_compress is True
    assert hot_signal.reason == "kv_usage_threshold"

    hold_signal = planner.build_signal(
        req_id="r1",
        estimated_cache_len=10,
        prefill_len=3,
        step=2,
        kv_usage=0.85,
    )
    assert hold_signal.should_compress is True
    assert hold_signal.reason == "kv_usage_threshold"

    cool_signal = planner.build_signal(
        req_id="r1",
        estimated_cache_len=10,
        prefill_len=3,
        step=3,
        kv_usage=0.79,
    )
    assert cool_signal.should_compress is False
    assert cool_signal.reason == "none"


def test_length_trigger_respects_prefill_outside_budget_mode():
    cfg = TriAttentionRuntimeConfig(
        kv_budget=16,
        divide_length=4,
        protect_prefill=True,
        include_prefill_in_budget=False,
    )
    planner = CompressionPlanner(cfg)

    # Threshold should be kv_budget + divide_length + prefill_len = 28
    under = planner.build_signal(
        req_id="r1",
        estimated_cache_len=27,
        prefill_len=8,
        step=1,
    )
    hit = planner.build_signal(
        req_id="r1",
        estimated_cache_len=28,
        prefill_len=8,
        step=2,
    )
    assert under.should_compress is False
    assert hit.should_compress is True
    assert hit.reason == "length_threshold"


def test_signal_propagates_scheduled_tokens():
    cfg = TriAttentionRuntimeConfig(kv_budget=16, divide_length=4)
    planner = CompressionPlanner(cfg)

    signal = planner.build_signal(
        req_id="r1",
        estimated_cache_len=20,
        prefill_len=8,
        step=1,
        scheduled_tokens=2048,
    )
    assert signal.scheduled_tokens == 2048
