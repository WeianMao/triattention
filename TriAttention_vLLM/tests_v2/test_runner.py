from types import SimpleNamespace

from triattention_v2.config import TriAttentionV2Config
from triattention_v2.runner import TriAttentionModelRunner
from triattention_v2.signals import CompressionSignal


def _new_req(req_id: str, prefill_len: int):
    return SimpleNamespace(
        req_id=req_id,
        prefill_token_ids=[0] * prefill_len,
        prompt_token_ids=None,
    )


def _scheduler_output(signal: CompressionSignal):
    return SimpleNamespace(
        scheduled_new_reqs=[_new_req(signal.req_id, signal.prefill_len)],
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        triattention_step=signal.step,
        triattention_signals={signal.req_id: signal},
    )


def _scheduler_output_many(signals: list[CompressionSignal]):
    return SimpleNamespace(
        scheduled_new_reqs=[_new_req(signal.req_id, signal.prefill_len) for signal in signals],
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        triattention_step=max(signal.step for signal in signals),
        triattention_signals={signal.req_id: signal for signal in signals},
    )


def test_runner_consumes_signal_and_skips_when_hook_missing():
    class BaseRunner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return "base-ok"

        def sample_tokens(self, grammar_output):
            return SimpleNamespace()

    config = TriAttentionV2Config(log_decisions=False)
    runner = TriAttentionModelRunner(base_runner=BaseRunner(), config=config)
    signal = CompressionSignal(
        req_id="r1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=256,
        step=1,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=64,
    )

    output = runner.execute_model(_scheduler_output(signal))
    assert output == "base-ok"

    state = runner.snapshot_states()["r1"]
    assert state.pending_triggers == 0
    assert state.compression_count == 0
    assert state.last_trigger_reason == "skipped:runner_hook_missing"


def test_runner_marks_compressed_when_hook_applies():
    class BaseRunner:
        def triattention_apply_compression(self, req_id, signal, scheduler_output):
            return {"applied": True, "reason": "unit-test", "cache_len_after": 111}

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return {"status": "ok"}

        def sample_tokens(self, grammar_output):
            return SimpleNamespace()

    config = TriAttentionV2Config(log_decisions=False)
    runner = TriAttentionModelRunner(base_runner=BaseRunner(), config=config)
    signal = CompressionSignal(
        req_id="r2",
        should_compress=True,
        reason="kv_usage_threshold",
        estimated_cache_len=300,
        step=2,
        kv_usage=0.99,
        protect_prefill=True,
        prefill_len=32,
    )

    out = runner.execute_model(_scheduler_output(signal))
    assert out == {"status": "ok"}

    state = runner.snapshot_states()["r2"]
    assert state.compression_count == 1
    assert state.pending_triggers == 0
    assert state.current_cache_len == 111
    assert state.last_trigger_reason == "applied"


def test_runner_attaches_events_on_sample_tokens():
    class BaseRunner:
        def triattention_apply_compression(self, req_id, signal, scheduler_output):
            return {"applied": False, "reason": "plan_only", "cache_len_after": 16}

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return None

        def sample_tokens(self, grammar_output):
            return SimpleNamespace(tag="out")

    config = TriAttentionV2Config(log_decisions=False)
    runner = TriAttentionModelRunner(base_runner=BaseRunner(), config=config)
    signal = CompressionSignal(
        req_id="r3",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=40,
        step=3,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=8,
    )

    runner.execute_model(_scheduler_output(signal))
    out = runner.sample_tokens(grammar_output=None)
    events = getattr(out, "triattention_compression_events")
    assert len(events) == 1
    assert events[0]["req_id"] == "r3"
    assert events[0]["status"] == "skipped"
    assert events[0]["reason"] == "plan_only"


def test_runner_batch_signals_keep_request_isolation():
    class BaseRunner:
        def triattention_apply_compression(self, req_id, signal, scheduler_output):
            if req_id == "rA":
                return {"applied": True, "reason": "hookA", "cache_len_after": 18}
            if req_id == "rB":
                return {"applied": False, "reason": "plan_only", "cache_len_after": 20}
            raise AssertionError(f"unexpected req_id={req_id}")

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return {"status": "ok"}

        def sample_tokens(self, grammar_output):
            return SimpleNamespace(tag="batch")

    config = TriAttentionV2Config(log_decisions=False)
    runner = TriAttentionModelRunner(base_runner=BaseRunner(), config=config)
    signal_a = CompressionSignal(
        req_id="rA",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=30,
        step=10,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=6,
    )
    signal_b = CompressionSignal(
        req_id="rB",
        should_compress=True,
        reason="kv_usage_threshold",
        estimated_cache_len=40,
        step=10,
        kv_usage=0.99,
        protect_prefill=True,
        prefill_len=4,
    )

    out = runner.execute_model(_scheduler_output_many([signal_a, signal_b]))
    assert out == {"status": "ok"}
    sample_out = runner.sample_tokens(grammar_output=None)
    events = getattr(sample_out, "triattention_compression_events")
    assert len(events) == 2

    states = runner.snapshot_states()
    state_a = states["rA"]
    state_b = states["rB"]

    assert state_a.compression_count == 1
    assert state_a.current_cache_len == 18
    assert state_a.last_trigger_reason == "applied"

    assert state_b.compression_count == 0
    assert state_b.current_cache_len == 40
    assert state_b.last_trigger_reason == "skipped:plan_only"


def test_runner_attaches_events_on_execute_model_output():
    class BaseRunner:
        def triattention_apply_compression(self, req_id, signal, scheduler_output):
            return {"applied": True, "reason": "hook", "cache_len_after": 21}

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return SimpleNamespace(tag="model_output")

        def sample_tokens(self, grammar_output):
            return SimpleNamespace(tag="sampler_output")

    config = TriAttentionV2Config(log_decisions=False)
    runner = TriAttentionModelRunner(base_runner=BaseRunner(), config=config)
    signal = CompressionSignal(
        req_id="rX",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=30,
        step=11,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=6,
    )

    model_out = runner.execute_model(_scheduler_output(signal))
    events = getattr(model_out, "triattention_compression_events")
    assert len(events) == 1
    assert events[0]["req_id"] == "rX"
    assert events[0]["status"] == "applied"
    assert events[0]["cache_len_after"] == 21
    assert isinstance(events[0]["details"], dict)

    # execute_model should clear pending events after attaching to output.
    sample_out = runner.sample_tokens(grammar_output=None)
    assert getattr(sample_out, "triattention_compression_events") == []


def test_runner_raises_on_triton_required_failure():
    class BaseRunner:
        def triattention_apply_compression(self, req_id, signal, scheduler_output):
            raise RuntimeError("TRIATTN_FATAL_TRITON_SCORING_REQUIRED:unit")

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return {"status": "ok"}

        def sample_tokens(self, grammar_output):
            return SimpleNamespace(tag="sampler_output")

    config = TriAttentionV2Config(log_decisions=False)
    runner = TriAttentionModelRunner(base_runner=BaseRunner(), config=config)
    signal = CompressionSignal(
        req_id="rF",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=30,
        step=12,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=6,
    )

    try:
        runner.execute_model(_scheduler_output(signal))
        assert False, "expected fatal Triton error to propagate"
    except RuntimeError as exc:
        assert "TRIATTN_FATAL_TRITON_SCORING_REQUIRED" in str(exc)


def test_runner_strict_mode_raises_when_hook_missing():
    class BaseRunner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return "base-ok"

        def sample_tokens(self, grammar_output):
            return SimpleNamespace()

    config = TriAttentionV2Config(
        log_decisions=False,
        enable_experimental_kv_compaction=True,
    )
    runner = TriAttentionModelRunner(base_runner=BaseRunner(), config=config)
    signal = CompressionSignal(
        req_id="rS",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=256,
        step=21,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=64,
    )

    try:
        runner.execute_model(_scheduler_output(signal))
    except RuntimeError as exc:
        assert "TRIATTN_FATAL_TRITON_SCORING_REQUIRED:unexpected_skip" in str(exc)
        assert "runner_hook_missing" in str(exc)
    else:
        raise AssertionError("expected strict mode to fail on missing compression hook")


def test_runner_strict_mode_raises_on_executor_exception():
    class BaseRunner:
        def triattention_apply_compression(self, req_id, signal, scheduler_output):
            raise ValueError("boom")

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return {"status": "ok"}

        def sample_tokens(self, grammar_output):
            return SimpleNamespace(tag="sampler_output")

    config = TriAttentionV2Config(
        log_decisions=False,
        enable_experimental_kv_compaction=True,
    )
    runner = TriAttentionModelRunner(base_runner=BaseRunner(), config=config)
    signal = CompressionSignal(
        req_id="rE",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=300,
        step=22,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=16,
    )

    try:
        runner.execute_model(_scheduler_output(signal))
    except RuntimeError as exc:
        assert "TRIATTN_FATAL_TRITON_SCORING_REQUIRED:executor_exception" in str(exc)
        assert "type=ValueError" in str(exc)
    else:
        raise AssertionError("expected strict mode to fail on executor exception")


def test_runner_effective_overrides_only_when_scheduled_batch_contains_compressed_req():
    class BaseRunner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            del scheduler_output, intermediate_tensors
            return {"status": "ok"}

        def sample_tokens(self, grammar_output):
            del grammar_output
            return SimpleNamespace(tag="sampler_output")

    runner = TriAttentionModelRunner(
        base_runner=BaseRunner(),
        config=TriAttentionV2Config(log_decisions=False),
    )
    runner.state_store.ensure(req_id="rX", prefill_len=0, protect_prefill=True)
    runner.state_store.mark_compressed(req_id="rX", step=1, cache_len=16)

    out_other = SimpleNamespace(num_scheduled_tokens={"r1": 1})
    out_same = SimpleNamespace(num_scheduled_tokens={"rX": 1})

    assert runner._needs_effective_input_overrides(out_other) is False
    assert runner._needs_effective_input_overrides(out_same) is True
