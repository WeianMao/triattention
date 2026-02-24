from triattention_v2.state import RequestStateStore


def test_state_lifecycle():
    store = RequestStateStore()
    state = store.ensure(req_id="r1", prefill_len=100, protect_prefill=True)
    assert state.req_id == "r1"
    assert state.prefill_len == 100
    assert state.mode == "protect_prefill"

    store.update_cache_len("r1", 128)
    assert store.get("r1").current_cache_len == 128
    assert store.get("r1").current_cache_len_semantics == "estimated_with_scheduled"

    store.mark_trigger("r1", reason="length_threshold", step=10)
    state = store.get("r1")
    assert state.pending_triggers == 1
    assert state.last_trigger_reason == "length_threshold"
    assert state.last_compression_step == 10

    store.mark_compressed("r1", step=11, cache_len=64)
    state = store.get("r1")
    assert state.compression_count == 1
    assert state.pending_triggers == 0
    assert state.current_cache_len == 64
    assert state.current_cache_len_semantics == "effective_pre_step"
    assert state.current_cache_len_step == 11
    assert state.last_compression_step == 11
    assert state.last_trigger_reason == "applied"

    store.mark_preempted("r1")
    assert store.get("r1").is_preempted is True
    store.mark_resumed("r1")
    assert store.get("r1").is_preempted is False

    store.remove("r1")
    assert store.get("r1") is None


def test_ensure_updates_prefill_and_mode():
    store = RequestStateStore()
    store.ensure(req_id="r2", prefill_len=50, protect_prefill=True)
    state = store.ensure(req_id="r2", prefill_len=30, protect_prefill=False)
    assert state.prefill_len == 50
    assert state.mode == "trim_prefill"


def test_mark_compression_skipped_consumes_pending():
    store = RequestStateStore()
    store.ensure(req_id="r3", prefill_len=10, protect_prefill=True)
    store.mark_trigger("r3", reason="length_threshold", step=3)
    assert store.get("r3").pending_triggers == 1

    store.mark_compression_skipped("r3", reason="runner_hook_missing", step=4)
    state = store.get("r3")
    assert state.pending_triggers == 0
    assert state.last_compression_step == 4
    assert state.last_trigger_reason == "skipped:runner_hook_missing"
