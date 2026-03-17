from types import SimpleNamespace

from triattention_runtime.runner_state_updates import (
    cleanup_finished_requests,
    consume_runner_signals,
    mark_preemptions,
    mark_resumed,
    register_new_requests,
)
from triattention_runtime.signals import CompressionSignal


class _FakeStateStore:
    def __init__(self, initial_states=None):
        self.calls = []
        self._states = dict(initial_states or {})

    def ensure(self, **kwargs):
        req_id = kwargs["req_id"]
        self.calls.append(("ensure", kwargs))
        state = self._states.get(req_id)
        if state is None:
            state = SimpleNamespace(
                mode="decode",
                compression_count=0,
                current_cache_len=0,
            )
            self._states[req_id] = state
        return state

    def remove(self, req_id):
        self.calls.append(("remove", req_id))
        self._states.pop(req_id, None)

    def mark_preempted(self, req_id):
        self.calls.append(("mark_preempted", req_id))

    def mark_resumed(self, req_id):
        self.calls.append(("mark_resumed", req_id))

    def update_cache_len(self, req_id, cache_len, step=None):
        self.calls.append(("update_cache_len", req_id, cache_len, step))
        state = self._states.get(req_id)
        if state is not None:
            state.current_cache_len = cache_len

    def mark_trigger(self, req_id, reason, step):
        self.calls.append(("mark_trigger", req_id, reason, step))


def test_runner_state_updates_lifecycle_helpers():
    store = _FakeStateStore()
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="r1", prefill_token_ids=[1, 2, 3], prompt_token_ids=None),
            SimpleNamespace(req_id="r2", prefill_token_ids=None, prompt_token_ids=[7]),
        ],
        finished_req_ids=["r2"],
        preempted_req_ids=["r3"],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=["r4"]),
    )

    register_new_requests(state_store=store, scheduler_output=scheduler_output, protect_prefill=True)
    cleanup_finished_requests(state_store=store, scheduler_output=scheduler_output)
    mark_preemptions(state_store=store, scheduler_output=scheduler_output)
    mark_resumed(state_store=store, scheduler_output=scheduler_output)

    assert ("remove", "r2") in store.calls
    assert ("mark_preempted", "r3") in store.calls
    assert ("mark_resumed", "r4") in store.calls
    ensure_calls = [c for c in store.calls if c[0] == "ensure"]
    assert len(ensure_calls) == 2
    assert ensure_calls[0][1]["prefill_len"] == 3
    assert ensure_calls[1][1]["prefill_len"] == 1


def test_mark_resumed_falls_back_to_req_ids():
    store = _FakeStateStore()
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["r10", "r11"]),
    )

    mark_resumed(state_store=store, scheduler_output=scheduler_output)

    assert ("mark_resumed", "r10") in store.calls
    assert ("mark_resumed", "r11") in store.calls


def test_register_new_requests_accepts_newrequestdata_len_field():
    store = _FakeStateStore()
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="r1", prompt_token_ids_len=7817),
        ],
        finished_req_ids=[],
        preempted_req_ids=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=[]),
    )

    register_new_requests(
        state_store=store,
        scheduler_output=scheduler_output,
        protect_prefill=False,
    )

    ensure_calls = [c for c in store.calls if c[0] == "ensure"]
    assert len(ensure_calls) == 1
    assert ensure_calls[0][1]["prefill_len"] == 7817


def test_consume_runner_signals_updates_state_and_returns_step():
    store = _FakeStateStore()
    signal = CompressionSignal(
        req_id="r1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=33,
        step=7,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=8,
    )
    scheduler_output = SimpleNamespace(
        triattention_step=11,
        triattention_signals={"r1": signal},
    )
    logger = SimpleNamespace(debug=lambda *args, **kwargs: None)

    step, signals = consume_runner_signals(
        state_store=store,
        scheduler_output=scheduler_output,
        last_step=10,
        logger=logger,
        log_decisions=True,
    )

    assert step == 11
    assert signals == {"r1": signal}
    assert ("update_cache_len", "r1", 33, 7) in store.calls
    assert ("mark_trigger", "r1", "length_threshold", 7) in store.calls


def test_consume_runner_signals_prefers_local_progress_for_compressed_request():
    store = _FakeStateStore(
        initial_states={
            "r1": SimpleNamespace(
                mode="decode",
                compression_count=2,
                current_cache_len=12000,
            )
        }
    )
    signal = CompressionSignal(
        req_id="r1",
        should_compress=False,
        reason="none",
        estimated_cache_len=18000,
        step=9,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=0,
        scheduled_tokens=1,
    )
    scheduler_output = SimpleNamespace(
        triattention_step=9,
        triattention_signals={"r1": signal},
    )
    logger = SimpleNamespace(debug=lambda *args, **kwargs: None)

    step, _ = consume_runner_signals(
        state_store=store,
        scheduler_output=scheduler_output,
        last_step=8,
        logger=logger,
        log_decisions=True,
    )

    assert step == 9
    # Local recurrence: 12000 + 1, instead of stale high 18000.
    assert ("update_cache_len", "r1", 12001, 9) in store.calls


def test_consume_runner_signals_ignores_stale_low_scheduler_estimate_after_compression():
    store = _FakeStateStore(
        initial_states={
            "r1": SimpleNamespace(
                mode="decode",
                compression_count=2,
                current_cache_len=12000,
            )
        }
    )
    signal = CompressionSignal(
        req_id="r1",
        should_compress=False,
        reason="none",
        estimated_cache_len=11000,
        step=10,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=0,
        scheduled_tokens=2,
    )
    scheduler_output = SimpleNamespace(
        triattention_step=10,
        triattention_signals={"r1": signal},
    )
    logger = SimpleNamespace(debug=lambda *args, **kwargs: None)

    step, _ = consume_runner_signals(
        state_store=store,
        scheduler_output=scheduler_output,
        last_step=9,
        logger=logger,
        log_decisions=True,
    )

    assert step == 10
    # Local recurrence should not regress to stale low 11000.
    assert ("update_cache_len", "r1", 12002, 10) in store.calls
