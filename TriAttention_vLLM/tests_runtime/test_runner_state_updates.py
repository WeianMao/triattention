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
    def __init__(self):
        self.calls = []

    def ensure(self, **kwargs):
        self.calls.append(("ensure", kwargs))
        return SimpleNamespace(mode="decode")

    def remove(self, req_id):
        self.calls.append(("remove", req_id))

    def mark_preempted(self, req_id):
        self.calls.append(("mark_preempted", req_id))

    def mark_resumed(self, req_id):
        self.calls.append(("mark_resumed", req_id))

    def update_cache_len(self, req_id, cache_len, step=None):
        self.calls.append(("update_cache_len", req_id, cache_len, step))

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
