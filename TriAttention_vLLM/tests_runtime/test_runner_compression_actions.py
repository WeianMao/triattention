from types import SimpleNamespace
import pytest

from triattention_runtime.runner_compression_actions import execute_runner_compression_actions
from triattention_runtime.signals import CompressionSignal


class _FakeStateStore:
    def __init__(self):
        self.calls = []

    def mark_compression_skipped(self, **kwargs):
        self.calls.append(("skip", kwargs))

    def mark_compressed(self, **kwargs):
        self.calls.append(("applied", kwargs))


def _signal() -> CompressionSignal:
    return CompressionSignal(
        req_id="r1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=64,
        step=3,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=8,
    )


def test_execute_runner_compression_actions_applied_event():
    state_store = _FakeStateStore()
    executor = SimpleNamespace(
        execute=lambda **kwargs: SimpleNamespace(
            applied=True,
            reason="kv_compacted",
            cache_len_after=32,
            details={"effective_tokens_before": 64, "budget_total": 32},
        )
    )
    logger = SimpleNamespace(debug=lambda *args, **kwargs: None, exception=lambda *a, **k: None)
    events = execute_runner_compression_actions(
        executor=executor,
        state_store=state_store,
        scheduler_output=SimpleNamespace(),
        signals={"r1": _signal()},
        strict_no_downgrade=True,
        allowed_strict_skip_reasons={"under_budget"},
        logger=logger,
        log_decisions=True,
    )
    assert len(events) == 1
    assert events[0]["status"] == "applied"
    assert events[0]["cache_len_after"] == 32
    assert state_store.calls[0][0] == "applied"


def test_execute_runner_compression_actions_raises_on_unexpected_skip_in_strict():
    state_store = _FakeStateStore()
    executor = SimpleNamespace(
        execute=lambda **kwargs: SimpleNamespace(
            applied=False,
            reason="compaction_failed:g0",
            cache_len_after=40,
            details={},
        )
    )
    logger = SimpleNamespace(debug=lambda *args, **kwargs: None, exception=lambda *a, **k: None)
    with pytest.raises(RuntimeError) as exc:
        execute_runner_compression_actions(
            executor=executor,
            state_store=state_store,
            scheduler_output=SimpleNamespace(),
            signals={"r1": _signal()},
            strict_no_downgrade=True,
            allowed_strict_skip_reasons={"under_budget"},
            logger=logger,
            log_decisions=False,
        )
    assert "unexpected_skip" in str(exc.value)
