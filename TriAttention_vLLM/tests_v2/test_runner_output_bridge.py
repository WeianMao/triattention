from types import SimpleNamespace

from triattention_v2.runner_output_bridge import (
    attach_execute_model_compression_events,
    attach_sample_tokens_compression_events,
    execute_base_model_with_effective_overrides,
)


def test_attach_execute_model_compression_events_clears_pending_when_attach_succeeds():
    output = SimpleNamespace()
    pending = [{"req_id": "r1", "status": "applied"}]
    out, remaining = attach_execute_model_compression_events(
        output=output,
        pending_events=pending,
    )
    assert out is output
    assert getattr(output, "triattention_compression_events") == pending
    assert remaining == []


def test_attach_execute_model_compression_events_keeps_pending_on_attach_failure():
    class _NoSetAttr:
        __slots__ = ()

    output = _NoSetAttr()
    pending = [{"req_id": "r1", "status": "applied"}]
    out, remaining = attach_execute_model_compression_events(
        output=output,
        pending_events=pending,
    )
    assert out is output
    assert remaining == pending


def test_attach_sample_tokens_compression_events_none_output_clears_pending():
    out, remaining = attach_sample_tokens_compression_events(
        output=None,
        pending_events=[{"req_id": "r1"}],
    )
    assert out is None
    assert remaining == []


def test_execute_base_model_with_effective_overrides_fails_if_overrides_not_consumed(monkeypatch):
    import triattention_v2.runner_output_bridge as mod

    class _BaseRunner:
        req_states = object()  # enable consumption assertion path

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            del scheduler_output, intermediate_tensors
            return {"ok": True}

    monkeypatch.setattr(
        mod,
        "prepare_effective_input_overrides",
        lambda **kwargs: SimpleNamespace(
            seq_base_map={1: 100},
            pos_delta_map={1: -10},
            single_seq_base=None,
            single_pos_delta=0,
        ),
    )

    class _Ctx:
        def __init__(self, overrides):
            self.overrides = overrides
        def __enter__(self):
            return self.overrides
        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mod, "active_effective_input_overrides", lambda overrides: _Ctx(overrides))
    monkeypatch.setattr(mod, "assert_effective_overrides_consumed", lambda: (_ for _ in ()).throw(RuntimeError("TRIATTN_EFFECTIVE_OVERRIDES_NOT_CONSUMED")))

    try:
        execute_base_model_with_effective_overrides(
            base_runner=_BaseRunner(),
            state_store=object(),
            scheduler_output=SimpleNamespace(),
            intermediate_tensors=None,
            use_effective_overrides=True,
        )
    except RuntimeError as exc:
        assert "TRIATTN_EFFECTIVE_OVERRIDES_NOT_CONSUMED" in str(exc)
    else:
        raise AssertionError("expected override consumption failure")


def test_execute_base_model_with_effective_overrides_skips_consumption_assert_for_lightweight_runner(monkeypatch):
    import triattention_v2.runner_output_bridge as mod

    class _BaseRunner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            del scheduler_output, intermediate_tensors
            return {"ok": True}

    monkeypatch.setattr(
        mod,
        "prepare_effective_input_overrides",
        lambda **kwargs: SimpleNamespace(
            seq_base_map={1: 100},
            pos_delta_map={1: -10},
            single_seq_base=None,
            single_pos_delta=0,
        ),
    )

    class _Ctx:
        def __init__(self, overrides):
            self.overrides = overrides
        def __enter__(self):
            return self.overrides
        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mod, "active_effective_input_overrides", lambda overrides: _Ctx(overrides))
    monkeypatch.setattr(mod, "assert_effective_overrides_consumed", lambda: (_ for _ in ()).throw(RuntimeError("should_not_be_called")))

    out = execute_base_model_with_effective_overrides(
        base_runner=_BaseRunner(),
        state_store=object(),
        scheduler_output=SimpleNamespace(),
        intermediate_tensors=None,
        use_effective_overrides=True,
    )
    assert out == {"ok": True}
