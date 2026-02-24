from __future__ import annotations

from triattention_v2.input_adapter import (
    EffectiveInputOverrides,
    active_effective_input_overrides,
    prepare_effective_input_overrides,
)


def test_effective_input_overrides_activate_and_clear(monkeypatch):
    calls: list[tuple[str, object]] = []

    import triattention_v2.input_adapter as mod

    monkeypatch.setattr(
        mod,
        "activate_effective_sparse_overrides",
        lambda **kwargs: calls.append(("activate_sparse", kwargs)),
    )
    monkeypatch.setattr(
        mod,
        "clear_effective_overrides",
        lambda: calls.append(("clear_all", None)),
    )

    overrides = EffectiveInputOverrides(
        seq_base_map={1: 100},
        pos_delta_map={1: -20},
        single_seq_base=100,
        single_pos_delta=-20,
    )
    overrides.activate()
    EffectiveInputOverrides.clear()

    assert calls[0][0] == "activate_sparse"
    assert calls[0][1]["seq_base_map"] == {1: 100}
    assert calls[0][1]["pos_delta_map"] == {1: -20}
    assert calls[0][1]["single_seq_base"] == 100
    assert calls[0][1]["single_pos_delta"] == -20
    assert calls[1] == ("clear_all", None)


def test_active_effective_input_overrides_context_clears_on_exit(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        EffectiveInputOverrides,
        "activate",
        lambda self: calls.append("activate"),
    )
    monkeypatch.setattr(
        EffectiveInputOverrides,
        "clear",
        staticmethod(lambda: calls.append("clear")),
    )

    overrides = EffectiveInputOverrides(
        seq_base_map=None,
        pos_delta_map=None,
        single_seq_base=None,
        single_pos_delta=0,
    )
    with active_effective_input_overrides(overrides):
        calls.append("body")

    assert calls == ["activate", "body", "clear"]


def test_prepare_effective_input_overrides_populates_expected_req_row_indices(monkeypatch):
    import triattention_v2.input_adapter as mod

    monkeypatch.setattr(
        mod,
        "build_effective_sparse_overrides",
        lambda **kwargs: ({1: 10}, {1: -2}, 10, -2),
    )

    base_runner = type(
        "_BR",
        (),
        {"req_states": type("_RS", (), {"req_id_to_index": {"rA": 5, "rB": 1}})()},
    )()
    scheduler_output = type("_SO", (), {"num_scheduled_tokens": {"rA": 2, "rB": 1}})()

    out = prepare_effective_input_overrides(
        base_runner=base_runner,
        state_store=object(),
        scheduler_output=scheduler_output,
    )
    assert out.expected_req_row_indices == (5, 1)
    assert out.expected_query_lens == (2, 1)


def test_prepare_effective_input_overrides_fails_if_overrides_active_but_req_index_unavailable(
    monkeypatch,
):
    import triattention_v2.input_adapter as mod

    monkeypatch.setattr(
        mod,
        "build_effective_sparse_overrides",
        lambda **kwargs: ({1: 10}, {1: -2}, 10, -2),
    )

    base_runner = type("_BR", (), {"req_states": None})()
    scheduler_output = type("_SO", (), {"num_scheduled_tokens": {"rA": 2}})()

    try:
        prepare_effective_input_overrides(
            base_runner=base_runner,
            state_store=object(),
            scheduler_output=scheduler_output,
        )
    except RuntimeError as exc:
        assert "TRIATTN_EXPECTED_REQ_ROW_INDEX_UNAVAILABLE" in str(exc)
    else:
        raise AssertionError("expected fail-fast when req_id_to_index is unavailable")
