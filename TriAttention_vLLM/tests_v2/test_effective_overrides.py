from __future__ import annotations

from types import SimpleNamespace

from triattention_v2.effective_overrides import (
    build_effective_positions_override_tensor,
    build_effective_seq_len_override_tensor,
    build_effective_sparse_overrides,
)
from triattention_v2.request_key_compat import get_scheduled_token_items


def test_build_effective_sparse_overrides_uses_pre_step_len_unless_compression_applied():
    req_states = SimpleNamespace(req_id_to_index={"r1": 3})
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={"r1": SimpleNamespace(num_computed_tokens=120)},
    )
    state_store = SimpleNamespace(
        get=lambda req_id: SimpleNamespace(
            current_cache_len=100,
            current_cache_len_semantics="estimated_with_scheduled",
            current_cache_len_step=6,
        )
        if req_id == "r1"
        else None
    )
    scheduler_output = SimpleNamespace(triattention_step=6, num_scheduled_tokens={"r1": 4})

    seq_map, pos_delta_map, single_seq_base, single_pos_delta = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert seq_map == {3: 96}
    assert pos_delta_map == {3: -24}
    assert single_seq_base == 96
    assert single_pos_delta == -24

    state_store2 = SimpleNamespace(
        get=lambda req_id: SimpleNamespace(
            current_cache_len=100,
            current_cache_len_semantics="effective_pre_step",
            current_cache_len_step=6,
        )
        if req_id == "r1"
        else None
    )
    seq_map2, pos_delta_map2, single_seq_base2, single_pos_delta2 = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store2,
        scheduler_output=scheduler_output,
        compression_events=[{"status": "applied", "req_id": "r1"}],
    )
    assert seq_map2 == {3: 100}
    assert pos_delta_map2 == {3: -20}
    assert single_seq_base2 == 100
    assert single_pos_delta2 == -20


def test_build_effective_override_tensors_use_effective_pre_step_bases():
    gpu = __import__("torch").tensor([10, 20, 30, 40], dtype=__import__("torch").long)
    req_states = SimpleNamespace(
        req_id_to_index={"r1": 1},
        num_computed_tokens=SimpleNamespace(gpu=gpu),
    )
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={"r1": SimpleNamespace(num_computed_tokens=120)},
    )
    state_store = SimpleNamespace(
        get=lambda req_id: SimpleNamespace(current_cache_len=100) if req_id == "r1" else None
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r1": 4})

    seq_override = build_effective_seq_len_override_tensor(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert seq_override is not None
    assert int(seq_override[1].item()) == 96
    # Non-scheduled rows remain copied from original tensor.
    assert int(seq_override[0].item()) == 10

    pos_override = build_effective_positions_override_tensor(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert pos_override is not None
    assert pos_override.tolist() == [96, 97, 98, 99]


def test_build_effective_sparse_overrides_supports_object_num_scheduled_token_keys():
    class _ReqKey:
        def __init__(self, req_id: str):
            self.req_id = req_id

    req_states = SimpleNamespace(req_id_to_index={"r1": 2})
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={"r1": SimpleNamespace(num_computed_tokens=50)},
    )
    state_store = SimpleNamespace(
        get=lambda req_id: SimpleNamespace(current_cache_len=40) if req_id == "r1" else None
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={_ReqKey("r1"): 3})

    seq_map, pos_delta_map, single_seq_base, single_pos_delta = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert seq_map == {2: 37}
    assert pos_delta_map == {2: -13}
    assert single_seq_base == 37
    assert single_pos_delta == -13


def test_build_effective_positions_override_tensor_preserves_mapping_iteration_order():
    req_states = SimpleNamespace(
        req_id_to_index={"r_big": 0, "r_small": 1},
        num_computed_tokens=SimpleNamespace(gpu=__import__("torch").tensor([0, 0], dtype=__import__("torch").long)),
    )
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={
            "r_big": SimpleNamespace(num_computed_tokens=100),
            "r_small": SimpleNamespace(num_computed_tokens=200),
        },
    )
    state_store = SimpleNamespace(
        get=lambda req_id: (
            SimpleNamespace(current_cache_len=90)
            if req_id == "r_big"
            else SimpleNamespace(current_cache_len=180)
        )
    )
    # Deliberately put larger qlen first; old code sorted by qlen and would reorder.
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r_big": 3, "r_small": 1})

    pos_override = build_effective_positions_override_tensor(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert pos_override is not None
    # r_big first => [87,88,89], then r_small => [179]
    assert pos_override.tolist() == [87, 88, 89, 179]


def test_get_scheduled_token_items_caches_per_scheduler_output_instance():
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r1": 2, "r2": 1})
    items1 = get_scheduled_token_items(scheduler_output)
    items2 = get_scheduled_token_items(scheduler_output)
    assert items1 == [( "r1", "r1", 2), ("r2", "r2", 1)]
    assert items2 is items1


def test_build_effective_sparse_overrides_omits_zero_delta_rows():
    req_states = SimpleNamespace(req_id_to_index={"r1": 0, "r2": 1})
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={
            "r1": SimpleNamespace(num_computed_tokens=10),
            "r2": SimpleNamespace(num_computed_tokens=20),
        },
    )
    states = {
        # r1 effective_before_step = 11 - 1 = 10 => delta == 0
        "r1": SimpleNamespace(current_cache_len=11),
        # r2 effective_before_step = 18 - 1 = 17 => delta == -3
        "r2": SimpleNamespace(current_cache_len=18),
    }
    state_store = SimpleNamespace(
        has_active_compressed_requests=lambda: True,
        get=lambda req_id: states.get(req_id),
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r1": 1, "r2": 1})

    seq_map, pos_delta_map, single_seq_base, single_pos_delta = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert seq_map == {1: 17}
    assert pos_delta_map == {1: -3}
    assert single_seq_base == 17
    assert single_pos_delta == -3


def test_build_effective_sparse_overrides_skips_never_compressed_requests():
    req_states = SimpleNamespace(req_id_to_index={"r1": 0})
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={"r1": SimpleNamespace(num_computed_tokens=123)},
    )
    state_store = SimpleNamespace(
        has_active_compressed_requests=lambda: True,  # some other req may be compressed
        get=lambda req_id: SimpleNamespace(current_cache_len=999, compression_count=0),
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r1": 1})

    seq_map, pos_delta_map, single_seq_base, single_pos_delta = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert seq_map is None
    assert pos_delta_map is None
    assert single_seq_base is None
    assert single_pos_delta == 0


def test_build_effective_sparse_overrides_caches_result_on_scheduler_output():
    req_states = SimpleNamespace(req_id_to_index={"r1": 3})
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={"r1": SimpleNamespace(num_computed_tokens=120)},
    )
    calls = {"get": 0}

    def _get(req_id: str):
        calls["get"] += 1
        return SimpleNamespace(current_cache_len=100)

    state_store = SimpleNamespace(
        has_active_compressed_requests=lambda: True,
        get=_get,
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r1": 4})

    out1 = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    out2 = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert out2 is out1
    assert calls["get"] == 1


def test_build_effective_sparse_overrides_prefers_request_state_semantics_marker():
    req_states = SimpleNamespace(req_id_to_index={"r1": 0})
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={"r1": SimpleNamespace(num_computed_tokens=120)},
    )
    state_store = SimpleNamespace(
        has_active_compressed_requests=lambda: True,
        get=lambda req_id: SimpleNamespace(
            current_cache_len=100,
            current_cache_len_semantics="effective_pre_step",
            current_cache_len_step=9,
            compression_count=1,
        ),
    )
    scheduler_output = SimpleNamespace(
        triattention_step=9,
        num_scheduled_tokens={"r1": 4},
    )

    seq_map, pos_delta_map, single_seq_base, single_pos_delta = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert seq_map == {0: 100}
    assert pos_delta_map == {0: -20}
    assert single_seq_base == 100
    assert single_pos_delta == -20


def test_build_effective_sparse_overrides_ignores_stale_effective_marker_from_prior_step():
    req_states = SimpleNamespace(req_id_to_index={"r1": 0})
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={"r1": SimpleNamespace(num_computed_tokens=120)},
    )
    state_store = SimpleNamespace(
        has_active_compressed_requests=lambda: True,
        get=lambda req_id: SimpleNamespace(
            current_cache_len=100,
            current_cache_len_semantics="effective_pre_step",
            current_cache_len_step=8,  # stale marker; current scheduler step is 9
            compression_count=1,
        ),
    )
    scheduler_output = SimpleNamespace(
        triattention_step=9,
        num_scheduled_tokens={"r1": 4},
    )

    seq_map, pos_delta_map, single_seq_base, single_pos_delta = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=[],
    )
    assert seq_map == {0: 96}
    assert pos_delta_map == {0: -24}
    assert single_seq_base == 96
    assert single_pos_delta == -24


def test_build_effective_sparse_overrides_fails_for_compressed_request_without_semantics_marker():
    req_states = SimpleNamespace(req_id_to_index={"r1": 0})
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={"r1": SimpleNamespace(num_computed_tokens=120)},
    )
    state_store = SimpleNamespace(
        has_active_compressed_requests=lambda: True,
        get=lambda req_id: SimpleNamespace(
            current_cache_len=100,
            compression_count=1,
        ),
    )
    scheduler_output = SimpleNamespace(triattention_step=9, num_scheduled_tokens={"r1": 4})

    try:
        build_effective_sparse_overrides(
            base_runner=base_runner,
            state_store=state_store,
            scheduler_output=scheduler_output,
            compression_events=[],
        )
    except RuntimeError as exc:
        assert "TRIATTN_EFFECTIVE_BASE_STATE_SEMANTICS_MISSING" in str(exc)
    else:
        raise AssertionError("expected fail-fast on missing state semantics marker")


def test_build_effective_sparse_overrides_fails_on_event_state_mismatch():
    req_states = SimpleNamespace(req_id_to_index={"r1": 0})
    base_runner = SimpleNamespace(
        req_states=req_states,
        requests={"r1": SimpleNamespace(num_computed_tokens=120)},
    )
    state_store = SimpleNamespace(
        has_active_compressed_requests=lambda: False,
        get=lambda req_id: None,
    )
    scheduler_output = SimpleNamespace(triattention_step=9, num_scheduled_tokens={"r1": 4})

    try:
        build_effective_sparse_overrides(
            base_runner=base_runner,
            state_store=state_store,
            scheduler_output=scheduler_output,
            compression_events=[{"status": "applied", "req_id": "r1"}],
        )
    except RuntimeError as exc:
        assert "TRIATTN_COMPRESSION_EVENT_STATE_MISMATCH" in str(exc)
    else:
        raise AssertionError("expected fail-fast on compression event/state mismatch")
