from types import SimpleNamespace

from triattention_runtime.runner_struct_compat import (
    resolve_req_id_to_index,
    resolve_request_state_view,
)


def test_resolve_req_id_to_index_prefers_req_states_when_populated():
    base_runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={"reqA": 3}),
        input_batch=SimpleNamespace(req_id_to_index={"reqA": 7}),
    )

    mapping, source = resolve_req_id_to_index(base_runner)

    assert source == "req_states"
    assert mapping == {"reqA": 3}


def test_resolve_req_id_to_index_falls_back_to_input_batch_when_req_states_empty():
    base_runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={}),
        input_batch=SimpleNamespace(req_id_to_index={"reqA": 7}),
    )

    mapping, source = resolve_req_id_to_index(base_runner)

    assert source == "input_batch"
    assert mapping == {"reqA": 7}


def test_resolve_request_state_view_falls_back_to_input_batch_proxy():
    block_tables = SimpleNamespace(
        append_block_ids=lambda req_index, block_ids, overwrite=False: None,
        apply_staged_writes=lambda: None,
    )
    base_runner = SimpleNamespace(
        requests={},
        req_states=SimpleNamespace(
            req_id_to_index={},
            num_computed_tokens=SimpleNamespace(gpu=None),
        ),
        input_batch=SimpleNamespace(req_id_to_index={"reqA": 1}),
        block_tables=block_tables,
    )

    view, source = resolve_request_state_view(base_runner, "reqA")

    assert source == "input_batch_proxy"
    assert view is not None
    assert view.req_id == "reqA"
    assert view.req_index == 1
