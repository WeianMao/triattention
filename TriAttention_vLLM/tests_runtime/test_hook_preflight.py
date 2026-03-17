from types import SimpleNamespace
from triattention_runtime.hook_preflight import (
    HookCompactionInputs,
    HookRequestContext,
    resolve_hook_compaction_inputs,
    resolve_hook_request_context,
)


def test_resolve_hook_request_context_success_and_runtime_state():
    req_state = SimpleNamespace(block_ids=([0],))
    state_store = {"r1": SimpleNamespace(effective_len=10)}
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        _triattention_state_store=state_store,
    )

    out = resolve_hook_request_context(base_runner=base_runner, req_id="r1")
    assert isinstance(out, HookRequestContext)
    assert out.req_state is req_state
    assert out.req_runtime_state is state_store["r1"]


def test_resolve_hook_request_context_missing_requests():
    base_runner = SimpleNamespace(requests=None)
    out = resolve_hook_request_context(base_runner=base_runner, req_id="r1")
    assert out == {"applied": False, "reason": "missing_requests"}


def test_resolve_hook_compaction_inputs_success_normalizes_block_ids():
    base_runner = SimpleNamespace(
        kv_caches=[],
        cache_config=SimpleNamespace(block_size=16),
    )
    out = resolve_hook_compaction_inputs(
        base_runner=base_runner,
        original_block_ids_by_group=([0, 1], (2, 3)),
    )
    assert isinstance(out, HookCompactionInputs)
    assert out.block_size == 16
    assert out.mutable_block_ids_by_group == [[0, 1], [2, 3]]


def test_resolve_hook_compaction_inputs_invalid_block_ids_container():
    base_runner = SimpleNamespace(
        kv_caches=[],
        cache_config=SimpleNamespace(block_size=16),
    )
    out = resolve_hook_compaction_inputs(
        base_runner=base_runner,
        original_block_ids_by_group="not-a-container",
    )
    assert out == {"applied": False, "reason": "invalid_block_ids_container"}
