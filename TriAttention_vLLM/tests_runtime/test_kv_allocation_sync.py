from types import SimpleNamespace

from triattention_runtime.kv_allocation_sync import (
    EFFECTIVE_KV_OFFSET_ATTR,
    EFFECTIVE_NUM_COMPUTED_ATTR,
    clear_request_allocation_sync_state,
    prepare_request_effective_num_computed,
    resolve_request_effective_num_computed,
    update_request_effective_kv_offset,
)


def test_update_offset_sets_positive_offset():
    req = SimpleNamespace(num_computed_tokens=77)
    offset = update_request_effective_kv_offset(request=req, cache_len_after=32)
    assert offset == 45
    assert getattr(req, EFFECTIVE_KV_OFFSET_ATTR) == 45


def test_prepare_effective_num_computed_from_offset():
    req = SimpleNamespace(num_computed_tokens=120)
    setattr(req, EFFECTIVE_KV_OFFSET_ATTR, 40)
    effective = prepare_request_effective_num_computed(req)
    assert effective == 80
    assert getattr(req, EFFECTIVE_NUM_COMPUTED_ATTR) == 80
    assert resolve_request_effective_num_computed(req) == 80


def test_prepare_clears_stale_offset_when_logical_resets():
    req = SimpleNamespace(num_computed_tokens=0)
    setattr(req, EFFECTIVE_KV_OFFSET_ATTR, 10)
    prepare_request_effective_num_computed(req)
    assert not hasattr(req, EFFECTIVE_KV_OFFSET_ATTR)
    assert not hasattr(req, EFFECTIVE_NUM_COMPUTED_ATTR)


def test_clear_request_allocation_sync_state():
    req = SimpleNamespace(num_computed_tokens=10)
    setattr(req, EFFECTIVE_KV_OFFSET_ATTR, 4)
    setattr(req, EFFECTIVE_NUM_COMPUTED_ATTR, 6)
    clear_request_allocation_sync_state(req)
    assert not hasattr(req, EFFECTIVE_KV_OFFSET_ATTR)
    assert not hasattr(req, EFFECTIVE_NUM_COMPUTED_ATTR)
