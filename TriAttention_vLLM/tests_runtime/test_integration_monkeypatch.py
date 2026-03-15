from types import SimpleNamespace

import triattention_runtime.integration_monkeypatch as monkeypatch_mod
from triattention_runtime.integration_monkeypatch import (
    _batch_queue_has_pending_compression_boundary,
    _patched_kv_cache_allocate_slots,
    _refresh_scheduler_stats_kv_usage,
    _scheduler_output_has_compression_boundary,
)


def test_refresh_scheduler_stats_kv_usage_updates_all_stats_entries():
    stats_a = SimpleNamespace(kv_cache_usage=0.9)
    stats_b = SimpleNamespace(kv_cache_usage=0.8)
    outputs = {
        0: SimpleNamespace(scheduler_stats=stats_a),
        1: SimpleNamespace(scheduler_stats=stats_b),
        2: SimpleNamespace(),
    }

    _refresh_scheduler_stats_kv_usage(outputs, 0.25)

    assert stats_a.kv_cache_usage == 0.25
    assert stats_b.kv_cache_usage == 0.25


def test_refresh_scheduler_stats_kv_usage_ignores_non_dict_outputs():
    _refresh_scheduler_stats_kv_usage([], 0.25)


def test_patched_kv_cache_allocate_slots_uses_effective_num_computed(monkeypatch):
    observed = {}

    def _fake_orig(
        _self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
    ):
        observed["num_computed_tokens"] = request.num_computed_tokens
        observed["num_new_tokens"] = num_new_tokens
        return "ok"

    monkeypatch.setattr(monkeypatch_mod, "_ORIG_KVCACHE_ALLOCATE_SLOTS", _fake_orig)

    req = SimpleNamespace(
        num_computed_tokens=120,
        _triattention_effective_num_computed_tokens=80,
    )
    out = _patched_kv_cache_allocate_slots(
        SimpleNamespace(),
        req,
        num_new_tokens=1,
    )

    assert out == "ok"
    assert observed["num_computed_tokens"] == 80
    assert req.num_computed_tokens == 120


def test_patched_kv_cache_allocate_slots_keeps_default_path(monkeypatch):
    observed = {}

    def _fake_orig(
        _self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
    ):
        observed["num_computed_tokens"] = request.num_computed_tokens
        return "ok"

    monkeypatch.setattr(monkeypatch_mod, "_ORIG_KVCACHE_ALLOCATE_SLOTS", _fake_orig)

    req = SimpleNamespace(num_computed_tokens=120)
    out = _patched_kv_cache_allocate_slots(
        SimpleNamespace(),
        req,
        num_new_tokens=1,
    )

    assert out == "ok"
    assert observed["num_computed_tokens"] == 120


def test_patched_kv_cache_allocate_slots_accepts_positional_extra_args(monkeypatch):
    observed = {}

    def _fake_orig(
        _self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
    ):
        observed["num_computed_tokens"] = request.num_computed_tokens
        observed["num_new_computed_tokens"] = num_new_computed_tokens
        observed["num_lookahead_tokens"] = num_lookahead_tokens
        return "ok"

    monkeypatch.setattr(monkeypatch_mod, "_ORIG_KVCACHE_ALLOCATE_SLOTS", _fake_orig)

    req = SimpleNamespace(
        num_computed_tokens=120,
        _triattention_effective_num_computed_tokens=80,
    )
    out = _patched_kv_cache_allocate_slots(
        SimpleNamespace(),
        req,
        1,
        7,
        None,
        2,
    )

    assert out == "ok"
    assert observed["num_computed_tokens"] == 80
    assert observed["num_new_computed_tokens"] == 7
    assert observed["num_lookahead_tokens"] == 2
    assert req.num_computed_tokens == 120


def test_scheduler_output_has_compression_boundary_detects_should_compress_signal():
    signal = SimpleNamespace(should_compress=True)
    scheduler_output = SimpleNamespace(triattention_signals={"r1": signal})

    assert _scheduler_output_has_compression_boundary(scheduler_output) is True


def test_scheduler_output_has_compression_boundary_ignores_empty_or_falsey_signals():
    scheduler_output = SimpleNamespace(
        triattention_signals={"r1": SimpleNamespace(should_compress=False)}
    )
    assert _scheduler_output_has_compression_boundary(scheduler_output) is False
    assert _scheduler_output_has_compression_boundary(SimpleNamespace()) is False


def test_batch_queue_has_pending_compression_boundary_detects_marked_scheduler_output():
    batch_queue = [
        (object(), SimpleNamespace(_triattention_force_boundary_sync=True), object()),
        (object(), SimpleNamespace(), object()),
    ]

    assert _batch_queue_has_pending_compression_boundary(batch_queue) is True


def test_batch_queue_has_pending_compression_boundary_ignores_unmarked_entries():
    batch_queue = [
        (object(), SimpleNamespace(), object()),
        (object(), SimpleNamespace(_triattention_force_boundary_sync=False), object()),
    ]

    assert _batch_queue_has_pending_compression_boundary(batch_queue) is False
