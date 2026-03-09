from types import SimpleNamespace

from triattention_runtime.integration_monkeypatch import (
    _refresh_scheduler_stats_kv_usage,
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
