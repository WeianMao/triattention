import os
from contextlib import contextmanager
from pathlib import Path

from triattention_runtime.config import TriAttentionRuntimeConfig


@contextmanager
def _patched_env(values: dict[str, str]):
    old_values: dict[str, str | None] = {}
    for key, value in values.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def test_config_defaults_are_valid():
    cfg = TriAttentionRuntimeConfig()
    cfg.validate()


def test_config_from_env():
    env_values = {
        "TRIATTN_RUNTIME_KV_BUDGET": "1024",
        "TRIATTN_RUNTIME_DIVIDE_LENGTH": "64",
        "TRIATTN_RUNTIME_PROTECT_PREFILL": "false",
        "TRIATTN_RUNTIME_ENABLE_KV_USAGE_TRIGGER": "true",
        "TRIATTN_RUNTIME_KV_USAGE_TRIGGER": "0.95",
        "TRIATTN_RUNTIME_KV_USAGE_RELEASE": "0.85",
        "TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM": "true",
        "TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING": "false",
        "TRIATTN_RUNTIME_FAIL_ON_EFFECTIVE_LEN_REGRESSION": "true",
        "TRIATTN_RUNTIME_EFFECTIVE_LEN_REGRESSION_RATIO": "0.8",
        "TRIATTN_RUNTIME_EFFECTIVE_LEN_GUARD_DIVIDE_MULTIPLES": "3",
        "TRIATTN_RUNTIME_LOG_DECISIONS": "false",
        "TRIATTN_RUNTIME_PRUNING_MODE": "per_layer",
        "TRIATTN_RUNTIME_ALLOW_PER_LAYER_MODE": "true",
        "TRIATTN_RUNTIME_PER_HEAD_SELECTION_SEMANTICS": "hf_aligned_global_per_head",
        "TRIATTN_RUNTIME_LAYER_PERHEAD_AGGREGATION": "mean",
        "TRIATTN_RUNTIME_PER_LAYER_AGGREGATION": "pure_mean",
        "TRIATTN_RUNTIME_WINDOW_SIZE": "96",
        "TRIATTN_RUNTIME_SPARSE_STATS_PATH": "/tmp/fake_stats.pt",
    }
    with _patched_env(env_values):
        cfg = TriAttentionRuntimeConfig.from_env()

    assert cfg.kv_budget == 1024
    assert cfg.divide_length == 64
    assert cfg.protect_prefill is False
    assert cfg.enable_kv_usage_trigger is True
    assert cfg.kv_usage_trigger == 0.95
    assert cfg.kv_usage_release == 0.85
    assert cfg.enable_experimental_block_reclaim is True
    assert cfg.require_triton_scoring is False
    assert cfg.fail_on_effective_len_regression is True
    assert cfg.effective_len_regression_ratio == 0.8
    assert cfg.effective_len_guard_divide_multiples == 3
    assert cfg.log_decisions is False
    assert cfg.pruning_mode == "per_layer"
    assert cfg.allow_per_layer_mode is True
    assert cfg.per_head_selection_semantics == "hf_aligned_global_per_head"
    assert cfg.layer_perhead_aggregation == "mean"
    assert cfg.per_layer_aggregation == "pure_mean"
    assert cfg.window_size == 96
    assert cfg.sparse_stats_path == Path("/tmp/fake_stats.pt")


def test_per_layer_mode_requires_explicit_opt_in():
    cfg = TriAttentionRuntimeConfig(pruning_mode="per_layer")
    try:
        cfg.validate()
    except ValueError as exc:
        assert "disabled by default" in str(exc)
        assert "ALLOW_PER_LAYER_MODE" in str(exc)
    else:
        raise AssertionError("expected ValueError for per_layer without opt-in")


def test_per_layer_mode_is_allowed_with_explicit_opt_in():
    cfg = TriAttentionRuntimeConfig(
        pruning_mode="per_layer",
        allow_per_layer_mode=True,
    )
    cfg.validate()


def test_invalid_hysteresis_raises():
    cfg = TriAttentionRuntimeConfig(
        enable_kv_usage_trigger=True,
        kv_usage_trigger=0.85,
        kv_usage_release=0.90,
    )
    try:
        cfg.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid hysteresis")


def test_invalid_per_head_selection_semantics_raises():
    cfg = TriAttentionRuntimeConfig(per_head_selection_semantics="invalid_mode")
    try:
        cfg.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid per_head_selection_semantics")


def test_invalid_effective_len_regression_ratio_raises():
    cfg = TriAttentionRuntimeConfig(effective_len_regression_ratio=0.0)
    try:
        cfg.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid effective_len_regression_ratio")


def test_invalid_effective_len_guard_multiples_raises():
    cfg = TriAttentionRuntimeConfig(effective_len_guard_divide_multiples=0)
    try:
        cfg.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid effective_len_guard_divide_multiples")
