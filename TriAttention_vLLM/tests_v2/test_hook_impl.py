from types import SimpleNamespace
import tempfile

import torch

from triattention_v2.config import TriAttentionV2Config
from triattention_v2.kv_compaction import register_kv_layout_axis_hint
from triattention_v2.hook_impl import (
    install_runner_compression_hook,
    make_runner_compression_hook,
)
from triattention_v2.signals import CompressionSignal


def _signal() -> CompressionSignal:
    return CompressionSignal(
        req_id="r1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=64,
        step=1,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=8,
    )


def test_hook_plan_only():
    req_state = SimpleNamespace(
        num_computed_tokens=40,
        block_ids=([0, 1, 2],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(kv_budget=16, enable_experimental_kv_compaction=False)
    hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
    out = hook(req_id="r1", signal=_signal(), scheduler_output=SimpleNamespace())
    assert out["applied"] is False
    assert out["reason"] == "plan_only"
    assert out["cache_len_after"] == 16


def test_hook_compaction_mode():
    kv_cache = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 4, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=32,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )
    hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
    out = hook(req_id="r1", signal=_signal(), scheduler_output=SimpleNamespace())
    assert out["applied"] is True
    assert out["reason"].startswith("kv_compacted")
    assert out["cache_len_after"] == 16
    assert out["effective_tokens_before"] == 32
    assert out["budget_total"] == 16


def test_hook_emits_block_reclaim_and_trims_req_block_ids():
    kv_cache = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 4, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=64,
        block_ids=([0, 1, 2, 3],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        enable_experimental_kv_compaction=True,
        enable_experimental_block_reclaim=True,
        require_triton_scoring=False,
    )
    hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
    out = hook(req_id="r1", signal=_signal(), scheduler_output=SimpleNamespace())
    assert out["applied"] is True
    assert out["cache_len_after"] == 16
    reclaim = out.get("block_reclaim")
    assert isinstance(reclaim, dict)
    assert reclaim["mode"] == "truncate_tail"
    assert reclaim["groups"][0]["gid"] == 0
    assert reclaim["groups"][0]["block_ids_after"] == [0]
    assert reclaim["groups"][0]["block_ids_removed"] == [1, 2, 3]
    assert req_state.block_ids[0] == [0]


def test_install_hook_if_missing():
    base_runner = SimpleNamespace()
    cfg = TriAttentionV2Config()
    install_runner_compression_hook(base_runner=base_runner, config=cfg)
    assert callable(getattr(base_runner, "triattention_apply_compression"))


def test_hook_multi_group_compaction():
    kv_a = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)
    kv_b = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)

    layer_a = SimpleNamespace(kv_cache=[kv_a])
    layer_b = SimpleNamespace(kv_cache=[kv_b])

    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer_a"]),
            SimpleNamespace(layer_names=["layer_b"]),
        ]
    )
    compilation_config = SimpleNamespace(
        static_forward_context={
            "layer_a": layer_a,
            "layer_b": layer_b,
        }
    )
    req_state = SimpleNamespace(
        num_computed_tokens=32,
        block_ids=([0, 1], [0, 1]),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_a, kv_b],
        kv_cache_config=kv_cache_config,
        compilation_config=compilation_config,
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )
    hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
    out = hook(req_id="r1", signal=_signal(), scheduler_output=SimpleNamespace())
    assert out["applied"] is True
    assert out["reason"].startswith("kv_compacted")
    assert out["cache_len_after"] == 16


def test_hook_raises_on_inconsistent_cache_len_after_across_groups():
    import triattention_v2.hook_impl as hook_impl_module

    kv_a = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)
    kv_b = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)

    layer_a = SimpleNamespace(kv_cache=[kv_a])
    layer_b = SimpleNamespace(kv_cache=[kv_b])
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer_a"]),
            SimpleNamespace(layer_names=["layer_b"]),
        ]
    )
    compilation_config = SimpleNamespace(
        static_forward_context={"layer_a": layer_a, "layer_b": layer_b}
    )
    req_state = SimpleNamespace(
        num_computed_tokens=32,
        block_ids=([0, 1], [0, 1]),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_a, kv_b],
        kv_cache_config=kv_cache_config,
        compilation_config=compilation_config,
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )

    old_compact = hook_impl_module.compact_request_kv_in_place
    old_resolve_groups = hook_impl_module._resolve_group_tensors
    calls = {"n": 0}

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, keep_token_indices, total_tokens, preserve_dropped_tokens
        calls["n"] += 1
        return 16 if calls["n"] == 1 else 15

    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    hook_impl_module._resolve_group_tensors = lambda _base_runner: {
        0: [(0, kv_a)],
        1: [(1, kv_b)],
    }
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        try:
            hook(
                req_id="r1",
                signal=_signal(),
                scheduler_output=SimpleNamespace(num_scheduled_tokens={"r1": 1}),
            )
        except RuntimeError as exc:
            assert "inconsistent_cache_len_after" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for inconsistent group cache_len_after")
    finally:
        hook_impl_module.compact_request_kv_in_place = old_compact
        hook_impl_module._resolve_group_tensors = old_resolve_groups


def test_hook_compaction_trim_prefill_mode():
    kv_cache = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 4, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=20,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    signal = CompressionSignal(
        req_id="r1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=20,
        step=1,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=8,
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )
    hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
    out = hook(req_id="r1", signal=signal, scheduler_output=SimpleNamespace())
    assert out["applied"] is True
    assert out["reason"].startswith("kv_compacted")
    # trim-prefill mode should retain only the latest kv_budget tokens.
    assert out["cache_len_after"] == 16


def test_hook_accepts_tuple_structures():
    kv_a = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)

    layer_a = SimpleNamespace(kv_cache=[kv_a])
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=(SimpleNamespace(layer_names=("layer_a",)),)
    )
    compilation_config = SimpleNamespace(
        static_forward_context={"layer_a": layer_a}
    )
    req_state = SimpleNamespace(
        num_computed_tokens=20,
        block_ids=((0, 1),),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_a],
        kv_cache_config=kv_cache_config,
        compilation_config=compilation_config,
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )
    hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
    out = hook(req_id="r1", signal=_signal(), scheduler_output=SimpleNamespace())
    assert out["applied"] is True
    assert out["reason"].startswith("kv_compacted")
    assert out["cache_len_after"] == 16


def test_hook_uses_hf_global_per_head_selector_once_per_group():
    import triattention_v2.hook_impl as hook_impl_module

    kv_a = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)
    kv_b = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)

    layer_a = SimpleNamespace(kv_cache=[kv_a])
    layer_b = SimpleNamespace(kv_cache=[kv_b])
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer_a", "layer_b"])]
    )
    compilation_config = SimpleNamespace(
        static_forward_context={"layer_a": layer_a, "layer_b": layer_b}
    )
    req_state = SimpleNamespace(num_computed_tokens=32, block_ids=([0, 1],))
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_a, kv_b],
        kv_cache_config=kv_cache_config,
        compilation_config=compilation_config,
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        pruning_mode="per_head",
        per_head_selection_semantics="hf_aligned_global_per_head",
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )

    calls = {"layer": 0, "group": 0}

    def _fake_layer_selector(**_kwargs):
        calls["layer"] += 1
        return {"mode": "shared", "indices": list(range(16))}

    def _fake_group_selector(**_kwargs):
        calls["group"] += 1
        assert _kwargs.get("layer_inputs") is None
        layer_kv_iter = _kwargs.get("layer_kv_iter")
        layer_input_iter = _kwargs.get("layer_input_iter")
        if callable(layer_kv_iter):
            realized = list(layer_kv_iter())
            assert len(realized) == 2
            assert all(len(item) == 4 for item in realized)
        else:
            assert callable(layer_input_iter)
            realized = list(layer_input_iter())
            assert len(realized) == 2
        return {
            "mode": "per_head",
            "indices": [list(range(16))],
            "semantic": "hf_aligned_global_per_head",
        }

    old_builder = hook_impl_module._build_speckv_selector
    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_layer_selector, _fake_group_selector, "enabled")
    )
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        out = hook(req_id="r1", signal=_signal(), scheduler_output=SimpleNamespace())
    finally:
        hook_impl_module._build_speckv_selector = old_builder

    assert out["applied"] is True
    assert out["reason"] == "kv_compacted:per_head:hf_aligned_global_per_head"
    assert calls["group"] == 1
    assert calls["layer"] == 0


def test_hook_uses_hf_global_per_head_selector_paged_group_when_supported():
    import triattention_v2.hook_impl as hook_impl_module

    kv_a = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)
    kv_b = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)

    layer_a = SimpleNamespace(kv_cache=[kv_a])
    layer_b = SimpleNamespace(kv_cache=[kv_b])
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer_a", "layer_b"])]
    )
    compilation_config = SimpleNamespace(
        static_forward_context={"layer_a": layer_a, "layer_b": layer_b}
    )
    req_state = SimpleNamespace(num_computed_tokens=32, block_ids=([0, 1],))
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_a, kv_b],
        kv_cache_config=kv_cache_config,
        compilation_config=compilation_config,
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        pruning_mode="per_head",
        per_head_selection_semantics="hf_aligned_global_per_head",
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )

    calls = {"layer": 0, "group": 0}

    def _fake_layer_selector(**_kwargs):
        calls["layer"] += 1
        return {"mode": "shared", "indices": list(range(16))}

    def _fake_group_selector(**_kwargs):
        calls["group"] += 1
        assert _kwargs.get("layer_inputs") is None
        assert _kwargs.get("layer_input_iter") is None
        layer_kv_iter = _kwargs.get("layer_kv_iter")
        assert callable(layer_kv_iter)
        realized = list(layer_kv_iter())
        assert len(realized) == 2
        assert all(len(item) == 4 for item in realized)
        return {
            "mode": "per_head",
            "indices": [list(range(16))],
            "semantic": "hf_aligned_global_per_head",
        }

    setattr(_fake_group_selector, "_supports_paged_group", True)

    old_builder = hook_impl_module._build_speckv_selector
    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_layer_selector, _fake_group_selector, "enabled")
    )
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        out = hook(req_id="r1", signal=_signal(), scheduler_output=SimpleNamespace())
    finally:
        hook_impl_module._build_speckv_selector = old_builder

    assert out["applied"] is True
    assert out["reason"] == "kv_compacted:per_head:hf_aligned_global_per_head"
    assert calls["group"] == 1
    assert calls["layer"] == 0


def test_hook_legacy_per_head_keeps_layer_local_selector():
    import triattention_v2.hook_impl as hook_impl_module

    kv_a = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)
    kv_b = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(2, 4, 16, 1, 2)

    layer_a = SimpleNamespace(kv_cache=[kv_a])
    layer_b = SimpleNamespace(kv_cache=[kv_b])
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer_a", "layer_b"])]
    )
    compilation_config = SimpleNamespace(
        static_forward_context={"layer_a": layer_a, "layer_b": layer_b}
    )
    req_state = SimpleNamespace(num_computed_tokens=32, block_ids=([0, 1],))
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_a, kv_b],
        kv_cache_config=kv_cache_config,
        compilation_config=compilation_config,
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        pruning_mode="per_head",
        per_head_selection_semantics="legacy_layer_local",
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )

    calls = {"layer": 0, "group": 0}

    def _fake_layer_selector(**_kwargs):
        calls["layer"] += 1
        return {"mode": "per_head", "indices": [list(range(16))]}

    def _fake_group_selector(**_kwargs):
        calls["group"] += 1
        raise AssertionError("group selector should not run in legacy mode")

    old_builder = hook_impl_module._build_speckv_selector
    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_layer_selector, _fake_group_selector, "enabled")
    )
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        out = hook(req_id="r1", signal=_signal(), scheduler_output=SimpleNamespace())
    finally:
        hook_impl_module._build_speckv_selector = old_builder

    assert out["applied"] is True
    assert out["reason"] == "kv_compacted:per_head"
    assert calls["group"] == 0
    assert calls["layer"] == 2


def test_hook_uses_effective_tokens_for_compaction_work():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=120,
        block_ids=([0, 1, 2, 3, 4, 5, 6, 7],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )

    signal = CompressionSignal(
        req_id="r1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=20,
        step=1,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=8,
    )

    called = {"gather_total_tokens": None, "compact_total_tokens": None}
    old_gather = hook_impl_module.gather_request_k_dense
    old_compact = hook_impl_module.compact_request_kv_in_place
    old_builder = hook_impl_module._build_speckv_selector

    def _fake_layer_selector(
        keys_dense,
        total_tokens,
        prefill_len,
        protect_prefill,
        layer_idx,
        round_start,
        budget_total,
    ):
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_layer_selector, None, "enabled")
    )

    def _fake_gather_request_k_dense(kv_cache, block_ids, block_size, total_tokens):
        called["gather_total_tokens"] = int(total_tokens)
        return torch.zeros((1, 1, total_tokens, 2), dtype=torch.float32)

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del preserve_dropped_tokens
        called["compact_total_tokens"] = int(total_tokens)
        return len(keep_token_indices)

    hook_impl_module.gather_request_k_dense = _fake_gather_request_k_dense
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        out = hook(req_id="r1", signal=signal, scheduler_output=SimpleNamespace())
    finally:
        hook_impl_module.gather_request_k_dense = old_gather
        hook_impl_module.compact_request_kv_in_place = old_compact
        hook_impl_module._build_speckv_selector = old_builder

    assert out["applied"] is True
    # Hook executes pre-step compaction; when scheduler_output lacks explicit
    # num_scheduled_tokens, best-effort default is 1 decode token.
    assert called["gather_total_tokens"] == 19
    assert called["compact_total_tokens"] == 19


def test_hook_clamps_effective_tokens_to_block_capacity_in_paged_path():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=130,
        block_ids=([0, 1, 2, 3, 4, 5, 6, 7],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )

    signal = CompressionSignal(
        req_id="r1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=130,  # intentionally above physical capacity 8*16=128
        step=1,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=8,
    )

    called = {"selector_total_tokens": None, "compact_total_tokens": None}
    old_compact = hook_impl_module.compact_request_kv_in_place
    old_builder = hook_impl_module._build_speckv_selector

    def _fake_paged_selector(
        keys_dense,
        kv_cache,
        block_ids,
        block_size,
        total_tokens,
        prefill_len,
        protect_prefill,
        layer_idx,
        round_start,
        budget_total,
    ):
        del (
            keys_dense,
            kv_cache,
            block_ids,
            block_size,
            prefill_len,
            protect_prefill,
            layer_idx,
            round_start,
        )
        called["selector_total_tokens"] = int(total_tokens)
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    setattr(_fake_paged_selector, "_supports_paged", True)
    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_paged_selector, None, "enabled")
    )

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, keep_token_indices, preserve_dropped_tokens
        called["compact_total_tokens"] = int(total_tokens)
        return min(total_tokens, cfg.kv_budget)

    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        out = hook(req_id="r1", signal=signal, scheduler_output=SimpleNamespace())
    finally:
        hook_impl_module.compact_request_kv_in_place = old_compact
        hook_impl_module._build_speckv_selector = old_builder

    assert out["applied"] is True
    assert called["selector_total_tokens"] == 128
    assert called["compact_total_tokens"] == 128


def test_hook_fail_fast_when_effective_len_regresses_to_full_history():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=120,
        block_ids=([0, 1, 2, 3, 4, 5, 6, 7],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        enable_experimental_block_reclaim=True,
        require_triton_scoring=False,
        fail_on_effective_len_regression=True,
        effective_len_regression_ratio=0.9,
        effective_len_guard_divide_multiples=2,
    )

    old_gather = hook_impl_module.gather_request_k_dense
    old_compact = hook_impl_module.compact_request_kv_in_place
    old_builder = hook_impl_module._build_speckv_selector

    def _fake_layer_selector(
        keys_dense,
        total_tokens,
        prefill_len,
        protect_prefill,
        layer_idx,
        round_start,
        budget_total,
    ):
        del keys_dense, prefill_len, protect_prefill, layer_idx, round_start
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_layer_selector, None, "enabled")
    )

    def _fake_gather_request_k_dense(kv_cache, block_ids, block_size, total_tokens):
        del kv_cache, block_ids, block_size
        return torch.zeros((1, 1, total_tokens, 2), dtype=torch.float32)

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, keep_token_indices, preserve_dropped_tokens
        # Keep physical block ids unchanged so second call can still exercise
        # effective_len regression guard against near-full-history length.
        return int(max(total_tokens, len(block_ids) * block_size - 1))

    hook_impl_module.gather_request_k_dense = _fake_gather_request_k_dense
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        # First call succeeds and marks request as compressed once.
        out = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=20,
                step=1,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert out["applied"] is True

        # Second call presents near-full-history effective length and should fail fast.
        try:
            hook(
                req_id="r1",
                signal=CompressionSignal(
                    req_id="r1",
                    should_compress=True,
                    reason="length_threshold",
                    estimated_cache_len=118,
                    step=2,
                    kv_usage=None,
                    protect_prefill=True,
                    prefill_len=0,
                ),
                scheduler_output=SimpleNamespace(),
            )
        except RuntimeError as exc:
            assert "TRIATTN_FATAL_TRITON_SCORING_REQUIRED:effective_len_regressed" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for effective_len regression")
    finally:
        hook_impl_module.gather_request_k_dense = old_gather
        hook_impl_module.compact_request_kv_in_place = old_compact
        hook_impl_module._build_speckv_selector = old_builder


def test_effective_len_guard_allows_one_step_async_slack():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=25,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
        fail_on_effective_len_regression=True,
        effective_len_regression_ratio=0.9,
        effective_len_guard_divide_multiples=2,
    )

    old_builder = hook_impl_module._build_speckv_selector
    old_compact = hook_impl_module.compact_request_kv_in_place

    def _fake_selector(**kwargs):
        total_tokens = int(kwargs["total_tokens"])
        budget_total = int(kwargs["budget_total"])
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, total_tokens, preserve_dropped_tokens
        return len(keep_token_indices)

    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_selector, None, "enabled")
    )
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        first = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=20,
                step=1,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert first["applied"] is True

        # guard_upper = 16 + 2*4 = 24. effective=25 is only +1 and should be
        # tolerated as scheduler/runner async skew.
        second = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=26,
                step=2,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert second["applied"] is True
    finally:
        hook_impl_module._build_speckv_selector = old_builder
        hook_impl_module.compact_request_kv_in_place = old_compact


def test_effective_len_guard_allows_small_block_granularity_overflow():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=20,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
        fail_on_effective_len_regression=True,
        effective_len_regression_ratio=0.9,
        effective_len_guard_divide_multiples=2,
    )

    old_builder = hook_impl_module._build_speckv_selector
    old_compact = hook_impl_module.compact_request_kv_in_place

    def _fake_selector(**kwargs):
        total_tokens = int(kwargs["total_tokens"])
        budget_total = int(kwargs["budget_total"])
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, total_tokens, preserve_dropped_tokens
        return len(keep_token_indices)

    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_selector, None, "enabled")
    )
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        first = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=20,
                step=1,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert first["applied"] is True

        # guard_upper = 16 + 2*4 = 24. A +4 overflow (to 28) should be tolerated
        # under block-size slack (16) rather than treated as fatal regression.
        req_state.num_computed_tokens = 28
        second = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=28,
                step=2,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert second["applied"] is True
    finally:
        hook_impl_module._build_speckv_selector = old_builder
        hook_impl_module.compact_request_kv_in_place = old_compact


def test_effective_len_guard_allows_block_plus_one_overflow():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=20,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
        fail_on_effective_len_regression=True,
        effective_len_regression_ratio=0.9,
        effective_len_guard_divide_multiples=2,
    )

    old_builder = hook_impl_module._build_speckv_selector
    old_compact = hook_impl_module.compact_request_kv_in_place

    def _fake_selector(**kwargs):
        total_tokens = int(kwargs["total_tokens"])
        budget_total = int(kwargs["budget_total"])
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, total_tokens, preserve_dropped_tokens
        return len(keep_token_indices)

    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_selector, None, "enabled")
    )
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        first = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=20,
                step=1,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert first["applied"] is True

        # guard_upper = 24, block_size = 16, scheduled_tokens = 1.
        # A +17 overflow (41) should be tolerated by block+scheduled slack.
        req_state.num_computed_tokens = 41
        second = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=41,
                step=2,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"r1": 1}),
        )
        assert second["applied"] is True
    finally:
        hook_impl_module._build_speckv_selector = old_builder
        hook_impl_module.compact_request_kv_in_place = old_compact


def test_effective_len_guard_allows_block_plus_two_with_estimate_skew():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=20,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
        fail_on_effective_len_regression=True,
        effective_len_regression_ratio=0.9,
        effective_len_guard_divide_multiples=2,
    )

    old_builder = hook_impl_module._build_speckv_selector
    old_compact = hook_impl_module.compact_request_kv_in_place

    def _fake_selector(**kwargs):
        total_tokens = int(kwargs["total_tokens"])
        budget_total = int(kwargs["budget_total"])
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, total_tokens, preserve_dropped_tokens
        return len(keep_token_indices)

    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_selector, None, "enabled")
    )
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        first = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=20,
                step=1,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert first["applied"] is True

        # guard_upper = 24, block_size = 16, overflow = +18 (to 42).
        # estimated_cache_len is +3 ahead (45), so block+estimate slack should pass.
        req_state.num_computed_tokens = 42
        second = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=45,
                step=2,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"r1": 1}),
        )
        assert second["applied"] is True
    finally:
        hook_impl_module._build_speckv_selector = old_builder
        hook_impl_module.compact_request_kv_in_place = old_compact


def test_effective_len_guard_disabled_in_no_reclaim_mode():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=20,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        enable_experimental_block_reclaim=False,
        require_triton_scoring=False,
        fail_on_effective_len_regression=True,
        effective_len_regression_ratio=0.9,
        effective_len_guard_divide_multiples=2,
    )

    old_builder = hook_impl_module._build_speckv_selector
    old_compact = hook_impl_module.compact_request_kv_in_place

    def _fake_selector(**kwargs):
        total_tokens = int(kwargs["total_tokens"])
        budget_total = int(kwargs["budget_total"])
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, total_tokens, preserve_dropped_tokens
        return len(keep_token_indices)

    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_selector, None, "enabled")
    )
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        out = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=20,
                step=1,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert out["applied"] is True

        # Same shape as regression test, but reclaim is disabled so guard should
        # not abort the no-reclaim A/B path.
        req_state.num_computed_tokens = 118
        second = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=118,
                step=2,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert second["applied"] is True
    finally:
        hook_impl_module._build_speckv_selector = old_builder
        hook_impl_module.compact_request_kv_in_place = old_compact


def test_hook_uses_pre_step_effective_len_and_absolute_round_start():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=20,  # absolute progress before current step
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=8,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )

    old_builder = hook_impl_module._build_speckv_selector
    old_compact = hook_impl_module.compact_request_kv_in_place

    seen: dict[str, int] = {}

    def _fake_selector(**kwargs):
        seen["total_tokens"] = int(kwargs["total_tokens"])
        seen["round_start"] = int(kwargs["round_start"])
        total_tokens = int(kwargs["total_tokens"])
        budget_total = int(kwargs["budget_total"])
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    setattr(_fake_selector, "_supports_paged", True)

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, preserve_dropped_tokens
        seen["compact_total_tokens"] = int(total_tokens)
        return len(keep_token_indices)

    hook_impl_module._build_speckv_selector = (
        lambda _cfg, base_runner=None: (_fake_selector, None, "enabled")
    )
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        out = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                # Scheduler estimate includes the currently scheduled decode token (+1).
                estimated_cache_len=21,
                step=1,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"r1": 1}),
        )
        assert out["applied"] is True
        # Compaction must use pre-step cache length (20), not estimated post-step length (21).
        assert seen["total_tokens"] == 20
        assert seen["compact_total_tokens"] == 20
        # round_start should track absolute decode progress before this step.
        assert seen["round_start"] == 20
    finally:
        hook_impl_module._build_speckv_selector = old_builder
        hook_impl_module.compact_request_kv_in_place = old_compact


def test_effective_len_guard_allows_block_plus_skew_plus_scheduled():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 8 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 8, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=20,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
        fail_on_effective_len_regression=True,
        effective_len_regression_ratio=0.9,
        effective_len_guard_divide_multiples=2,
    )

    old_builder = hook_impl_module._build_speckv_selector
    old_compact = hook_impl_module.compact_request_kv_in_place

    def _fake_selector(**kwargs):
        total_tokens = int(kwargs["total_tokens"])
        budget_total = int(kwargs["budget_total"])
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, total_tokens, preserve_dropped_tokens
        return len(keep_token_indices)

    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_selector, None, "enabled")
    )
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        first = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=20,
                step=1,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert first["applied"] is True

        # guard_upper = 24. Overflow to 44 means +20.
        # estimate skew = (47 - 44) = 3, scheduled_tokens = 1.
        # block + skew + scheduled = 16 + 3 + 1 = 20, so this should pass.
        req_state.num_computed_tokens = 44
        second = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=47,
                step=2,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"r1": 1}),
        )
        assert second["applied"] is True
    finally:
        hook_impl_module._build_speckv_selector = old_builder
        hook_impl_module.compact_request_kv_in_place = old_compact


def test_hook_requires_triton_selector_when_enabled():
    kv_cache = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 4, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=32,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=True,
        sparse_stats_path=None,
    )
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)
        hook(req_id="r1", signal=_signal(), scheduler_output=SimpleNamespace())
    except RuntimeError as exc:
        assert "TRIATTN_FATAL_TRITON_SCORING_REQUIRED" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when Triton selector is unavailable")


def test_hook_defers_recompression_until_budget_plus_divide():
    import triattention_v2.hook_impl as hook_impl_module

    kv_cache = torch.arange(2 * 4 * 16 * 1 * 2, dtype=torch.float32).view(
        2, 4, 16, 1, 2
    )
    req_state = SimpleNamespace(
        num_computed_tokens=32,
        block_ids=([0, 1],),
    )
    base_runner = SimpleNamespace(
        requests={"r1": req_state},
        kv_caches=[kv_cache],
        cache_config=SimpleNamespace(block_size=16),
    )
    cfg = TriAttentionV2Config(
        kv_budget=16,
        divide_length=8,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )

    old_builder = hook_impl_module._build_speckv_selector
    old_compact = hook_impl_module.compact_request_kv_in_place

    calls = {"compact": 0}

    def _fake_selector(**kwargs):
        total_tokens = int(kwargs["total_tokens"])
        budget_total = int(kwargs["budget_total"])
        return {
            "mode": "shared",
            "indices": list(range(min(total_tokens, budget_total))),
        }

    def _fake_compact_request_kv_in_place(
        kv_cache,
        block_ids,
        block_size,
        keep_token_indices,
        total_tokens,
        preserve_dropped_tokens=True,
    ):
        del kv_cache, block_ids, block_size, total_tokens, preserve_dropped_tokens
        calls["compact"] += 1
        return len(keep_token_indices)

    hook_impl_module._build_speckv_selector = (
        lambda _cfg: (_fake_selector, None, "enabled")
    )
    hook_impl_module.compact_request_kv_in_place = _fake_compact_request_kv_in_place
    try:
        hook = make_runner_compression_hook(base_runner=base_runner, config=cfg)

        # First call should compact.
        first = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=32,
                step=1,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert first["applied"] is True

        # After first compaction: budget=16, divide=8, local threshold=24.
        # effective=23 should be deferred and must not compact.
        req_state.num_computed_tokens = 23
        second = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=23,
                step=2,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert second["applied"] is False
        assert second["reason"] == "under_budget"

        # Crossing threshold should compact again.
        req_state.num_computed_tokens = 24
        third = hook(
            req_id="r1",
            signal=CompressionSignal(
                req_id="r1",
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=24,
                step=3,
                kv_usage=None,
                protect_prefill=True,
                prefill_len=0,
            ),
            scheduler_output=SimpleNamespace(),
        )
        assert third["applied"] is True
        assert calls["compact"] == 2
    finally:
        hook_impl_module._build_speckv_selector = old_builder
        hook_impl_module.compact_request_kv_in_place = old_compact


def test_selector_hf_global_per_head_uses_attention_head_scores_and_group_max():
    import triattention.compressor as compressor_module
    import triattention.scoring as scoring_module
    import triattention_v2.hook_impl as hook_impl_module

    class _FakeCompressor:
        def __init__(self, _cfg):
            self.head_stats = {
                0: {
                    "q_abs_mean": torch.ones((8, 2), dtype=torch.float32),
                    "q_mean_complex": torch.zeros((8, 2, 2), dtype=torch.float32),
                }
            }
            self.freq_scale_sq = torch.ones((1, 8, 2), dtype=torch.float32)
            self.omega = torch.ones((2,), dtype=torch.float32)
            self.offsets = torch.ones((1,), dtype=torch.float32)

        def _lazy_init(self):
            return None

    calls: dict[str, list[int]] = {"heads": []}
    # 8 sampled attention heads -> 2 KV heads (group size = 4)
    # Group0 heads [0..3] peak at token 1, group1 heads [4..7] peak at token 2.
    table = torch.tensor(
        [
            [9, 1, 1, 1, 1, 1],
            [8, 2, 2, 2, 2, 2],
            [7, 3, 3, 3, 3, 3],
            [1, 10, 1, 1, 1, 1],
            [1, 1, 11, 1, 1, 1],
            [2, 2, 8, 2, 2, 2],
            [3, 3, 7, 3, 3, 3],
            [4, 4, 6, 4, 4, 4],
        ],
        dtype=torch.float32,
    )

    def _fake_compute_scores_triton(
        key_states,
        cache_positions,
        head_stats,
        omega,
        offsets,
        freq_scale_sq,
        config,
        round_start=None,
        trig_cache=None,
    ):
        del (
            cache_positions,
            head_stats,
            omega,
            offsets,
            freq_scale_sq,
            config,
            round_start,
            trig_cache,
        )
        calls["heads"].append(int(key_states.shape[1]))
        seq_len = int(key_states.shape[2])
        return table[:, :seq_len].unsqueeze(0).contiguous()

    old_compressor = compressor_module.TriAttentionCompressor
    old_compute = scoring_module.compute_scores_triton
    compressor_module.TriAttentionCompressor = _FakeCompressor
    scoring_module.compute_scores_triton = _fake_compute_scores_triton
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt") as stats_file:
            cfg = TriAttentionV2Config(
                kv_budget=1,
                pruning_mode="per_head",
                per_head_selection_semantics="hf_aligned_global_per_head",
                sparse_stats_path=stats_file.name,
                sparse_normalize_scores=False,
                window_size=0,
                enable_experimental_kv_compaction=True,
                require_triton_scoring=False,
            )
            _, group_selector, status = hook_impl_module._build_speckv_selector(cfg)
            assert status == "enabled"
            layer_selector, _, _ = hook_impl_module._build_speckv_selector(cfg)
            assert getattr(layer_selector, "_supports_paged", False) is True
            assert callable(group_selector)
            keys_dense = torch.zeros((1, 2, 6, 4), dtype=torch.float32)
            out = group_selector(
                layer_inputs=[(0, keys_dense)],
                total_tokens=6,
                prefill_len=0,
                protect_prefill=False,
                round_start=5,
                budget_total=1,
            )
    finally:
        compressor_module.TriAttentionCompressor = old_compressor
        scoring_module.compute_scores_triton = old_compute

    assert out is not None
    assert out["mode"] == "per_head"
    assert calls["heads"] == [8]
    indices = out["indices"]
    assert torch.equal(indices[:, 0], torch.tensor([1, 2], dtype=torch.long))


def test_selector_hf_per_layer_requires_explicit_opt_in():
    import triattention_v2.hook_impl as hook_impl_module

    cfg = TriAttentionV2Config(
        pruning_mode="per_layer",
        sparse_stats_path="/tmp/unused.pt",
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )
    try:
        hook_impl_module._build_speckv_selector(cfg)
    except RuntimeError as exc:
        message = str(exc)
        assert "per_layer_mode_disabled" in message
        assert "allow_per_layer_mode=True" in message
    else:
        raise AssertionError("expected RuntimeError for per_layer without opt-in")


def test_selector_hf_per_layer_per_head_uses_attention_head_scores_before_grouping():
    import triattention.compressor as compressor_module
    import triattention.scoring as scoring_module
    import triattention_v2.hook_impl as hook_impl_module

    class _FakeCompressor:
        def __init__(self, _cfg):
            self.head_stats = {
                0: {
                    "q_abs_mean": torch.ones((8, 2), dtype=torch.float32),
                    "q_mean_complex": torch.zeros((8, 2, 2), dtype=torch.float32),
                }
            }
            self.freq_scale_sq = torch.ones((1, 8, 2), dtype=torch.float32)
            self.omega = torch.ones((2,), dtype=torch.float32)
            self.offsets = torch.ones((1,), dtype=torch.float32)

        def _lazy_init(self):
            return None

    calls: dict[str, list[int]] = {"heads": []}
    table = torch.tensor(
        [
            [9, 1, 1, 1, 1, 1],
            [8, 2, 2, 2, 2, 2],
            [7, 3, 3, 3, 3, 3],
            [1, 10, 1, 1, 1, 1],
            [1, 1, 11, 1, 1, 1],
            [2, 2, 8, 2, 2, 2],
            [3, 3, 7, 3, 3, 3],
            [4, 4, 6, 4, 4, 4],
        ],
        dtype=torch.float32,
    )

    def _fake_compute_scores_triton(
        key_states,
        cache_positions,
        head_stats,
        omega,
        offsets,
        freq_scale_sq,
        config,
        round_start=None,
        trig_cache=None,
    ):
        del (
            cache_positions,
            head_stats,
            omega,
            offsets,
            freq_scale_sq,
            config,
            round_start,
            trig_cache,
        )
        calls["heads"].append(int(key_states.shape[1]))
        seq_len = int(key_states.shape[2])
        return table[:, :seq_len].unsqueeze(0).contiguous()

    old_compressor = compressor_module.TriAttentionCompressor
    old_compute = scoring_module.compute_scores_triton
    compressor_module.TriAttentionCompressor = _FakeCompressor
    scoring_module.compute_scores_triton = _fake_compute_scores_triton
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt") as stats_file:
            cfg = TriAttentionV2Config(
                kv_budget=1,
                pruning_mode="per_layer_per_head",
                layer_perhead_aggregation="max",
                sparse_stats_path=stats_file.name,
                sparse_normalize_scores=False,
                window_size=0,
                enable_experimental_kv_compaction=True,
                require_triton_scoring=False,
            )
            layer_selector, _, status = hook_impl_module._build_speckv_selector(cfg)
            assert status == "enabled"
            keys_dense = torch.zeros((1, 2, 6, 4), dtype=torch.float32)
            out = layer_selector(
                keys_dense=keys_dense,
                total_tokens=6,
                prefill_len=0,
                protect_prefill=False,
                layer_idx=0,
                round_start=5,
                budget_total=1,
            )
    finally:
        compressor_module.TriAttentionCompressor = old_compressor
        scoring_module.compute_scores_triton = old_compute

    assert out is not None
    assert out["mode"] == "per_head"
    assert calls["heads"] == [8]
    indices = out["indices"]
    assert torch.equal(indices[:, 0], torch.tensor([1, 2], dtype=torch.long))


def test_selector_hf_per_layer_per_head_layer_group_aggregation_mean_is_configurable():
    import triattention.compressor as compressor_module
    import triattention.scoring as scoring_module
    import triattention_v2.hook_impl as hook_impl_module

    class _FakeCompressor:
        def __init__(self, _cfg):
            self.head_stats = {
                0: {
                    "q_abs_mean": torch.ones((4, 2), dtype=torch.float32),
                    "q_mean_complex": torch.zeros((4, 2, 2), dtype=torch.float32),
                }
            }
            self.freq_scale_sq = torch.ones((1, 4, 2), dtype=torch.float32)
            self.omega = torch.ones((2,), dtype=torch.float32)
            self.offsets = torch.ones((1,), dtype=torch.float32)

        def _lazy_init(self):
            return None

    table = torch.tensor(
        [
            [101, 0, 70],
            [0, 100, 70],
            [5, 9, 1],
            [4, 8, 2],
        ],
        dtype=torch.float32,
    )

    def _fake_compute_scores_triton(
        key_states,
        cache_positions,
        head_stats,
        omega,
        offsets,
        freq_scale_sq,
        config,
        round_start=None,
        trig_cache=None,
    ):
        del (
            cache_positions,
            head_stats,
            omega,
            offsets,
            freq_scale_sq,
            config,
            round_start,
            trig_cache,
        )
        seq_len = int(key_states.shape[2])
        return table[:, :seq_len].unsqueeze(0).contiguous()

    old_compressor = compressor_module.TriAttentionCompressor
    old_compute = scoring_module.compute_scores_triton
    compressor_module.TriAttentionCompressor = _FakeCompressor
    scoring_module.compute_scores_triton = _fake_compute_scores_triton
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt") as stats_file:
            base_kwargs = dict(
                kv_budget=1,
                pruning_mode="per_layer_per_head",
                sparse_stats_path=stats_file.name,
                sparse_normalize_scores=False,
                window_size=0,
                enable_experimental_kv_compaction=True,
                require_triton_scoring=False,
            )
            cfg_max = TriAttentionV2Config(**base_kwargs, layer_perhead_aggregation="max")
            cfg_mean = TriAttentionV2Config(**base_kwargs, layer_perhead_aggregation="mean")
            selector_max, _, status_max = hook_impl_module._build_speckv_selector(cfg_max)
            selector_mean, _, status_mean = hook_impl_module._build_speckv_selector(cfg_mean)
            assert status_max == "enabled"
            assert status_mean == "enabled"
            keys_dense = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
            out_max = selector_max(
                keys_dense=keys_dense,
                total_tokens=3,
                prefill_len=0,
                protect_prefill=False,
                layer_idx=0,
                round_start=2,
                budget_total=1,
            )
            out_mean = selector_mean(
                keys_dense=keys_dense,
                total_tokens=3,
                prefill_len=0,
                protect_prefill=False,
                layer_idx=0,
                round_start=2,
                budget_total=1,
            )
    finally:
        compressor_module.TriAttentionCompressor = old_compressor
        scoring_module.compute_scores_triton = old_compute

    assert out_max is not None and out_mean is not None
    idx_max = out_max["indices"]
    idx_mean = out_mean["indices"]
    assert isinstance(idx_max, torch.Tensor)
    assert isinstance(idx_mean, torch.Tensor)
    assert int(idx_max[0, 0].item()) == 0
    assert int(idx_mean[0, 0].item()) == 2


def test_selector_hf_global_per_head_paged_matches_dense_with_normalize():
    import triattention.compressor as compressor_module
    import triattention.scoring as scoring_module
    import triattention_v2.hook_impl as hook_impl_module

    class _FakeCompressor:
        def __init__(self, _cfg):
            head_stats = {
                "q_abs_mean": torch.ones((8, 2), dtype=torch.float32),
                "q_mean_complex": torch.zeros((8, 2, 2), dtype=torch.float32),
            }
            self.head_stats = {0: head_stats, 1: head_stats}
            self.freq_scale_sq = torch.ones((2, 8, 2), dtype=torch.float32)
            self.omega = torch.ones((2,), dtype=torch.float32)
            self.offsets = torch.ones((1,), dtype=torch.float32)

        def _lazy_init(self):
            return None

    def _fake_compute_scores_triton(
        key_states,
        cache_positions,
        head_stats,
        omega,
        offsets,
        freq_scale_sq,
        config,
        round_start=None,
        trig_cache=None,
    ):
        del (
            cache_positions,
            head_stats,
            omega,
            offsets,
            freq_scale_sq,
            config,
            round_start,
            trig_cache,
        )
        # key_states[0, 0, :, 0] encodes logical token id in our test fixture.
        token_ids = key_states[0, 0, :, 0].to(dtype=torch.float32)
        num_heads = int(key_states.shape[1])
        out = []
        for head in range(num_heads):
            head_bias = float(head + 1)
            # Piecewise shift makes chunk-local normalization diverge from full-seq normalization.
            scores = torch.where(
                token_ids < 4.0,
                token_ids + 50.0 + head_bias,
                token_ids * 0.25 + head_bias,
            )
            out.append(scores)
        return torch.stack(out, dim=0).unsqueeze(0).contiguous()

    def _make_kv(total_tokens: int, block_size: int, offset: float) -> torch.Tensor:
        num_blocks = (total_tokens + block_size - 1) // block_size
        kv = torch.zeros((2, num_blocks, block_size, 2, 2), dtype=torch.float32)
        for token in range(total_tokens):
            b = token // block_size
            o = token % block_size
            value = float(token) + offset
            kv[0, b, o, :, 0] = value  # key dim0 carries token id
        if num_blocks == 2:
            register_kv_layout_axis_hint(kv, 0)
        return kv

    old_compressor = compressor_module.TriAttentionCompressor
    old_compute = scoring_module.compute_scores_triton
    compressor_module.TriAttentionCompressor = _FakeCompressor
    scoring_module.compute_scores_triton = _fake_compute_scores_triton
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt") as stats_file:
            cfg = TriAttentionV2Config(
                kv_budget=2,
                pruning_mode="per_head",
                per_head_selection_semantics="hf_aligned_global_per_head",
                sparse_stats_path=stats_file.name,
                sparse_normalize_scores=True,
                window_size=0,
                enable_experimental_kv_compaction=True,
                require_triton_scoring=False,
            )
            _, group_selector, status = hook_impl_module._build_speckv_selector(cfg)
            assert status == "enabled"
            assert callable(group_selector)

            total_tokens = 8
            block_size = 4
            kv0 = _make_kv(total_tokens, block_size, offset=0.0)
            kv1 = _make_kv(total_tokens, block_size, offset=0.5)

            dense0 = hook_impl_module.gather_request_k_dense(
                kv_cache=kv0,
                block_ids=[0, 1],
                block_size=block_size,
                total_tokens=total_tokens,
            )
            dense1 = hook_impl_module.gather_request_k_dense(
                kv_cache=kv1,
                block_ids=[0, 1],
                block_size=block_size,
                total_tokens=total_tokens,
            )

            dense_out = group_selector(
                layer_inputs=[(0, dense0), (1, dense1)],
                layer_input_iter=None,
                layer_kv_iter=None,
                total_tokens=total_tokens,
                prefill_len=0,
                protect_prefill=False,
                round_start=7,
                budget_total=2,
            )
            paged_out = group_selector(
                layer_inputs=None,
                layer_input_iter=None,
                layer_kv_iter=lambda: iter(
                    [
                        (0, kv0, [0, 1], block_size),
                        (1, kv1, [0, 1], block_size),
                    ]
                ),
                total_tokens=total_tokens,
                prefill_len=0,
                protect_prefill=False,
                round_start=7,
                budget_total=2,
            )
    finally:
        compressor_module.TriAttentionCompressor = old_compressor
        scoring_module.compute_scores_triton = old_compute

    assert dense_out is not None and paged_out is not None
    assert dense_out["mode"] == "per_head"
    assert paged_out["mode"] == "per_head"
    assert torch.equal(
        dense_out["indices"],
        paged_out["indices"],
    )


def test_selector_hf_per_layer_paged_matches_dense_with_normalize():
    import triattention.compressor as compressor_module
    import triattention.scoring as scoring_module
    import triattention_v2.hook_impl as hook_impl_module

    class _FakeCompressor:
        def __init__(self, _cfg):
            self.head_stats = {
                0: {
                    "q_abs_mean": torch.ones((2, 2), dtype=torch.float32),
                    "q_mean_complex": torch.zeros((2, 2, 2), dtype=torch.float32),
                }
            }
            self.freq_scale_sq = torch.ones((1, 2, 2), dtype=torch.float32)
            self.omega = torch.ones((2,), dtype=torch.float32)
            self.offsets = torch.ones((1,), dtype=torch.float32)

        def _lazy_init(self):
            return None

    def _fake_compute_scores_triton(
        key_states,
        cache_positions,
        head_stats,
        omega,
        offsets,
        freq_scale_sq,
        config,
        round_start=None,
        trig_cache=None,
    ):
        del (
            cache_positions,
            head_stats,
            omega,
            offsets,
            freq_scale_sq,
            config,
            round_start,
            trig_cache,
        )
        token_ids = key_states[0, 0, :, 0].to(dtype=torch.float32)
        num_heads = int(key_states.shape[1])
        out = []
        for head in range(num_heads):
            head_bias = float(head + 1)
            scores = torch.where(
                token_ids < 6.0,
                token_ids * 2.0 + head_bias,
                token_ids * 0.1 + 10.0 + head_bias,
            )
            out.append(scores)
        return torch.stack(out, dim=0).unsqueeze(0).contiguous()

    def _make_kv(total_tokens: int, block_size: int) -> torch.Tensor:
        num_blocks = (total_tokens + block_size - 1) // block_size
        kv = torch.zeros((2, num_blocks, block_size, 2, 2), dtype=torch.float32)
        for token in range(total_tokens):
            b = token // block_size
            o = token % block_size
            kv[0, b, o, :, 0] = float(token)
        if num_blocks == 2:
            register_kv_layout_axis_hint(kv, 0)
        return kv

    old_compressor = compressor_module.TriAttentionCompressor
    old_compute = scoring_module.compute_scores_triton
    compressor_module.TriAttentionCompressor = _FakeCompressor
    scoring_module.compute_scores_triton = _fake_compute_scores_triton
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt") as stats_file:
            cfg = TriAttentionV2Config(
                kv_budget=4,
                pruning_mode="per_layer",
                allow_per_layer_mode=True,
                sparse_stats_path=stats_file.name,
                sparse_normalize_scores=True,
                window_size=0,
                divide_length=128,
                enable_experimental_kv_compaction=True,
                require_triton_scoring=False,
            )
            layer_selector, _, status = hook_impl_module._build_speckv_selector(cfg)
            assert status == "enabled"
            assert callable(layer_selector)
            assert getattr(layer_selector, "_supports_paged", False) is True

            total_tokens = 8
            block_size = 4
            kv = _make_kv(total_tokens, block_size)
            dense = hook_impl_module.gather_request_k_dense(
                kv_cache=kv,
                block_ids=[0, 1],
                block_size=block_size,
                total_tokens=total_tokens,
            )
            dense_out = layer_selector(
                keys_dense=dense,
                kv_cache=None,
                block_ids=None,
                block_size=None,
                total_tokens=total_tokens,
                prefill_len=0,
                protect_prefill=False,
                layer_idx=0,
                round_start=5,
                budget_total=4,
            )
            paged_out = layer_selector(
                keys_dense=None,
                kv_cache=kv,
                block_ids=[0, 1],
                block_size=block_size,
                total_tokens=total_tokens,
                prefill_len=0,
                protect_prefill=False,
                layer_idx=0,
                round_start=5,
                budget_total=4,
            )
    finally:
        compressor_module.TriAttentionCompressor = old_compressor
        scoring_module.compute_scores_triton = old_compute

    assert dense_out is not None and paged_out is not None
    assert dense_out["mode"] == "shared"
    assert paged_out["mode"] == "shared"
    assert torch.equal(
        dense_out["indices"],
        paged_out["indices"],
    )


def test_selector_reduces_stats_heads_to_runtime_heads_for_legacy_path():
    import triattention.compressor as compressor_module
    import triattention.scoring as scoring_module
    import triattention_v2.hook_impl as hook_impl_module

    class _FakeCompressor:
        def __init__(self, _cfg):
            self.head_stats = {
                0: {
                    "q_abs_mean": torch.arange(8 * 2, dtype=torch.float32).view(8, 2),
                    "q_mean_complex": torch.zeros((8, 2, 2), dtype=torch.float32),
                }
            }
            self.freq_scale_sq = torch.ones((1, 8, 2), dtype=torch.float32)
            self.omega = torch.ones((2,), dtype=torch.float32)
            self.offsets = torch.ones((1,), dtype=torch.float32)

        def _lazy_init(self):
            return None

    calls: dict[str, list[int]] = {"heads": []}

    def _fake_compute_scores_triton(
        key_states,
        cache_positions,
        head_stats,
        omega,
        offsets,
        freq_scale_sq,
        config,
        round_start=None,
        trig_cache=None,
    ):
        del cache_positions, omega, offsets, freq_scale_sq, config, round_start, trig_cache
        calls["heads"].append(int(key_states.shape[1]))
        assert int(head_stats["q_abs_mean"].shape[0]) == int(key_states.shape[1])
        seq_len = int(key_states.shape[2])
        return torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len).expand(
            1, key_states.shape[1], seq_len
        )

    old_compressor = compressor_module.TriAttentionCompressor
    old_compute = scoring_module.compute_scores_triton
    compressor_module.TriAttentionCompressor = _FakeCompressor
    scoring_module.compute_scores_triton = _fake_compute_scores_triton
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt") as stats_file:
            cfg = TriAttentionV2Config(
                kv_budget=2,
                pruning_mode="per_head",
                per_head_selection_semantics="legacy_layer_local",
                sparse_stats_path=stats_file.name,
                sparse_normalize_scores=False,
                window_size=0,
                enable_experimental_kv_compaction=True,
                require_triton_scoring=False,
            )
            layer_selector, _, status = hook_impl_module._build_speckv_selector(cfg)
            assert status == "enabled"
            assert callable(layer_selector)
            assert getattr(layer_selector, "_supports_paged", False) is True
            keys_dense = torch.zeros((1, 2, 6, 4), dtype=torch.float32)
            out = layer_selector(
                keys_dense=keys_dense,
                total_tokens=6,
                prefill_len=0,
                protect_prefill=False,
                layer_idx=0,
                round_start=5,
                budget_total=2,
            )
    finally:
        compressor_module.TriAttentionCompressor = old_compressor
        scoring_module.compute_scores_triton = old_compute

    assert out is not None
    assert out["mode"] == "per_head"
    assert calls["heads"] == [2]


def test_selector_paged_streaming_merge_topk_clamps_k_to_current_pool():
    import triattention.compressor as compressor_module
    import triattention.scoring as scoring_module
    import triattention_v2.hook_impl as hook_impl_module

    class _FakeCompressor:
        def __init__(self, _cfg):
            self.head_stats = {
                0: {
                    "q_abs_mean": torch.ones((1, 2), dtype=torch.float32),
                    "q_mean_complex": torch.zeros((1, 2, 2), dtype=torch.float32),
                }
            }
            self.freq_scale_sq = torch.ones((1, 1, 2), dtype=torch.float32)
            self.omega = torch.ones((2,), dtype=torch.float32)
            self.offsets = torch.ones((1,), dtype=torch.float32)

        def _lazy_init(self):
            return None

    def _fake_compute_scores_triton(
        key_states,
        cache_positions,
        head_stats,
        omega,
        offsets,
        freq_scale_sq,
        config,
        round_start=None,
        trig_cache=None,
    ):
        del (
            cache_positions,
            head_stats,
            omega,
            offsets,
            freq_scale_sq,
            config,
            round_start,
            trig_cache,
        )
        seq_len = int(key_states.shape[2])
        num_heads = int(key_states.shape[1])
        base = torch.arange(seq_len, dtype=torch.float32, device=key_states.device)
        return base.view(1, 1, seq_len).expand(1, num_heads, seq_len).contiguous()

    old_compressor = compressor_module.TriAttentionCompressor
    old_compute = scoring_module.compute_scores_triton
    compressor_module.TriAttentionCompressor = _FakeCompressor
    scoring_module.compute_scores_triton = _fake_compute_scores_triton
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt") as stats_file:
            cfg = TriAttentionV2Config(
                kv_budget=2200,
                pruning_mode="per_layer",
                allow_per_layer_mode=True,
                sparse_stats_path=stats_file.name,
                sparse_normalize_scores=False,
                divide_length=128,
                window_size=0,
                enable_experimental_kv_compaction=True,
                require_triton_scoring=False,
            )
            layer_selector, _, status = hook_impl_module._build_speckv_selector(cfg)
            assert status == "enabled"
            assert getattr(layer_selector, "_supports_paged", False) is True

            total_tokens = 2300
            block_size = 16
            num_blocks = (total_tokens + block_size - 1) // block_size
            kv_cache = torch.zeros((2, num_blocks, block_size, 1, 2), dtype=torch.float32)
            block_ids = list(range(num_blocks))
            out = layer_selector(
                keys_dense=None,
                kv_cache=kv_cache,
                block_ids=block_ids,
                block_size=block_size,
                total_tokens=total_tokens,
                prefill_len=0,
                protect_prefill=False,
                layer_idx=0,
                round_start=0,
                budget_total=2200,
            )
    finally:
        compressor_module.TriAttentionCompressor = old_compressor
        scoring_module.compute_scores_triton = old_compute

    assert out is not None
    assert out["mode"] == "shared"
    assert len(out["indices"]) == 2200
