from types import SimpleNamespace

import torch

from triattention_v2.kv_compaction import (
    clear_kv_layout_axis_hints_for_tests,
    gather_request_kv_dense,
)
from triattention_v2.kv_group_resolver import resolve_group_tensors


def _make_fake_layer(kv_tensor: torch.Tensor, layer_idx: int = 0):
    return SimpleNamespace(layer_idx=layer_idx, kv_cache=[kv_tensor])


def _make_fake_runner(kv_tensor: torch.Tensor, backend_cls: type):
    layer_name = "model.layers.0"
    kv_group = SimpleNamespace(layer_names=[layer_name])
    kv_cache_config = SimpleNamespace(kv_cache_groups=[kv_group])
    compilation_config = SimpleNamespace(
        static_forward_context={layer_name: _make_fake_layer(kv_tensor)}
    )
    attn_group = SimpleNamespace(backend=backend_cls())
    return SimpleNamespace(
        kv_cache_config=kv_cache_config,
        compilation_config=compilation_config,
        attn_groups=[attn_group],
    )


def test_resolve_group_tensors_registers_flash_layout_hint_for_ambiguous_shape():
    clear_kv_layout_axis_hints_for_tests()
    kv = torch.arange(2 * 2 * 4 * 1 * 2, dtype=torch.float32).view(2, 2, 4, 1, 2)

    class FlashAttentionBackend:
        pass

    FlashAttentionBackend.__module__ = "vllm.v1.attention.backends.flash_attn"

    runner = _make_fake_runner(kv, FlashAttentionBackend)
    groups = resolve_group_tensors(runner)
    assert 0 in groups and groups[0][0][1] is kv

    keys, values = gather_request_kv_dense(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        total_tokens=6,
    )
    assert torch.equal(keys[0, 0, 5], kv[0, 1, 1, 0])
    assert torch.equal(values[0, 0, 5], kv[1, 1, 1, 0])


def test_resolve_group_tensors_registers_triton_layout_hint_for_ambiguous_shape():
    clear_kv_layout_axis_hints_for_tests()
    kv = torch.arange(2 * 2 * 4 * 1 * 2, dtype=torch.float32).view(2, 2, 4, 1, 2)

    class TritonAttentionBackend:
        pass

    TritonAttentionBackend.__module__ = "vllm.v1.attention.backends.triton_attn"

    runner = _make_fake_runner(kv, TritonAttentionBackend)
    groups = resolve_group_tensors(runner)
    assert 0 in groups and groups[0][0][1] is kv

    keys, values = gather_request_kv_dense(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        total_tokens=6,
    )
    assert torch.equal(keys[0, 0, 5], kv[1, 0, 1, 0])
    assert torch.equal(values[0, 0, 5], kv[1, 1, 1, 0])
