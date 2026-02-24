"""Resolve vLLM KV cache tensors grouped by kv-cache group/layer for TriAttention V2."""

from __future__ import annotations

import re
from typing import Any

import torch

from .kv_compaction import register_kv_layout_axis_hint


def infer_layer_idx(layer_name: str, layer_obj: Any, fallback_idx: int) -> int:
    for attr in ("layer_idx", "layer_id", "idx"):
        value = getattr(layer_obj, attr, None)
        if isinstance(value, int):
            return value
    matches = re.findall(r"\d+", layer_name)
    if matches:
        return int(matches[-1])
    return fallback_idx


def _infer_kv_axis_from_group_backend(base_runner: Any, gid: int) -> int | None:
    attn_groups = getattr(base_runner, "attn_groups", None)
    if not isinstance(attn_groups, (list, tuple)):
        return None
    if gid < 0 or gid >= len(attn_groups):
        return None
    group = attn_groups[gid]
    backend = getattr(group, "backend", None)
    if backend is None:
        return None

    backend_cls = backend if isinstance(backend, type) else backend.__class__
    module_name = str(getattr(backend_cls, "__module__", ""))
    cls_name = str(getattr(backend_cls, "__name__", ""))
    ident = f"{module_name}.{cls_name}".lower()

    if "flash_attn" in ident:
        return 0
    if "triton_attn" in ident:
        return 1
    return None


def resolve_group_tensors(base_runner: Any) -> dict[int, list[tuple[int, torch.Tensor]]]:
    """Resolve kv cache tensors for each kv cache group.

    Returns:
        gid -> list of (layer_idx, kv_cache_tensor)
    """
    group_tensors: dict[int, list[tuple[int, torch.Tensor]]] = {}

    kv_cache_config = getattr(base_runner, "kv_cache_config", None)
    compilation_config = getattr(base_runner, "compilation_config", None)
    static_forward_context = (
        getattr(compilation_config, "static_forward_context", None)
        if compilation_config is not None
        else None
    )

    if kv_cache_config is None or not isinstance(static_forward_context, dict):
        fallback = getattr(base_runner, "kv_caches", None)
        if isinstance(fallback, list):
            tensors = [
                (idx, t)
                for idx, t in enumerate(fallback)
                if isinstance(t, torch.Tensor)
            ]
            if tensors:
                group_tensors[0] = tensors
        return group_tensors

    kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not isinstance(kv_cache_groups, (list, tuple)):
        return group_tensors

    for gid, group in enumerate(kv_cache_groups):
        layer_names = getattr(group, "layer_names", None)
        if not isinstance(layer_names, (list, tuple)):
            continue
        tensors: list[tuple[int, torch.Tensor]] = []
        seen_ptrs: set[int] = set()
        for local_idx, layer_name in enumerate(layer_names):
            layer = static_forward_context.get(layer_name)
            if layer is None:
                continue
            kv_cache_list = getattr(layer, "kv_cache", None)
            if not isinstance(kv_cache_list, list) or not kv_cache_list:
                continue
            tensor = kv_cache_list[0]
            if not isinstance(tensor, torch.Tensor):
                continue
            ptr = tensor.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            tensors.append(
                (
                    infer_layer_idx(
                        layer_name=layer_name,
                        layer_obj=layer,
                        fallback_idx=local_idx,
                    ),
                    tensor,
                )
            )
        if tensors:
            kv_axis_hint = _infer_kv_axis_from_group_backend(base_runner=base_runner, gid=gid)
            if kv_axis_hint is not None:
                for _layer_idx, tensor in tensors:
                    try:
                        register_kv_layout_axis_hint(tensor, kv_axis_hint)
                    except ValueError:
                        # Best effort registration only; compaction path will fail-fast if
                        # an ambiguous layout cannot be safely disambiguated.
                        pass
            group_tensors[gid] = tensors
    return group_tensors
