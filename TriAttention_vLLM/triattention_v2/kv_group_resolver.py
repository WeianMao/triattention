"""Resolve vLLM KV cache tensors grouped by kv-cache group/layer for TriAttention V2."""

from __future__ import annotations

import re
from typing import Any

import torch


def infer_layer_idx(layer_name: str, layer_obj: Any, fallback_idx: int) -> int:
    for attr in ("layer_idx", "layer_id", "idx"):
        value = getattr(layer_obj, attr, None)
        if isinstance(value, int):
            return value
    matches = re.findall(r"\d+", layer_name)
    if matches:
        return int(matches[-1])
    return fallback_idx


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
            group_tensors[gid] = tensors
    return group_tensors
