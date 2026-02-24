"""In-place KV compaction primitives for Phase 1B prototype."""

from __future__ import annotations

from typing import Iterable

import torch

_CONSECUTIVE_SPAN_CACHE_MAX = 8192
_CONSECUTIVE_SPAN_CACHE: dict[tuple[int, int, int], tuple[int, int] | None] = {}


def build_keep_token_indices(
    total_tokens: int,
    kv_budget: int,
    prefill_len: int,
    protect_prefill: bool,
    include_prefill_in_budget: bool = True,
) -> list[int] | None:
    """Build ordered keep indices for one request.

    Returns:
    - None: impossible under constraints (e.g., prefill_len > kv_budget).
    - list[int]: sorted ascending keep token indices.
    """
    effective_budget = kv_budget
    if protect_prefill and not include_prefill_in_budget:
        effective_budget += max(prefill_len, 0)
    effective_budget = min(total_tokens, effective_budget)

    if total_tokens <= effective_budget:
        return list(range(total_tokens))

    if protect_prefill:
        if include_prefill_in_budget and prefill_len > effective_budget:
            return None
        keep_prefill = list(range(prefill_len))
        tail_need = max(0, effective_budget - prefill_len)
        tail_start = max(prefill_len, total_tokens - tail_need)
        keep_tail = list(range(tail_start, total_tokens))
        return keep_prefill + keep_tail

    start = max(0, total_tokens - effective_budget)
    return list(range(start, total_tokens))


def _split_kv_axes(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return key/value views in shape [num_blocks, block_size, H, D]."""
    if kv_cache.ndim != 5:
        raise ValueError(f"Unsupported kv_cache ndim={kv_cache.ndim}, expect 5")
    if kv_cache.shape[0] == 2:
        return kv_cache[0], kv_cache[1]
    if kv_cache.shape[1] == 2:
        return kv_cache[:, 0], kv_cache[:, 1]
    raise ValueError(
        "Unsupported KV layout for compaction: expected a dimension with size 2"
    )


def _token_slot(block_ids: list[int], block_size: int, token_idx: int) -> tuple[int, int]:
    block_idx = token_idx // block_size
    if block_idx >= len(block_ids):
        raise IndexError(
            f"token_idx={token_idx} requires block_idx={block_idx}, "
            f"but only {len(block_ids)} blocks exist"
        )
    return block_ids[block_idx], token_idx % block_size


def _as_block_ids_tensor(
    block_ids: list[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(block_ids, torch.Tensor):
        return block_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(block_ids, device=device, dtype=torch.long)


def _resolve_token_slots(
    block_ids: list[int] | torch.Tensor,
    block_size: int,
    token_indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_ids_t = _as_block_ids_tensor(block_ids, device=device)
    logical_block_idx = torch.div(token_indices, block_size, rounding_mode="floor")
    if logical_block_idx.numel() > 0:
        max_required = int(logical_block_idx.max().item())
        if max_required >= int(block_ids_t.numel()):
            raise IndexError(
                f"token indices require block_idx={max_required}, "
                f"but only {int(block_ids_t.numel())} blocks exist"
            )
    src_blocks = block_ids_t[logical_block_idx]
    src_off = torch.remainder(token_indices, block_size)
    return src_blocks, src_off


def _resolve_token_slots_contiguous_range(
    block_ids: list[int] | torch.Tensor,
    block_size: int,
    *,
    start_token: int,
    num_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve slots for contiguous token range [start_token, start_token + num_tokens)."""
    block_ids_t = _as_block_ids_tensor(block_ids, device=device)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
    max_required = (start_token + num_tokens - 1) // block_size
    if max_required >= int(block_ids_t.numel()):
        raise IndexError(
            f"token range requires block_idx={max_required}, "
            f"but only {int(block_ids_t.numel())} blocks exist"
        )
    token_indices = torch.arange(
        start_token,
        start_token + num_tokens,
        device=device,
        dtype=torch.long,
    )
    logical_block_idx = torch.div(token_indices, block_size, rounding_mode="floor")
    src_blocks = block_ids_t[logical_block_idx]
    src_off = torch.remainder(token_indices, block_size)
    return src_blocks, src_off


def _consecutive_block_span(
    block_ids: list[int] | torch.Tensor,
) -> tuple[int, int] | None:
    if isinstance(block_ids, torch.Tensor):
        block_ids_t = block_ids.to(dtype=torch.long).flatten()
        if block_ids_t.numel() == 0:
            return None
        cache_key = (
            int(block_ids_t.data_ptr()),
            int(block_ids_t.numel()),
            int(block_ids_t.device.index) if block_ids_t.device.type == "cuda" else -1,
        )
        if cache_key in _CONSECUTIVE_SPAN_CACHE:
            return _CONSECUTIVE_SPAN_CACHE[cache_key]

        count = int(block_ids_t.numel())
        start = int(block_ids_t[0].item())
        if count == 1:
            span: tuple[int, int] | None = (start, 1)
        else:
            end = int(block_ids_t[-1].item())
            # Quick range gate before full adjacency check.
            if end - start + 1 != count:
                span = None
            else:
                deltas = block_ids_t[1:] - block_ids_t[:-1]
                span = (start, count) if bool((deltas == 1).all().item()) else None

        if len(_CONSECUTIVE_SPAN_CACHE) >= _CONSECUTIVE_SPAN_CACHE_MAX:
            _CONSECUTIVE_SPAN_CACHE.pop(next(iter(_CONSECUTIVE_SPAN_CACHE)))
        _CONSECUTIVE_SPAN_CACHE[cache_key] = span
        return span

    if not block_ids:
        return None
    start = int(block_ids[0])
    for idx, block_id in enumerate(block_ids):
        if int(block_id) != start + idx:
            return None
    return start, len(block_ids)


def _try_dense_token_view(
    cache_thd: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    total_tokens: int,
) -> torch.Tensor | None:
    """Try to return a token-major view [T, H, D] without gather copy."""
    span = _consecutive_block_span(block_ids)
    if span is None:
        return None
    start, num_blocks = span
    if start < 0:
        return None
    end = start + num_blocks
    if end > int(cache_thd.shape[0]):
        return None
    block_slice = cache_thd[start:end]
    token_view = block_slice.reshape(-1, block_slice.shape[2], block_slice.shape[3])
    if total_tokens > int(token_view.shape[0]):
        return None
    return token_view[:total_tokens]


def gather_request_kv_dense(
    kv_cache: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    block_size: int,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one request KV from paged cache into dense [1, H, T, D] tensors."""
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be > 0, got {total_tokens}")

    key_cache, value_cache = _split_kv_axes(kv_cache)
    key_fast = _try_dense_token_view(
        cache_thd=key_cache,
        block_ids=block_ids,
        total_tokens=total_tokens,
    )
    value_fast = _try_dense_token_view(
        cache_thd=value_cache,
        block_ids=block_ids,
        total_tokens=total_tokens,
    )
    if key_fast is not None and value_fast is not None:
        return key_fast.transpose(0, 1).unsqueeze(0), value_fast.transpose(0, 1).unsqueeze(0)

    src_blocks, src_off = _resolve_token_slots_contiguous_range(
        block_ids=block_ids,
        block_size=block_size,
        start_token=0,
        num_tokens=total_tokens,
        device=key_cache.device,
    )
    keys_thd = key_cache[src_blocks, src_off]
    values_thd = value_cache[src_blocks, src_off]
    keys = keys_thd.transpose(0, 1).unsqueeze(0)
    values = values_thd.transpose(0, 1).unsqueeze(0)
    return keys, values


def gather_request_k_dense(
    kv_cache: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    block_size: int,
    total_tokens: int,
) -> torch.Tensor:
    """Gather one request K from paged cache into dense [1, H, T, D] tensor."""
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be > 0, got {total_tokens}")

    key_cache, _ = _split_kv_axes(kv_cache)
    key_fast = _try_dense_token_view(
        cache_thd=key_cache,
        block_ids=block_ids,
        total_tokens=total_tokens,
    )
    if key_fast is not None:
        return key_fast.transpose(0, 1).unsqueeze(0)

    src_blocks, src_off = _resolve_token_slots_contiguous_range(
        block_ids=block_ids,
        block_size=block_size,
        start_token=0,
        num_tokens=total_tokens,
        device=key_cache.device,
    )
    keys_thd = key_cache[src_blocks, src_off]
    return keys_thd.transpose(0, 1).unsqueeze(0)


def gather_request_k_dense_range(
    kv_cache: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    block_size: int,
    start_token: int,
    num_tokens: int,
) -> torch.Tensor:
    """Gather request K sub-range into dense [1, H, T_chunk, D] tensor."""
    if start_token < 0:
        raise ValueError(f"start_token must be >= 0, got {start_token}")
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}")

    key_cache, _ = _split_kv_axes(kv_cache)
    span = _try_dense_token_view(
        cache_thd=key_cache,
        block_ids=block_ids,
        total_tokens=start_token + num_tokens,
    )
    if span is not None:
        return span[start_token : start_token + num_tokens].transpose(0, 1).unsqueeze(0)

    src_blocks, src_off = _resolve_token_slots_contiguous_range(
        block_ids=block_ids,
        block_size=block_size,
        start_token=start_token,
        num_tokens=num_tokens,
        device=key_cache.device,
    )
    keys_thd = key_cache[src_blocks, src_off]
    return keys_thd.transpose(0, 1).unsqueeze(0)


def compact_request_kv_in_place(
    kv_cache: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    block_size: int,
    keep_token_indices: Iterable[int] | torch.Tensor,
    total_tokens: int,
    preserve_dropped_tokens: bool = True,
) -> int:
    """Compact KV for a single request in-place.

    Behavior:
    - Reorders tokens into [kept..., dropped...] permutation in-place.
    - Preserves the full token multiset (no tail zeroing).
    - Does not modify block_ids or scheduler metadata.

    Note:
    We intentionally avoid writing zero tails while request logical length is
    still `total_tokens`, otherwise dropped entries continue participating in
    attention softmax as zero-K tokens and corrupt generation quality.
    """
    if isinstance(keep_token_indices, torch.Tensor):
        keep_tensor = keep_token_indices.to(device=kv_cache.device, dtype=torch.long).flatten()
    else:
        keep_tensor = torch.as_tensor(
            list(keep_token_indices),
            device=kv_cache.device,
            dtype=torch.long,
        )
    if keep_tensor.numel() == 0:
        return 0

    key_cache, value_cache = _split_kv_axes(kv_cache)
    if keep_tensor.numel() > 0:
        min_idx = int(keep_tensor.min().item())
        max_idx = int(keep_tensor.max().item())
        if min_idx < 0 or max_idx >= total_tokens:
            raise IndexError(
                f"keep token index out of range: min={min_idx}, max={max_idx}, "
                f"expected [0, {total_tokens})"
            )
    # TopK path always yields unique indices; fallback path should also be unique.
    if int(torch.unique(keep_tensor).numel()) != int(keep_tensor.numel()):
        raise ValueError("keep_token_indices must not contain duplicates")

    keep_count = int(keep_tensor.numel())
    device = key_cache.device

    if preserve_dropped_tokens:
        if keep_count >= total_tokens:
            return keep_count
        prefix = torch.arange(keep_count, device=device, dtype=torch.long)
        if torch.equal(keep_tensor, prefix):
            # Identity permutation, no copy needed.
            return keep_count
        all_tokens = torch.arange(total_tokens, device=device, dtype=torch.long)
        tail_mask = torch.ones(total_tokens, device=device, dtype=torch.bool)
        tail_mask[keep_tensor] = False
        perm_tensor = torch.cat([keep_tensor, all_tokens[tail_mask]], dim=0)
        dst_tokens = torch.arange(total_tokens, device=device, dtype=torch.long)
    else:
        perm_tensor, dst_tokens = _build_fill_hole_placement_shared(
            keep_tensor=keep_tensor,
            keep_count=keep_count,
            device=device,
        )
        if perm_tensor.numel() == 0:
            return keep_count

    src_blocks, src_off = _resolve_token_slots(
        block_ids=block_ids,
        block_size=block_size,
        token_indices=perm_tensor,
        device=device,
    )
    gathered_keys = key_cache[src_blocks, src_off].clone()
    gathered_values = value_cache[src_blocks, src_off].clone()

    dst_blocks, dst_off = _resolve_token_slots(
        block_ids=block_ids,
        block_size=block_size,
        token_indices=dst_tokens,
        device=device,
    )
    key_cache[dst_blocks, dst_off] = gathered_keys
    value_cache[dst_blocks, dst_off] = gathered_values

    return keep_count


def compact_request_kv_in_place_per_head(
    kv_cache: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    block_size: int,
    keep_token_indices_per_head: list[list[int]] | torch.Tensor,
    total_tokens: int,
    preserve_dropped_tokens: bool = True,
) -> int:
    """Compact KV in-place using independent keep indices for each KV head.

    Reorder each head with [kept..., dropped...] permutation while preserving
    all tokens per head (no tail zeroing).
    """
    key_cache, value_cache = _split_kv_axes(kv_cache)
    device = key_cache.device
    num_kv_heads = key_cache.shape[2]
    if isinstance(keep_token_indices_per_head, torch.Tensor):
        keep_tensor = keep_token_indices_per_head.to(device=device, dtype=torch.long)
    else:
        if not keep_token_indices_per_head:
            return 0
        keep_tensor = torch.as_tensor(
            keep_token_indices_per_head,
            device=device,
            dtype=torch.long,
        )
    if keep_tensor.ndim != 2:
        raise ValueError(
            f"keep_token_indices_per_head must be 2D, got shape={tuple(keep_tensor.shape)}"
        )
    if int(keep_tensor.shape[0]) != int(num_kv_heads):
        raise ValueError(
            "keep_token_indices_per_head head count mismatch: "
            f"expected {num_kv_heads}, got {int(keep_tensor.shape[0])}"
        )
    keep_count = int(keep_tensor.shape[1])
    if keep_count <= 0:
        return 0
    min_idx = int(keep_tensor.min().item())
    max_idx = int(keep_tensor.max().item())
    if min_idx < 0 or max_idx >= total_tokens:
        raise IndexError(
            f"keep token index out of range: min={min_idx}, max={max_idx}, "
            f"expected [0, {total_tokens})"
        )
    # TopK path yields per-row unique indices; enforce this assumption to avoid
    # expensive Python-side dedup per compression step.
    sorted_keep = torch.sort(keep_tensor, dim=1).values
    if keep_count > 1 and bool((sorted_keep[:, 1:] == sorted_keep[:, :-1]).any().item()):
        raise ValueError("per-head keep indices must not contain duplicates")

    if preserve_dropped_tokens:
        prefix = torch.arange(keep_count, device=device, dtype=torch.long).unsqueeze(0)
        prefix = prefix.expand(num_kv_heads, -1)
        if torch.equal(keep_tensor, prefix):
            # Identity permutation for all heads, no copy needed.
            return keep_count
        all_tokens = torch.arange(total_tokens, device=device, dtype=torch.long).unsqueeze(0)
        all_tokens = all_tokens.expand(num_kv_heads, -1)
        tail_mask = torch.ones(
            (num_kv_heads, total_tokens),
            device=device,
            dtype=torch.bool,
        )
        tail_mask.scatter_(1, keep_tensor, False)
        tail_tokens = all_tokens[tail_mask].view(num_kv_heads, total_tokens - keep_count)
        perm_tensor = torch.cat([keep_tensor, tail_tokens], dim=1)
        dst_tokens = torch.arange(total_tokens, device=device, dtype=torch.long)
        src_blocks, src_off = _resolve_token_slots(
            block_ids=block_ids,
            block_size=block_size,
            token_indices=perm_tensor,
            device=device,
        )
        head_idx = torch.arange(num_kv_heads, device=device, dtype=torch.long).unsqueeze(1)
        gathered_keys = key_cache[src_blocks, src_off, head_idx].clone()
        gathered_values = value_cache[src_blocks, src_off, head_idx].clone()

        dst_blocks, dst_off = _resolve_token_slots(
            block_ids=block_ids,
            block_size=block_size,
            token_indices=dst_tokens,
            device=device,
        )
        key_cache[dst_blocks, dst_off] = gathered_keys.permute(1, 0, 2).contiguous()
        value_cache[dst_blocks, dst_off] = gathered_values.permute(1, 0, 2).contiguous()
    else:
        src_tokens, dst_tokens_flat, head_idx = _build_fill_hole_placement_per_head(
            keep_tensor=keep_tensor,
            keep_count=keep_count,
            num_kv_heads=num_kv_heads,
            device=device,
        )
        if src_tokens.numel() == 0:
            return keep_count

        src_blocks, src_off = _resolve_token_slots(
            block_ids=block_ids,
            block_size=block_size,
            token_indices=src_tokens,
            device=device,
        )
        dst_blocks, dst_off = _resolve_token_slots(
            block_ids=block_ids,
            block_size=block_size,
            token_indices=dst_tokens_flat,
            device=device,
        )
        gathered_keys = key_cache[src_blocks, src_off, head_idx].clone()
        gathered_values = value_cache[src_blocks, src_off, head_idx].clone()
        key_cache[dst_blocks, dst_off, head_idx] = gathered_keys
        value_cache[dst_blocks, dst_off, head_idx] = gathered_values

    return keep_count


def _build_fill_hole_placement_shared(
    *,
    keep_tensor: torch.Tensor,
    keep_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build src/dst token placement for shared fill-hole compaction.

    The returned mapping moves only tail survivors into free prefix slots,
    intentionally allowing permutation (unordered prefix) to minimize copies.
    """
    in_prefix = keep_tensor < keep_count
    occupied_prefix = keep_tensor[in_prefix]
    free_mask = torch.ones(keep_count, device=device, dtype=torch.bool)
    if occupied_prefix.numel() > 0:
        free_mask[occupied_prefix] = False
    dst_tokens = torch.arange(keep_count, device=device, dtype=torch.long)[free_mask]
    src_tokens = keep_tensor[~in_prefix]
    if int(dst_tokens.numel()) != int(src_tokens.numel()):
        raise RuntimeError(
            "fill_in_place_mismatch: "
            f"dst={int(dst_tokens.numel())} src={int(src_tokens.numel())}"
        )
    return src_tokens, dst_tokens


def _build_fill_hole_placement_per_head(
    *,
    keep_tensor: torch.Tensor,
    keep_count: int,
    num_kv_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build flattened per-head fill-hole placement.

    Returns `(src_tokens, dst_tokens, head_idx)` with one move per row entry.
    """
    if keep_count <= 0 or num_kv_heads <= 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty, empty

    in_prefix = keep_tensor < keep_count
    src_mask = ~in_prefix

    prefix_positions = torch.arange(keep_count, device=device, dtype=torch.long)
    prefix_positions_2d = prefix_positions.unsqueeze(0).expand(num_kv_heads, -1)
    free_mask = torch.ones((num_kv_heads, keep_count), device=device, dtype=torch.bool)
    if bool(in_prefix.any().item()):
        head_ids_all = torch.arange(num_kv_heads, device=device, dtype=torch.long)
        head_ids_2d = head_ids_all.unsqueeze(1).expand(num_kv_heads, keep_count)
        occupied_heads = head_ids_2d[in_prefix]
        occupied_cols = keep_tensor[in_prefix]
        free_mask[occupied_heads, occupied_cols] = False

    src_counts = src_mask.sum(dim=1)
    dst_counts = free_mask.sum(dim=1)
    if not torch.equal(src_counts, dst_counts):
        mismatch = (src_counts != dst_counts).nonzero(as_tuple=False)
        head = int(mismatch[0].item()) if mismatch.numel() > 0 else -1
        src_n = int(src_counts[head].item()) if head >= 0 else -1
        dst_n = int(dst_counts[head].item()) if head >= 0 else -1
        raise RuntimeError(
            "fill_in_place_per_head_mismatch: "
            f"head={head} dst={dst_n} src={src_n}"
        )

    if not bool(src_mask.any().item()):
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty, empty

    head_idx_2d = torch.arange(num_kv_heads, device=device, dtype=torch.long).unsqueeze(1)
    head_idx_2d = head_idx_2d.expand(num_kv_heads, keep_count)
    return (
        keep_tensor[src_mask],
        prefix_positions_2d[free_mask],
        head_idx_2d[src_mask],
    )
