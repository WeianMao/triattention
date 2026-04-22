"""Token-level KV cache gather, compact-in-place, and slot reclamation.

Unlike the vLLM integration which operates on 5-D paged blocks, sglang
uses a flat 3-D token-level KV pool (``[total_tokens, H, D]``).  This
makes gather and compact operations significantly simpler — they reduce
to ``index_select`` and ``index_put`` on the pool buffers.

Design note — per-head pruning
------------------------------
TriAttention supports per-head scoring where different KV heads may
rank tokens differently.  However, sglang's KV pool stores all heads
for a given token slot in the same row (``k_buffer[layer][slot] →
[H, D]``).  This means physically moving data at the *slot* level
always moves *all* heads at once.

For Phase 1 we implement **unified compaction** only: all heads share
the same set of kept token positions.  Per-head scoring still happens
upstream (in the compressor/selector), but the final keep indices
passed here must be a single 1-D tensor representing the union /
intersection of per-head decisions.

If true per-head physical separation is needed (different heads
keeping different subsets and wanting independent slot layouts), it
would require either (a) separate KV buffers per head, or (b) a
virtual indexing layer on top of ``req_to_token``.  Both are
significant architectural changes recorded in ``decisions/pending.md``
as P-019.

Public functions
----------------
gather_request_k_dense
    Extract a dense key tensor for a single request from the token pool.
compact_request_kv_in_place
    Re-arrange retained KV entries to form a contiguous prefix (unified,
    all heads share the same kept positions).
compact_request_kv_in_place_per_head
    Re-arrange retained KV entries per-head: each KV head independently
    keeps its own top-K token positions via advanced indexing.
reclaim_freed_slots
    Return evicted token slots to the pool allocator.
"""

from __future__ import annotations

from typing import List, Optional

import torch


def gather_request_k_dense(
    k_buffers: List[torch.Tensor],
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    effective_len: int,
    target_layers: Optional[List[int]] = None,
) -> torch.Tensor:
    """Gather all key vectors for a request into a dense tensor.

    Reads the token-slot mapping for this request out of the
    ``req_to_token`` table, then uses those slot indices to index into
    each layer's ``k_buffer``.

    Parameters
    ----------
    k_buffers : list[Tensor]
        Per-layer key cache tensors, each of shape
        ``[total_tokens, num_kv_heads, head_dim]``.
    req_to_token : Tensor
        Shape ``[max_reqs, max_context_len]``, dtype int32.
        ``req_to_token[req_pool_idx, i]`` gives the physical slot
        index in the KV pool for the *i*-th token of this request.
    req_pool_idx : int
        Row index in ``req_to_token`` for the target request.
    effective_len : int
        Number of valid (non-evicted) token positions for this
        request.  Only the first ``effective_len`` entries of the
        ``req_to_token`` row are meaningful.
    target_layers : list[int] or None
        If provided, only gather from these layer indices.  If None,
        gather from all layers.

    Returns
    -------
    Tensor
        Dense key tensor of shape
        ``[num_layers, num_kv_heads, effective_len, head_dim]``.

        The ``num_kv_heads`` and ``head_dim`` dimensions come from
        ``k_buffers[0].shape[1:]``.  The ``num_layers`` dimension
        corresponds to ``len(target_layers)`` or ``len(k_buffers)``.

        This tensor is freshly allocated contiguous memory suitable
        for passing to the scoring kernel (which expects
        ``[batch, num_kv_heads, seq_len, head_dim]`` — the caller
        should unsqueeze a batch dimension if needed).
    """
    if effective_len <= 0:
        raise ValueError(
            f"effective_len must be > 0, got {effective_len}"
        )

    # effective_len upper bound validation.
    max_context_len = req_to_token.shape[1]
    if effective_len > max_context_len:
        raise ValueError(
            f"effective_len ({effective_len}) exceeds req_to_token "
            f"column count ({max_context_len})"
        )

    # Physical slot indices for this request's valid tokens.
    slot_indices = req_to_token[req_pool_idx, :effective_len].long()

    # Slot index bounds check.
    pool_size = k_buffers[0].shape[0]
    slot_min = int(slot_indices.min().item())
    slot_max = int(slot_indices.max().item())
    if slot_min < 0 or slot_max >= pool_size:
        raise ValueError(
            f"slot_indices out of bounds: min={slot_min}, max={slot_max}, "
            f"k_buffer pool size={pool_size}"
        )

    layers = target_layers if target_layers is not None else list(range(len(k_buffers)))
    num_layers = len(layers)

    # Infer shapes from the first buffer.
    sample_buf = k_buffers[layers[0]]
    num_kv_heads = sample_buf.shape[1]
    head_dim = sample_buf.shape[2]

    # Allocate output: [num_layers, num_kv_heads, effective_len, head_dim]
    out = torch.empty(
        (num_layers, num_kv_heads, effective_len, head_dim),
        dtype=sample_buf.dtype,
        device=sample_buf.device,
    )

    for out_idx, layer_idx in enumerate(layers):
        # k_buffers[layer_idx] is [total_tokens, H, D].
        # index_select on dim=0 gathers [effective_len, H, D].
        gathered = k_buffers[layer_idx].index_select(0, slot_indices)
        # Transpose to [H, effective_len, D] then write into output.
        out[out_idx] = gathered.permute(1, 0, 2)

    return out


def compact_request_kv_in_place(
    k_buffers: List[torch.Tensor],
    v_buffers: List[torch.Tensor],
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    keep_indices: torch.Tensor,
    effective_len: int,
    target_layers: Optional[List[int]] = None,
) -> int:
    """Compact retained KV entries in-place within the token pool.

    After compaction the first ``budget`` slots of the request's
    ``req_to_token`` row contain the retained entries in their original
    relative order (order must be preserved).

    The operation is logically:

    1. Map ``keep_indices`` (positions within the request's logical
       sequence) to physical slot indices via ``req_to_token``.
    2. For each layer, copy the kept K and V data from their current
       slots into the first ``budget`` slots of the request's mapping.
    3. Update ``req_to_token`` so the first ``budget`` entries point
       to the slots now holding the kept data.

    Parameters
    ----------
    k_buffers, v_buffers : list[Tensor]
        Per-layer key/value cache tensors, each of shape
        ``[total_tokens, num_kv_heads, head_dim]``.
    req_to_token : Tensor
        Shape ``[max_reqs, max_context_len]``, dtype int32.
    req_pool_idx : int
        Row index in ``req_to_token`` for the target request.
    keep_indices : Tensor
        1-D tensor of token *positions* (within the request, 0-indexed)
        to retain, length ``budget``.  Must be sorted in ascending
        order to preserve the original token order.  All values must
        be in ``[0, effective_len)``.
    effective_len : int
        Current number of valid tokens before compaction.
    target_layers : list[int] or None
        If provided, only compact these layer indices.  If None,
        compact all layers.

    Returns
    -------
    int
        The number of retained tokens (``budget``), equal to
        ``len(keep_indices)``.
    """
    budget = keep_indices.shape[0]
    if budget <= 0:
        return 0
    if budget > effective_len:
        raise ValueError(
            f"budget ({budget}) exceeds effective_len ({effective_len})"
        )

    device = k_buffers[0].device

    # Ensure keep_indices is sorted (order-preserving invariant).
    keep_indices = keep_indices.to(device=device, dtype=torch.long)
    if budget > 1:
        sorted_ki = torch.sort(keep_indices).values
        if not torch.equal(keep_indices, sorted_ki):
            raise ValueError(
                "keep_indices must be sorted in ascending order "
                "to preserve original token order"
            )

    # Validate range.
    if int(keep_indices.min().item()) < 0:
        raise IndexError("keep_indices contains negative values")
    if int(keep_indices.max().item()) >= effective_len:
        raise IndexError(
            f"keep_indices max ({int(keep_indices.max().item())}) "
            f">= effective_len ({effective_len})"
        )

    # Map request-local positions to physical slot indices.
    all_slots = req_to_token[req_pool_idx, :effective_len].long().to(device)
    kept_slots = all_slots[keep_indices]  # [budget]

    # Destination slots: the first ``budget`` entries of the mapping.
    dst_slots = all_slots[:budget]  # [budget]

    # Check for identity: if the kept tokens are already at positions
    # 0..budget-1, no data movement is needed.
    identity_positions = torch.arange(budget, device=device, dtype=torch.long)
    if torch.equal(keep_indices, identity_positions):
        # Already a contiguous prefix — no copies needed.
        # But we still need to update req_to_token for the tail.
        # (Tail clearing is done by reclaim_freed_slots.)
        return budget

    layers = target_layers if target_layers is not None else list(range(len(k_buffers)))

    # Copy kept data to destination slots. We do this in two phases to
    # handle potential overlap between source and destination slot sets:
    # first read all kept data into a temporary buffer, then write back.
    for layer_idx in layers:
        k_buf = k_buffers[layer_idx]
        v_buf = v_buffers[layer_idx]

        # Read kept data: [budget, H, D]
        k_kept = k_buf[kept_slots].clone()
        v_kept = v_buf[kept_slots].clone()

        # Write to destination slots.
        k_buf[dst_slots] = k_kept
        v_buf[dst_slots] = v_kept

    # Update the req_to_token mapping: first ``budget`` entries now
    # point to ``dst_slots`` (which were already in those positions,
    # but the KV data in those slots has been overwritten with kept
    # data).  The mapping values themselves don't change for the
    # first ``budget`` entries — the physical slots stay the same,
    # only their *content* has been updated.
    #
    # Actually, we need to reconsider: the destination slots are
    # req_to_token[req_pool_idx, 0:budget] — those slots already
    # exist in the mapping.  After overwriting their content with
    # the kept tokens' data, these slots now logically represent the
    # kept tokens.  The mapping doesn't need to change for positions
    # 0..budget-1.
    #
    # The tail (positions budget..effective_len-1) will be handled
    # by reclaim_freed_slots.

    return budget


# Per-head compaction: each KV head independently retains its own set
# of token positions within shared physical slots.
def compact_request_kv_in_place_per_head(
    k_buffers: List[torch.Tensor],
    v_buffers: List[torch.Tensor],
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    keep_indices_per_head: torch.Tensor,  # [num_kv_heads, budget]
    effective_len: int,
    target_layers: Optional[List[int]] = None,
) -> int:
    """Compact retained KV entries in-place with per-head token selection.

    Unlike ``compact_request_kv_in_place`` where all heads share the
    same set of kept token positions, this function allows each KV head
    to independently retain its own top-K positions.  After compaction,
    ``dst_slots[i]`` for head *h* contains the KV data of whatever
    original token head *h* selected at rank *i*.  Different heads in
    the same slot may therefore originate from different tokens.

    The attention kernel is unaffected because it computes per-head dot
    products independently — it never assumes that all heads in a slot
    originate from the same token.

    Parameters
    ----------
    k_buffers, v_buffers : list[Tensor]
        Per-layer key/value cache tensors, each ``[total_tokens, H, D]``.
    req_to_token : Tensor
        ``[max_reqs, max_context_len]``, dtype int32.
    req_pool_idx : int
        Row in ``req_to_token`` for the target request.
    keep_indices_per_head : Tensor
        Shape ``[num_kv_heads, budget]``.  Each row contains the token
        positions (0-indexed, sorted ascending) that head *h* retains.
        Must be dtype int64.
    effective_len : int
        Number of valid tokens before compaction.
    target_layers : list[int] or None
        If provided, only compact these layers.  Default: all layers.

    Returns
    -------
    int
        The budget (``keep_indices_per_head.shape[1]``).
    """
    # Dtype guard — downstream advanced indexing requires int64.
    device = k_buffers[0].device
    keep_indices_per_head = keep_indices_per_head.to(
        device=device, dtype=torch.int64
    )

    num_kv_heads = keep_indices_per_head.shape[0]
    budget = keep_indices_per_head.shape[1]

    if budget <= 0:
        return 0
    if budget > effective_len:
        raise ValueError(
            f"budget ({budget}) exceeds effective_len ({effective_len})"
        )

    # Validate range for every head.
    ki_min = int(keep_indices_per_head.min().item())
    ki_max = int(keep_indices_per_head.max().item())
    if ki_min < 0:
        raise IndexError("keep_indices_per_head contains negative values")
    if ki_max >= effective_len:
        raise IndexError(
            f"keep_indices_per_head max ({ki_max}) "
            f">= effective_len ({effective_len})"
        )

    # Validate sorted ascending for each head.
    if budget > 1:
        sorted_ki = torch.sort(keep_indices_per_head, dim=-1).values
        if not torch.equal(keep_indices_per_head, sorted_ki):
            raise ValueError(
                "Each row of keep_indices_per_head must be sorted "
                "ascending to preserve original token order"
            )

    # Map request-local positions to physical slot indices.
    all_slots = req_to_token[req_pool_idx, :effective_len].long().to(device)
    dst_slots = all_slots[:budget]  # [budget]

    # Identity check — if every head keeps positions 0..budget-1, no
    # data movement is needed.
    identity_positions = torch.arange(budget, device=device, dtype=torch.int64)
    identity_row = identity_positions.unsqueeze(0).expand(num_kv_heads, -1)
    if torch.equal(keep_indices_per_head, identity_row):
        return budget

    layers = (
        target_layers
        if target_layers is not None
        else list(range(len(k_buffers)))
    )

    # Step A — map per-head position indices to slot indices.
    kept_slots_per_head = all_slots[keep_indices_per_head]  # [H, budget]

    # Build head index tensor for advanced indexing.
    head_idx = torch.arange(
        num_kv_heads, device=device
    ).unsqueeze(1).expand(-1, budget)  # [H, budget]

    # Destination slot tensor expanded for per-head scatter.
    dst_expanded = dst_slots.unsqueeze(0).expand(num_kv_heads, -1)  # [H, budget]

    for layer_idx in layers:
        k_buf = k_buffers[layer_idx]  # [total_tokens, H, D]
        v_buf = v_buffers[layer_idx]

        # Step B — per-head gather via advanced indexing.
        # k_buf[kept_slots_per_head[h,i], head_idx[h,i], :] selects
        # head h's data from the slot that head h wants to keep at rank i.
        k_gathered = k_buf[kept_slots_per_head, head_idx, :].clone()
        v_gathered = v_buf[kept_slots_per_head, head_idx, :].clone()

        # Step C — per-head scatter back to dst slots.
        k_buf[dst_expanded, head_idx, :] = k_gathered
        v_buf[dst_expanded, head_idx, :] = v_gathered

    return budget


def reclaim_freed_slots(
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    budget: int,
    effective_len: int,
    allocator: object,
) -> torch.Tensor:
    """Free evicted token slots back to the pool allocator.

    Releases ``req_to_token[req_pool_idx, budget:effective_len]`` via
    the allocator's ``free()`` method and zeroes the vacated entries
    to prevent dangling references.

    Parameters
    ----------
    req_to_token : Tensor
        Shape ``[max_reqs, max_context_len]``, dtype int32.
    req_pool_idx : int
        Row index for the target request.
    budget : int
        Number of retained tokens (the compaction frontier).
    effective_len : int
        Number of valid tokens before compaction.
    allocator : object
        Token pool allocator with a ``free(indices: Tensor)`` method.
        In sglang this is ``TokenToKVPoolAllocator``.

    Returns
    -------
    Tensor
        The freed slot indices (for diagnostics or logging).
        Empty tensor if nothing was freed.
    """
    if budget >= effective_len:
        # Nothing to free — no compression happened or budget equals
        # effective length.
        return torch.empty(0, dtype=torch.int64, device=req_to_token.device)

    # Slot indices that are being released.
    freed_slots = req_to_token[req_pool_idx, budget:effective_len].long()

    if freed_slots.numel() == 0:
        return freed_slots

    # Filter out zero-valued slots to prevent double-free.
    # Slot 0 is the padded/dummy slot in sglang's allocator (reserved during
    # clear()), and zeroed entries in req_to_token indicate already-reclaimed
    # slots.  Passing them to allocator.free() would corrupt the free list.
    nonzero_mask = freed_slots != 0
    freed_slots = freed_slots[nonzero_mask]

    if freed_slots.numel() == 0:
        return freed_slots

    # Return slots to the allocator.
    allocator.free(freed_slots)

    # Zero out the released entries in req_to_token to prevent
    # dangling references.  This is a safety measure — subsequent
    # code should respect ``effective_len`` boundaries, but zeroing
    # provides defense in depth.
    req_to_token[req_pool_idx, budget:effective_len] = 0

    return freed_slots
