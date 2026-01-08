# Bug: Perhead Pruning Cross-Layer Grouping Issue

## Summary

The current per-head pruning implementation groups sampled attention heads by KV head index **across all layers**, resulting in 196 heads per KV head group. This makes the per-head mode produce nearly identical results to global mode.

## Affected Code

**File:** `R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py`
**Lines:** 432-458

```python
if self.use_per_head_pruning:
    # Group sampled attention heads by KV head
    kv_head_groups = {}
    for i, (layer, attn_head) in enumerate(self.sampled_heads):
        kv_head = attn_head // max(1, self.num_key_value_groups)  # ⚠️ IGNORES layer!
        if kv_head not in kv_head_groups:
            kv_head_groups[kv_head] = []
        kv_head_groups[kv_head].append(i)

    # For each KV head, aggregate scores and perform independent top-k selection
    for kv_head_idx in range(self.num_key_value_heads):
        indices = kv_head_groups[kv_head_idx]
        group_scores = per_head_scores[indices]  # [196, seq_len] - TOO MANY!
        aggregated = group_scores.max(dim=0).values  # [seq_len]
        keep_indices_for_head = aggregated.topk(keep_count).indices
```

## Model Configuration (Qwen-7B / DeepSeek-R1-Distill-Qwen-7B)

| Parameter | Value |
|-----------|-------|
| num_attention_heads | 28 |
| num_key_value_heads | 4 |
| num_key_value_groups | 7 (28 / 4) |
| num_layers | 28 |

## sampled_heads Structure

- Total: 784 sampled heads
- Format: `[(layer_0, head_0), (layer_0, head_1), ..., (layer_27, head_27)]`
- All 28 layers × 28 attention heads

## Current (Buggy) Grouping Result

The grouping uses `attn_head // 7` and **ignores layer index**:

| KV Head | Attention Heads | Layers | Total Sampled Heads |
|---------|----------------|--------|---------------------|
| 0 | 0, 1, 2, 3, 4, 5, 6 | all 28 | **196** |
| 1 | 7, 8, 9, 10, 11, 12, 13 | all 28 | **196** |
| 2 | 14, 15, 16, 17, 18, 19, 20 | all 28 | **196** |
| 3 | 21, 22, 23, 24, 25, 26, 27 | all 28 | **196** |

## Why Results Are Identical

### Perhead Mode
- Each KV head: `max` aggregation over 196 heads → top-k selection

### Global Mode
- All heads: `max` aggregation over 784 heads → top-k selection

### Mathematical Equivalence

Due to the `max` aggregation property:
```
max(196 heads in KV group 0) ≈ contribution to max(784 heads)
```

When you take `max` over a large enough subset (196 out of 784), the result is statistically very similar to the global max. This makes the per-head selection produce nearly identical token sets as global selection.

## Root Cause

The grouping logic at line 436:
```python
kv_head = attn_head // max(1, self.num_key_value_groups)
```

This only considers `attn_head` index and completely ignores `layer` index, causing all 28 layers to be mixed into each KV head group.

## Expected Behavior

Per-head pruning should allow each KV head to select **different** tokens based on its specific attention patterns. The current design dilutes this by aggregating across too many (196) attention heads.

## Potential Fixes

1. **Per-layer per-head**: Only group attention heads within the same layer
2. **Single-layer sampling**: Use statistics from a representative layer instead of all layers
3. **Reduced aggregation**: Use fewer sampled heads per KV head
4. **Different aggregation**: Change from `max` to something that preserves per-head variance

## Reproduction

Run experiments with and without `--per-head-pruning`:
```bash
# Without perhead
bash run_speckv_aime24_qwen_norm_aligned_budget.sh

# With perhead
bash run_speckv_aime24_qwen_norm_aligned_budget_perhead.sh
```

Compare results - they will be nearly identical due to this bug.

## Date Discovered

2026-01-07

## Status

**RESOLVED** - Fixed in `speckv_rkv_style.py` on 2026-01-08.

## Fix Applied

Changed aggregation strategy in `_select_per_head_independent()`:
- **Before**: `max` over 196 heads (all 28 layers mixed) → nearly identical to global mode
- **After**: `mean(max(7 heads per layer))` → each layer contributes equally

Key changes:
1. Group by `(layer, kv_head)` tuple instead of just `kv_head`
2. Compute `max` within each layer's 7-head group
3. Average the per-layer max scores across layers

This preserves per-head variance while maintaining the 2D return format for KV cache gather.
