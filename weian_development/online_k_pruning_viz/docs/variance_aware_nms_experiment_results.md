# Variance-Aware NMS Experiment Results

## Overview

This document records the experimental results of the Variance-Aware NMS (Non-Maximum Suppression) approach for KV cache compression. This method uses Q-magnitude percentile weights with conservative weight selection, as described in `variance_aware_nms.md`.

## Experimental Setup

- **Trace**: `qid0003_trace34` (DeepSeek R1 Qwen3 8B)
- **Sequence Length**: 10,938 tokens
- **Heads**: 10 sampled heads from `hybrid_sample_heads_lowret_top10.json`
- **Parameters**: max_keys=2048, round_window=64, aggregation=mean
- **Epsilon**: Fixed at 0 (as per design)
- **Total K tokens**: 109,380 (seq_len × heads)

## Key Design Principles

### Conservative Weight Selection (Section 6.5)

The variance-aware NMS uses Q-magnitude percentile weights with conservative selection:
- **w_low** = low percentile of |Q| (e.g., 20th percentile)
- **w_high** = high percentile of |Q| (e.g., 80th percentile)

Selection rule:
- If `per_freq_score > 0` (contributes to suppression) → use **w_low** (underestimate contribution)
- If `per_freq_score ≤ 0` (opposes suppression) → use **w_high** (amplify resistance)

**IMPORTANT**: This requires `low_percentile ≤ high_percentile`. Violating this reverses the conservative principle and creates an aggressive strategy instead.

## Results Summary

### Valid Configurations (low_percentile ≤ high_percentile)

| 配置 | Retention | Δ Retention | Total Drops | 占所有 K % | 状态 |
|------|-----------|-------------|-------------|------------|------|
| baseline | **0.9668** | - | 0 | 0.00% | - |
| p20/p80 | **0.9668** | 0.0000 | 57 | 0.05% | ✅ 正确 |
| p30/p70 | **0.9668** | 0.0000 | 322 | 0.29% | ✅ 正确 |
| p40/p60 | **0.9665** | -0.0003 | 1,607 | 1.47% | ✅ 正确 |
| **p50/p50** | **0.9658** | **-0.0010** | **6,311** | **5.77%** | ✅ **推荐** |

### Invalid Configurations (low_percentile > high_percentile)

These configurations violate the conservative principle and should NOT be used:

| 配置 | Retention | Δ Retention | Total Drops | 占所有 K % | 状态 |
|------|-----------|-------------|-------------|------------|------|
| p55/p45 | 0.9640 | -0.0028 | 11,339 | 10.37% | ❌ 违反原则 |
| p60/p40 | 0.9613 | -0.0055 | 19,643 | 17.96% | ❌ 违反原则 |
| p65/p35 | 0.9531 | -0.0137 | 32,128 | 29.37% | ❌ 违反原则 |
| p70/p30 | 0.9346 | -0.0322 | 49,680 | 45.42% | ❌ 违反原则 |

**Note**: Invalid configurations now raise `ValueError` due to the assertion added in `compute_q_magnitude_percentile_weights()`.

## Comparison with Spectrum-Aware NMS

| Method | Retention | Δ Retention | Total Drops | 占所有 K % |
|--------|-----------|-------------|-------------|------------|
| Baseline | 0.9668 | - | 0 | 0.00% |
| **Variance-Aware p50/p50** | **0.9658** | **-0.0010** | **6,311** | **5.77%** |
| Spectrum-Aware (amplitude) | 0.9312 | -0.0356 | 61,140 | 55.90% |
| Spectrum-Aware (meanvec) | 0.8322 | -0.1346 | 101,174 | 92.50% |

## Per-Head Analysis (p50/p50)

| Layer | Head | Retention | Drops |
|-------|------|-----------|-------|
| 3 | 7 | 0.9995 | 5 |
| 9 | 19 | 0.9200 | 132 |
| 17 | 25 | 0.8139 | 20 |
| 24 | 0 | 0.9984 | 283 |
| 24 | 11 | 0.9976 | 2,251 |
| 24 | 14 | 0.9653 | 370 |
| 31 | 30 | 0.9753 | 135 |
| 32 | 27 | 1.0000 | 1,738 |
| 33 | 0 | 1.0000 | 772 |
| 34 | 6 | 0.9980 | 605 |

### Key Observations

1. **Layer 17 Head 25** has the lowest baseline retention (0.8139) and received minimal drops (20), showing the conservative approach protects vulnerable heads.

2. **Layer 24 heads** received the most drops (2,251 + 370 + 283 = 2,904), but maintained high retention, indicating these heads have more redundant keys.

3. **Layer 32-34 heads** with perfect/near-perfect retention can tolerate significant drops (1,738 + 772 + 605 = 3,115) without impact.

## Analysis

### Why Variance-Aware NMS Works Better

1. **Conservative by design**: The percentile-based weight selection ensures that suppression only happens when it's clearly justified across the Q-magnitude distribution.

2. **Protects vulnerable frequencies**: Using w_low for positive scores and w_high for negative scores creates a "worst-case" analysis that prevents over-aggressive suppression.

3. **No normalization issues**: With ε=0, the decision boundary is simply `score > 0`, avoiding the cross-head normalization problems that affected spectrum-aware NMS.

### Trade-off Analysis

| Configuration | Drop Rate | Retention Loss | Efficiency Ratio |
|---------------|-----------|----------------|------------------|
| p20/p80 | 0.05% | 0.00% | ∞ (perfect) |
| p30/p70 | 0.29% | 0.00% | ∞ (perfect) |
| p40/p60 | 1.47% | 0.03% | 49:1 |
| p50/p50 | 5.77% | 0.10% | 57.7:1 |

The p50/p50 configuration offers the best practical trade-off: dropping 5.77% of K tokens with only 0.10% retention loss.

## Conclusions

1. **Variance-aware NMS successfully preserves retention** while enabling meaningful KV compression, unlike spectrum-aware NMS which degraded retention significantly.

2. **The conservative weight selection principle is critical**: Configurations violating `low_percentile ≤ high_percentile` become aggressive strategies and cause retention degradation.

3. **Recommended configuration: p50/p50** (`--low-percentile 50 --high-percentile 50`):
   - Drops 5.77% of K tokens
   - Retention loss only 0.10% (0.9668 → 0.9658)
   - Safe margin well within 1% threshold

4. **For maximum safety, use p40/p60**:
   - Drops 1.47% of K tokens
   - Retention loss only 0.03%

## Files Generated

- `attention_pruning_case_study_hybrid_rounds_xtrace_nms_variance.py` - Main experiment script
- `run_nms_variance_p50_p50.sh` - Recommended experiment runner
- `results/nms_variance_*/` - Per-configuration metrics and visualizations

## Code Safeguards

An assertion was added to `compute_q_magnitude_percentile_weights()` to prevent invalid configurations:

```python
if low_percentile > high_percentile:
    raise ValueError(
        f"low_percentile ({low_percentile}) must be <= high_percentile ({high_percentile}). "
        f"Violating this breaks the conservative weight selection principle. "
        f"See docs/variance_aware_nms.md Section 6.5."
    )
```

## Future Work

1. Test on additional traces to validate generalization
2. Explore per-layer adaptive percentile settings
3. Investigate the relationship between head characteristics and optimal percentile values
4. Consider combining with other KV compression techniques
