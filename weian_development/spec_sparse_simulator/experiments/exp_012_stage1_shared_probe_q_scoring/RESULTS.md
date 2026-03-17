# Exp 012: Shared Probe Q Scoring - Full Experiment Log

## Overview

This experiment explores different scoring mechanisms for the shared probe architecture, progressing through multiple stages with iterative improvements.

## Stage Evolution

### Stage 2: Magnitude-Aware Scoring (Baseline)

**Architecture**:
- K-side: `s_K^(b) = P_rot_b^T * K + u_b^T * m^K + k_bias_b`
- Q-side: `s_Q^(b) = tilde_w_b^T * d_b + v_b^T * m^Q + c_b`
- Magnitude: `m_f = sqrt(x_{2f}^2 + x_{2f+1}^2)`

**Parameters**: 41,216 total

**Results (25 epochs, K-means init)**:
| K | Hit Rate |
|---|----------|
| 50 | 95.40% |
| 500 | 97.85% |
| 1000 | 98.24% |
| Loss | 0.105 |

**Miss Case Analysis (K=50, 100 epochs)**:
| Metric | Count | Percentage |
|--------|-------|------------|
| Total queries | 16,442 | 100% |
| Misses | 662 | 4.03% |
| Type A (Key Network) | 85 | 12.84% of misses |
| Type B (Query Network) | 577 | 87.16% of misses |

**Finding**: 87% of errors were Query Network routing issues.

---

### Stage 3: L2 Normalization (Symmetric)

**Key Changes**:
1. L2 normalize Q to unit norm before distance computation
2. L2 normalize probe vectors for both K and Q sides
3. Q-side magnitude sensing uses normalized Q
4. K-side magnitude sensing uses original K (unchanged)

**Architecture**:
- Added `l2_normalize(x)` helper: `x / (||x|| + eps)`
- K-side: Uses normalized probes, original K for magnitude
- Q-side: Uses normalized Q and normalized probes for distance, normalized Q for magnitude

**Results (25 epochs, non-normalized K-means init)**:
| K | Hit Rate |
|---|----------|
| 50 | 98.50% |
| 500 | 98.72% |
| 1000 | 98.91% |
| Loss | 0.072 |

**Miss Case Analysis (Test Set, K=50)**:
| Metric | Count | Percentage |
|--------|-------|------------|
| Total queries | 16,442 | 100% |
| Misses | 247 | 1.50% |
| Type A (Key Network) | 193 | 78.14% of misses |
| Type B (Query Network) | 54 | 21.86% of misses |

**Improvement over Stage 2**:
- Total misses reduced from 662 to 247 (62.7% reduction)
- Type B errors reduced from 577 to 54 (90.6% reduction)
- However, Type A errors increased from 85 to 193

---

### Stage 3 Attempt 1: Cosine Similarity (Rejected)

**Hypothesis**: Replace distance with cosine similarity for Q-side scoring.

**Change**:
- Distance: `d_{b,f} = ||P_rot_{b,f} - Q_f||`
- Cosine: `cos_{b,f} = P_rot_{b,f} · Q_normalized_f`

**Result**: K=50 dropped from 98.50% to 98.49%

**Decision**: Rolled back - no improvement.

---

### Stage 3 Attempt 2: Asymmetric Normalization (Rejected)

**Hypothesis**: K-side probe normalization causing Type A errors. Try normalizing only Q-side probes.

**Change**: K-side uses `normalize=False`, Q-side uses `normalize=True`

**Results**:
| K | Hit Rate |
|---|----------|
| 50 | 93.84% |

**Miss Case Analysis**:
| Metric | Count | Percentage |
|--------|-------|------------|
| Misses | 1,013 | 6.16% |
| Type A | 81 | 8% of misses |
| Type B | 932 | 92% of misses |

**Decision**: Rolled back - asymmetric normalization breaks Q-K consistency, causing massive routing errors.

---

### Stage 3c: Normalized K-means Initialization (Current)

**Hypothesis**: K-means initialization uses raw Q vectors, but network operates in normalized space. Align initialization with normalized space.

**Change**: Add L2 normalization before K-means clustering in `compute_kmeans_init.py`:
```python
Q_relative = apply_rope_rotation(round_Q, -ref_pos)
Q_relative = l2_normalize(Q_relative)  # NEW: normalize before K-means
```

**K-means Inertia**:
- Before (raw vectors): ~12,000
- After (normalized): 3,228 (vectors on unit sphere, tighter clusters)

**Results (25 epochs)**:

| Dataset | K=50 | K=500 | K=1000 | Loss |
|---------|------|-------|--------|------|
| Test Set | 98.49% | 98.61% | 98.70% | 0.037 |
| Train Set | 99.31% | 99.55% | 99.68% | 0.037 |

**Miss Case Analysis - Test Set (K=50)**:
| Metric | Count | Percentage |
|--------|-------|------------|
| Total queries | 16,442 | 100% |
| Misses | 249 | 1.51% |
| Type A (Key Network) | 184 | 73.90% of misses |
| Type B (Query Network) | 65 | 26.10% of misses |

**Miss Case Analysis - Train Set (K=50)**:
| Metric | Count | Percentage |
|--------|-------|------------|
| Total queries | 9,810 | 100% |
| Misses | 68 | 0.69% |
| Type A (Key Network) | 0 | 0% of misses |
| Type B (Query Network) | 68 | 100% of misses |

**Best Rank Statistics**:
| Statistic | Train Set | Test Set |
|-----------|-----------|----------|
| Min | 0 | 0 |
| Max | 3 | 5,052 |
| Mean | 0.22 | 561.22 |
| Median | 0.0 | 206.0 |

---

## Key Findings

### 1. L2 Normalization Impact
- Stage 3 L2 normalization reduced misses by 62.7% vs Stage 2
- Fixed Query Network routing (Type B: 577 → 54)
- But increased Key Network errors (Type A: 85 → 193)

### 2. Normalized K-means Initialization
- Loss improved significantly: 0.072 → 0.037 (48.6% reduction)
- Test set hit rate unchanged (~98.5%)
- Trade-off: Type A slightly reduced (193 → 184), Type B slightly increased (54 → 65)

### 3. Generalization Gap
- **Train set**: 99.31% hit rate, **0 Type A errors**
- **Test set**: 98.49% hit rate, 184 Type A errors
- Key Network achieves near-perfect performance on training data
- Generalization bottleneck is Key Network, not Query Network

### 4. Symmetric Normalization is Essential
- Asymmetric normalization (K-side raw, Q-side normalized) causes catastrophic routing failures
- Q and K networks must operate in the same space for consistency

### 5. Multi-Bin Strategy (from Stage 2)
- Top-4 bins × 200 keys = 99.51% hit rate (with 800 keys/query budget)
- Multi-bin consistently outperforms single-bin at same key budget

---

## Files

| File | Description |
|------|-------------|
| `model.py` | Stage 3 model with L2 normalization |
| `train.py` | Training script with K-means initialization |
| `evaluate.py` | Standard evaluation (top-1 bin) |
| `evaluate_topk_bins.py` | Multi-bin evaluation |
| `analyze_miss_cases.py` | Error analysis script |
| `compute_kmeans_init.py` | K-means probe initialization (normalized) |
| `config.yaml` | Experiment configuration |

---

## Ablation: Number of Probes

Testing whether fewer probes can achieve similar performance with reduced parameters.

| Probes | Parameters | K-means Inertia | Loss | K=50 | K=500 | K=1000 |
|--------|------------|-----------------|------|------|-------|--------|
| **128** | 41,216 | 3,228 | 0.037 | 98.49% | 98.61% | 98.70% |
| **64** | 20,608 | 3,825 | 0.046 | 98.50% | 98.66% | 98.77% |
| **32** | 10,304 | 5,110 | 0.057 | 98.49% | 98.59% | 98.76% |

**Findings**:
1. Hit rate is nearly identical across all probe counts (~98.5% at K=50)
2. 64 probes slightly outperforms 128 probes
3. Parameter reduction is significant: 32 probes uses 75% fewer parameters
4. Loss increases with fewer probes but doesn't translate to lower hit rate

**Recommendation**: Use **64 probes** as default - same performance with 50% fewer parameters.

---

## Conclusions

1. **L2 normalization is beneficial** - significantly reduces routing errors
2. **Initialization alignment matters less than expected** - normalized K-means reduces loss but doesn't improve hit rate
3. **Key Network generalization is the bottleneck** - perfect on training set, but errors on test set
4. **Probe count has minimal impact** - 32/64/128 probes all achieve ~98.5% hit rate
5. **Recommended config**: 64 probes with L2 normalization and normalized K-means init
