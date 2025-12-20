# Exp 012 Stage 2: Magnitude-Aware Scoring Results

## Model Architecture

**Stage 2 完整模长感知评分**：
- K-side: `s_K^(b) = P_rot_b^T * K + u_b^T * m^K + k_bias_b`
- Q-side: `s_Q^(b) = tilde_w_b^T * d_b + v_b^T * m^Q + c_b`
- Magnitude: `m_f = sqrt(x_{2f}^2 + x_{2f+1}^2)`

### Parameters (41,216 total)
| Component | Shape | Count |
|-----------|-------|-------|
| Shared probes P | 128 × 128 | 16,384 |
| Q distance weights w | 128 × 64 | 8,192 |
| Q magnitude weights v | 128 × 64 | 8,192 |
| Q bias c | 128 | 128 |
| K magnitude weights u | 128 × 64 | 8,192 |
| K bias | 128 | 128 |

## Training Results

### Stage 1 vs Stage 2 Comparison (25 epochs, K-means init)

| K | Stage 1 | Stage 2 | Improvement |
|---|---------|---------|-------------|
| 50 | 95.38% | 95.40% | +0.02% |
| 500 | 97.35% | 97.85% | +0.50% |
| 1000 | 97.83% | 98.24% | +0.41% |
| Loss | 0.118 | 0.105 | -11.6% |

### 100 Epochs Training

| K | 25 epochs | 100 epochs | Change |
|---|-----------|------------|--------|
| 50 | 95.40% | 95.97% | +0.57% |
| 500 | 97.85% | 97.28% | -0.57% |
| 1000 | 98.24% | 97.57% | -0.67% |
| Loss | 0.105 | 0.086 | -18% |

**Note**: 100 epochs shows signs of overfitting - K=50 improves but K=500/1000 degrade.

## Miss Case Analysis (K=50, 100 epochs model)

### Summary
| Metric | Count | Percentage |
|--------|-------|------------|
| Total queries | 16,442 | 100% |
| Recent hits | 426 | 2.59% |
| Bin hits | 15,354 | 93.38% |
| **Misses** | **662** | **4.03%** |

### Error Type Classification
| Type | Count | % of Misses | Description |
|------|-------|-------------|-------------|
| Type A | 85 | 12.84% | Key Network issue - argmax ranks poorly in ALL bins |
| Type B | 577 | **87.16%** | Query Network issue - argmax ranks well in another bin |

**Key Finding**: 87% of errors are Query Network routing issues. The argmax key exists in a good bin (median best_rank = 0), but the Query network selects the wrong bin.

## Multi-Bin Evaluation

### K=50 keys per bin, varying number of bins
| Bins | Hit Rate | Keys/Query |
|------|----------|------------|
| 1 | 95.97% | 50 |
| 2 | 98.66% | 100 |
| 4 | 99.14% | 200 |
| 8 | 99.25% | 400 |
| 16 | 99.34% | 800 |
| 32 | 99.39% | 1600 |

### Fixed 800 keys budget
| Configuration | Hit Rate |
|---------------|----------|
| 800 keys × 1 bin | 97.51% |
| 400 keys × 2 bins | 99.26% |
| **200 keys × 4 bins** | **99.51%** |
| 100 keys × 8 bins | 99.43% |
| 50 keys × 16 bins | 99.34% |

**Optimal**: 4 bins × 200 keys achieves 99.51% hit rate with 800 keys/query.

## Conclusions

1. **Stage 2 magnitude features provide marginal improvement** over Stage 1 (+0.5% at K=500/1000)

2. **Query network is the bottleneck**: 87% of misses are routing errors where the argmax key ranks well in another bin

3. **Multi-bin strategy is highly effective**:
   - Top-16 bins: 99.34% (vs 95.97% for top-1)
   - Optimal: 4 bins × 200 keys = 99.51%
   - Multi-bin outperforms single-bin even with same total key budget

4. **Training recommendations**:
   - 25 epochs is optimal; 100 epochs causes overfitting
   - K-means initialization is essential (solved bin collapse)

## Files

- `model.py`: Stage 2 model with magnitude features
- `train.py`: Training script with K-means initialization
- `evaluate.py`: Standard evaluation (top-1 bin)
- `evaluate_topk_bins.py`: Multi-bin evaluation
- `analyze_miss_cases.py`: Error analysis script
- `compute_kmeans_init.py`: K-means probe initialization
