# Experiment 002: Multi-Bin Key Assignment Sanity Check

## Overview

Verify Multi-Bin Key Assignment loss functions with **softmax over keys (dim=0)**.

This experiment tests a fundamentally different approach from exp_001:
- Keys can belong to multiple bins (multi-bin membership)
- TopK inference selects top K keys from the predicted bin
- Two new loss functions designed for this setting

## Key Differences from exp_001

### Softmax Direction

| Aspect | exp_001 | exp_002 |
|--------|---------|---------|
| **Key softmax** | dim=1 (over bins) | dim=0 (over keys) |
| **Meaning** | Each key belongs to one bin | Each bin selects from all keys |
| **P[:, b] sums to** | Not applicable | 1.0 (probability over keys) |
| **P[k, :] sums to** | 1.0 (probability over bins) | Variable (not normalized) |

### Inference Method

| Aspect | exp_001 | exp_002 |
|--------|---------|---------|
| **Key selection** | All keys in same bin | TopK keys from bin |
| **Keys per query** | Variable (bin size) | Fixed (K parameter) |
| **Hit criterion** | Q and K in same bin (argmax) | K in TopK of Q's bin |

### Loss Functions

| exp_001 | exp_002 |
|---------|---------|
| Bidirectional CE + Linear Repel | Attraction Loss (NLL) |
| Bidirectional CE + Log Repel | Bidirectional CE (normalized) |
| Baseline (CE only) | - |

## Loss Functions

### Attraction Loss (NLL-based)

From [doc 06 Section 3.2](../../docs/06_multi_bin_key_assignment.md):

```
match_prob[q] = Σ_b p_q[b] * P[k(q), b]
loss = -log(match_prob).mean()
```

**Intuition**: Probability that the query's predicted bin selects its argmax key.

### Bidirectional Cross-Entropy (with Normalization)

From [doc 06 Section 3.4](../../docs/06_multi_bin_key_assignment.md):

Since `P[k, :]` does NOT sum to 1, we normalize first:

```
P_norm[k, :] = P[k, :] / Σ_b P[k, b]
loss = CE(p_q, P_norm[k(q)]) + CE(P_norm[k(q)], p_q)
```

**Implementation**: Uses log-space computation (logsumexp) for numerical stability.

## TopK Values

From [doc 06 Section 4.1](../../docs/06_multi_bin_key_assignment.md), test three K values:

| K | Keys per Query | Computation Reduction |
|---|----------------|----------------------|
| 50 | 50 | 99.17% |
| 500 | 500 | 91.67% |
| 1000 | 1000 | 83.33% |

## Usage

### Run with config file

```bash
cd weian_development/spec_sparse_simulator/experiments/exp_002_multi_bin_sanity_check
python run.py --config config.yaml
```

### Run with custom parameters

```bash
python run.py \
  --num_queries 6000 \
  --num_keys 6000 \
  --num_bins 128 \
  --epochs 1000 \
  --lr 0.01 \
  --topk_k 50 500 1000
```

### Run single TopK value

```bash
python run.py --topk_k 500
```

## Expected Output

```
output/
├── figures/
│   ├── loss_curves_*.png           # Loss convergence for all experiments
│   ├── topk_hit_rate_comparison_*.png  # Hit rate across K values
│   ├── metrics_comparison_*.png    # Overall metrics comparison
│   └── heatmap_*.png               # Bin distribution heatmaps
├── results/
│   └── metrics_*.json              # Numerical results
└── mock_data/
    └── mock_data_q6000_k6000.pt    # Cached Q-K relationships
```

## Results

### TopK Sweep Results

| K | Loss Function | TopK Hit Rate | Keys/Query | Comp. Reduction |
|---|---------------|---------------|------------|-----------------|
| 50 | attraction_nll | TODO | 50 | 99.17% |
| 50 | bidirectional_ce | TODO | 50 | 99.17% |
| 500 | attraction_nll | TODO | 500 | 91.67% |
| 500 | bidirectional_ce | TODO | 500 | 91.67% |
| 1000 | attraction_nll | TODO | 1000 | 83.33% |
| 1000 | bidirectional_ce | TODO | 1000 | 83.33% |

### Key Observations

**Expected trends**:
1. TopK Hit Rate should increase with K (more keys selected = higher chance of including argmax key)
2. Both loss functions should converge (attraction_nll may be simpler)
3. No need for repel term since TopK inherently limits selection

**Analysis**:
- TODO: Fill in after running experiments

## Reference Documents

- **Design Spec**: [docs/06_multi_bin_key_assignment.md](../../docs/06_multi_bin_key_assignment.md)
- **Reference Experiment**: [exp_001_sanity_check](../exp_001_sanity_check/)
- **Training & Labels**: [docs/04_training_and_labels.md](../../docs/04_training_and_labels.md)
