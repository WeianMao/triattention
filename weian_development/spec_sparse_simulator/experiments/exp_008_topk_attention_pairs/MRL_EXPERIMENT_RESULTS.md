# Exp 010: Masked Ranking Loss (MRL) Ablation Results

## Overview

This experiment implements and evaluates **Masked Ranking Loss (MRL)** as an alternative to the original Attraction Loss in exp_008. MRL constructs K independent training scenarios where each scenario's target is "hit top-1 among remaining keys" by progressively masking higher-ranked keys.

## Configuration

- **Loss Type**: `mrl` (Masked Ranking Loss)
- **K Values Tested**: 1, 2, 4, 8, 16, 32, 64
- **Training**: 25 epochs, lr=0.001, Adam optimizer
- **Evaluation**: TopK Hit Rate for K=50, 500, 1000

## Results

| K | K=50 Hit% | K=500 Hit% | K=1000 Hit% | Active Bins | Entropy |
|---|-----------|------------|-------------|-------------|---------|
| 1 | 98.62% | 98.92% | 99.16% | 1/128 | 0.0083 |
| 2 | 98.68% | 99.14% | 99.35% | 7/128 | 0.0924 |
| 4 | 98.71% | 99.14% | 99.29% | 8/128 | 0.1304 |
| 8 | 98.74% | 99.34% | 99.53% | 10/128 | 0.3011 |
| **16** | **98.82%** | **99.36%** | **99.60%** | 13/128 | 0.4313 |
| 32 | 98.64% | 99.28% | 99.45% | 16/128 | 0.5090 |
| 64 | 96.28% | 97.91% | 98.61% | 19/128 | 0.5472 |

## Key Findings

### 1. Hit Rate Performance
- **Best configuration**: K=16 achieves highest hit rates (98.82% / 99.36% / 99.60%)
- K=8 is also excellent (98.74% / 99.34% / 99.53%)
- K=64 shows degraded hit rate (~2% drop) due to optimization complexity

### 2. Bin Collapse Mitigation
- **K=1**: Severe collapse (1/128 active bins, Entropy=0.0083)
- **K=16**: Good diversity (13/128 active bins, Entropy=0.4313)
- **K=64**: Best diversity (19/128 active bins, Entropy=0.5472)
- Larger K values significantly improve bin diversity

### 3. Numerical Stability Fix
Initial implementation had NaN issues for K>16 due to division by near-zero values.

**Root Cause**: When logit scale increases during training, top-K keys dominate probability mass, causing `remaining_prob = 1.0 - sum(masked_probs)` to approach zero.

**Fix Applied**:
```python
# 1. Increased eps: 1e-8 → 1e-6
# 2. Clamp remaining_prob
remaining_prob = remaining_prob.clamp(min=eps)
# 3. Clamp normalized values
normalized_key_prob = normalized_key_prob.clamp(max=1.0)
```

## Comparison with Attraction Loss (exp_008)

| Method | K | K=50 Hit% | K=1000 Hit% | Active Bins |
|--------|---|-----------|-------------|-------------|
| Attraction | 1 | 98.62% | 99.16% | 1/128 |
| Attraction | 10 | 98.68% | 99.26% | 16/128 |
| **MRL** | **16** | **98.82%** | **99.60%** | 13/128 |
| MRL | 64 | 96.28% | 98.61% | 19/128 |

MRL achieves comparable or slightly better hit rate with similar bin diversity characteristics.

## Conclusions

1. **MRL is viable**: Achieves comparable performance to Attraction Loss
2. **Optimal K=16**: Best balance of hit rate and bin diversity
3. **Trade-off exists**: K>32 improves bin diversity but reduces hit rate
4. **Numerical stability**: Requires careful eps and clamping for large K values

## Usage

```bash
# Run MRL with specific K value
python run.py --mode ablation_mrl --topk_values 16

# Run full ablation
python run.py --mode ablation_mrl --topk_values 1,2,4,8,16,32,64
```

## Files Modified

- `train.py`: Added `compute_masked_ranking_loss()` function with numerical stability fix
- `config.yaml`: Added `loss_type` parameter ("attraction" or "mrl")
- `run.py`: Added `ablation_mrl` mode

---
*Experiment completed: 2024-12-18*
