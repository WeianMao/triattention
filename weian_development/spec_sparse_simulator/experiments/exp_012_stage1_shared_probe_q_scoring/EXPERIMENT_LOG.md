# Experiment Log: L2 Normalization Ablation Study

## Background

Module 2 (learned probe scoring) was underperforming compared to Hybrid Frequency baseline.
This log documents the investigation to find the root cause.

## Key Finding: L2 Normalization Hurts Performance

The original Module 2 design applied L2 normalization to:
1. Probe vectors before RoPE rotation
2. Query vectors before distance computation

This differs from Hybrid Frequency which preserves the original amplitude information.

## Experiment Results (num_bins=1, initialization only, no training)

| Setting | K=50 | K=500 | K=1000 |
|---------|------|-------|--------|
| **With L2 norm** | 50.51% | 96.14% | 98.77% |
| **Without L2 norm** | 86.61% | 98.69% | 98.95% |
| Hybrid Frequency baseline | 99.05% | 99.60% | 99.76% |

**K=50 improved by ~36 percentage points** after removing L2 normalization.

## Code Changes

### `model.py`
- Added `use_l2_norm` parameter to:
  - `DistanceBasedQueryScorer.__init__()` and `forward()`
  - `Module2KeyNetwork.__init__()` and `forward()`
  - `Module2QueryNetwork.__init__()`
  - `Module2Network.__init__()`
  - `create_model()`
- Default is `use_l2_norm=True` for backward compatibility

### `compute_kmeans_init.py`
- Added `use_l2_norm` parameter to `compute_kmeans_init()`
- Controls whether K-means is run on L2-normalized or raw Q_relative vectors

### `test_init_only.py`
- Added `use_l2_norm` toggle in initialization section
- Default set to `False` for new experiments

## Why L2 Normalization Hurts

Module 2 formula (with L2 norm):
```
score = P_normalized_rot · K + u · m^K + bias
```

Hybrid Frequency formula (no L2 norm):
```
score = Σ_f |Q_mean_f| * |K_f| * cos(phi_f + delta * omega_f) + extra
```

The key difference: Hybrid Frequency preserves `|Q_mean_f|` (per-frequency amplitude),
while L2 normalization compresses this information across all frequencies.

## Next Steps

1. Test with num_bins=128 without L2 norm
2. Explore different probe initialization methods
3. Potentially train the model without L2 norm

## Timeline

- 2025-12-20: Identified L2 normalization as key performance bottleneck
