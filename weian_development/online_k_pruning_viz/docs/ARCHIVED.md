# Archived Scripts

This document records the scripts that were deleted from this directory and their differences from the kept script `attention_pruning_case_study_hybrid_rounds_xtrace.py`.

## Kept Script

**`attention_pruning_case_study_hybrid_rounds_xtrace.py`** - The canonical version with:
- Cross-trace statistics (`--stats-trace` for using different trace's Q statistics)
- Round-based cache maintenance (prune every `--prune-every` tokens)
- Geometric offset grid for future position scoring
- Hybrid scoring: `base_scores = (amp * cos_phase).sum() + extra.sum()`
- Extra term: `(q_abs_mean - q_mean_abs) * k_abs`

---

## Deleted Scripts

### 1. `attention_pruning_case_study.py`
**Difference**: Most basic version
- Uses current trace Q for statistics (no cross-trace support)
- Per-query scoring (not round-based)
- No extra term in scoring formula
- Scoring: `scores = (amp * torch.cos(phase)).sum()`

### 2. `attention_pruning_case_study_hybrid.py`
**Difference**: Added hybrid scoring
- Introduced extra term: `extra = (q_abs_mean - q_mean_abs) * k_abs`
- Still per-query scoring (not round-based)
- No cross-trace support

### 3. `attention_pruning_case_study_hybrid09.py`
**Difference**: Variant with coefficient 1.1
- Modified boost calculation: `boost = (1.1 * q_abs_mean - q_mean_abs).clamp_min(0.0)`
- Per-query scoring
- No cross-trace support

### 4. `attention_pruning_case_study_hybrid_sampled.py`
**Difference**: Added head sampling
- Added `--heads-json` for loading sampled heads from JSON file
- Added retention metrics aggregation
- Per-query scoring
- No cross-trace support

### 5. `attention_pruning_case_study_hybrid_sampled_xtrace.py`
**Difference**: First version with cross-trace support
- Added `--stats-trace` argument for cross-trace Q statistics
- Per-query scoring (not round-based)
- Intermediate step before round-based version

### 6. `attention_pruning_case_study_hybrid_rounds_gaussian.py`
**Difference**: Gaussian distribution modeling
- Full Gaussian stats with `GaussianFrequencyStats` dataclass (mean, covariance)
- Quantile-based scoring: `mean_term + sqrt(variance) * alpha`
- Added `--quantile-p` parameter for tail probability
- Uses `scipy.stats.norm.ppf()` for quantile computation

### 7. `attention_pruning_case_study_hybrid_rounds_gaussian_combined.py`
**Difference**: "Method 5.2" - Combined Gaussian
- Sums frequency components first, then applies Gaussian quantile
- Different aggregation order than `_gaussian.py`
- Key function: `score_keys_for_round_gaussian_combined()`

### 8. `attention_pruning_case_study_hybrid_rounds_xtrace_masked.py`
**Difference**: High-frequency masking
- Masks frequency components where `period < delta * 0.5`
- Core logic:
  ```python
  safe_periods = (2 * math.pi) / omega_abs
  freq_mask = period_grid >= delta_threshold
  cos_phase = torch.cos(phase) * freq_mask
  ```
- Fixed threshold scale at 0.5

### 9. `attention_pruning_case_study_hybrid_rounds_xtrace_masked_tuned.py`
**Difference**: Tunable masking parameters
- Added `--period-threshold-scale` (default 0.5, configurable)
- Added `--mask-extra-term` option to also mask extra term
- Extension of `_masked.py` with more flexibility

---

## Evolution Path

```
attention_pruning_case_study.py (basic)
    │
    ├── attention_pruning_case_study_hybrid.py (+ extra term)
    │       │
    │       ├── attention_pruning_case_study_hybrid09.py (1.1 coefficient variant)
    │       │
    │       └── attention_pruning_case_study_hybrid_sampled.py (+ head sampling)
    │               │
    │               └── attention_pruning_case_study_hybrid_sampled_xtrace.py (+ cross-trace)
    │                       │
    │                       └── attention_pruning_case_study_hybrid_rounds_xtrace.py (+ round-based) [KEPT]
    │                               │
    │                               ├── _gaussian.py (Gaussian quantile scoring)
    │                               │       │
    │                               │       └── _gaussian_combined.py (Method 5.2)
    │                               │
    │                               └── _masked.py (high-freq masking)
    │                                       │
    │                                       └── _masked_tuned.py (tunable params)
```
