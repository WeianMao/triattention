# Spectrum-Aware NMS Experiment Results

## Overview

This document records the experimental results of the Spectrum-Aware NMS (Non-Maximum Suppression) approach for KV cache compression, testing whether frequency-weighted projection coverage can improve key selection quality.

## Experimental Setup

- **Trace**: `qid0003_trace34` (DeepSeek R1 Qwen3 8B)
- **Heads**: 10 sampled heads from `hybrid_sample_heads_lowret_top10.json`
- **Parameters**: max_keys=2048, round_window=64, aggregation=mean
- **Epsilon**: Fixed at 0 (as per brainstorm design)

## Experiment Configurations

| Setting | NMS Enabled | Energy Method | GPU |
|---------|-------------|---------------|-----|
| Baseline | No | N/A | 1 |
| Amplitude NMS | Yes | amplitude | 2 |
| Causal NMS | Yes | causal | 3 |

### Energy Methods

1. **Amplitude**: `E_f = E[|q_f| * |k_f|]` using unrotated Q/K - measures frequency band importance by magnitude correlation
2. **Causal**: Causal attention weighted per frequency using rotated Q/K - measures actual attention contribution per frequency band

## Results Summary

| Setting | Overall Retention | NMS Drop Rate | Total NMS Drops |
|---------|------------------|---------------|-----------------|
| Baseline | **96.68%** | 0% | 0 |
| Amplitude NMS | 93.12% | 35.75% | 61,140 |
| Causal NMS | 73.39% | 60.24% | 103,016 |

## Per-Head Analysis

### Retention Rates by Head

| Layer | Head | Baseline | Amplitude NMS | Causal NMS | Amplitude Δ | Causal Δ |
|-------|------|----------|---------------|------------|-------------|----------|
| 3 | 7 | 99.95% | 81.90% | 71.44% | -18.05% | -28.51% |
| 9 | 19 | 92.00% | 88.45% | 77.66% | -3.55% | -14.34% |
| 17 | 25 | 81.39% | 77.40% | 31.18% | -3.99% | -50.21% |
| 24 | 0 | 99.84% | 99.44% | 98.72% | -0.40% | -1.12% |
| 24 | 11 | 99.76% | 96.40% | 93.69% | -3.36% | -6.07% |
| 24 | 14 | 96.53% | 91.83% | 68.28% | -4.70% | -28.25% |
| 31 | 30 | 97.53% | 96.69% | 91.63% | -0.84% | -5.90% |
| 32 | 27 | 100.00% | 99.73% | 99.73% | -0.27% | -0.27% |
| 33 | 0 | 100.00% | 99.53% | **2.65%** | -0.47% | **-97.35%** |
| 34 | 6 | 99.80% | 99.78% | 98.95% | -0.02% | -0.85% |

### Key Observations

1. **Baseline achieves highest retention** (96.68%): The original score-based pruning without NMS preserves the most keys that match the argmax selection.

2. **Amplitude NMS shows moderate degradation** (-3.56%): The amplitude-based frequency weighting causes some important keys to be suppressed, but maintains reasonable retention for most heads.

3. **Causal NMS causes severe degradation** (-23.29%): The causal energy weighting leads to much more aggressive suppression, particularly problematic for specific heads:
   - Layer 33, Head 0: Catastrophic drop from 100% to 2.65%
   - Layer 17, Head 25: Drop from 81.39% to 31.18%

4. **Layer 24 heads are most robust**: Heads in layer 24 show the smallest degradation across both NMS methods, suggesting their attention patterns are more localized in frequency space.

5. **High NMS drop rates don't correlate with retention**: Even with lower drop rates (amplitude: 35.75%), retention still suffers, indicating the NMS is suppressing the wrong keys.

## Analysis

### Why NMS Hurts Performance

The spectrum-aware NMS approach assumes that keys with similar frequency signatures can substitute for each other. However, the experimental results suggest this assumption is flawed:

1. **Frequency similarity ≠ Attention equivalence**: Two keys may have similar frequency content but serve different semantic roles in the attention pattern.

2. **Argmax selection is highly specific**: The oracle argmax selection chooses keys based on their actual attention contribution, which depends on the specific query-key interaction, not just the key's frequency profile.

3. **Causal weighting amplifies errors**: The causal method weights frequencies by their attention contribution, but this creates a feedback loop where high-energy frequencies dominate the suppression decision, leading to over-aggressive pruning.

### Per-Head Variation

The dramatic variation across heads (e.g., Layer 32 Head 27 barely affected vs. Layer 33 Head 0 nearly destroyed) suggests:

- Different heads operate in different frequency regimes
- A one-size-fits-all NMS approach cannot capture this diversity
- Head-adaptive or layer-adaptive frequency weighting might be needed

## Conclusions

1. **Spectrum-Aware NMS does not improve retention** in its current form. Both energy methods (amplitude and causal) reduce retention compared to baseline.

2. **The causal method is particularly harmful**, likely because it over-weights high-energy frequencies that don't necessarily correspond to important keys.

3. **The approach may need fundamental changes**:
   - Consider query-dependent suppression instead of key-only analysis
   - Adaptive epsilon tuning per head/layer
   - Alternative coverage metrics that better capture attention semantics

4. **Recommendation**: Do not deploy spectrum-aware NMS for KV cache compression without significant redesign of the coverage score formulation.

## Files Generated

- `attention_pruning_case_study_hybrid_rounds_xtrace_nms.py` - Main experiment script
- `run_nms_baseline.sh`, `run_nms_amplitude.sh`, `run_nms_causal.sh` - Experiment runners
- `results/baseline/`, `results/nms_amplitude/`, `results/nms_causal/` - Per-head metrics and visualizations

## Future Work

1. Investigate query-conditioned frequency weighting
2. Test adaptive epsilon based on frequency energy distribution
3. Explore layer/head-specific NMS parameters
4. Consider alternative suppression criteria beyond frequency projection
