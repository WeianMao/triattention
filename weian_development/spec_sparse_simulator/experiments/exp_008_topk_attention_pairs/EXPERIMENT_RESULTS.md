# Experiment 007: Probe Activation Loss (Loss1) Ablation Results

## Experiment Date: 2025-12-18

## Overview

This document records the evaluation results for Loss1 (Probe Activation Loss) ablation study, including new Key Activation metrics designed to measure whether Loss1 successfully activates bins in the Key Network.

## Evaluation Setup

- **Training Data**: `qid0003_trace34/qk.pt`
- **Test Data**: `qid0008_trace46/qk.pt` (cross-trace validation)
- **Head**: layer=33, head=0
- **Model**: Module2Network (128 bins, 64 freqs, 3 kernels)

---

## Part 1: Key Activation Metrics (Top-50)

These metrics measure whether the Key Network places Ground Truth (GT) keys in the Top-50 of each bin.

**Ground Truth Definition**: Keys that are argmax attention targets for queries in each round.

| λ值 | Recall% | AvgBinsGT | AvgGT/Bin | Gini | Entropy |
|-----|---------|-----------|-----------|------|---------|
| **0.0 (baseline)** | **76.95%** | 78.72 | 0.6444 | 0.2787 | 0.9715 |
| 0.01 | 58.98% | 127.66 | 1.0465 | 0.0101 | 1.0000 |
| 0.05 | 54.58% | 127.81 | 1.0571 | 0.0108 | 1.0000 |
| 0.1 | 54.58% | 127.81 | 1.0612 | 0.0117 | 1.0000 |
| 0.5 | 54.24% | 127.84 | 1.0676 | 0.0116 | 1.0000 |

### Metrics Explanation

- **Recall%**: Percentage of GT keys found in ANY bin's Top-50 (higher = better coverage)
- **AvgBinsGT**: Average number of bins containing GT keys per round (out of 128)
- **AvgGT/Bin**: Average GT keys per bin
- **Gini**: Inequality measure (0=uniform, 1=concentrated)
- **Entropy**: Distribution uniformity (1=uniform, 0=collapsed)

### Key Findings

1. **Baseline Recall is only 76.95%**: ~23% of GT keys are NOT in any bin's Top-50
2. **Loss1 DECREASES Recall**: From 76.95% → 54-59% (counterintuitive!)
3. **Loss1 DOES activate more bins**: AvgBinsGT increases from 78.72 → 127+
4. **Loss1 makes distribution more uniform**: Gini drops from 0.28 → 0.01

### Interpretation

Loss1 makes the Key Network distribute **all keys** more uniformly across bins, but this **dilutes the GT keys** - they are no longer concentrated in the top positions of fewer bins. Instead, they get spread thin across all bins, often pushed out of the Top-50 by other keys.

---

## Part 2: Hit Rate vs Number of Bins Selected

Testing how hit rate changes when Query Network selects different numbers of bins.

### Baseline (λ=0.0), Top-50 keys per bin

| #Bins | HitRate% | BinHitRate% | AvgKeys/Query |
|-------|----------|-------------|---------------|
| 1     | 98.49    | 95.90       | 178           |
| 2     | 98.62    | 96.03       | 228           |
| 4     | 98.71    | 96.12       | 328           |
| 8     | 98.77    | 96.18       | 528           |
| 16    | 98.88    | 96.29       | 928           |
| 32    | 99.06    | 96.47       | 1728          |
| 64    | 99.26    | 96.67       | 3328          |
| **128** | **99.37** | **96.78** | **6528**    |

### Key Findings

1. **Top-2 bin achieves 98.62%** - matches original ablation results
2. **Upper bound with all 128 bins is 99.37%** - only +0.75% over Top-2
3. **~0.63% of queries cannot be hit** even with all bins' Top-50 keys

---

## Part 3: Original Hit Rate Ablation (from previous runs)

| λ值 | K=50 Hit% | K=500 Hit% | K=1000 Hit% |
|-----|-----------|------------|-------------|
| 0.0 | 98.62 | 98.92 | 99.16 |
| 0.01 | 98.59 | 98.90 | 99.09 |
| 0.05 | 98.60 | 98.94 | 99.11 |
| 0.1 | 98.61 | 98.95 | 99.09 |
| 0.5 | 98.64 | 99.03 | 99.17 |

### Observation

Hit rates are nearly identical across all λ values, confirming that Loss1 does not improve final hit rate despite activating more bins.

---

## Conclusions

### Loss1 Design Issue

Loss1 was designed to activate "dead" bins by forcing the Key Network to place argmax keys into bins that receive low query routing probability. However:

1. **Loss1 works on the wrong objective**: It makes key distribution more uniform, but this dilutes GT keys rather than concentrating them where they matter.

2. **Recall drops significantly**: GT keys being spread across all bins means fewer of them appear in any single bin's Top-50.

3. **Hit rate unchanged**: Despite activating more bins, the Query Network still achieves similar hit rates because:
   - Queries already select the right bins
   - But the bins now have diluted key rankings

### Potential Improvements

1. **Focus on GT keys specifically**: Loss should encourage GT keys (not all argmax keys) to rank higher in their assigned bins.

2. **Consider Query-Key co-optimization**: Loss1 only optimizes Key Network, but Query Network needs to learn to route to newly activated bins.

3. **Different activation strategy**: Instead of making all keys uniform, keep GT keys concentrated while activating empty bins with relevant keys.

---

## Files Generated

- `evaluate_key_activation.py`: New evaluation script for Key Activation metrics
- `evaluate_num_bins_sweep.py`: Script to test hit rate with different bin counts
- `output/ablation/key_activation_metrics.json`: Detailed metrics results
- `output/ablation/lambda_*/num_bins_sweep_results.json`: Per-lambda bin sweep results
