# KV Cache Budget vs Accuracy Comparison

## Overview

This figure compares the performance of different KV Cache compression methods on the **Qwen3-8B** model across three mathematical reasoning benchmarks. The goal is to show how accuracy changes as we allocate more memory (budget) to the KV Cache.

## Figure Description

The combined figure (`fig_kv_budget_accuracy_combined.png`) contains three panels arranged horizontally:

### Panel (A): MATH Benchmark

- **X-axis**: KV Cache Budget (512, 1024, 2048, 3072 tokens)
- **Y-axis**: Accuracy (%), ranging approximately from 45% to 70%
- **Red dashed horizontal line**: FullKV baseline at 69.6% (no compression, full attention)
- **Blue line with circle markers**: R-KV method
- **Green line with square markers**: SpecKV method

**Trend Description**:
- At the lowest budget (512 tokens), R-KV achieves only 46.4% while SpecKV achieves 56.0% — a **9.6 percentage point advantage** for SpecKV.
- At budget=1024, SpecKV (68.4%) still leads R-KV (60.4%) by **8 points**.
- At budget=2048, both methods approach the FullKV baseline: SpecKV (69.2%) nearly matches FullKV (69.6%), while R-KV reaches 68.2%.
- At budget=3072, SpecKV (70.0%) **slightly exceeds** FullKV (69.6%), while R-KV plateaus at 67.8%.

**Key Insight**: On MATH, SpecKV consistently outperforms R-KV across all budget levels, with the largest advantage at low budgets. SpecKV can match or exceed full attention performance with only 2048-3072 tokens of KV Cache.

---

### Panel (B): AIME24 Benchmark

- **X-axis**: KV Cache Budget (512, 1024, 2048, 3072, 4096 tokens)
- **Y-axis**: Accuracy (%), ranging approximately from 5% to 60%
- **Red dashed horizontal line**: FullKV baseline at 57.1%
- **Blue line with circle markers**: R-KV method
- **Green line with square markers**: SpecKV method

**Trend Description**:
- At budget=512, both methods perform poorly: R-KV at 6.7%, SpecKV at 7.5%.
- As budget increases, SpecKV pulls ahead significantly:
  - Budget=1024: SpecKV (25.8%) vs R-KV (10.4%) — **15.4 point gap**
  - Budget=2048: SpecKV (42.1%) vs R-KV (25.4%) — **16.7 point gap**
  - Budget=3072: SpecKV (50.0%) vs R-KV (38.8%) — **11.2 point gap**
  - Budget=4096: SpecKV (54.6%) vs R-KV (49.2%) — **5.4 point gap**
- At maximum budget (4096), SpecKV reaches 54.6%, approaching but not exceeding FullKV (57.1%).

**Key Insight**: AIME24 is more challenging and requires larger KV Cache budgets. SpecKV maintains a consistent advantage over R-KV throughout, with the gap being largest in the mid-budget range (1024-2048 tokens).

---

### Panel (C): AIME25 Benchmark

- **X-axis**: KV Cache Budget (512, 1024, 2048, 3072, 4096 tokens)
- **Y-axis**: Accuracy (%), ranging approximately from 5% to 45%
- **Red dashed horizontal line**: FullKV baseline at 40.8%
- **Blue line with circle markers**: R-KV method
- **Green line with square markers**: SpecKV method

**Trend Description**:
- At budget=512, both methods are near random: R-KV at 6.0%, SpecKV at 8.3%.
- SpecKV consistently outperforms R-KV:
  - Budget=1024: SpecKV (15.4%) vs R-KV (8.8%) — **6.6 point gap**
  - Budget=2048: SpecKV (32.9%) vs R-KV (17.5%) — **15.4 point gap**
  - Budget=3072: SpecKV (40.8%) vs R-KV (21.7%) — **19.1 point gap**
  - Budget=4096: SpecKV (43.3%) vs R-KV (31.7%) — **11.6 point gap**
- At budget=3072, SpecKV (40.8%) **exactly matches** FullKV (40.8%).
- At budget=4096, SpecKV (43.3%) **exceeds** FullKV (40.8%) by 2.5 points.

**Key Insight**: On AIME25, SpecKV not only matches but surpasses full attention performance at higher budgets (3072+). The performance gap over R-KV is largest at mid-to-high budgets, reaching nearly 20 percentage points at budget=3072.

---

## Overall Conclusions

1. **SpecKV consistently outperforms R-KV** across all three benchmarks and all budget levels tested.

2. **The advantage is most pronounced at low-to-mid budgets** (512-2048 tokens), where SpecKV can be 10-20 percentage points better than R-KV.

3. **SpecKV can match or exceed FullKV performance** with significantly less memory:
   - MATH: Matches FullKV at budget=2048, exceeds at budget=3072
   - AIME24: Approaches FullKV (within 2.5%) at budget=4096
   - AIME25: Matches FullKV at budget=3072, exceeds at budget=4096

4. **Harder benchmarks require more budget**: MATH (easiest) saturates around 2048 tokens, while AIME24/AIME25 (harder) continue improving up to 4096 tokens.

5. **R-KV struggles at low budgets**, particularly on AIME benchmarks where it remains below 10% accuracy until budget exceeds 1024 tokens.

---

## Data Source

- Model: Qwen3-8B (DeepSeek-R1-0528-Qwen3-8B)
- Data file: `paper_visualizations/Materials/SpecKV Experiment Data - Performance.csv`
- Script: `paper_visualizations/scripts/kv_budget_accuracy_curve.py`

## Visual Style

- Background: Light gray-purple (#E7E7F0)
- Grid: White lines, alpha=0.7
- FullKV baseline: Red dashed line (#E24A33)
- R-KV: Blue with circle markers
- SpecKV: Green with square markers
- Panel labels: Bold (A), (B), (C) in top-left corners
