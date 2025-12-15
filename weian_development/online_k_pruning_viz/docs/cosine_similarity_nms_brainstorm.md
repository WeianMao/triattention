# Cosine Similarity NMS for KV Cache Compression

## 1. Project Goal & Ablation Study Design

### 1.1 High-Level Objective

本项目的核心目标是验证 **基于 Cosine Similarity 的 NMS 抑制** 对 KV cache 压缩的有效性。

### 1.2 Ablation Study Principle: NMS-Only Isolated Flow

**关键原则**：新脚本采用**仅 NMS（NMS-only）**的实验设计，移除所有前置压缩算法，只保留 NMS 本身的效果：

| 组件 | 原 hybrid 脚本 | NMS-only 脚本 | 说明 |
|-----|---------------|--------------|------|
| 打分函数 | `score_keys_for_round` | **移除** | 不再使用频率打分 |
| Top-K 裁剪 | `trim_to_max_keys` | **移除** | 不再有容量限制 |
| 前置压缩 | 有 | **无** | 凸显 NMS 本身效果 |
| **NMS 抑制** | 无 | **唯一压缩手段** | 核心变量 |
| 可视化 | heatmap + argmax | **保持一致** | 不修改 |
| 评估指标 | retention rate | **保持一致** | 不修改 |

**目的**：
1. 排除前置打分/Top-K 对 NMS 效果的干扰
2. 验证 Cosine Similarity NMS 作为**独立** KV 压缩算法的有效性
3. 便于分析 NMS 的压缩行为（drop rate, 哪些 K 被删除）

**实验流程**（每轮）：
```
每轮开始时 (每 round_window 个 token):
    1. 累积本轮新 K 到缓存
    2. 轮末执行一次 Cosine Similarity NMS
       - 对当前所有 K（历史 + 新）计算 SA_cos_sim
       - 对于相似度超过阈值且模长比满足条件的 K 对，删除模长较小的 K
    3. 无 Top-K 限制，仅依赖 NMS 自身判定
```

**参考实现**：`attention_pruning_case_study_nms_variance_isolated.py`

---

## 2. Background & Motivation

### 2.1 Problem Statement

在 KV cache 中，如果两个 K 在频率域中高度相似，保留两个没有意义——它们对不同 Q 的贡献模式几乎相同。

### 2.2 Core Insight

借鉴目标检测中的 NMS (Non-Maximum Suppression)：
- **目标检测 NMS**：两个框如果 IoU 高，保留得分大的框，**删除**得分小的框
- **KV Cache NMS**：两个 K 如果 **cosine similarity 高**，且模长差距足够大，保留模长大的 K，**删除**模长小的 K

### 2.3 Why Spectrum-Aware?

在 RoPE 中存在多个频段，但通过可视化发现：
- 不同频段的能量分布不均匀
- 某些频段对 attention 的贡献远大于其他频段
- 简单的 cosine similarity 对所有频段一视同仁，无法捕捉这种频段重要性差异

因此需要 **Spectrum-Aware Cosine Similarity**：按频段重要性加权的相似度度量。

### 2.4 Why Cosine Similarity over Projection Coverage?

之前尝试的投影覆盖方法存在问题：
- 投影条件 $|A| \cdot \cos(\theta) > |B|$ 过于严格或过于宽松
- 依赖绝对模长比较，对不同 head 的数值范围敏感
- 实验结果表明该方法不够有效

**Cosine Similarity 的优势**：
- **归一化度量**：cos_sim ∈ [-1, 1]，跨 head 可比
- **直接语义**：高相似度 = 方向接近 = 信息冗余
- **阈值易设定**：0.9, 0.95, 0.99 等直观阈值
- **与 NMS 原始思想更接近**：IoU 本质也是相似度度量

---

## 3. Spectrum-Aware Cosine Similarity

### 3.1 Notation

- $k^{rot}_i \in \mathbb{C}^{F}$：**经过 RoPE 旋转后**的 K 向量（原始 qk.pt 中的数据，转为复数表示）
- $q^{rot}_i \in \mathbb{C}^{F}$：**经过 RoPE 旋转后**的 Q 向量
- $F$：频段数 = head_dim / 2
- $\text{Real}(\cdot)$：取复数的实部

**设计选择**：所有计算都在 **RoPE 旋转后的空间**进行。

**理由**：
- 实际 attention 点积发生在 RoPE 旋转后的 Q/K
- 在旋转后空间做 NMS 更贴近真实注意力计算
- 避免 invert_rope 带来的额外计算和误差

### 3.2 Standard Cosine Similarity

$$\text{cos\_sim}(k^{rot}_i, k^{rot}_j) = \frac{\text{Real}\left(\sum_f k^{rot,(f)}_i \cdot \overline{k^{rot,(f)}_j}\right)}{|k^{rot}_i| \cdot |k^{rot}_j|}$$

### 3.3 Spectrum-Aware Cosine Similarity

引入频段权重 $w_f$：

$$\text{SA\_cos\_sim}(k^{rot}_i, k^{rot}_j) = \frac{\sum_f w_f \cdot \text{Real}\left(k^{rot,(f)}_i \cdot \overline{k^{rot,(f)}_j}\right)}{\sqrt{\sum_f w_f |k^{rot,(f)}_i|^2} \cdot \sqrt{\sum_f w_f |k^{rot,(f)}_j|^2}}$$

### 3.4 Frequency Weight: Q-Magnitude Median

**核心思想**：注意力点积结果 = Q 模长 × K 模长 × cos(夹角)。在计算频段权重时，应该考虑 Q 在各频段的模长分布。

**权重定义**：使用 Q 在各频段的**中位数模长**作为权重：

$$w_f = \text{median}_{t}\left(|q^{rot,(f)}_t|\right)$$

其中 $t$ 遍历 stats_trace 中的所有 token 位置。

**为什么用中位数**：
- 中位数对异常值稳健
- 代表该频段 Q 的"典型"贡献能力
- 高 $w_f$ 的频段在 attention 中更重要，相似度计算应更关注这些频段

**实现**：
```python
# q_rotated_complex: [seq_len, freq_count] - RoPE旋转后的Q，转为复数
q_magnitudes = torch.abs(q_rotated_complex)  # [seq_len, freq_count]
w_f = torch.median(q_magnitudes, dim=0).values  # [freq_count]

# 归一化（可选，便于解释）
w_f = w_f / w_f.sum()
```

**计算时机**：在脚本开始时从 stats_trace 计算一次，整个实验过程中保持不变。

---

## 4. NMS Implementation: Cosine Similarity Based

### 4.1 Requirement: Sparsity via Hard Drop

我们需要的是 **sparsity**，即真正删除冗余的 K，而不是只降低分数。

### 4.2 NMS Logic with Cosine Similarity

**核心思想**：如果两个 K 的 SA_cos_sim 超过阈值，**且**模长差距足够大，删除模长较小的 K。

**抑制条件**：A 抑制 B 当且仅当满足**两个条件**：

1. **相似度条件**：$\text{SA\_cos\_sim}(A, B) > \tau$
   - $\tau$ 是相似度阈值（如 0.9）
   - 保证 A 和 B 方向足够接近

2. **模长比条件**：$\|B\|_w < \|A\|_w \cdot \alpha$
   - $\alpha$ 是模长比阈值（如 0.9）
   - 保证 B 的模长显著小于 A

**为什么需要模长比条件**：
- 如果只用 $\|A\|_w > \|B\|_w$，当两个 K 模长非常相近时，哪个被删除取决于微小的数值差异，不稳定
- 加入 $\alpha$ 因子后，只有当 B 显著弱于 A 时（$\|B\| < 0.9 \|A\|$），B 才会被删除
- 避免误删模长相近但都重要的 K

### 4.3 Weighted Norm Definition

**加权模长**：使用与 SA_cos_sim 相同的频段权重：

$$\|K\|_w = \sqrt{\sum_f w_f \cdot |k^{rot,(f)}|^2}$$

其中 $w_f$ 是 Q-magnitude 中位数权重。

**设计一致性**：模长计算和相似度计算使用相同的权重，保持语义一致。

### 4.4 Mathematical Formulation

**抑制条件完整表述**：A 抑制 B 当：

$$\text{SA\_cos\_sim}(A, B) > \tau \quad \text{AND} \quad \|B\|_w < \alpha \cdot \|A\|_w$$

其中：
- $\tau$：相似度阈值（默认 0.9）
- $\alpha$：模长比阈值（默认 0.9）
- $\|K\|_w$：Q-magnitude 加权范数

### 4.5 Fast Parallel NMS (Matrix Implementation)

**重要说明**：我们使用的是 **Fast NMS**，不是严格的传统 NMS。

**Fast NMS 特点**：
- 一次性计算所有 pair 的相似度和支配关系
- C 可能被 B 杀死，即使 B 也被 A 杀死
- 比传统迭代 NMS 更激进（删除更多）
- **我们接受这种"误杀"**

**算法**：
```python
def fast_parallel_nms_cosine(
    k_complex: torch.Tensor,      # [N, F] K 向量（RoPE旋转后，复数表示）
    freq_weights: torch.Tensor,   # [F] 频段权重（Q-magnitude 中位数）
    sim_threshold: float = 0.9,   # 相似度阈值 τ
    norm_ratio: float = 0.9,      # 模长比阈值 α
) -> torch.Tensor:
    """
    Fast Parallel NMS：基于 Cosine Similarity 的并行删除

    A 抑制 B 当:
    1. SA_cos_sim(A, B) > sim_threshold
    2. ||B||_w < norm_ratio * ||A||_w

    Returns:
        keep_mask: [N] bool tensor, True = 保留
    """
    N, F = k_complex.shape

    # 1. 计算加权模长
    k_abs_sq = torch.abs(k_complex) ** 2  # [N, F]
    weighted_norm_sq = (k_abs_sq * freq_weights).sum(dim=1)  # [N]
    weighted_norm = torch.sqrt(weighted_norm_sq + 1e-10)  # [N]

    # 2. 计算加权点积 (分子)
    #    Real(A · conj(B)) for all pairs, weighted by freq_weights
    dot_product = torch.einsum('af,bf->abf', k_complex, k_complex.conj()).real  # [N, N, F]
    weighted_dot = (dot_product * freq_weights.view(1, 1, -1)).sum(dim=2)  # [N, N]

    # 3. 计算 SA_cos_sim = weighted_dot / (||A||_w * ||B||_w)
    norm_product = weighted_norm.unsqueeze(1) * weighted_norm.unsqueeze(0)  # [N, N]
    cos_sim = weighted_dot / (norm_product + 1e-10)  # [N, N]

    # 4. 条件1: 相似度超过阈值
    similar = cos_sim > sim_threshold  # [N, N]

    # 5. 条件2: ||B||_w < α * ||A||_w
    #    即 A 的模长显著大于 B
    #    suppresses[a, b] = True 表示 A 可能抑制 B
    norm_ratio_satisfied = weighted_norm.unsqueeze(1) * norm_ratio > weighted_norm.unsqueeze(0)
    # norm_ratio_satisfied[a, b] = True when ||A||_w * α > ||B||_w
    # 即 ||B||_w < α * ||A||_w

    # 6. 两个条件同时满足
    suppresses = similar & norm_ratio_satisfied  # [N, N]

    # 7. 排除自己抑制自己（对角线）
    suppresses.fill_diagonal_(False)

    # 8. 如果 B 被任何 A 抑制，则删除 B
    is_suppressed = suppresses.any(dim=0)  # [N]: is_suppressed[b] = any A suppresses B
    keep_mask = ~is_suppressed

    return keep_mask
```

**时间复杂度**：O(N² × F)，完全可并行化（GPU friendly）

### 4.6 Hyperparameter Selection

**相似度阈值 τ**：
- `0.99`：非常严格，只删除几乎相同的 K
- `0.95`：适中
- `0.90`：推荐起点
- `0.85`：宽松

**模长比阈值 α**：
- `0.95`：保守，只删除明显弱的 K
- `0.90`：推荐起点
- `0.80`：激进

**推荐**：从 τ=0.9, α=0.9 开始实验。

### 4.7 Incremental NMS: Optional Computation Optimization

> **Note**: 本节描述的是一种**可选的性能优化**。建议先用 Section 4.5 的简单实现验证算法正确性，必要时再考虑此优化。

由于历史 K 之间已经做过 NMS，只需计算涉及新 K 的块：

```
                    历史 K (H)     新 K (N)
                 ┌─────────────┬───────────┐
   历史 K (H)    │   SKIP      │ 历史抑制  │
                 │ (已做过NMS) │  新K检查  │
                 ├─────────────┼───────────┤
   新 K (N)      │ 新K抑制     │  新K互相  │
                 │ 历史K检查   │  抑制检查 │
                 └─────────────┴───────────┘
```

**需要计算**：
- 历史K 抑制 新K (H × N)
- 新K 抑制 历史K (N × H)
- 新K 抑制 新K (N × N)

```python
def incremental_fast_nms_cosine(
    historical_k: torch.Tensor,   # [H, F] 历史 K 复数
    new_k: torch.Tensor,          # [N, F] 新 K 复数
    freq_weights: torch.Tensor,   # [F] 频段权重
    sim_threshold: float = 0.9,
    norm_ratio: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    增量 Fast NMS（基于 Cosine Similarity）

    Returns:
        historical_keep: [H] bool, 历史 K 是否保留
        new_keep: [N] bool, 新 K 是否保留
    """
    H, N, F = historical_k.shape[0], new_k.shape[0], historical_k.shape[1]

    # 计算加权模长
    def weighted_norm(k):
        k_abs_sq = torch.abs(k) ** 2
        return torch.sqrt((k_abs_sq * freq_weights).sum(dim=1) + 1e-10)

    hist_norm = weighted_norm(historical_k)  # [H]
    new_norm = weighted_norm(new_k)  # [N]

    # 计算 cos_sim 的辅助函数
    def compute_cos_sim(k1, k2, norm1, norm2):
        dot = torch.einsum('af,bf->abf', k1, k2.conj()).real  # [N1, N2, F]
        weighted_dot = (dot * freq_weights.view(1, 1, -1)).sum(dim=2)  # [N1, N2]
        norm_prod = norm1.unsqueeze(1) * norm2.unsqueeze(0)  # [N1, N2]
        return weighted_dot / (norm_prod + 1e-10)

    # 1. 历史K vs 新K: 检查历史K是否抑制新K
    cos_sim_hist_new = compute_cos_sim(historical_k, new_k, hist_norm, new_norm)  # [H, N]
    similar_hist_new = cos_sim_hist_new > sim_threshold
    # hist_norm[h] * norm_ratio > new_norm[n] => new suppressed by hist
    hist_suppresses_new = similar_hist_new & (hist_norm.unsqueeze(1) * norm_ratio > new_norm.unsqueeze(0))
    new_suppressed_by_hist = hist_suppresses_new.any(dim=0)  # [N]

    # 2. 新K vs 历史K: 检查新K是否抑制历史K
    cos_sim_new_hist = cos_sim_hist_new.T  # [N, H] (对称)
    similar_new_hist = cos_sim_new_hist > sim_threshold
    # new_norm[n] * norm_ratio > hist_norm[h] => hist suppressed by new
    new_suppresses_hist = similar_new_hist & (new_norm.unsqueeze(1) * norm_ratio > hist_norm.unsqueeze(0))
    hist_suppressed_by_new = new_suppresses_hist.any(dim=0)  # [H]

    # 3. 新K vs 新K
    cos_sim_new_new = compute_cos_sim(new_k, new_k, new_norm, new_norm)  # [N, N]
    similar_new_new = cos_sim_new_new > sim_threshold
    new_suppresses_new = similar_new_new & (new_norm.unsqueeze(1) * norm_ratio > new_norm.unsqueeze(0))
    new_suppresses_new.fill_diagonal_(False)
    new_suppressed_by_new = new_suppresses_new.any(dim=0)  # [N]

    # 4. 汇总
    new_keep = ~(new_suppressed_by_hist | new_suppressed_by_new)
    historical_keep = ~hist_suppressed_by_new

    return historical_keep, new_keep
```

### 4.8 Complexity Analysis

| 方法 | 时间复杂度 | 空间复杂度 |
|-----|-----------|-----------|
| 简单实现 (每轮全矩阵) | O((H+N)² × F) | O((H+N)² × F) |
| **优化实现 (Incremental)** | O((H×N + N²) × F) | O((H×N + N²) × F) |

---

## 5. Experimental Design

### 5.1 Isolated NMS Flow

**核心原则**：NMS-only 实验，移除所有前置压缩算法。

**对比实验**：
- **Baseline**：NMS disabled，无任何压缩（所有 K 保留）
- **Test**：NMS enabled，仅依赖 NMS 的 coverage 判定

**参考脚本**：`attention_pruning_case_study_nms_variance_isolated.py`

### 5.2 Experiment Variables

| 参数 | 选项 | 说明 |
|-----|------|------|
| `--nms-enabled` | True/False | 是否启用 NMS |
| `--sim-threshold` | 0.85 / 0.90 / 0.95 / 0.99 | 相似度阈值 τ |
| `--norm-ratio` | 0.80 / 0.85 / 0.90 / 0.95 | 模长比阈值 α |

### 5.3 Automated Hyperparameter Search

**搜索空间**：
```python
SEARCH_SPACE = {
    "sim_threshold": [0.85, 0.90, 0.95, 0.99],
    "norm_ratio": [0.80, 0.85, 0.90, 0.95],
}
```

**自动搜索脚本设计**：
```bash
# run_cosine_nms_sweep.sh
#!/bin/bash
SIM_THRESHOLDS=("0.85" "0.90" "0.95" "0.99")
NORM_RATIOS=("0.80" "0.85" "0.90" "0.95")

# Baseline (无 NMS)
python attention_pruning_case_study_cosine_nms_isolated.py \
    $INPUT_ROOT --trace $TRACE --stats-trace $STATS_TRACE \
    --output-root results/baseline

# NMS variants
for sim in "${SIM_THRESHOLDS[@]}"; do
    for ratio in "${NORM_RATIOS[@]}"; do
        python attention_pruning_case_study_cosine_nms_isolated.py \
            $INPUT_ROOT --trace $TRACE --stats-trace $STATS_TRACE \
            --nms-enabled --sim-threshold $sim --norm-ratio $ratio \
            --output-root results/nms_sim${sim}_ratio${ratio}
    done
done
```

**搜索结果汇总**：
```
| sim_threshold | norm_ratio | retention_rate | nms_drop_rate | total_drops |
|---------------|------------|----------------|---------------|-------------|
| baseline      | -          | 100.00%        | 0%            | 0           |
| 0.90          | 0.90       | 96.58%         | 3.69%         | 6311        |
| 0.90          | 0.85       | 95.20%         | 5.12%         | ...         |
| 0.95          | 0.90       | 98.10%         | 1.85%         | ...         |
| ...           | ...        | ...            | ...           | ...         |
```

### 5.4 Evaluation Metrics

1. **Retention Rate**: argmax 命中率（与 baseline 对比）
2. **NMS Drop Rate**: 被 NMS 删除的 K 比例
3. **Total Drops**: 总删除数量
4. **Per-Head Analysis**: 每个 head 的压缩行为
5. **Attention Heatmap**: 可视化对比

### 5.5 Additional Metrics for NMS

新增 NMS 相关指标（记录在 metrics JSON 中）：
- `nms_drop_rate`: 总 drop 数 / 总 round 数
- `nms_total_drops`: 总删除数量
- `nms_drop_count_per_head`: 每个 head 的删除数量
- `avg_cos_sim_of_dropped`: 被删除 K 对的平均相似度（诊断阈值选择）
- `cos_sim_distribution`: 所有 K 对相似度的分布统计

---

## 6. Design Decisions (Confirmed)

### 6.1 频段权重计算策略

**决定**：使用 Q-magnitude 中位数作为频段权重。

**计算时机**：每个 head 在脚本开始时从 stats_trace 计算一次，整个实验过程中**保持不变**。

**实现**：
```python
# 在脚本开始时，加载 stats_trace 后计算
q_rotated = stats_data["q"][layer, head]  # RoPE 旋转后的 Q
q_complex = to_complex_pairs(q_rotated)   # [seq_len, freq_count]
freq_weights = torch.median(torch.abs(q_complex), dim=0).values  # [freq_count]
freq_weights = freq_weights / freq_weights.sum()  # 归一化
```

### 6.2 计算空间

**决定**：所有计算都在 **RoPE 旋转后的空间**进行。

**理由**：
- 与实际 attention 计算一致
- 避免 invert_rope 带来的额外计算

### 6.3 NMS 作用范围

**决定**：全局 NMS，每轮轮末对所有 K 进行 NMS 检查。

### 6.4 NMS 类型

**决定**：Fast Parallel NMS（真正删除被支配的 K）

### 6.5 抑制条件

**决定**：双条件抑制

**A 抑制 B 当**：
1. SA_cos_sim(A, B) > τ
2. ||B||_w < α × ||A||_w

**优势**：
- 避免模长相近时的不稳定判定
- 只删除"显著弱于"强者的 K

### 6.6 默认超参数

- **sim_threshold (τ)**：0.9
- **norm_ratio (α)**：0.9

---

## 7. Implementation Checklist

### 7.1 Code Structure

```
weian_development/online_k_pruning_viz/
├── attention_pruning_case_study_nms_variance_isolated.py  # 参考实现
├── attention_pruning_case_study_cosine_nms_isolated.py    # 新脚本（待创建）
├── run_cosine_nms_sweep.sh                                # 自动超参数搜索脚本
└── docs/
    ├── spectrum_aware_nms_brainstorm.md      # 投影覆盖方法 (archived)
    ├── variance_aware_nms_isolated_effect.md # Variance-aware 实验报告
    └── cosine_similarity_nms_brainstorm.md   # 本文档
```

### 7.2 New Arguments to Add

```python
parser.add_argument("--nms-enabled", action="store_true", default=False,
                    help="Enable cosine similarity NMS at each round end")
parser.add_argument("--sim-threshold", type=float, default=0.9,
                    help="Cosine similarity threshold τ (0.85-0.99)")
parser.add_argument("--norm-ratio", type=float, default=0.9,
                    help="Norm ratio threshold α for suppression (0.80-0.95)")
```

### 7.3 Key Implementation Notes

1. **NMS-only 流程**：
   - 移除 `score_keys_for_round` 调用
   - 移除 `trim_to_max_keys` 调用
   - 每轮只执行 NMS

2. **新增函数**：
   - `compute_q_magnitude_weights()`: 计算 Q 中位数权重
   - `spectrum_aware_cos_sim()`: 计算 SA_cos_sim
   - `fast_parallel_nms_cosine()`: 核心 NMS 算法
   - `incremental_fast_nms_cosine()`: 增量优化版本（可选）

3. **权重计算时机**：
   - 脚本开始时从 stats_trace 计算
   - 每个 head 独立权重
   - 整个实验过程中不更新

4. **指标记录**：
   - `nms_drop_rate`, `nms_total_drops`, `nms_drop_count_per_head`
   - `sim_threshold`, `norm_ratio` 记录在 metrics JSON

---

## 8. Comparison: Cosine Similarity vs Projection Coverage vs Variance-Aware

| 方面 | Cosine Similarity | Projection Coverage | Variance-Aware |
|------|-------------------|---------------------|----------------|
| **抑制条件** | cos_sim > τ AND \|B\| < α\|A\| | proj(A on B) > \|B\| | coverage_score > 0 |
| **归一化** | 自然归一化 ∈ [-1, 1] | 依赖绝对值 | 基于 sign 判定 |
| **阈值设定** | 直观 (τ, α) | 需要调整 ε | 无阈值 (ε=0) |
| **权重来源** | Q-magnitude 中位数 | 频谱能量 | Q-magnitude 百分位 |
| **模长处理** | 显式比较 + ratio | 隐式在 proj 中 | 隐式在 coverage 中 |

---

## 9. References

- Original NMS: Neubeck & Van Gool, "Efficient Non-Maximum Suppression", ICPR 2006
- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
- Variance-Aware NMS: `docs/variance_aware_nms.md`
- Isolated Experiment Design: `docs/variance_aware_nms_isolated_effect.md`
