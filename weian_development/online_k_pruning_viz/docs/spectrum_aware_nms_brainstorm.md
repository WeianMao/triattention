# Spectrum-Aware NMS for KV Cache Compression

## 1. Project Goal & Ablation Study Design

### 1.1 High-Level Objective

本项目的核心目标是验证 **Spectrum-Aware NMS 抑制** 对 KV cache 压缩的有效性。

### 1.2 Ablation Study Principle

**关键原则**：新脚本 `attention_pruning_case_study_hybrid_rounds_xtrace_nms.py` 与原脚本 `attention_pruning_case_study_hybrid_rounds_xtrace.py` 必须构成严格的消融实验：

| 组件 | 原脚本 | 新脚本 | 说明 |
|-----|--------|--------|------|
| 打分函数 | `score_keys_for_round` | **保持一致** | 不修改 |
| 可视化 | heatmap + argmax | **保持一致** | 不修改 |
| 评估指标 | retention rate | **保持一致** | 不修改 |
| 输入输出 | qk.pt, metadata | **保持一致** | 不修改 |
| round-based 框架 | simulate_round_pruning | **保持一致** | 不修改 |
| **NMS 抑制** | 无 | **新增** | 唯一变量 |

**目的**：
1. 验证 Spectrum-Aware NMS 作为 KV 压缩预处理步骤是否有效
2. 对比不同能量统计方法 (amplitude vs causal) 的效果
3. 调优 NMS 阈值等超参数

---

## 2. Background & Motivation

### 2.1 Problem Statement

在现有的 round-based KV cache 压缩框架中，每轮压缩通过打分函数选择 top-k 个 K 保留。但该方法忽略了 K 之间的冗余性：如果两个 K 在频率域中高度相似，保留两个没有意义。

### 2.2 Core Insight

借鉴目标检测中的 NMS (Non-Maximum Suppression)：
- **目标检测 NMS**：两个框如果 IoU 高，保留得分大的框，**删除**得分小的框
- **KV Cache NMS**：两个 K 如果角度相似，保留模长大的 K，**删除**模长小的 K

### 2.3 Why Spectrum-Aware?

在 RoPE 中存在多个频段，但通过可视化发现：
- 不同频段的能量分布不均匀
- 某些频段对 attention 的贡献远大于其他频段
- 简单的 cosine similarity 无法捕捉这种频段重要性差异

因此需要 **Spectrum-Aware Cosine Similarity**：按频段能量加权的相似度度量。

---

## 3. Spectrum-Aware Cosine Similarity (Legacy - DO NOT IMPLEMENT)

> **⚠️ WARNING - 本节内容与本文档主体无关，请勿实现！**
>
> 本节内容为早期设计草案，已被 **完全废弃**。当前的 NMS 实现使用的是 **逐频段投影覆盖计算**（见 Section 5.4），而**不是**这里定义的全局加权 cosine similarity。
>
> **给后续实验者的说明**：
> - ❌ 不要实现本节中的 `SA_cos_sim` 公式
> - ❌ 不要将本节内容与 Section 5 的内容混淆
> - ✅ 请直接跳到 Section 4（Energy Statistics）和 Section 5（NMS Implementation）
>
> 保留本节仅供历史参考，了解设计演变过程。

### 3.1 Notation

- $\tilde{k}_i \in \mathbb{C}^{F}$：**去除 RoPE 后**的 K 向量（通过 `invert_rope` 得到，再用 `to_complex_pairs` 转为复数表示）
- $F$：频段数 = head_dim / 2
- $\text{Real}(\cdot)$：取复数的实部（**注意：这里 Real 表示实部，不是 RoPE**）

### 3.2 Standard Cosine Similarity

$$\text{cos\_sim}(\tilde{k}_i, \tilde{k}_j) = \frac{\text{Real}\left(\sum_f \tilde{k}_i^{(f)} \cdot \overline{\tilde{k}_j^{(f)}}\right)}{|\tilde{k}_i| \cdot |\tilde{k}_j|}$$

### 3.3 Spectrum-Aware Cosine Similarity

引入频段权重 $w_f$（归一化后的频段能量）：

$$\text{SA\_cos\_sim}(\tilde{k}_i, \tilde{k}_j) = \frac{\sum_f w_f \cdot \text{Real}\left(\tilde{k}_i^{(f)} \cdot \overline{\tilde{k}_j^{(f)}}\right)}{\sqrt{\sum_f w_f |\tilde{k}_i^{(f)}|^2} \cdot \sqrt{\sum_f w_f |\tilde{k}_j^{(f)}|^2}}$$

其中 $w_f = \frac{E_f}{\sum_{f'} E_{f'}}$，$E_f$ 是频段 $f$ 的能量。

---

## 4. Energy Statistics Methods

### 4.1 Method A: E[|q| * |k|] - Amplitude Product

**公式**（所有向量均为去除 RoPE 后的复数表示）：
$$E_f = \mathbb{E}_{t} \left[|\tilde{q}_t^{(f)}| \cdot |\tilde{k}_t^{(f)}|\right]$$

**特点**：
- 直接反映 Q-K 在每个频段的交互强度
- 不考虑相位，只关注幅度
- 计算简单：O(n × F)

**实现**：
```python
# q_complex, k_complex: [seq_len, freq_count] - 去除RoPE后的复数表示
q_abs = torch.abs(q_complex)  # [seq_len, freq_count]
k_abs = torch.abs(k_complex)  # [seq_len, freq_count]
energy_per_freq = (q_abs * k_abs).mean(dim=0)  # [freq_count]
weights = energy_per_freq / energy_per_freq.sum()
```

### 4.2 Method B: Causal Attention Weighted (Using Rotated Q/K)

**问题**：如果 Q 和 K 都去除 RoPE 旋转后再点乘，结果与实际 attention score 不一致。

**原因**：实际 attention 计算的是 `rotated_q · rotated_k`，其中 RoPE 引入了位置相关的相位：
$$q^{rot}_i = \tilde{q}_i \cdot e^{i \cdot \text{pos}_i \cdot \omega}$$
$$k^{rot}_j = \tilde{k}_j \cdot e^{i \cdot \text{pos}_j \cdot \omega}$$

因此，实际 attention score 在频段 $f$ 的贡献是：
$$\text{Real}\left(q^{rot,(f)}_i \cdot \overline{k^{rot,(f)}_j}\right) = |\tilde{q}_i^{(f)}| \cdot |\tilde{k}_j^{(f)}| \cdot \cos\left(\phi_{ij}^{(f)} + (\text{pos}_i - \text{pos}_j) \cdot \omega_f\right)$$

其中 $\phi_{ij}^{(f)} = \angle(\tilde{q}_i^{(f)} \cdot \overline{\tilde{k}_j^{(f)}})$ 是 unrotated q/k 的相位差。

**公式**（使用经过 RoPE 旋转后的 Q/K，即原始 qk.pt 中的数据）：

$$E_f = \frac{\sum_{i=1}^{n} \sum_{j=1}^{i} \text{Real}\left(q^{rot,(f)}_i \cdot \overline{k^{rot,(f)}_j}\right)}{\sum_{i=1}^{n} i}$$

其中：
- $q^{rot}, k^{rot}$ 是**经过 RoPE 旋转后**的 Q/K（直接从 qk.pt 加载，转为复数表示）
- $i, j$ 是 token 位置索引
- $j \leq i$ 表示 causal mask
- 分母 $\frac{n(n+1)}{2}$ 是 causal mask 中有效位置数

**特点**：
- 直接反映实际 attention 在每个频段的贡献
- 考虑 RoPE 引入的位置相关相位
- 计算复杂度：O(n² × F)

**实现**：
```python
# q_rotated, k_rotated: [seq_len, head_dim] - 原始的经过RoPE旋转的Q/K
# 转为复数表示
q_complex = to_complex_pairs(q_rotated)  # [seq_len, freq_count]
k_complex = to_complex_pairs(k_rotated)  # [seq_len, freq_count]

# 计算每对 (q_i, k_j) 在每个频段的实际 attention 贡献
attn_contrib = torch.einsum('if,jf->ijf', q_complex, k_complex.conj()).real  # [n, n, F]

# Causal mask: 只保留 j <= i 的位置
causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # [n, n]

# 应用 causal mask
masked = attn_contrib * causal_mask.unsqueeze(-1)  # [n, n, F]

# 对每个频段求均值
num_valid_pairs = causal_mask.sum()  # = n*(n+1)/2
energy_per_freq = masked.sum(dim=(0, 1)) / num_valid_pairs  # [freq_count]

# 归一化
weights = energy_per_freq / energy_per_freq.sum()
```

**注意**：Method A 使用去除 RoPE 后的向量，Method B 使用原始旋转后的向量。两者的 "能量" 含义不同：
- Method A：频段的"潜在"交互能力（不考虑位置）
- Method B：频段的"实际"attention 贡献（包含位置信息）

### 4.3 Recommendation

- **Method A (amplitude)** 作为默认选项：简单高效，与 NMS 的 unrotated K 表示一致
- **Method B (causal)** 作为对比实验：更准确但计算量大，需注意 rotated vs unrotated 的一致性

---

## 5. NMS Implementation: Projection Coverage

### 5.1 Requirement: Sparsity via Hard Drop

我们需要的是 **sparsity**，即真正删除冗余的 K，而不是只降低分数。

### 5.2 Geometric Intuition: Projection Coverage

**核心思想**：如果 A 在 B 方向上的投影"覆盖"了 B，那么 B 就是冗余的，可以被删除。

**单频段情况**：
```
        A
       /
      /
     /  A' (A 在 B 方向的投影)
    /   |
   -----+----→ B
        |
        如果 |A'| > |B|，则 B 被 A 覆盖
```

**算法步骤**：
1. 有向量 A 和 B（从原点出发）
2. 计算 A 在 B 方向上的投影 A'：$|A'| = |A| \cdot \cos(\theta_{AB})$
3. 如果 $|A'| > |B| + \epsilon$，则 B 被 A 抑制

### 5.3 Mathematical Formulation (Single Frequency)

**单频段**：A 抑制 B 当且仅当：

$$|A| \cdot \cos(\theta_{AB}) > |B| + \epsilon$$

即 A 在 B 方向的投影长度超过 B 的模长（加上一个 margin）。

### 5.4 Multi-Frequency Extension

**多频段情况**：对每个频段单独计算 "coverage score"，然后按频谱能量加权求和。

**定义 per-frequency coverage score**：

$$\text{score}_f(A, B) = |A^{(f)}| \cdot \cos(\theta_{AB}^{(f)}) - |B^{(f)}|$$

其中 $\cos(\theta_{AB}^{(f)}) = \frac{\text{Real}(A^{(f)} \cdot \overline{B^{(f)}})}{|A^{(f)}| \cdot |B^{(f)}|}$ 是 A 和 B 在频段 $f$ 的夹角余弦。

**公式**：
$$\text{total\_score}(A, B) = \sum_f w_f \cdot \left( |A^{(f)}| \cdot \cos(\theta_{AB}^{(f)}) - |B^{(f)}| \right)$$

**设计理由**：
- 逐频段计算 cos_sim，更精确地反映每个频段的覆盖关系
- 不同频段可能有不同的方向关系
- 按频谱能量加权求和

**最终公式**：

$$\text{coverage\_score}(A, B) = \sum_f w_f \cdot \left( \frac{\text{Real}(A^{(f)} \cdot \overline{B^{(f)}})}{|B^{(f)}|} - |B^{(f)}| \right)$$

化简（注意 $|A^{(f)}| \cdot \cos(\theta) = \frac{\text{Real}(A \cdot \bar{B})}{|B|}$）：

$$\text{coverage\_score}(A, B) = \sum_f w_f \cdot \left( \frac{\text{Real}(\tilde{k}_A^{(f)} \cdot \overline{\tilde{k}_B^{(f)}})}{|\tilde{k}_B^{(f)}|} - |\tilde{k}_B^{(f)}| \right)$$

**A 抑制 B 当**：$\text{coverage\_score}(A, B) > \epsilon$

### 5.5 Normalization Problem (Deferred)

> **Note**: 本节内容暂时搁置。当前实现强制 $\epsilon = 0$，不允许修改。

**问题**：不同 head 的 K 模长数值范围差异很大。同一个 $\epsilon$ 在不同 head 下意义完全不同。

**当前决定**：$\epsilon = 0$（**强制，不可配置**）

**原因**：在未解决 normalization 问题之前，非零 ε 没有意义。

**未来方向**：见 `docs/variance_aware_nms.md`（独立文档）

### 5.6 Fast Parallel NMS (Matrix Implementation)

**重要说明**：我们使用的是 **Fast NMS**，不是严格的传统 NMS。

**Fast NMS 特点**：
- 一次性计算所有 pair 的支配关系
- C 可能被 B 杀死，即使 B 也被 A 杀死
- 比传统迭代 NMS 更激进（删除更多）
- **我们接受这种"误杀"**

**算法**（逐频段 cos_sim）：
```python
def fast_parallel_nms(
    k_complex: torch.Tensor,      # [N, F] 去RoPE后的K (复数)
    freq_weights: torch.Tensor,   # [F] 频段权重
) -> torch.Tensor:
    """
    Fast Parallel NMS：基于投影覆盖的并行删除

    coverage_score(A, B) = Σ_f w_f * (Real(A_f * conj(B_f)) / |B_f| - |B_f|)
    A 抑制 B 当 coverage_score(A, B) > 0

    Note: epsilon 强制为 0（见 Section 5.5）

    Returns:
        keep_mask: [N] bool tensor, True = 保留
    """
    N, F = k_complex.shape

    # 1. 计算每个 K 每个频段的模长
    k_abs = torch.abs(k_complex)  # [N, F]
    k_abs_safe = k_abs.clamp(min=1e-8)  # 避免除零

    # 2. 计算 A 在 B 方向的投影（逐频段）
    #    proj[a, b, f] = Real(k_a^f * conj(k_b^f)) / |k_b^f|
    #                  = |k_a^f| * cos(theta_ab^f)
    # 使用 einsum: Real(A * conj(B)) for all pairs
    real_dot = torch.einsum('af,bf->abf', k_complex, k_complex.conj()).real  # [N, N, F]
    proj_on_b = real_dot / k_abs_safe.unsqueeze(0)  # [N, N, F]: proj[a,b,f] = A在B方向投影

    # 3. 计算逐频段的 coverage score: proj - |B|
    #    score[a, b, f] = proj[a, b, f] - |k_b^f|
    per_freq_score = proj_on_b - k_abs.unsqueeze(0)  # [N, N, F]

    # 4. 按频谱能量加权求和
    #    total_score[a, b] = Σ_f w_f * score[a, b, f]
    coverage_score = (per_freq_score * freq_weights.view(1, 1, -1)).sum(dim=2)  # [N, N]

    # 5. A 抑制 B 当 coverage_score[a, b] > 0 (epsilon固定为0)
    suppresses = coverage_score > 0  # [N, N]

    # 6. 排除自己抑制自己（对角线）
    suppresses.fill_diagonal_(False)

    # 7. 如果 B 被任何 A 抑制，则删除 B
    is_suppressed = suppresses.any(dim=0)  # [N]: is_suppressed[b] = any A suppresses B
    keep_mask = ~is_suppressed

    return keep_mask
```

**时间复杂度**：O(N² × F)，完全可并行化（GPU friendly）

**内存优化**：对于大 N，可以分块计算避免 O(N² × F) 的中间张量。

### 5.7 Incremental NMS: Optional Computation Optimization

> **Note**: 本节描述的是一种**可选的性能优化**。建议先用 Section 5.6 的简单实现验证算法正确性，必要时再考虑此优化。

由于历史 K 之间已经做过 NMS，只需计算涉及新 K 的块：

```
                    历史 K (H)     新 K (N)
                 ┌─────────────┬───────────┐
   历史 K (H)    │   SKIP      │ 历史被新K │
                 │ (已做过NMS) │  抑制检查 │
                 ├─────────────┼───────────┤
   新 K (N)      │ 新K被历史K  │  新K互相  │
                 │  抑制检查   │  抑制检查 │
                 └─────────────┴───────────┘
```

**需要计算**：
- 历史K 抑制 新K (H × N)：检查新 K 是否被历史 K 抑制
- 新K 抑制 历史K (N × H)：检查历史 K 是否被新 K 抑制
- 新K 抑制 新K (N × N)：新 K 之间互相检查

**复杂度**：从 O((H+N)² × F) 降到 O((H×N + N²) × F)

```python
def incremental_fast_nms(
    historical_k: torch.Tensor,   # [H, F] 历史 K (复数)
    new_k: torch.Tensor,          # [N, F] 新 K (复数)
    freq_weights: torch.Tensor,   # [F] 频段权重
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    增量 Fast NMS（基于投影覆盖）

    coverage_score(A, B) = Σ_f w_f * (Real(A_f * conj(B_f)) / |B_f| - |B_f|)
    A 抑制 B 当 coverage_score(A, B) > 0

    Note: epsilon 强制为 0（见 Section 5.5）

    Returns:
        historical_keep: [H] bool, 历史 K 是否保留
        new_keep: [N] bool, 新 K 是否保留
    """
    H, N, F = historical_k.shape[0], new_k.shape[0], historical_k.shape[1]

    # 预处理：计算模长
    hist_abs = torch.abs(historical_k)  # [H, F]
    hist_abs_safe = hist_abs.clamp(min=1e-8)
    new_abs = torch.abs(new_k)  # [N, F]
    new_abs_safe = new_abs.clamp(min=1e-8)

    # 1. 历史K 抑制 新K: coverage_score[h, n] = Σ_f w_f * (Real(hist_h * conj(new_n)) / |new_n| - |new_n|)
    real_dot_hist_new = torch.einsum('hf,nf->hnf', historical_k, new_k.conj()).real  # [H, N, F]
    proj_hist_on_new = real_dot_hist_new / new_abs_safe.unsqueeze(0)  # [H, N, F]
    score_hist_new = proj_hist_on_new - new_abs.unsqueeze(0)  # [H, N, F]
    coverage_hist_new = (score_hist_new * freq_weights.view(1, 1, -1)).sum(dim=2)  # [H, N]
    new_suppressed_by_hist = (coverage_hist_new > 0).any(dim=0)  # [N]

    # 2. 新K 抑制 历史K: coverage_score[n, h] = Σ_f w_f * (Real(new_n * conj(hist_h)) / |hist_h| - |hist_h|)
    real_dot_new_hist = real_dot_hist_new.transpose(0, 1)  # [N, H, F] (Real(A*conj(B)) = Real(B*conj(A)))
    proj_new_on_hist = real_dot_new_hist / hist_abs_safe.unsqueeze(0)  # [N, H, F]
    score_new_hist = proj_new_on_hist - hist_abs.unsqueeze(0)  # [N, H, F]
    coverage_new_hist = (score_new_hist * freq_weights.view(1, 1, -1)).sum(dim=2)  # [N, H]
    hist_suppressed_by_new = (coverage_new_hist > 0).any(dim=0)  # [H]

    # 3. 新K 抑制 新K: coverage_score[n, m] = Σ_f w_f * (Real(new_n * conj(new_m)) / |new_m| - |new_m|)
    real_dot_new_new = torch.einsum('nf,mf->nmf', new_k, new_k.conj()).real  # [N, N, F]
    proj_new_on_new = real_dot_new_new / new_abs_safe.unsqueeze(0)  # [N, N, F]
    score_new_new = proj_new_on_new - new_abs.unsqueeze(0)  # [N, N, F]
    coverage_new_new = (score_new_new * freq_weights.view(1, 1, -1)).sum(dim=2)  # [N, N]
    coverage_new_new.fill_diagonal_(float('-inf'))  # 排除自己
    new_suppressed_by_new = (coverage_new_new > 0).any(dim=0)  # [N]

    # 4. 汇总
    new_keep = ~(new_suppressed_by_hist | new_suppressed_by_new)
    historical_keep = ~hist_suppressed_by_new

    return historical_keep, new_keep
```

### 5.8 Complexity Analysis (For Optional Incremental Optimization)

| 方法 | 时间复杂度 | 空间复杂度 |
|-----|-----------|-----------|
| 简单实现 (每轮全矩阵) | O((H+N)² × F) | O((H+N)² × F) |
| **优化实现 (Incremental)** | O((H×N + N²) × F) | O((H×N + N²) × F) |

**典型场景**：H=2048, N=64 (round_window)
- 简单实现: (2048+64)² ≈ 4.5M 次计算
- 优化实现: 2048×64 + 64² ≈ 135K 次计算
- **加速比: ~33x**

**建议**：对于实验验证，简单实现的性能通常足够。仅在处理大规模数据时考虑优化实现。

---

## 6. Integration into Round-Based Pruning

### 6.1 Modified Round Workflow

```
每轮开始时 (每 round_window 个 token):
    1. [NEW] Spectrum-Aware NMS Hard Drop
       - 对当前所有 K（历史 + 新）进行 Fast Parallel NMS
       - 计算投影覆盖分数 coverage_score（见 Section 5.4）
       - 删除被其他 K 覆盖的 K（真正从 cache 中移除）
       - 使用预计算的频段权重 freq_weights（见 Section 8.1）

    2. [EXISTING - 保持不变] 打分函数 (score_keys_for_round)
       - 基于频率域预测未来 attention

    3. [EXISTING - 保持不变] Top-K 选择
       - 保留分数最高的 max_keys 个 K
```

**Note**: Section 5.7 中描述的 Incremental NMS 是一种**可选的计算优化**，用于减少重复计算。核心算法仍然是 Section 5.6 中的 Fast Parallel NMS。实现时可以选择：
- **简单实现**：每轮对所有 K 调用 `fast_parallel_nms()`
- **优化实现**：使用 `incremental_fast_nms()` 跳过历史 K 之间的重复计算

---

## 7. Experimental Design

### 7.1 Experiment Variables

| 参数 | 选项 | 说明 |
|-----|------|------|
| `--nms-enabled` | True/False | 是否启用 NMS |
| `--energy-method` | amplitude / causal | 能量统计方法 |

> **Note**: `epsilon` 固定为 0（见 Section 5.5），不作为搜索参数。

### 7.2 Automated Hyperparameter Search

**不要手动执行实验**，使用自动搜索脚本。

**搜索空间**：
```python
SEARCH_SPACE = {
    "energy_method": ["amplitude", "causal"],
    # epsilon 固定为 0，不搜索
}
```

**自动搜索脚本设计**：
```bash
# run_nms_sweep.sh
#!/bin/bash
ENERGY_METHODS=("amplitude" "causal")

# Baseline (无 NMS)
python attention_pruning_case_study_hybrid_rounds_xtrace_nms.py \
    $INPUT_ROOT --trace $TRACE --stats-trace $STATS_TRACE \
    --output-root results/baseline

# NMS variants
for method in "${ENERGY_METHODS[@]}"; do
    python attention_pruning_case_study_hybrid_rounds_xtrace_nms.py \
        $INPUT_ROOT --trace $TRACE --stats-trace $STATS_TRACE \
        --nms-enabled --energy-method $method \
        --output-root results/nms_${method}
done
```

**搜索结果汇总**：脚本应自动生成汇总表：
```
| energy_method | retention_rate | nms_drop_rate |
|---------------|----------------|---------------|
| baseline      | 0.85           | 0%            |
| amplitude     | 0.82           | 15%           |
| causal        | 0.84           | 10%           |
```

### 7.3 Evaluation (与原脚本完全一致)

1. **Retention Rate**: argmax 命中率
2. **Attention Heatmap**: 可视化对比
3. **Per-layer Statistics**: 每层的压缩效果
4. **Metrics JSON**: retention_metrics.json

### 7.4 Additional Metrics for NMS

新增 NMS 相关指标（记录在 metrics JSON 中）：
- `nms_drop_rate`: 每轮被 NMS 删除的 K 的比例
- `nms_drop_count_per_round`: 每轮删除的 K 数量列表
- `avg_shadow_depth`: 被删除 K 的平均 shadow depth（用于分析 epsilon 选择）

---

## 8. Design Decisions (Confirmed)

### 8.1 频段权重计算策略

**决定**：每个 head 在脚本开始时计算一次频段权重，之后在整个 round-based pruning 过程中**保持不变**。

**与原脚本一致**：这与 `attention_pruning_case_study_hybrid_rounds_xtrace.py` 的设计完全一致——在脚本开始时从 `stats_trace` 加载/计算统计量，然后在所有 round 中重复使用，不再更新。

**理由**：
- 使用 stats_trace 预计算的频段能量权重
- 避免每轮重复计算的开销
- 频段能量分布是 head 的固有属性，不随 round 变化

**实现时机**：
```python
# 在脚本开始时，加载 stats_trace 后立即计算
freq_weights = compute_frequency_energy_weights(stats_trace, method=args.energy_method)
# 之后在所有 round 中使用同一个 freq_weights，不再更新
```

### 8.2 NMS 作用范围

**决定**：全局 NMS，每轮对所有 K 进行 NMS 检查。

**简单实现**：每轮直接对所有 K 调用 `fast_parallel_nms()`。

**可选优化**（Incremental NMS，见 Section 5.7）：利用历史 K 之间已经做过 NMS 的事实，只需计算：
1. 新 K vs 历史 K
2. 新 K vs 新 K
3. 历史 K vs 新 K（检查历史 K 是否被新 K 支配）

**建议**：先用简单实现验证算法正确性，性能瓶颈明显时再考虑优化实现。

### 8.3 NMS 类型

**决定**：Fast Parallel NMS（真正删除被支配的 K）

**理由**：
- 需要实现 sparsity，而非只调整分数
- Fast NMS 允许"误杀"（已被杀的 K 仍可杀其他 K），这是可接受的
- 完全可并行化，GPU friendly

### 8.4 抑制条件

**决定**：使用投影覆盖条件（Projection Coverage）

**公式**：
$$\text{coverage\_score}(A, B) = \sum_f w_f \cdot \left( \frac{\text{Real}(A^{(f)} \cdot \overline{B^{(f)}})}{|B^{(f)}|} - |B^{(f)}| \right) > \epsilon$$

**几何意义**：A 在 B 方向的投影超过了 B 的模长 → A "覆盖" 了 B → B 冗余

**理由**：
- 逐频段计算投影，更精确
- 按频谱能量加权求和
- 物理意义清晰：A 覆盖 B

### 8.5 Epsilon 设置

**当前决定**：$\epsilon = 0$（**强制，不可配置**）

**原因**：不同 head 的数值范围差异大，非零 ε 无法跨 head 使用。

**未来改进**：见 `docs/variance_aware_nms.md`（独立文档）中的 normalization 方案。

---

## 9. Implementation Checklist

### 9.1 Code Structure

```
weian_development/online_k_pruning_viz/
├── attention_pruning_case_study_hybrid_rounds_xtrace.py      # 原始脚本 (不修改)
├── attention_pruning_case_study_hybrid_rounds_xtrace_nms.py  # 新脚本
├── run_nms_sweep.sh                                          # 自动超参数搜索脚本
└── docs/
    ├── spectrum_aware_nms_brainstorm.md  # 本文档
    └── variance_aware_nms.md             # Variance-Aware NMS (Future Work)
```

### 9.2 New Arguments to Add

```python
parser.add_argument("--nms-enabled", action="store_true", default=False,
                    help="Enable spectrum-aware NMS before scoring")
parser.add_argument("--energy-method", choices=["amplitude", "causal"], default="amplitude",
                    help="Method for computing frequency band energy weights")
# Note: epsilon 固定为 0，不提供参数配置（见 Section 5.5）
```

### 9.3 Key Implementation Notes

1. **严格保持与原脚本一致的部分**：
   - `score_keys_for_round` 函数
   - `compute_pooled_attention` 函数
   - `save_comparison_figure` 函数
   - 所有可视化和指标计算
   - 命令行参数（NMS 相关参数之外）

2. **新增部分（仅 NMS）**：
   - 频段能量权重计算函数 `compute_frequency_energy_weights()`
   - Fast NMS 函数 `fast_parallel_nms()`（核心算法）
   - 可选：`incremental_fast_nms()`（性能优化版本）
   - 在 `simulate_round_pruning` 每轮开始处调用 NMS

3. **能量权重计算时机**（与原脚本一致）：
   - 在脚本开始时、加载 stats_trace 后一次性计算
   - 每个 head 有独立的权重
   - 之后在所有 round 中重复使用，**不再更新**

4. **NMS 指标记录**：
   - 在 metrics JSON 中新增 `nms_drop_rate`, `nms_drop_count_per_round`

---

## 10. References

- Original NMS: Neubeck & Van Gool, "Efficient Non-Maximum Suppression", ICPR 2006
- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
