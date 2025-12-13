# Q-Magnitude Weighted NMS for KV Cache Compression

> 本文档是 KV Cache 压缩中 NMS (Non-Maximum Suppression) 方法的完整设计文档。
>
> **本文档完全独立**，不依赖于其他文档。

---

## 1. Project Goal & Ablation Study Design

### 1.1 High-Level Objective

本项目的核心目标是验证 **Q-Magnitude Weighted NMS 抑制** 对 KV cache 压缩的有效性。

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
1. 验证 Q-Magnitude Weighted NMS 作为 KV 压缩预处理步骤是否有效
2. 对比不同权重计算方法的效果
3. 调优 percentile 等超参数

---

## 2. Background & Motivation

### 2.1 Problem Statement

在现有的 round-based KV cache 压缩框架中，每轮压缩通过打分函数选择 top-k 个 K 保留。但该方法忽略了 K 之间的冗余性：如果两个 K 在频率域中高度相似，保留两个没有意义。

### 2.2 Core Insight

借鉴目标检测中的 NMS (Non-Maximum Suppression)：
- **目标检测 NMS**：两个框如果 IoU 高，保留得分大的框，**删除**得分小的框
- **KV Cache NMS**：两个 K 如果角度相似，保留模长大的 K，**删除**模长小的 K

### 2.3 Why Frequency-Aware?

在 RoPE 中存在多个频段，但通过可视化发现：
- 不同频段的重要性不均匀
- 某些频段对 attention 的贡献远大于其他频段
- 简单的 cosine similarity 无法捕捉这种频段重要性差异

因此需要 **加权的相似度度量**：按频段重要性加权。

---

## 3. 核心设计：为什么用 Q 模长作为权重

### 3.1 早期错误设计（已废弃）

早期设计使用"频段能量"作为权重：

$$w_f = \frac{E_f}{\sum_{f'} E_{f'}}$$

其中 $E_f$ 是频段 $f$ 的"能量"（通过 amplitude product 或 causal attention 计算）。

### 3.2 为什么这是错的

考虑 attention 的计算过程：

$$\text{score}_{ij} = q_i \cdot k_j = |q_i| \cdot |k_j| \cdot \cos(\theta_{ij})$$

在我们的 NMS coverage score 公式中（见 Section 5.3）：

$$\text{coverage\_score}(A, B) = \sum_f w_f \cdot \left( \frac{\text{Real}(A^{(f)} \cdot \overline{B^{(f)}})}{|B^{(f)}| + \varepsilon_{\text{den}}} - |B^{(f)}| \right)$$

**关键观察**：
- **K 的模长已经体现在公式中**：投影 $\frac{\text{Real}(A \cdot \overline{B})}{|B|}$ 和被投影向量 $|B|$ 都是 K 的模长
- **唯一没考虑到的变量是 Q 的模长 $|Q|$**

**正确的理解**：
- coverage_score 判断的是 "A 能否覆盖 B"
- 但实际 attention 中，不同 Q 对 K 的"敏感度"不同
- 模长大的 Q 对 K 之间的差异更敏感
- 因此，权重应该反映 **Q 的模长分布**，而不是某种"能量"

### 3.3 正确的权重设计

**权重定义**：

$$w_f = |\tilde{q}^{(f)}|$$

其中 $|\tilde{q}^{(f)}|$ 是 Q 在频段 $f$ 的模长。

**注意**：这里**不需要归一化**。因为当 $\epsilon = 0$ 时，判断条件是 `coverage_score > 0`，归一化只是将 score 除以一个正常数 $\sum_{f'} |q^{(f')}|$，不影响正负号判断。

**物理意义**：
- 模长大的 Q 频段对 attention 的贡献更大
- 在这些频段，K 之间的差异更重要
- NMS 应该更重视这些频段的覆盖关系

---

## 4. Notation

- $k^{rot} \in \mathbb{R}^{D}$：经过 RoPE 旋转后的 K 向量（直接从 qk.pt 加载）
- $\tilde{k} \in \mathbb{C}^{F}$：去除 RoPE 后的 K 向量（通过 `invert_rope` 得到，再用 `to_complex_pairs` 转为复数表示）
- $\tilde{q} \in \mathbb{C}^{F}$：去除 RoPE 后的 Q 向量
- $F$：频段数 = head_dim / 2
- $\text{Real}(\cdot)$：取复数的实部
- $\overline{z}$：复数 $z$ 的共轭

**重要说明**：
- **NMS 在 RoPE 旋转后的空间执行**：使用 $k^{rot}$ 转为复数后进行计算
- **权重统计在去 RoPE 空间执行**：使用 $\tilde{q}$ 计算 Q 模长分布

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

### 5.3 Mathematical Formulation

#### 5.3.1 Single Frequency

**单频段**：A 抑制 B 当且仅当：

$$|A| \cdot \cos(\theta_{AB}) > |B| + \epsilon$$

即 A 在 B 方向的投影长度超过 B 的模长（加上一个 margin）。

#### 5.3.2 Multi-Frequency Extension

**多频段情况**：对每个频段单独计算 "coverage score"，然后按 Q 模长加权求和。

**定义 per-frequency coverage score**（在 RoPE 旋转后的 K 空间，记作 $k^{rot}$；后续公式中的 $A,B$ 均为 $k^{rot}$ 转为复数后的表示）：

$$\text{score}_f(A, B) = |A^{(f)}| \cdot \cos(\theta_{AB}^{(f)}) - |B^{(f)}|$$

其中 $\cos(\theta_{AB}^{(f)}) = \frac{\text{Real}(A^{(f)} \cdot \overline{B^{(f)}})}{|A^{(f)}| \cdot |B^{(f)}|}$ 是 A 和 B 在频段 $f$ 的夹角余弦。

**公式**：
$$\text{total\_score}(A, B) = \sum_f w_f \cdot \left( |A^{(f)}| \cdot \cos(\theta_{AB}^{(f)}) - |B^{(f)}| \right)$$

**设计理由**：
- 逐频段计算 cos_sim，更精确地反映每个频段的覆盖关系
- 不同频段可能有不同的方向关系
- 按 Q 模长加权求和

**最终公式**：

$$\text{coverage\_score}(A, B) = \sum_f w_f \cdot \left( \frac{\text{Real}(A^{(f)} \cdot \overline{B^{(f)}})}{|B^{(f)}| + \varepsilon_{\text{den}}} - |B^{(f)}| \right)$$

其中 $\varepsilon_{\text{den}}$ 为防止除零/爆数的极小正数（实现中使用 $1\times 10^{-5}$）。

化简（注意 $|A^{(f)}| \cdot \cos(\theta) = \frac{\text{Real}(A \cdot \bar{B})}{|B| + \varepsilon_{\text{den}}}$）：

$$\text{coverage\_score}(A, B) = \sum_f w_f \cdot \left( \frac{\text{Real}(k^{rot,(f)}_A \cdot \overline{k^{rot,(f)}_B})}{|k^{rot,(f)}_B| + \varepsilon_{\text{den}}} - |k^{rot,(f)}_B| \right)$$

**A 抑制 B 当**：$\text{coverage\_score}(A, B) > \epsilon$

### 5.4 Epsilon and Normalization Problem

**问题**：不同 head 的 K 模长数值范围差异很大。同一个 $\epsilon$ 在不同 head 下意义完全不同。

**具体例子**：
- Head A 的 K 模长范围：$[1, 10]$
- Head B 的 K 模长范围：$[100, 1000]$
- 同一个 $\epsilon = 0.1$ 在 Head A 中很显著，在 Head B 中几乎无影响

**当前决定**：$\epsilon = 0$（**强制，不可配置**）

**原因**：在未解决 normalization 问题之前，非零 ε 没有意义。

**未来方向**：见 Section 10（Normalization Ideas）

### 5.5 Fast Parallel NMS (Matrix Implementation)

**重要说明**：我们使用的是 **Fast NMS**，不是严格的传统 NMS。

**Fast NMS 特点**：
- 一次性计算所有 pair 的支配关系
- C 可能被 B 杀死，即使 B 也被 A 杀死
- 比传统迭代 NMS 更激进（删除更多）
- **我们接受这种"误杀"**

**时间复杂度**：O(N² × F)，完全可并行化（GPU friendly）

**内存优化**：对于大 N，可以分块计算避免 O(N² × F) 的中间张量。

### 5.6 Incremental NMS: Optional Computation Optimization

> **Note**: 本节描述的是一种**可选的性能优化**。建议先用简单实现验证算法正确性，必要时再考虑此优化。

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

**典型场景**：H=2048, N=64 (round_window)
- 简单实现: (2048+64)² ≈ 4.5M 次计算
- 优化实现: 2048×64 + 64² ≈ 135K 次计算
- **加速比: ~33x**

---

## 6. Variance-Aware: Conservative Weight Selection

### 6.1 Problem Statement

当前方法假设每个频段的权重 $w_f$ 是固定的（从 stats_trace 预计算）。但实际上：

1. **Q 的模长不是固定的**：不同 query position 下，每个频段的 Q 模长会有变化
2. **存在方差**：同一频段的 Q 模长在不同 position 下有一个范围
3. **极端值问题**：某些 query 的 Q 模长可能处于 top 5% 或 bottom 5% 的极端值

**核心问题**：当前的判断 `coverage_score > 0` 只考虑了 Q 模长的期望值，没有考虑方差可能导致的极端情况。

### 6.2 Intuition

考虑这样的情况：
- 频段 f 的平均 Q 模长是 $E[|q^{(f)}|] = 0.1$
- 但它的方差很大，95% 区间是 $[0.01, 0.5]$

如果用期望值作为权重计算 coverage_score，在极端情况下（top 5%），这个频段的 Q 模长可能是 0.5，远超期望值。

**问题**：如果 B 只在某些极端 Q 模长情况下"逃脱" A 的覆盖，我们是否仍应该删除 B？

**可能的答案**：
- **保守策略**：只有当 B 在 95% 的 Q 模长情况下都被 A 覆盖时，才删除 B
- **激进策略**：只要 B 在期望 Q 模长情况下被 A 覆盖，就删除 B

### 6.3 Design Goal

**B 只有在即使极端 Q 模长情况下也被 A 覆盖时，才被抑制。**

这意味着我们需要使用"最不利于抑制"的 Q 模长权重来计算 coverage_score。

### 6.4 Key Insight: Score Sign Matters

回顾 coverage_score 公式：

$$\text{coverage\_score}(A, B) = \sum_f w_f \cdot \underbrace{\left( \frac{\text{Real}(A^{(f)} \cdot \overline{B^{(f)}})}{|B^{(f)}| + \varepsilon_{\text{den}}} - |B^{(f)}| \right)}_{\text{per\_freq\_score}_f}$$

**关键观察**：每个频段的贡献 $\text{per\_freq\_score}_f$ 可正可负：
- **正分**（$\text{per\_freq\_score}_f > 0$）：A 的投影覆盖了 B，贡献于抑制
- **负分**（$\text{per\_freq\_score}_f < 0$）：A 的投影不足，反对抑制

### 6.5 Conservative Weight Selection

**原则**：使用最不利于抑制的权重。

对于每个频段 f：
- 若 $\text{per\_freq\_score}_f > 0$（贡献抑制）→ 使用 $w_f^{\text{low}}$（低估其贡献）
- 若 $\text{per\_freq\_score}_f < 0$（反对抑制）→ 使用 $w_f^{\text{high}}$（放大其阻力）

### 6.6 Percentile Statistics（基于 Q 模长分布）

对每个频段 f，从 stats_trace 中统计 **Q 的模长分布**：

- $w_f^{\text{low}}$ = 5th percentile of $|q^{(f)}|$ across all queries
- $w_f^{\text{high}}$ = 95th percentile of $|q^{(f)}|$ across all queries

**注意**：**不需要归一化**。因为 $\epsilon = 0$，判断条件是 `coverage_score > 0`，归一化不影响正负号判断。

### 6.7 Final Formula

$$\text{conservative\_coverage\_score}(A, B) = \sum_f w_f^{*} \cdot \text{per\_freq\_score}_f$$

其中：
$$w_f^{*} = \begin{cases}
w_f^{\text{low}} & \text{if } \text{per\_freq\_score}_f > 0 \\
w_f^{\text{high}} & \text{if } \text{per\_freq\_score}_f \leq 0
\end{cases}$$

**判断条件**：
$$\text{A 抑制 B 当且仅当 } \text{conservative\_coverage\_score}(A, B) > 0$$

### 6.8 Geometric Interpretation

- 对于"帮助抑制"的频段，我们假设 Q 模长较小（取 low percentile），从而低估该频段的重要性
- 对于"阻止抑制"的频段，我们假设 Q 模长较大（取 high percentile），从而放大该频段的阻力
- 只有当在这种最不利情况下 A 仍能覆盖 B，我们才抑制 B

**结果**：更保守的 NMS，减少误删。

---

## 7. Implementation

### 7.1 Precomputation: Q-Magnitude Percentile Statistics

> **重要**：percentile 权重在**脚本开始时计算一次**，之后在所有 round 中**保持不变，不再更新**。这与 `attention_pruning_case_study_hybrid_rounds_xtrace.py` 的设计完全一致。

```python
def compute_q_magnitude_percentile_weights(
    q_complex: torch.Tensor,  # [num_samples, freq_count] Q 向量（去除RoPE后的复数表示）
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算每个频段的 Q 模长分布的 low 和 high percentile 权重。

    注意：此函数应在脚本开始时调用一次，返回的权重在所有 round 中重复使用。
    注意：不需要归一化，因为 epsilon=0 时判断条件是 >0，归一化不影响正负号。

    Args:
        q_complex: 从 stats_trace 加载的 Q 向量（去除RoPE后的复数表示）
        low_percentile: 低百分位数 (default 5%)
        high_percentile: 高百分位数 (default 95%)

    Returns:
        w_low: [freq_count] 5th percentile Q-magnitude weights
        w_high: [freq_count] 95th percentile Q-magnitude weights
    """
    # 计算每个 Q 在每个频段的模长
    q_magnitudes = torch.abs(q_complex)  # [num_samples, freq_count]

    # 计算每个频段的 percentile（不需要归一化）
    w_low = torch.quantile(q_magnitudes, low_percentile / 100.0, dim=0)  # [F]
    w_high = torch.quantile(q_magnitudes, high_percentile / 100.0, dim=0)  # [F]

    return w_low, w_high
```

**调用时机**：
```python
# 在脚本开始时，加载 stats_trace 后立即计算
# 需要从 stats_trace 中提取 Q 向量，转为复数表示
q_complex = to_complex_pairs(invert_rope(stats_q))  # 去除RoPE后转为复数
w_low, w_high = compute_q_magnitude_percentile_weights(q_complex)
# 之后在所有 round 中使用同一个 w_low, w_high，不再更新
```

### 7.2 Variance-Aware Fast NMS

```python
def variance_aware_fast_nms(
    k_complex: torch.Tensor,      # [N, F] RoPE旋转后的K (原始 qk.pt) 转为复数
    w_low: torch.Tensor,          # [F] 5th percentile Q-magnitude weights
    w_high: torch.Tensor,         # [F] 95th percentile Q-magnitude weights
) -> torch.Tensor:
    """
    Variance-Aware Fast Parallel NMS：使用保守的 Q 模长权重选择

    对于每对 (A, B)，每个频段 f：
    - 若 per_freq_score[a, b, f] > 0：使用 w_low[f]（假设 Q 模长小）
    - 若 per_freq_score[a, b, f] <= 0：使用 w_high[f]（假设 Q 模长大）

    A 抑制 B 当 conservative_coverage_score(A, B) > 0

    Returns:
        keep_mask: [N] bool tensor, True = 保留
    """
    N, F = k_complex.shape

    # 1. 计算每个 K 每个频段的模长
    k_abs = torch.abs(k_complex)  # [N, F]
    k_abs_safe = k_abs.clamp(min=1e-5)  # 避免除零/爆数

    # 2. 计算 A 在 B 方向的投影（逐频段）
    #    proj[a, b, f] = Real(k_a^f * conj(k_b^f)) / |k_b^f|
    #                  = |k_a^f| * cos(theta_ab^f)
    real_dot = torch.einsum('af,bf->abf', k_complex, k_complex.conj()).real  # [N, N, F]
    proj_on_b = real_dot / k_abs_safe.unsqueeze(0)  # [N, N, F]

    # 3. 计算逐频段的 coverage score
    #    score[a, b, f] = proj[a, b, f] - |k_b^f|
    per_freq_score = proj_on_b - k_abs.unsqueeze(0)  # [N, N, F]

    # 4. 根据 score 符号选择权重
    #    positive score → use w_low (Q 模长小时该频段贡献小)
    #    negative score → use w_high (Q 模长大时该频段阻力大)
    weights = torch.where(
        per_freq_score > 0,
        w_low.view(1, 1, -1),   # 正分用 low
        w_high.view(1, 1, -1),  # 负分用 high
    )  # [N, N, F]

    # 5. 加权求和
    conservative_score = (per_freq_score * weights).sum(dim=2)  # [N, N]

    # 6. A 抑制 B 当 conservative_score > 0
    suppresses = conservative_score > 0  # [N, N]
    suppresses.fill_diagonal_(False)

    # 7. 如果 B 被任何 A 抑制，则删除 B
    is_suppressed = suppresses.any(dim=0)
    keep_mask = ~is_suppressed

    return keep_mask
```

### 7.3 Incremental Version (Optional Performance Optimization)

> **Note**: 本节描述的是一种**可选的性能优化**。建议先用 Section 7.2 的简单实现验证算法正确性，必要时再考虑此优化。

```python
def variance_aware_incremental_nms(
    historical_k: torch.Tensor,   # [H, F] 历史 K (RoPE 旋转后) 复数
    new_k: torch.Tensor,          # [N, F] 新 K (RoPE 旋转后) 复数
    w_low: torch.Tensor,          # [F] 5th percentile Q-magnitude weights
    w_high: torch.Tensor,         # [F] 95th percentile Q-magnitude weights
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variance-Aware 增量 Fast NMS（可选优化版本）

    Returns:
        historical_keep: [H] bool, 历史 K 是否保留
        new_keep: [N] bool, 新 K 是否保留
    """
    H, N, F = historical_k.shape[0], new_k.shape[0], historical_k.shape[1]

    # 预处理
    hist_abs = torch.abs(historical_k)
    hist_abs_safe = hist_abs.clamp(min=1e-5)
    new_abs = torch.abs(new_k)
    new_abs_safe = new_abs.clamp(min=1e-5)

    # --- Block 1: 历史K 抑制 新K ---
    real_dot_hist_new = torch.einsum('hf,nf->hnf', historical_k, new_k.conj()).real
    proj_hist_on_new = real_dot_hist_new / new_abs_safe.unsqueeze(0)
    score_hist_new = proj_hist_on_new - new_abs.unsqueeze(0)  # [H, N, F]

    weights_hist_new = torch.where(
        score_hist_new > 0,
        w_low.view(1, 1, -1),
        w_high.view(1, 1, -1),
    )
    coverage_hist_new = (score_hist_new * weights_hist_new).sum(dim=2)
    new_suppressed_by_hist = (coverage_hist_new > 0).any(dim=0)

    # --- Block 2: 新K 抑制 历史K ---
    real_dot_new_hist = real_dot_hist_new.transpose(0, 1)
    proj_new_on_hist = real_dot_new_hist / hist_abs_safe.unsqueeze(0)
    score_new_hist = proj_new_on_hist - hist_abs.unsqueeze(0)  # [N, H, F]

    weights_new_hist = torch.where(
        score_new_hist > 0,
        w_low.view(1, 1, -1),
        w_high.view(1, 1, -1),
    )
    coverage_new_hist = (score_new_hist * weights_new_hist).sum(dim=2)
    hist_suppressed_by_new = (coverage_new_hist > 0).any(dim=0)

    # --- Block 3: 新K 抑制 新K ---
    real_dot_new_new = torch.einsum('nf,mf->nmf', new_k, new_k.conj()).real
    proj_new_on_new = real_dot_new_new / new_abs_safe.unsqueeze(0)
    score_new_new = proj_new_on_new - new_abs.unsqueeze(0)  # [N, N, F]

    weights_new_new = torch.where(
        score_new_new > 0,
        w_low.view(1, 1, -1),
        w_high.view(1, 1, -1),
    )
    coverage_new_new = (score_new_new * weights_new_new).sum(dim=2)
    coverage_new_new.fill_diagonal_(float('-inf'))
    new_suppressed_by_new = (coverage_new_new > 0).any(dim=0)

    # 汇总
    new_keep = ~(new_suppressed_by_hist | new_suppressed_by_new)
    historical_keep = ~hist_suppressed_by_new

    return historical_keep, new_keep
```

---

## 8. Integration into Round-Based Pruning

### 8.1 Modified Round Workflow

```
每轮开始时 (每 round_window 个 token):
    1. [NEW] Q-Magnitude Weighted NMS Hard Drop
       - 对当前所有 K（历史 + 新）进行 Fast Parallel NMS
       - 计算投影覆盖分数 coverage_score（见 Section 5.3）
       - 使用 Variance-Aware 权重选择（见 Section 6）
       - 删除被其他 K 覆盖的 K（真正从 cache 中移除）
       - 使用预计算的 Q 模长 percentile 权重（见 Section 7.1）

    2. [EXISTING - 保持不变] 打分函数 (score_keys_for_round)
       - 基于频率域预测未来 attention

    3. [EXISTING - 保持不变] Top-K 选择
       - 保留分数最高的 max_keys 个 K
```

**实现选择**：
- **简单实现**：每轮对所有 K 调用 `variance_aware_fast_nms()`
- **优化实现**：使用 `variance_aware_incremental_nms()` 跳过历史 K 之间的重复计算

**Online 约束说明**：NMS 只作用于"当前已在缓存中的 K"，其中"新 K"指上一轮生成并已加入缓存的 K。未来 token 的 K/查询均不可见，不应参与 NMS。

---

## 9. Experimental Design

### 9.1 Experiment Variables

| 参数 | 选项 | 说明 |
|-----|------|------|
| `--nms-enabled` | True/False | 是否启用 NMS |
| `--low-percentile` | 1, 5, 10, 25 | Q 模长 low percentile |
| `--high-percentile` | 75, 90, 95, 99 | Q 模长 high percentile |

> **Note**: `epsilon` 固定为 0（见 Section 5.4），不作为搜索参数。

### 9.2 Automated Hyperparameter Search

**不要手动执行实验**，使用自动搜索脚本。

**搜索空间**：
```python
SEARCH_SPACE = {
    "low_percentile": [5, 10, 25],
    "high_percentile": [75, 90, 95],
    # epsilon 固定为 0，不搜索
}
```

### 9.3 Evaluation (与原脚本完全一致)

1. **Retention Rate**: argmax 命中率
2. **Attention Heatmap**: 可视化对比
3. **Per-layer Statistics**: 每层的压缩效果
4. **Metrics JSON**: retention_metrics.json

### 9.4 Additional Metrics for NMS

新增 NMS 相关指标（记录在 metrics JSON 中）：
- `nms_drop_rate`: 每轮被 NMS 删除的 K 的比例
- `nms_drop_count_per_round`: 每轮删除的 K 数量列表

---

## 10. Normalization Ideas (Future Work)

### 10.1 Per-Head Mean Normalization

$$\text{normalized\_score}(A, B) = \frac{\text{coverage\_score}(A, B)}{\bar{|K|}_{\text{head}}}$$

### 10.2 Z-Score Normalization

1. 计算该 head 中所有 pair 的 coverage_score 分布
2. 计算 $\mu_{\text{head}}, \sigma_{\text{head}}$
3. $\text{z\_score}(A, B) = \frac{\text{coverage\_score}(A, B) - \mu_{\text{head}}}{\sigma_{\text{head}}}$

**判断条件**：$\text{z\_score}(A, B) > z_{\text{threshold}}$

### 10.3 Percentile-Based Threshold

1. 计算该 head 中所有 pair 的 coverage_score
2. 使用百分位数作为阈值：如 "score > 90th percentile → suppress"

**优点**：不需要手动设置 ε，自动适应不同 head 的数值范围。

### 10.4 Relative Margin

将 ε 定义为相对于 |B| 的比例：
$$\text{coverage\_score}(A, B) > \epsilon \cdot |B|_w$$

---

## 11. Design Decisions (Confirmed)

### 11.1 权重计算策略

**决定**：每个 head 在脚本开始时计算一次 Q 模长 percentile 权重，之后在整个 round-based pruning 过程中**保持不变**。

**与原脚本一致**：这与 `attention_pruning_case_study_hybrid_rounds_xtrace.py` 的设计完全一致——在脚本开始时从 `stats_trace` 加载/计算统计量，然后在所有 round 中重复使用，不再更新。

**理由**：
- 使用 stats_trace 预计算的 Q 模长分布
- 避免每轮重复计算的开销
- Q 模长分布是 head 的固有属性，不随 round 变化

### 11.2 NMS 作用范围

**决定**：全局 NMS，每轮对所有 K 进行 NMS 检查。

**简单实现**：每轮直接对所有 K 调用 `variance_aware_fast_nms()`。

**可选优化**（Incremental NMS，见 Section 7.3）：利用历史 K 之间已经做过 NMS 的事实。

**建议**：先用简单实现验证算法正确性，性能瓶颈明显时再考虑优化实现。

### 11.3 NMS 类型

**决定**：Fast Parallel NMS（真正删除被支配的 K）

**理由**：
- 需要实现 sparsity，而非只调整分数
- Fast NMS 允许"误杀"（已被杀的 K 仍可杀其他 K），这是可接受的
- 完全可并行化，GPU friendly

### 11.4 抑制条件

**决定**：使用投影覆盖条件（Projection Coverage）+ Variance-Aware 权重选择

**公式**：
$$\text{conservative\_coverage\_score}(A, B) = \sum_f w_f^{*} \cdot \text{per\_freq\_score}_f > 0$$

**几何意义**：A 在 B 方向的投影超过了 B 的模长（在最不利的 Q 模长情况下）→ A "覆盖" 了 B → B 冗余

### 11.5 Epsilon 设置

**当前决定**：$\epsilon = 0$（**强制，不可配置**）

**原因**：不同 head 的数值范围差异大，非零 ε 无法跨 head 使用。

---

## 12. Implementation Checklist

### 12.1 Code Structure

```
weian_development/online_k_pruning_viz/
├── attention_pruning_case_study_hybrid_rounds_xtrace.py      # 原始脚本 (不修改)
├── attention_pruning_case_study_hybrid_rounds_xtrace_nms.py  # 新脚本
├── run_nms_sweep.sh                                          # 自动超参数搜索脚本
└── docs/
    └── variance_aware_nms.md  # 本文档（完整独立）
```

### 12.2 New Arguments to Add

```python
parser.add_argument("--nms-enabled", action="store_true", default=False,
                    help="Enable Q-magnitude weighted NMS before scoring")
parser.add_argument("--low-percentile", type=float, default=5.0,
                    help="Low percentile for Q-magnitude weights (default: 5)")
parser.add_argument("--high-percentile", type=float, default=95.0,
                    help="High percentile for Q-magnitude weights (default: 95)")
# Note: epsilon 固定为 0，不提供参数配置（见 Section 5.4）
```

### 12.3 Key Implementation Notes

1. **严格保持与原脚本一致的部分**：
   - `score_keys_for_round` 函数
   - `compute_pooled_attention` 函数
   - `save_comparison_figure` 函数
   - 所有可视化和指标计算
   - 命令行参数（NMS 相关参数之外）

2. **新增部分（仅 NMS）**：
   - Q 模长 percentile 权重计算函数 `compute_q_magnitude_percentile_weights()`
   - Variance-Aware Fast NMS 函数 `variance_aware_fast_nms()`（核心算法）
   - 可选：`variance_aware_incremental_nms()`（性能优化版本）
   - 在 `simulate_round_pruning` 每轮开始处调用 NMS

3. **权重计算时机**（与原脚本一致）：
   - 在脚本开始时、加载 stats_trace 后一次性计算
   - 每个 head 有独立的权重
   - 之后在所有 round 中重复使用，**不再更新**

4. **NMS 指标记录**：
   - 在 metrics JSON 中新增 `nms_drop_rate`, `nms_drop_count_per_round`

---

## 13. Hyperparameters Summary

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `low_percentile` | 5.0 | 用于正分频段的权重（假设 Q 模长小） |
| `high_percentile` | 95.0 | 用于负分频段的权重（假设 Q 模长大） |
| `epsilon` | 0 (固定) | 抑制阈值（不可配置） |

**调参方向**：
- 更保守：降低 `low_percentile`（如 1%），提高 `high_percentile`（如 99%）
- 更激进：提高 `low_percentile`（如 25%），降低 `high_percentile`（如 75%）

---

## 14. Open Questions

1. **Q 模长统计应该基于什么数据？**
   - 整个 stats_trace？
   - 滑动窗口？
   - 在线估计？

2. **高方差频段应该被惩罚还是放大？**
   - 惩罚：因为不稳定
   - 放大：因为可能捕捉重要信号

3. **Normalization 应该在什么粒度？**
   - Per-head？
   - Per-layer？
   - Global？

4. **如何验证本方法的效果？**
   - 需要设计特定的测试用例
   - 对比有无 variance-aware 的 retention rate

---

## 15. References

- Original NMS: Neubeck & Van Gool, "Efficient Non-Maximum Suppression", ICPR 2006
- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
