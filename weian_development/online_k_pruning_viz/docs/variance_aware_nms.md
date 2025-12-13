# Variance-Aware NMS (Future Work)

> 本文档从 `spectrum_aware_nms_brainstorm.md` 原 Section 10 中提取，详细描述 Variance-Aware NMS 的设计。
>
> **前置知识**：本文档假设读者已熟悉主文档中的以下内容：
> - Section 4: Energy Statistics Methods（能量统计方法）
> - Section 5: NMS Implementation: Projection Coverage（投影覆盖 NMS 实现）
>
> **注意**：请勿参考主文档 Section 3（Legacy - DO NOT IMPLEMENT），该节内容已废弃，与本文档无关。

## 1. Problem Statement

当前方法假设每个频段的能量权重 $w_f$ 是固定的（从 stats_trace 预计算）。但实际上：

1. **能量不是固定的**：不同 query 下，每个频段的实际能量贡献会有变化
2. **存在方差**：同一频段的能量在不同 query/position 下有一个范围
3. **极端值问题**：某些频段可能出现 top 5% 或 bottom 5% 的极端值

**核心问题**：当前的判断 `coverage_score > 0` 只考虑了期望值，没有考虑方差可能导致的极端情况。

## 2. Intuition

考虑这样的情况：
- 频段 f 的平均能量是 $E[w_f] = 0.1$
- 但它的方差很大，95% 区间是 $[0.01, 0.5]$

当前算法用 $w_f = 0.1$ 计算 coverage_score。但在极端情况下（top 5%），这个频段的贡献可能是 0.5，远超期望值。

**问题**：如果 B 只在某些极端情况下"逃脱" A 的覆盖，我们是否仍应该删除 B？

**可能的答案**：
- **保守策略**：只有当 B 在 95% 的情况下都被 A 覆盖时，才删除 B
- **激进策略**：只要 B 在期望情况下被 A 覆盖，就删除 B

---

## 3. Recommended Approach: Percentile-Based Energy Weights

### 3.1 Design Goal

**B 只有在即使极端情况下也被 A 覆盖时，才被抑制。**

这意味着我们需要使用"最不利于抑制"的能量权重来计算 coverage_score。

### 3.2 Key Insight: Score Sign Matters

回顾 coverage_score 公式：

$$\text{coverage\_score}(A, B) = \sum_f w_f \cdot \underbrace{\left( \frac{\text{Real}(A^{(f)} \cdot \overline{B^{(f)}})}{|B^{(f)}| + \varepsilon_{\text{den}}} - |B^{(f)}| \right)}_{\text{per\_freq\_score}_f}$$

其中 $\varepsilon_{\text{den}}$ 为防止除零/爆数的极小正数（实现中使用 $1\times 10^{-5}$）。

**关键观察**：每个频段的贡献 $\text{per\_freq\_score}_f$ 可正可负：
- **正分**（$\text{per\_freq\_score}_f > 0$）：A 的投影覆盖了 B，贡献于抑制
- **负分**（$\text{per\_freq\_score}_f < 0$）：A 的投影不足，反对抑制

### 3.3 Conservative Weight Selection

**原则**：使用最不利于抑制的权重。

对于每个频段 f：
- 若 $\text{per\_freq\_score}_f > 0$（贡献抑制）→ 使用 $w_f^{\text{low}}$（低估其贡献）
- 若 $\text{per\_freq\_score}_f < 0$（反对抑制）→ 使用 $w_f^{\text{high}}$（放大其阻力）

### 3.4 Percentile Statistics

对每个频段 f，从 stats_trace 中统计：

- $w_f^{\text{low}}$ = 5th percentile of energy weights for frequency f
- $w_f^{\text{high}}$ = 95th percentile of energy weights for frequency f

**归一化**：与均值版本类似，但分别对 low 和 high 进行归一化。

### 3.5 Final Formula

$$\text{conservative\_coverage\_score}(A, B) = \sum_f w_f^{*} \cdot \text{per\_freq\_score}_f$$

其中：
$$w_f^{*} = \begin{cases}
w_f^{\text{low}} & \text{if } \text{per\_freq\_score}_f > 0 \\
w_f^{\text{high}} & \text{if } \text{per\_freq\_score}_f \leq 0
\end{cases}$$

**判断条件**：
$$\text{A 抑制 B 当且仅当 } \text{conservative\_coverage\_score}(A, B) > 0$$

### 3.6 Geometric Interpretation

- 对于"帮助抑制"的频段，我们假设它们的贡献被低估（取 low percentile）
- 对于"阻止抑制"的频段，我们假设它们的贡献被高估（取 high percentile）
- 只有当在这种最不利情况下 A 仍能覆盖 B，我们才抑制 B

**结果**：更保守的 NMS，减少误删。

---

## 4. Implementation

### 4.1 Precomputation: Percentile Statistics

> **重要**：与基础版本一致，percentile 权重在**脚本开始时计算一次**，之后在所有 round 中**保持不变，不再更新**。这与 `attention_pruning_case_study_hybrid_rounds_xtrace.py` 的设计完全一致。

```python
def compute_percentile_weights(
    stats_trace: torch.Tensor,  # [num_samples, freq_count] 能量统计
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算每个频段的 low 和 high percentile 能量权重。

    注意：此函数应在脚本开始时调用一次，返回的权重在所有 round 中重复使用。

    Args:
        stats_trace: 从 calibration data 统计的能量分布
        low_percentile: 低百分位数 (default 5%)
        high_percentile: 高百分位数 (default 95%)

    Returns:
        w_low: [freq_count] 5th percentile weights (normalized)
        w_high: [freq_count] 95th percentile weights (normalized)
    """
    # 计算每个频段的 percentile
    w_low_raw = torch.quantile(stats_trace, low_percentile / 100.0, dim=0)  # [F]
    w_high_raw = torch.quantile(stats_trace, high_percentile / 100.0, dim=0)  # [F]

    # 归一化
    w_low = w_low_raw / w_low_raw.sum()
    w_high = w_high_raw / w_high_raw.sum()

    return w_low, w_high
```

**调用时机**：
```python
# 在脚本开始时，加载 stats_trace 后立即计算
w_low, w_high = compute_percentile_weights(stats_trace)
# 之后在所有 round 中使用同一个 w_low, w_high，不再更新
```

### 4.2 Variance-Aware Fast NMS

```python
def variance_aware_fast_nms(
    k_complex: torch.Tensor,      # [N, F] RoPE旋转后的K (原始 qk.pt) 转为复数
    w_low: torch.Tensor,          # [F] 5th percentile weights
    w_high: torch.Tensor,         # [F] 95th percentile weights
) -> torch.Tensor:
    """
    Variance-Aware Fast Parallel NMS：使用保守的权重选择

    对于每对 (A, B)，每个频段 f：
    - 若 per_freq_score[a, b, f] > 0：使用 w_low[f]
    - 若 per_freq_score[a, b, f] <= 0：使用 w_high[f]

    A 抑制 B 当 conservative_coverage_score(A, B) > 0

    Returns:
        keep_mask: [N] bool tensor, True = 保留
    """
    N, F = k_complex.shape

    # 1. 计算每个 K 每个频段的模长
    k_abs = torch.abs(k_complex)  # [N, F]
    k_abs_safe = k_abs.clamp(min=1e-5)  # 避免除零/爆数

    # 2. 计算 A 在 B 方向的投影（逐频段）
    real_dot = torch.einsum('af,bf->abf', k_complex, k_complex.conj()).real  # [N, N, F]
    proj_on_b = real_dot / k_abs_safe.unsqueeze(0)  # [N, N, F]

    # 3. 计算逐频段的 coverage score
    per_freq_score = proj_on_b - k_abs.unsqueeze(0)  # [N, N, F]

    # 4. 根据 score 符号选择权重
    #    positive score → use w_low (underestimate contribution)
    #    negative score → use w_high (overestimate resistance)
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

### 4.3 Incremental Version (Optional Performance Optimization)

> **Note**: 本节描述的是一种**可选的性能优化**。建议先用 Section 4.2 的简单实现验证算法正确性，必要时再考虑此优化。

```python
def variance_aware_incremental_nms(
    historical_k: torch.Tensor,   # [H, F] 历史 K (RoPE 旋转后) 复数
    new_k: torch.Tensor,          # [N, F] 新 K (RoPE 旋转后) 复数
    w_low: torch.Tensor,          # [F] 5th percentile weights
    w_high: torch.Tensor,         # [F] 95th percentile weights
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

## 5. Hyperparameters

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `low_percentile` | 5.0 | 用于正分频段的权重 |
| `high_percentile` | 95.0 | 用于负分频段的权重 |

**调参方向**：
- 更保守：降低 `low_percentile`（如 1%），提高 `high_percentile`（如 99%）
- 更激进：提高 `low_percentile`（如 25%），降低 `high_percentile`（如 75%）

---

## 6. Integration Notes

### 6.1 与基础版本的关系

Variance-Aware NMS 是基础版本的**保守增强**：

| 版本 | 权重选择 | 特点 |
|-----|---------|------|
| 基础版本 | 固定 $w_f$ (均值) | 简单，可能误删 |
| Variance-Aware | 根据 score 符号选择 $w_f^{low}$ 或 $w_f^{high}$ | 保守，减少误删 |

### 6.2 何时使用

- **先验证基础版本**：如果基础版本效果好（retention rate 没有明显下降），不需要 Variance-Aware
- **若观察到误删问题**：引入 Variance-Aware 版本

### 6.3 统计量来源与计算时机

需要从 stats_trace 中提取 percentile 统计：
- 需要足够多的样本来估计分布
- 建议样本数 >= 1000

**计算时机**（与基础版本及原脚本一致）：
- 在脚本开始时、加载 stats_trace 后**一次性计算** `w_low` 和 `w_high`
- 之后在所有 round 中重复使用，**不再更新**

---

## 7. Alternative Approaches (For Reference)

### 7.1 Approach A: Robust Coverage Score with Variance

**思路**：将 coverage_score 的计算扩展为考虑方差的版本。

期望 coverage score：
$$\mu_{AB} = \sum_f E[w_f] \cdot \text{score}_f(A, B)$$

方差项（近似）：
$$\sigma_{AB}^2 \approx \sum_f \text{Var}(w_f) \cdot \text{score}_f(A, B)^2$$

**Robust 判断条件**：
$$\mu_{AB} - z \cdot \sigma_{AB} > 0$$

其中 $z$ 是置信度参数（如 $z=1.96$ 对应 95% 置信区间）。

**问题**：需要假设正态分布，实际可能不成立。

### 7.2 Approach C: Per-Frequency Variance Penalty

**思路**：高方差频段的贡献应该被"打折扣"。

**修改权重**：
$$w_f^{adjusted} = \frac{E[w_f]}{1 + \alpha \cdot \text{CV}(w_f)}$$

其中 $\text{CV}(w_f) = \sigma_f / \mu_f$ 是变异系数。

**问题**：引入额外超参数 $\alpha$，难以调参。

---

## 8. Normalization Problem (Future Work)

**问题**：不同 head 的 K 模长数值范围差异很大。

**具体例子**：
- Head A 的 K 模长范围：$[1, 10]$
- Head B 的 K 模长范围：$[100, 1000]$
- 同一个 $\epsilon = 0.1$ 在 Head A 中很显著，在 Head B 中几乎无影响

**当前解决方案**：$\epsilon = 0$（只判断 > 0）

### 8.1 Potential Normalization Ideas

#### Idea 1: Per-Head Mean Normalization

$$\text{normalized\_score}(A, B) = \frac{\text{coverage\_score}(A, B)}{\bar{|K|}_{\text{head}}}$$

#### Idea 2: Z-Score Normalization

1. 计算该 head 中所有 pair 的 coverage_score 分布
2. 计算 $\mu_{\text{head}}, \sigma_{\text{head}}$
3. $\text{z\_score}(A, B) = \frac{\text{coverage\_score}(A, B) - \mu_{\text{head}}}{\sigma_{\text{head}}}$

**判断条件**：$\text{z\_score}(A, B) > z_{\text{threshold}}$

#### Idea 3: Percentile-Based Threshold

1. 计算该 head 中所有 pair 的 coverage_score
2. 使用百分位数作为阈值：如 "score > 90th percentile → suppress"

**优点**：不需要手动设置 ε，自动适应不同 head 的数值范围。

#### Idea 4: Relative Margin

将 ε 定义为相对于 |B| 的比例：
$$\text{coverage\_score}(A, B) > \epsilon \cdot |B|_w$$

---

## 9. Open Questions

1. **方差统计应该基于什么数据？**
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

4. **如何验证方差感知版本的效果？**
   - 需要设计特定的测试用例
   - 对比有无方差感知的 retention rate
