# 加权探针激活损失设计 (Weighted Probe Activation Loss)

## 1. 问题分析

### 当前实现的局限性

当前的 Probe Activation Loss 公式：

$$\mathcal{L}_{\text{activation}} = -\frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \frac{1}{|\mathcal{K}^+|} \sum_{k \in \mathcal{K}^+} \log \sigma_{d,k}$$

**问题**：所有正样本 key ($\mathcal{K}^+$ = 当前 batch 的 argmax keys) 被**平等对待**。

但实际上，这些 key 应该有优先级区分：
- 有些 key 在活跃探针中排名很高（"重要" key）
- 有些 key 在活跃探针中排名较低（"次要" key）

死亡探针应该**优先学习**那些"重要" key，这样其 key 排序能与活跃探针更一致。

---

## 2. 改进方案

### 2.1 核心思想

用**加权交叉熵**替代标准交叉熵，权重反映每个 key 的"全局重要性"。

**重要性定义**：一个 key 在所有探针中获得的最大概率。

直觉：如果某个 key 在某个活跃探针中概率很高，说明它是"重要" key，死亡探针也应该优先学习它。

### 2.2 权重计算

设 Key Network 输出概率矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{N_k \times N_p}$：
- $N_k$ = 历史 key 数量
- $N_p$ = 探针数量 (num_bins)
- $\sigma_{k,p}$ = key $k$ 在探针 $p$ 上的 softmax 概率（沿 key 维度归一化）

**Step 1: Max Pooling over probes**

$$m_k = \max_{p \in \{1,...,N_p\}} \sigma_{k,p} \quad \forall k$$

得到向量 $\mathbf{m} \in \mathbb{R}^{N_k}$，其中 $m_k$ 表示 key $k$ 在所有探针中获得的最大概率。

**Step 2: 在正样本集合上归一化**

$$w_k = \frac{m_k}{\sum_{k' \in \mathcal{K}^+} m_{k'}} \quad \forall k \in \mathcal{K}^+$$

注意：
- 仅在当前 batch 的正样本 $\mathcal{K}^+$ 上归一化
- 确保 $\sum_{k \in \mathcal{K}^+} w_k = 1$
- 非正样本的 key 不参与权重计算

**Step 3: Detach 权重**

$$w_k \leftarrow \text{detach}(w_k)$$

**重要**：权重不参与梯度计算。原因：
- 权重仅用于指导死亡探针的学习优先级
- 不应该让梯度通过权重反传到 Key Network
- 避免 Key Network 为了降低 activation loss 而人为调整概率分布

---

## 3. 数学公式

### 3.1 改进后的损失函数

$$\mathcal{L}_{\text{activation}}^{\text{weighted}} = -\frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \sum_{k \in \mathcal{K}^+} w_k \cdot \log \sigma_{d,k}$$

其中：
- $\mathcal{D}$ = 死亡探针集合
- $\mathcal{K}^+$ = 正样本 key 集合（当前 batch 的 argmax keys，且在历史 keys 范围内）
- $w_k$ = key $k$ 的归一化权重（detached）
- $\sigma_{d,k}$ = 死亡探针 $d$ 对 key $k$ 的 softmax 概率

### 3.2 对比

| 项目 | 原方案 | 改进方案 |
|------|--------|----------|
| 正样本权重 | 均匀 $\frac{1}{|\mathcal{K}^+|}$ | Max-pooled 概率归一化 $w_k$ |
| 含义 | 所有 argmax key 平等 | 高概率 key 优先级更高 |
| 权重梯度 | N/A | Detach，不反传 |
| 损失数量级 | $O(1)$ | $O(1)$（权重归一化保证） |

---

## 4. PyTorch 实现

```python
def compute_weighted_probe_activation_loss(
    key_logits,        # (num_keys, num_bins) - raw logits
    key_probs,         # (num_keys, num_bins) - softmax over keys (dim=0)
    batch_argmax_keys, # (batch_size,) - positive key indices
    num_bins,
    alpha_dead_threshold=0.05
):
    """
    Compute Weighted Probe Activation Loss.

    改进点：使用 max-pooled 概率作为正样本权重，
    让死亡探针优先学习在活跃探针中排名高的 key。
    """
    device = key_logits.device
    num_historical_keys = key_logits.size(0)

    # === Step 1: 死亡探针检测 (与原实现相同) ===
    # ... (省略，与原实现相同)

    # === Step 2: 筛选历史 keys 中的正样本 ===
    historical_mask = batch_argmax_keys < num_historical_keys
    historical_argmax_keys = batch_argmax_keys[historical_mask]

    if historical_argmax_keys.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    positive_keys = torch.unique(historical_argmax_keys)  # (|K+|,)

    # === Step 3: 计算加权权重 ===
    # Max pooling over probes: 每个 key 在所有探针中的最大概率
    max_probs, _ = key_probs.max(dim=1)  # (num_keys,)

    # 提取正样本的 max probs
    positive_max_probs = max_probs[positive_keys]  # (|K+|,)

    # 归一化得到权重 (在正样本集合上)
    weights = positive_max_probs / (positive_max_probs.sum() + 1e-8)  # (|K+|,)
    weights = weights.detach()  # 阻断梯度反传

    # === Step 4: 计算加权交叉熵 ===
    log_key_probs = F.log_softmax(key_logits, dim=0)  # (num_keys, num_bins)
    log_key_probs_dead = log_key_probs[:, dead_mask]  # (num_keys, |D|)
    log_probs_selected = log_key_probs_dead[positive_keys, :]  # (|K+|, |D|)

    # 加权求和: weights (|K+|,) @ log_probs (|K+|, |D|) → (|D|,)
    # 使用 einsum 或 broadcasting
    weighted_log_probs = (weights.unsqueeze(1) * log_probs_selected).sum(dim=0)  # (|D|,)

    # 对死亡探针取平均
    activation_loss = -weighted_log_probs.mean()

    return activation_loss
```

---

## 5. 预期效果

### 5.1 理论分析

- **高概率 key 获得更大权重**：在活跃探针中排名靠前的 key，死亡探针会优先学习
- **死亡探针与活跃探针对齐**：学习后的死亡探针 key 排序应与活跃探针更一致
- **渐进式激活**：死亡探针先学习"主流"重要 key，再逐步学习边缘 key

### 5.2 可能的问题

1. **权重偏向单一 key**：如果某个 key 的 max_prob 远高于其他 key，权重可能过于集中
   - 解决方案：可考虑使用 temperature softmax 或 log 变换平滑权重

2. **与 bin collapse 的交互**：如果 Query Network 已经 collapse 到单一探针，max_probs 可能都来自同一个探针
   - 这实际上可能是好事：确保死亡探针学习活跃探针的 key 排序

---

## 6. 实验计划

1. **实现加权版本** (`compute_weighted_probe_activation_loss`)
2. **与原版本对比消融**：
   - 原版：均匀权重
   - 改进版：max-pooled 权重
3. **观察指标**：
   - Hit Rate (K=50/500/1000)
   - 死亡探针数量变化
   - Bin utilization entropy
4. **超参数**：继续使用 `lambda_activation` 控制权重

---

## 7. 相关文件

- 原设计文档: [exp_007_anti_collapse_losses.md](../exp_007_anti_collapse_losses.md)
- 当前实现: [train.py](./train.py) - `compute_probe_activation_loss()`
- 消融实验结果: [output/ablation/](./output/ablation/)
