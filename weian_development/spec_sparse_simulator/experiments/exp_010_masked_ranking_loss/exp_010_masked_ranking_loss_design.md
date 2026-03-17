# 实验 010: Masked Ranking Loss 设计文档

## 1. 概述

**状态**: 实验完成 ✅ (2024-12-18)

**基准参考**: `exp_008_topk_attention_pairs`

**实验结果摘要**:

| K | K=50 Hit% | K=1000 Hit% | Active Bins | Entropy |
|---|-----------|-------------|-------------|---------|
| 1 | 98.62% | 99.16% | 1/128 | 0.0083 |
| 8 | 98.74% | 99.53% | 10/128 | 0.3011 |
| **16** | **98.82%** | **99.60%** | 13/128 | 0.4313 |
| 32 | 98.64% | 99.45% | 16/128 | 0.5090 |
| 64 | 96.28% | 98.61% | 19/128 | 0.5472 |

**结论**: K=16 为最佳配置，命中率最高且 bin diversity 适中。详细结果见 `exp_008_topk_attention_pairs/MRL_EXPERIMENT_RESULTS.md`

**核心改动**: 引入 Masked Ranking Loss，通过逐步 mask 高排名 key 来构造训练样本，使每个训练信号都对齐 top-1 目标。

**实现方式**: 扩展 exp_008（添加 `loss_type` 配置开关），而非独立实验目录。

**修改的文件**:
- `exp_008_topk_attention_pairs/train.py`: 添加 `compute_masked_ranking_loss()` 函数
- `exp_008_topk_attention_pairs/config.yaml`: 添加 `loss_type` 配置参数
- `exp_008_topk_attention_pairs/run.py`: 添加 `ablation_mrl` 实验模式

**运行方式**:
```bash
# 运行 MRL 消融实验（默认 K=1,2,4,8,16,32,64）
python run.py --mode ablation_mrl

# 自定义 K 值
python run.py --mode ablation_mrl --topk_values 1,2,4,8,16

# 单次训练（手动设置 config.yaml 中 loss_type: "mrl"）
python run.py --mode train
```

---

## 2. 背景：exp_008 的局限性

### 2.1 exp_008 回顾

在 exp_008 中，我们将 Ground Truth 从 Top-1 扩展到 Top-K：
- 对于每个 Query，选择 K 个最高 attention score 的 Keys
- 对这 K 个 (Query, Key) pairs **平等对待**，计算 attraction loss

### 2.2 问题：目标不对齐

虽然 exp_008 解决了 bin collapse 问题，但引入了新的问题：

| 方面 | 最终目标 | exp_008 训练目标 | Gap |
|------|----------|------------------|-----|
| 击中对象 | Top-1 Key | Top-K Keys（平等） | 监督信号包含 top-2~K，但评估只关心 top-1 |
| 优先级 | Top-1 最重要 | 所有 K 个 key 同等重要 | 没有体现排名的重要性差异 |

**核心矛盾**：我们训练模型去匹配所有 top-K keys，但评估时只关心能否击中 top-1。

---

## 3. 新想法：Masked Ranking Loss

### 3.1 核心直觉

**思路转变**：不是让模型同时匹配 K 个 keys，而是构造 K 个**独立的训练场景**，每个场景的目标都是"击中当前可用 keys 中的 top-1"。

具体做法：
- **场景 1**：所有 keys 可用 → 目标是击中原始 top-1
- **场景 2**：mask 掉 top-1 → 目标是击中"新的 top-1"（即原始 top-2）
- **场景 3**：mask 掉 top-1、top-2 → 目标是击中"新的 top-1"（即原始 top-3）
- ...以此类推

这样，**每个训练信号的目标都与最终评估目标对齐**：在当前可用的 keys 中击中 top-1。

### 3.2 Mask 的数学实现

#### 问题：如何体现"某个 key 被 mask"？

在我们的模型中：
- $P^{(K)}_j(b)$：Key $K_j$ 属于 bin $b$ 的概率（对 bins 做 softmax）
- 每个 bin $b$ 中，所有 keys 的概率贡献总和：$\text{TotalProb}(b) = \sum_k P^{(K)}_k(b)$

当我们 mask 掉某些 keys 后，这些 keys 对 bin 的概率贡献应该被移除。

**Mask 后的归一化概率**：

设 $\mathcal{M}$ 为被 mask 的 key 集合，对于剩余的 key $j$：

$$\tilde{P}^{(K)}_j(b) = \frac{P^{(K)}_j(b)}{\sum_{k \notin \mathcal{M}} P^{(K)}_k(b)}$$

**直观理解**：
- 分子：key $j$ 对 bin $b$ 的原始贡献
- 分母：所有未被 mask 的 keys 对 bin $b$ 的总贡献
- 结果：在可用 keys 中，key $j$ 对 bin $b$ 的**相对贡献**

### 3.3 完整公式推导

#### 符号定义

| 符号 | 含义 |
|------|------|
| $Q_i$ | 位置 $i$ 的 Query 向量 |
| $K_j$ | 位置 $j$ 的 Key 向量 |
| $\alpha_{i,j}$ | Query $i$ 到 Key $j$ 的 attention score |
| $K$ | 超参数：要使用的 top keys 数量（`topk_gt`） |
| $\mathcal{K}^+_i = \{k_1, k_2, ..., k_K\}$ | Query $i$ 的 Top-K key 索引，按 attention score 降序排列 |
| $\mathcal{M}_m = \{k_1, k_2, ..., k_{m-1}\}$ | 处理第 $m$ 个 key 时的 mask 集合（前 $m-1$ 个 key） |
| $p^{(Q)}_i(b)$ | Query $i$ 选择 bin $b$ 的概率 |
| $P^{(K)}_j(b)$ | Key $j$ 属于 bin $b$ 的概率 |
| $B$ | bin（探针）总数 |

#### Step 1: 计算每个 bin 的剩余 key 概率和

首先，预计算所有 keys 对每个 bin 的总贡献：

$$\text{TotalProb}(b) = \sum_{j=1}^{N_{keys}} P^{(K)}_j(b)$$

对于第 $m$ 个 key（$m$ 从 1 开始），mask 集合为 $\mathcal{M}_m = \{k_1, ..., k_{m-1}\}$。

剩余 keys 对 bin $b$ 的概率和：

$$R_m(b) = \text{TotalProb}(b) - \sum_{j \in \mathcal{M}_m} P^{(K)}_j(b)$$

特殊情况：
- 当 $m = 1$ 时，$\mathcal{M}_1 = \emptyset$，所以 $R_1(b) = \text{TotalProb}(b)$

#### Step 2: 计算 key $k_m$ 的归一化概率

在 mask 掉前 $m-1$ 个 keys 后，$k_m$ 在 bin $b$ 中的相对概率：

$$\tilde{P}^{(K)}_{k_m}(b) = \frac{P^{(K)}_{k_m}(b)}{R_m(b) + \epsilon}$$

其中 $\epsilon$ 是小常数（如 $10^{-8}$），用于防止除零。

#### Step 3: 计算 match probability

Query $i$ 与 key $k_m$（在 mask 场景下）的匹配概率：

$$p_{\text{match}}^{(i,m)} = \sum_{b=1}^{B} p^{(Q)}_i(b) \cdot \tilde{P}^{(K)}_{k_m}(b)$$

#### Step 4: 计算 Masked Ranking Loss

$$\mathcal{L}_{\text{MRL}} = -\frac{1}{N \cdot K} \sum_{i=1}^{N} \sum_{m=1}^{K} \log \left( p_{\text{match}}^{(i,m)} + \epsilon \right)$$

其中 $N$ 是 batch 中有效的 query 数量。

### 3.4 公式展开示例

以 Query $Q_i$ 和其 Top-3 Keys $\{k_1, k_2, k_3\}$ 为例：

**第 1 个 key（$k_1$，无 mask）**：
- Mask 集合：$\mathcal{M}_1 = \emptyset$
- 剩余概率：$R_1(b) = \text{TotalProb}(b)$
- 归一化概率：$\tilde{P}^{(K)}_{k_1}(b) = P^{(K)}_{k_1}(b) / \text{TotalProb}(b)$
- 匹配概率：$p_{\text{match}}^{(1)} = \sum_b p^{(Q)}_i(b) \cdot \tilde{P}^{(K)}_{k_1}(b)$

**第 2 个 key（$k_2$，mask 掉 $k_1$）**：
- Mask 集合：$\mathcal{M}_2 = \{k_1\}$
- 剩余概率：$R_2(b) = \text{TotalProb}(b) - P^{(K)}_{k_1}(b)$
- 归一化概率：$\tilde{P}^{(K)}_{k_2}(b) = P^{(K)}_{k_2}(b) / R_2(b)$
- 匹配概率：$p_{\text{match}}^{(2)} = \sum_b p^{(Q)}_i(b) \cdot \tilde{P}^{(K)}_{k_2}(b)$

**第 3 个 key（$k_3$，mask 掉 $k_1, k_2$）**：
- Mask 集合：$\mathcal{M}_3 = \{k_1, k_2\}$
- 剩余概率：$R_3(b) = \text{TotalProb}(b) - P^{(K)}_{k_1}(b) - P^{(K)}_{k_2}(b)$
- 归一化概率：$\tilde{P}^{(K)}_{k_3}(b) = P^{(K)}_{k_3}(b) / R_3(b)$
- 匹配概率：$p_{\text{match}}^{(3)} = \sum_b p^{(Q)}_i(b) \cdot \tilde{P}^{(K)}_{k_3}(b)$

---

## 4. 高效并行实现

### 4.1 关键观察：累积和优化

计算 $\sum_{j \in \mathcal{M}_m} P^{(K)}_j(b)$（被 mask 的 keys 的概率和）可以通过**累积和**高效完成。

设：
- $C_m(b) = \sum_{j=1}^{m} P^{(K)}_{k_j}(b)$

则：
- $R_1(b) = \text{TotalProb}(b)$
- $R_m(b) = \text{TotalProb}(b) - C_{m-1}(b)$ for $m > 1$

### 4.2 向量化实现

```python
def compute_masked_ranking_loss(key_probs, query_bin_probs, topk_keys, eps=1e-8):
    """
    高效并行计算 Masked Ranking Loss

    参数:
        key_probs: (num_keys, num_bins) - 所有 keys 的 bin 概率
        query_bin_probs: (num_queries, num_bins) - queries 的 bin 概率
        topk_keys: (num_queries, K) - 每个 query 的 top-K key 索引

    返回:
        标量 loss
    """
    num_queries, K = topk_keys.shape
    num_bins = key_probs.shape[1]

    # Step 1: 预计算每个 bin 的总 key 概率
    total_key_prob = key_probs.sum(dim=0)  # (num_bins,)

    # Step 2: 获取 top-K keys 的概率
    # topk_key_probs[i, m, b] = key_probs[topk_keys[i, m], b]
    topk_key_probs = key_probs[topk_keys]  # (num_queries, K, num_bins)

    # Step 3: 计算累积和（前 m 个 key 的概率和）
    cumsum = torch.cumsum(topk_key_probs, dim=1)  # (num_queries, K, num_bins)

    # Step 4: 构造 masked_sum（前 m-1 个 key 的概率和）
    # 通过 shift 实现：masked_sum[:, m, :] = cumsum[:, m-1, :]
    masked_sum = torch.zeros_like(cumsum)
    masked_sum[:, 1:, :] = cumsum[:, :-1, :]  # (num_queries, K, num_bins)

    # Step 5: 计算剩余 keys 的概率和
    # R_m(b) = TotalProb(b) - masked_sum[m](b)
    remaining_prob = total_key_prob.unsqueeze(0).unsqueeze(0) - masked_sum
    # remaining_prob: (num_queries, K, num_bins)

    # Step 6: 归一化 key 概率
    normalized_key_prob = topk_key_probs / (remaining_prob + eps)
    # normalized_key_prob: (num_queries, K, num_bins)

    # Step 7: 计算 match probability
    # match[i, m] = sum_b query_probs[i, b] * normalized_key_prob[i, m, b]
    match_prob = (query_bin_probs.unsqueeze(1) * normalized_key_prob).sum(dim=2)
    # match_prob: (num_queries, K)

    # Step 8: 计算 loss
    loss = -torch.log(match_prob + eps).mean()

    return loss
```

### 4.3 计算复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 预计算 total_key_prob | $O(N_k \cdot B)$ | $O(B)$ |
| 索引 topk_key_probs | $O(N_q \cdot K \cdot B)$ | $O(N_q \cdot K \cdot B)$ |
| 累积和 cumsum | $O(N_q \cdot K \cdot B)$ | $O(N_q \cdot K \cdot B)$ |
| 归一化 + match | $O(N_q \cdot K \cdot B)$ | $O(N_q \cdot K \cdot B)$ |
| **总计** | $O(N_q \cdot K \cdot B)$ | $O(N_q \cdot K \cdot B)$ |

其中：
- $N_q$：queries 数量
- $N_k$：keys 数量
- $K$：top-K 超参数
- $B$：bins 数量

**效率优势**：
1. **全并行**：所有操作都是向量化的，没有显式循环
2. **单次前向**：不需要重新运行模型的 forward，直接使用已有的概率
3. **累积和复用**：通过 cumsum 一次计算所有 mask 场景

---

## 5. 对比：exp_008 vs exp_010

| 方面 | exp_008 (Top-K Attraction) | exp_010 (Masked Ranking) |
|------|---------------------------|--------------------------|
| 训练样本 | K 个平等的 (Q, K) pairs | K 个层次化的 mask 场景 |
| 每个样本的目标 | 匹配某个 top-K key | 在可用 keys 中匹配 top-1 |
| 与最终目标对齐 | 部分对齐（包含非 top-1） | 完全对齐（每个场景都是 top-1） |
| Key 概率计算 | 原始概率 $P^{(K)}_k(b)$ | 归一化概率 $\tilde{P}^{(K)}_k(b)$ |
| 计算复杂度 | $O(N_q \cdot K \cdot B)$ | $O(N_q \cdot K \cdot B)$（相同） |
| 数据增强 | K 倍 | K 倍（但质量更高） |

---

## 6. 建议实验

### 6.1 超参数消融

在**对数坐标系**上探索 K 的影响：

| K 值 | 描述 | 预期效果 |
|------|------|----------|
| 1 | Baseline (无 mask) | 与 exp_007 top-1 相同 |
| 2 | 最小扩展 | 验证 mask 机制有效性 |
| 4 | 小规模 | 平衡增强与噪声 |
| 8 | 中等规模 | 更多训练信号 |
| 16 | 较大规模 | 丰富的 mask 场景 |
| 32 | 大规模 | 可能引入低 ranking 噪声 |
| 64 | 极大规模 | 边界测试 |

### 6.2 评估指标

与之前实验一致：
- Hit Rate @ K=50/500/1000
- Bin Collapse 指标（Active Bins, Entropy, Gini, Max Bin Usage）
- Training Loss 收敛曲线

### 6.3 对比基线

- **exp_007 Top-1**：原始单 key 方法
- **exp_008 Top-K**：平等对待多 key 方法
- **exp_010 MRL**：本实验的 Masked Ranking Loss

---

## 7. 配置变更

```yaml
# exp_010 config.yaml 新增内容
training:
  topk_gt: 8           # 默认值：使用 8 个 top keys
  loss_type: "mrl"     # 'attraction' (exp_008) 或 'mrl' (本实验)
```

---

## 8. 待确认问题

1. **归一化分母**：当 $R_m(b)$ 接近 0 时（即前 m-1 个 key 几乎垄断了 bin b），归一化会放大噪声。是否需要额外的稳定性处理？

2. **Loss 权重**：是否需要对不同 rank 的 key 使用不同权重？例如，top-1 的 loss 权重更大？

3. **与其他 Loss 的组合**：是否需要保留 exp_007 的 `lambda_activation`、`lambda_balance` 等可选 loss？

---

## 9. 参考

- 基准实验：`exp_008_topk_attention_pairs/`
- exp_008 设计文档：`../exp_008_topk_attention_pairs_design.md`
- exp_008 结论：`topk_gt=10` 最优（entropy=0.446，16 active bins）
