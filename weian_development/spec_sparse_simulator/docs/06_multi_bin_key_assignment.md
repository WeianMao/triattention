# Multi-Bin Key Assignment 设计方案

## 概述

本文档讨论一种改进的 Key Binning 方案：允许一个 Key 同时属于多个 Bin。

### 动机

当前设计中，每个 Key 只能属于一个 Bin（通过在 bin 维度上做 softmax）。这带来一个核心问题：

> **精准分 bin 可能是一件困难的事情**

如果一个 Key 被错误地分到了错误的 Bin，而 Query 预测了正确的 Bin，那么这个 Key 就无法被找到，导致 Argmax Hit Rate 下降。

### 新方案核心思想

1. **允许 Key 同时属于多个 Bin**：不再强制每个 Key 只属于一个 Bin
2. **保持 Loss 的两个目标**：
   - 同组的 Q-K 应该属于同一个 Bin（attraction）
   - 每个 Bin 里的 Key 越少越好（sparsity）
3. **推理时使用 TopK**：给定 Query 的 bin，取该 bin 中分数最高的 Top-K 个 Key 做 attention

---

## 1. 当前设计回顾

### 1.1 Key Binning（当前）

```python
# 当前设计：每个 Key 在 bin 维度上做 softmax
logits = neural_net_key(K)  # (num_bins,)
p_k = softmax(logits)       # 在 bin 维度上 softmax
bin_id = argmax(p_k)        # 每个 key 只属于一个 bin
```

**打分矩阵视角**：

设 `M[k, b]` 为 Key k 对 Bin b 的 logits（`num_keys × num_bins`）

当前设计：`P[k, :] = softmax(M[k, :])`，即每一**行**做 softmax

结果：每个 Key 的分布 `P[k, :]` 和为 1，Key 只能"选择"一个主要的 Bin

### 1.2 Query Routing（保持不变）

```python
# Query 仍然在 bin 维度上做 softmax（这一点不变）
logits = neural_net_query(Q)  # (num_bins,)
p_q = softmax(logits)         # 在 bin 维度上 softmax
bin_q = argmax(p_q)           # Query 选择一个 bin
```

---

## 2. 新方案：Softmax 沿 Key 维度

### 2.1 核心改变

将 softmax 的方向从"每个 Key 在 bin 上归一化"改为"**每个 Bin 在 key 上归一化**"

```python
# 新方案：每个 Bin 在 key 维度上做 softmax
logits = neural_net_key(all_keys)  # (num_keys, num_bins)
P = softmax(logits, dim=0)         # 在 key 维度上 softmax！
# P[:, b] 是 bin b 在所有 key 上的概率分布，和为 1
```

### 2.2 打分矩阵

`P[k, b] = exp(M[k, b]) / Σ_k' exp(M[k', b])`

- 每一**列** `P[:, b]` 和为 1（所有 key 对 bin b 的概率分布）
- 每一**行** `P[k, :]` **不**和为 1（key k 可以在多个 bin 中都有高分）

### 2.3 TopK 推理

```python
def topk_attention(Q, all_keys, key_probs, query_net, K):
    """
    TopK 推理

    Args:
        key_probs: (num_keys, num_bins) - softmax over keys for each bin
        K: TopK 参数
    """
    # Step 1: Query 选择 bin（不变）
    bin_q = argmax(softmax(query_net(Q)))

    # Step 2: 在该 bin 中选择 TopK keys
    # P[:, bin_q] 是该 bin 对所有 key 的概率分布
    scores = key_probs[:, bin_q]  # (num_keys,)
    topk_indices = topk(scores, K)

    # Step 3: 只与 TopK keys 做 attention
    relevant_keys = all_keys[topk_indices]
    return attention(Q, relevant_keys)
```

---

## 3. Loss Function 设计

### 3.1 符号定义

- `p_q ∈ Δ^{B-1}`: Query 的 bin 分布（softmax over bins）
- `P ∈ R^{K×B}`: Key 的 bin membership（softmax over keys for each bin）
- `k(q)`: Query q 的 argmax Key 索引

### 3.2 Attraction Loss（拉近同组 Q-K）

目标：让 Query 预测的 bin 对其 argmax Key 有高概率

```python
def attraction_loss_vectorized(p_q, P, query_to_key):
    """
    Attraction loss（向量化版本）

    Args:
        p_q: (num_queries, num_bins) - query bin distributions (softmax over bins)
        P: (num_keys, num_bins) - key bin scores (softmax over keys for each bin)
        query_to_key: (num_queries,) - 每个 query 的 argmax key 索引

    解释：
    - p_q[q, b] 是 query q 预测 bin b 的概率
    - P[k, b] 是 bin b 选中 key k 的概率
    - 我们希望 query 选的 bin 恰好选中其 argmax key
    - match_prob = Σ_b p_q[q, b] * P[k(q), b]
    """
    # 获取每个 query 对应的 key 的分数
    P_matched = P[query_to_key]  # (num_queries, num_bins)

    # 计算匹配概率：query bin 分布与 key bin 分数的点积
    match_prob = (p_q * P_matched).sum(dim=1)  # (num_queries,)

    # Negative log likelihood
    loss = -torch.log(match_prob + 1e-8).mean()

    return loss
```

**直觉**：这个 loss 可以理解为"Query 按其 bin 分布采样一个 bin，然后该 bin 采样一个 key，采样到正确 key 的概率"。

### 3.3 数值稳定性

上述公式在概率空间计算会有精度问题（小概率相乘后求和再取 log）。实现时应在 log 空间计算，使用 `log_softmax` + `torch.logsumexp`。

### 3.4 备选方案：双向交叉熵

参考 04_training_and_labels.md 中的双向交叉熵设计，可以用另一种 loss：

- **Key 视角**：如果 query 在某个 bin 有较大概率，key 在该 bin 也应有较大概率
- **Query 视角**：如果 key 在某个 bin 有较大概率，query 在该 bin 也应有较大概率

**归一化问题**：`P[k, :]` 不是概率分布（不和为 1），需要先归一化：

```
P_norm[k, :] = P[k, :] / Σ_b P[k, b]
```

**双向交叉熵公式**：

```
L_bidirectional = CE(p_q, P_norm) + CE(P_norm, p_q)
                = -Σ_b p_q[b] · log(P_norm[k, b]) - Σ_b P_norm[k, b] · log(p_q[b])
```

**数值稳定实现**：使用 `logsumexp` 在 log 空间计算归一化

```python
log_P = F.log_softmax(key_logits, dim=0)       # (num_keys, num_bins)
log_sum = torch.logsumexp(log_P, dim=1)        # (num_keys,)
log_P_norm = log_P - log_sum.unsqueeze(1)      # 归一化后的 log 概率
```

---

## 4. 待定事项

### 4.1 TopK 的 K 值选择

测试以下三种配置：
- K = 50
- K = 500
- K = 1000

### 4.2 Sparsity 实现

由于 Key 在 key 维度上做 softmax，Sparsity 通过 TopK 选择自然实现：
- 只对选中的 TopK keys 做 attention
- 未进入 TopK 的 key 直接忽略
- 不需要额外的 Sparsity Loss

### 4.3 与 Module 1 (Key Pruning) 的配合方式

待后续讨论。

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-16 | 初始化文档；定义问题动机；提出 Softmax over Keys 方案；设计 Attraction Loss |
| 2025-12-16 | 添加备选方案：双向交叉熵（需对 P[k,:] 归一化） |
| 2025-12-16 | 完善待定事项：TopK 值选择（50/500/1000）、Sparsity 实现说明 |
