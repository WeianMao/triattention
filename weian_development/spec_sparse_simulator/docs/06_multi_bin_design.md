# Multi-Bin Key Assignment 设计方案

## 动机

当前设计中，每个 Key 只能属于一个 Bin。如果 Key 被错误分 bin，而 Query 预测了正确 bin，该 Key 无法被找到。

**新方案**：允许一个 Key 同时属于多个 Bin。

---

## 核心改变

### 当前设计

```python
# 每个 Key 在 bin 维度上 softmax
P = softmax(logits, dim=1)  # 每行和为 1
bin_id = argmax(P[k, :])    # Key 只属于一个 bin
```

### 新方案

```python
# 每个 Bin 在 key 维度上 softmax
P = softmax(logits, dim=0)  # 每列和为 1
# P[:, b] 是 bin b 选择各 key 的概率分布
```

**结果**：`P[k, :]` 不再和为 1，Key 可以在多个 bin 中都有高分。

---

## TopK 推理

```python
def topk_attention(Q, all_keys, key_probs, query_net, K):
    bin_q = argmax(softmax(query_net(Q)))  # Query 选 bin
    scores = key_probs[:, bin_q]           # 该 bin 对各 key 的分数
    topk_indices = scores.topk(K).indices  # 选 TopK
    return attention(Q, all_keys[topk_indices])
```

---

## Loss Function

### Attraction Loss（推荐）

目标：Query 选的 bin 恰好选中其 argmax Key

```python
def attraction_loss(p_q, P, query_to_key):
    """
    p_q: (num_queries, num_bins) - query bin 分布
    P: (num_keys, num_bins) - softmax over keys for each bin
    """
    P_matched = P[query_to_key]  # 对应 key 的分数
    match_prob = (p_q * P_matched).sum(dim=1)
    return -torch.log(match_prob + 1e-8).mean()
```

**直觉**：Query 按 bin 分布采样 bin，bin 采样 key，采样到正确 key 的概率。

### 实验结论

| 方法 | K=50 | K=500 | K=1000 |
|------|------|-------|--------|
| **Attraction Loss** | 100% | 100% | 100% |
| 双向交叉熵 | 67.78% | 100% | 100% |

**推荐使用 Attraction Loss**。

---

## 待定事项

- [ ] TopK 的 K 值选择（50/500/1000）
- [ ] 与 Module 1 的配合方式
