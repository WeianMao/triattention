# Loss 设计与训练策略

## 当前阶段：Proof of Concept

| 项目 | 设置 |
|------|------|
| 训练数据 | 单个 trace 的 qk.pt |
| 测试数据 | 同一 trace（overfit 验证） |
| 成功标准 | Loss 收敛 |

---

## Module 1: Key Pruning

### 标签定义

```
label(K_i) = 0  若存在 Q_j (j >= round_start) 使得 argmax Attention(Q_j, K) == i
label(K_i) = 1  否则
```

- 使用 **argmax** 判定：只要有一个未来 Q attend 到该 K，标签为 0（保留）
- 标签为 1 表示该 Key 可以被 drop

### Loss Function

```python
# Binary Cross Entropy
loss = F.binary_cross_entropy(drop_probs[valid_mask], labels[valid_mask])
```

### 排除末尾 Key

训练时，位置 >= `seq_len - 1000` 的 Key 不计算 loss（避免标签噪声）。

---

## Module 2: Multi-Bin Sparse Attention

### Ground Truth

**Group 定义**：argmax 相同的 Query 与对应的 Key 构成一个 Group（一个 K 可对应多个 Q）

```python
query_to_key = attention_matrix.argmax(dim=1)  # (num_queries,)
```

### Loss Function：Attraction Loss

**目标**：Query 选的 bin 能选中其 argmax Key

```python
def attraction_loss(p_q, P, query_to_key):
    P_matched = P[query_to_key]  # (num_queries, num_bins)
    match_prob = (p_q * P_matched).sum(dim=1)  # (num_queries,)
    return -torch.log(match_prob + 1e-8).mean()
```

**数值稳定性**：实现时应在 log 空间计算，使用 `log_softmax` + `logsumexp`

### 实验结论

| 方法 | K=50 | K=500 | K=1000 |
|------|------|-------|--------|
| **Attraction Loss** | 100% | 100% | 100% |
| 双向交叉熵 | 67.78% | 100% | 100% |

**推荐使用 Attraction Loss**，在所有 K 值下均达到 100% TopK Hit Rate。

---

## 训练策略

### 模块分离

两个模块独立训练：

| 阶段 | 模块 |
|------|------|
| Stage 1 | Module 1 (Key Pruning) |
| Stage 2 | Module 2 (Multi-Bin) |

### 训练/推理不对齐

Module 1 存在不对齐：
- **训练**：保留所有 Key 计算 loss
- **推理**：预测概率 < threshold 的 Key 被丢弃

POC 阶段可接受。

### 优化器

```python
optimizer = Adam(params, lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

---

## 未来注意事项

### GQA 下 Key 丢弃决策

多 Query head 共享 KV head 时，使用 **AND 逻辑**：

```python
# 只有所有 Q head 都认为 drop，才最终 drop
final_drop = drop_decisions.all(dim=0)
```

### GQA 下分 Bin 策略

一个 KV head 对应多个 Q head 时：
- **Key 分 bin 网络**：每个 KV head 一个，被对应的多个 Q head 共享
- **Query 分 bin 网络**：每个 Q head 独立

### 数据增强

随机偏移 round 起始位置，避免位置过拟合：

```python
offset = random.randint(0, round_window - 1)
Q_aug, K_aug = Q[offset:], K[offset:]
```

---

## 待定事项

- [ ] 跨 trace 泛化性验证
- [ ] TopK 的 K 值选择
