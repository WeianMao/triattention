# Loss 设计与训练策略

## 当前阶段：Proof of Concept

| 项目 | 设置 |
|------|------|
| 训练数据 | 单个 trace 的 qk.pt |
| 测试数据 | 同一 trace（overfit 验证） |
| 成功标准 | Loss 收敛 |

---

## Module 1: Key Pruning

### Loss Function

```python
# Binary Cross Entropy
loss = F.binary_cross_entropy(drop_probs[valid_mask], labels[valid_mask])
```

### 排除末尾 Key

训练时，位置 >= `seq_len - 1000` 的 Key 不计算 loss（避免标签噪声）。

---

## Module 2: Bin-based Sparse Attention

### Ground Truth：Group 定义

若 Q_i 的 argmax attention 是 K_j，则 Q_i 和 K_j 属于同一 **group**。

```python
def build_groups(attention_matrix):
    # attention_matrix: (num_queries, num_history_keys)
    groups = defaultdict(list)
    for q_idx in range(attention_matrix.shape[0]):
        argmax_key = attention_matrix[q_idx].argmax()
        groups[argmax_key].append(q_idx)
    return groups
```

### Loss Function：双向交叉熵 + Linear Repel

```
L_total = L_attract + λ × L_repel
```

**Attract（拉近同 group）**：双向交叉熵
```python
def attract_loss(p, log_p, r, log_r, query_to_key):
    r_matched = r[query_to_key]
    log_r_matched = log_r[query_to_key]
    return -(p * log_r_matched).sum() - (r_matched * log_p).sum()
```

**Repel（推远非 group）**：Linear Repel
```python
def repel_loss(p, r, group_masks):
    s = torch.mm(p, r.T)  # collision probability
    return (s * (~group_masks).float()).sum()
```

### 关键发现（Sanity Check 实验）

1. **必须归一化**：大规模数据需 per-query 归一化 attract 和 repel 项
2. **Linear Repel 可行**：λ=10~20 时达到 100% Hit Rate
3. **Log Repel 不可用**：`log(r) < 0` 导致优化方向错误

### 推荐参数

| 参数 | 值 |
|------|-----|
| Loss 函数 | Linear Repel |
| λ | 10~20 |
| 归一化 | 必须 |

---

## 训练策略

### 模块分离

两个模块独立训练：

| 阶段 | 模块 |
|------|------|
| Stage 1 | Module 1 (Key Pruning) |
| Stage 2 | Module 2 (Binning) |

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

### 数据增强

随机偏移 round 起始位置，避免位置过拟合：

```python
offset = random.randint(0, round_window - 1)
Q_aug, K_aug = Q[offset:], K[offset:]
```

---

## 待定事项

- [ ] 跨 trace 泛化性验证
- [ ] 负载均衡 Loss（防止 bin 分布不均）
- [ ] Bin collapse 预防
