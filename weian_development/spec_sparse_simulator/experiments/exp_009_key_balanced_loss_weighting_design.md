# Exp 009: Key-Balanced Loss Weighting

## 问题背景 (Problem Background)

在 exp_006 及相关实验的训练过程中，模型对多个 Query-Key Pair 计算 Attraction Loss。然而，这些 Pair 在 Key 维度上存在**严重的不平衡分布**。

### 具体现象

假设当前 round 有 128 个 Query-Key Pair 参与 loss 计算，但这些 Pair 对应的 unique Key 数量远少于 128。例如：

```
Round 示例：
- 128 个 Query-Key Pairs
- 9 个 unique Keys
- 分布情况：
  - 8 个 Key 各对应 1 个 Pair（共 8 个 Pair）
  - 1 个 Key 对应 120 个 Pair（共 120 个 Pair）
```

### 问题分析

当前的 Attraction Loss 计算方式（参考 `exp_006_module2_reverse_cross_trace_validation/train.py:159-200`）：

```python
# 当前实现：对所有 valid query 计算 loss 并取平均
loss = -torch.log(match_prob + eps).mean()
```

这种**按 Pair 平均**的方式导致：
- 拥有大量 Pair 的 Key（高频 Key）在梯度中被过度表示
- 只有少量 Pair 的 Key（低频 Key）对梯度贡献不足
- 模型可能过度优化高频 Key 的 bin assignment，而忽略低频 Key
- 形成**长尾效应**：少数 Key 主导训练，多数 Key 被边缘化

## 解决方案 (Proposed Solution)

### 核心思想

引入 **Key-Balanced Loss Weighting**，通过对每个 Pair 的 loss 进行加权，抵消 Key 分布不平衡带来的偏差。

### 算法设计

#### Step 1: 统计每个 Key 的 Pair 数量

在每个 round 中，统计每个 unique Key 对应的 Pair 数量：

```
count_k = 每个 Key k 在当前 round 中作为 argmax_key 出现的次数
```

#### Step 2: 计算原始权重（倒数）

对每个 Pair，其原始权重为其对应 Key 的 Pair 数量的倒数：

```
raw_weight_i = 1 / count_{k_i}

其中 k_i 是第 i 个 Pair 对应的 argmax_key
```

直觉：高频 Key 的每个 Pair 权重更小，低频 Key 的每个 Pair 权重更大。

#### Step 3: 归一化权重

为确保加权 loss 的数值尺度与原始 loss 一致，对权重进行归一化：

```
weight_i = raw_weight_i / sum(raw_weight_j for all j)

约束：sum(weight_i) = 1
```

#### Step 4: 加权 Loss 计算

```
weighted_loss = sum(weight_i * loss_i)
```

### 数学形式化

设当前 round 有 N 个 valid Query-Key Pair，对应的 argmax_key 集合为 {k_1, k_2, ..., k_N}。

定义：
- `C(k)` = Key k 在当前 round 中出现的次数
- `L_i` = 第 i 个 Pair 的 loss（`-log(match_prob_i)`）

**原始 Loss**（当前实现）：
```
L_original = (1/N) * sum_{i=1}^{N} L_i
```

**Key-Balanced Loss**（提议方法）：
```
raw_w_i = 1 / C(k_i)
w_i = raw_w_i / sum_{j=1}^{N} raw_w_j
L_balanced = sum_{i=1}^{N} w_i * L_i
```

### 期望效果

| 场景 | 原始 Loss | Key-Balanced Loss |
|------|-----------|-------------------|
| 高频 Key (120 pairs) | 贡献 120/128 ≈ 94% 梯度 | 每 pair 权重 ∝ 1/120，总贡献与其他 Key 平衡 |
| 低频 Key (1 pair) | 贡献 1/128 ≈ 0.8% 梯度 | 权重 ∝ 1/1，获得合理比例的梯度贡献 |

**最终效果**：每个 unique Key 在训练中获得大致相等的影响力，消除长尾效应。

## 实现考虑 (Implementation Considerations)

### 实现位置

修改 `train.py` 中的 `compute_attraction_loss` 函数：

```python
def compute_attraction_loss_balanced(key_probs, query_bin_probs, argmax_keys, argmax_in_recent, eps=1e-8):
    """
    Compute Key-Balanced Attraction Loss.

    Changes from original:
    1. Count occurrences of each unique key
    2. Compute inverse-frequency weights
    3. Normalize weights to sum to 1
    4. Apply weighted mean instead of simple mean
    """
    # ... existing filtering logic ...

    # Step 1: Count key occurrences
    unique_keys, inverse_indices, counts = torch.unique(
        valid_argmax_keys, return_inverse=True, return_counts=True
    )

    # Step 2: Compute raw weights (inverse of count)
    raw_weights = 1.0 / counts[inverse_indices].float()  # (num_valid,)

    # Step 3: Normalize weights
    weights = raw_weights / raw_weights.sum()  # sum to 1

    # Step 4: Compute weighted loss
    per_pair_loss = -torch.log(match_prob + eps)  # (num_valid,)
    weighted_loss = (weights * per_pair_loss).sum()

    return weighted_loss
```

### 注意事项

1. **数值稳定性**：当某些 Key 只出现 1 次时，权重可能较大，需确保不会导致梯度爆炸
2. **边界情况**：如果所有 Pair 对应同一个 Key，权重均为 1/N，退化为原始 mean
3. **计算开销**：`torch.unique` 操作会增加少量计算开销，但相比整体训练可忽略

## 实验计划 (Experiment Plan)

### 对比实验

| 配置 | Loss 函数 | 目的 |
|------|----------|------|
| Baseline | 原始 Attraction Loss | 对照组 |
| Exp 009 | Key-Balanced Loss | 验证平衡效果 |

### 评估指标

1. **TopK Hit Rate**：验证整体性能是否提升或保持
2. **Per-Key Hit Rate 分布**：验证低频 Key 的命中率是否改善
3. **Loss 曲线对比**：观察训练稳定性变化
4. **Bin 分配均匀度**：验证 Key 在 bin 间的分配是否更均衡

### 预期结果

- 低频 Key 的命中率应有所提升
- 整体 TopK Hit Rate 应保持或略有提升
- 训练过程可能更稳定（避免被高频 Key 主导）

## 相关工作 (Related Work)

- **Focal Loss**：通过调整 hard/easy sample 权重解决类别不平衡
- **Class-Balanced Loss**：在分类任务中按类别频率加权
- **Inverse Frequency Weighting**：NLP 中常用的词频加权方法

本方案借鉴了 Inverse Frequency Weighting 的思想，但应用于 Query-Key Pair 的 Key 维度平衡。

## 参考实验 (References)

- [exp_006 README](./exp_006_module2_reverse_cross_trace_validation/README.md) - 原始训练实现
- [exp_006 train.py](./exp_006_module2_reverse_cross_trace_validation/train.py) - Attraction Loss 实现
