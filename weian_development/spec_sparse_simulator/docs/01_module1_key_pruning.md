# Module 1: Key Pruning (Drop KV)

## 目标

预测每个 Key 是否会被未来的 Query attend，丢弃预测为"不会被 attend"的 Key。

---

## 执行时机

每 128 次解码的 round 开头

---

## 算法流程

```python
# 在每个 round 开头执行
#
# ⚠️ 实现注意：以下代码使用循环仅为了表达算法逻辑清晰。
# 实际实现时应使用 tensor 操作，避免 Python 循环：
#   - 批量处理所有 Key：neural_net(kv_cache)  # shape: (n,)
#   - 使用 boolean indexing：kv_cache[predictions >= threshold]
#
def key_pruning(kv_cache, neural_net, threshold):
    """
    Input:
        kv_cache: 当前 KV Cache 中的所有 Key {K_1, K_2, ..., K_n}
        neural_net: Key Pruning 神经网络
        threshold: 保留阈值
    Output:
        保留的 Key 集合
    """
    retained_keys = []

    for K_i in kv_cache:
        # 预测 K_i 未来被 attend 的概率
        p_i = neural_net(K_i)  # 输出经过 Sigmoid

        if p_i >= threshold:
            retained_keys.append(K_i)
        # else: drop K_i

    return retained_keys
```

**向量化实现示例**：
```python
def key_pruning_vectorized(kv_cache, neural_net, threshold):
    """
    向量化版本，实际部署使用

    kv_cache: (num_keys, head_dim)
    """
    # 批量预测
    predictions = neural_net(kv_cache)  # (num_keys,)

    # Boolean indexing
    mask = predictions >= threshold
    retained_keys = kv_cache[mask]

    return retained_keys, mask
```

---

## 神经网络结构

```
Input: K (post-RoPE, 在旋转参考系下)
    │
    ▼
┌─────────────────────────────────┐
│   Kernel Encoding Layer        │
│   (详见 03_neural_network_      │
│    architecture.md)            │
│   Output: 128-dim vector       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   MLP Layer (1 层)              │
│   128 → hidden → 1             │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Sigmoid                      │
│   Output: p ∈ [0, 1]           │
└─────────────────────────────────┘
```

**特点**：比 Module 2 的网络稍重（多一层 MLP）

---

## 标签定义

### "被 attend 到" 的定义

一个 Key K_i 被认为"会被未来的 Query attend 到"，当且仅当：

> **存在**某个未来的 Query Q_j (j > current_round)，使得 K_i 是 Q_j 的 attention 最大值对应的 Key

形式化定义：
```
label(K_i) = 1  iff  ∃ Q_j (j > i): argmax_k Attention(Q_j, K_k) == i
label(K_i) = 0  otherwise
```

### 要点

- 使用 **argmax** 而非 threshold
- **只要有一个**未来的 Q attend 到这个 K，标签就为 1
- 这是一个相对严格的定义，可能会有较多的 negative samples

---

## Sparse Attention vs KV Cache 压缩

> **重要区别**：本项目是 **Sparse Attention**，不是 KV Cache 压缩。

| 对比项 | KV Cache 压缩 | Sparse Attention (本项目) |
|--------|--------------|--------------------------|
| **max_keys** | 有硬限制 | **不存在** |
| **Key 处理** | 物理删除 | 保留，但选择性计算 |
| **目标** | 减少显存 | 减少计算量 |

### Pruning 策略

采用 **Soft Pruning**：
- 只根据 threshold 决定是否参与后续 attention
- 不存在 max_keys 限制
- 被 "pruned" 的 Key 仍在显存中，只是不参与当前 round 的 attention 计算

```python
# Soft Pruning: 没有 max_keys 限制
mask = predictions >= threshold
# mask 决定哪些 Key 参与后续 attention，而非物理删除
```

---

## 评估指标

### 核心指标（最重要）

| 指标 | 定义 | 目标 | Baseline (Full Attention) |
|------|------|------|---------------------------|
| **Argmax Hit Rate** | Query 仍能 attend 到原 argmax Key 的比例 | 越高越好（>99%） | 100% |
| **Keys per Query** | 每个 Query 参与 attention 的平均 Key 数量 | 越低越好 | N（所有历史 Key） |
| **Computation Reduction** | 1 - (Keys per Query / N) | 越高越好 | 0% |

> **Argmax Hit Rate 是最关键指标**：如果 Query 无法 attend 到原来的 argmax Key，可能严重影响生成质量。

### 辅助指标

| 指标 | 定义 | 目标 |
|------|------|------|
| Retention Rate | 保留的 Key 数量 / 总 Key 数量 | 越低越好（更多压缩） |
| False Negative Rate | 被错误丢弃的"会被 attend"的 Key 比例 | 越低越好 |

### 指标计算示例

```python
def compute_module1_metrics(predictions, labels, threshold=0.5):
    """
    计算 Module 1 评估指标

    Args:
        predictions: (num_keys,) - 模型预测的保留概率
        labels: (num_keys,) - 真实标签（1=会被 attend，0=不会）
        threshold: 保留阈值
    """
    retained_mask = predictions >= threshold

    # Retention Rate
    retention_rate = retained_mask.sum() / len(predictions)

    # Argmax Hit Rate（关键指标）
    # 被标记为 1 的 Key 中，有多少被保留了？
    argmax_keys = labels == 1
    argmax_hit_rate = (retained_mask & argmax_keys).sum() / argmax_keys.sum()

    # False Negative Rate
    false_negatives = (~retained_mask & argmax_keys).sum()
    false_negative_rate = false_negatives / argmax_keys.sum()

    # Keys per Query（需要结合具体 round 计算）
    keys_per_query = retained_mask.sum()  # 简化：保留的 key 数量

    return {
        'retention_rate': retention_rate,
        'argmax_hit_rate': argmax_hit_rate,
        'false_negative_rate': false_negative_rate,
        'keys_per_query': keys_per_query,
    }
```

---

## 实验验证计划

### Phase A: Oracle Upper Bound

使用真实的 attention pattern（从 trace 中提取）作为 oracle：
1. 统计真实的 "被 attend 到" 的 Key
2. 计算 oracle pruning 的 retention rate
3. 评估理论上的最大压缩率

### Phase B: 神经网络验证

1. 训练神经网络
2. 对比 oracle vs 神经网络预测
3. 评估 False Negative Rate 对最终生成质量的影响

---

## 待定设计细节

- [ ] MLP 的 hidden dimension
- [ ] threshold 如何设定（固定 / 自适应 / 可学习）

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-14 | 初始化文档 |
| 2025-12-14 | 明确 Sparse Attention vs KV Cache 压缩区别；选择 Soft Pruning；添加向量化实现注释 |
| 2025-12-15 | 重构评估指标：添加 Argmax Hit Rate、Keys per Query、Computation Reduction 核心指标及 Full Attention baseline；添加指标计算代码；移除 Focal Loss |
