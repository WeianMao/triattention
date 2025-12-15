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
#   - 使用 boolean indexing：kv_cache[predictions < threshold]
#
def key_pruning(kv_cache, neural_net, threshold):
    """
    Input:
        kv_cache: 当前 KV Cache 中的所有 Key {K_1, K_2, ..., K_n}
        neural_net: Key Pruning 神经网络
        threshold: drop 阈值
    Output:
        保留的 Key 集合
    """
    retained_keys = []

    for K_i in kv_cache:
        # 预测 K_i 应该被 drop 的概率（即未来不会被 attend 的概率）
        p_i = neural_net(K_i)  # 输出经过 Sigmoid，表示 drop 概率

        if p_i < threshold:
            retained_keys.append(K_i)  # drop 概率低，保留
        # else: drop K_i（drop 概率高，丢弃）

    return retained_keys
```

**向量化实现示例**：
```python
def key_pruning_vectorized(kv_cache, neural_net, threshold):
    """
    向量化版本，实际部署使用

    kv_cache: (num_keys, head_dim)
    """
    # 批量预测 drop 概率
    drop_probs = neural_net(kv_cache)  # (num_keys,)

    # Boolean indexing: 保留 drop 概率低的 Key
    retain_mask = drop_probs < threshold
    retained_keys = kv_cache[retain_mask]

    return retained_keys, retain_mask
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
│   (p = drop 概率)               │
└─────────────────────────────────┘
```

**特点**：比 Module 2 的网络稍重（多一层 MLP）

**输出语义**：
- p 接近 1 → Key 很可能不会被未来 Query attend → 应该 drop
- p 接近 0 → Key 很可能会被未来 Query attend → 应该保留

---

## 标签定义

### Drop 标签定义

> **标签 = drop 概率的目标值**

一个 Key K_i 的标签定义如下：
- **label = 0**（不应 drop）：存在某个未来的 Query 会 attend 到这个 Key
- **label = 1**（应该 drop）：没有任何未来的 Query 会 attend 到这个 Key

形式化定义：
```
label(K_i) = 0  iff  ∃ Q_j (j > i): argmax_k Attention(Q_j, K_k) == i  (会被 attend，不 drop)
label(K_i) = 1  otherwise  (不会被 attend，drop)
```

### 要点

- 使用 **argmax** 而非 threshold
- **只要有一个**未来的 Q attend 到这个 K，标签就为 0（不 drop）
- label = 1 表示应该 drop，对应模型输出的 drop 概率目标为 1
- 这是一个相对严格的定义，"应该保留"（label=0）的 Key 可能较少

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
# drop_probs: 模型预测的 drop 概率
# retain_mask: drop 概率低于阈值的 Key 被保留
retain_mask = drop_probs < threshold
# retain_mask 决定哪些 Key 参与后续 attention，而非物理删除
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
def compute_module1_metrics(drop_probs, labels, threshold=0.5):
    """
    计算 Module 1 评估指标

    Args:
        drop_probs: (num_keys,) - 模型预测的 drop 概率
        labels: (num_keys,) - 真实标签（0=会被 attend 应保留，1=不会被 attend 应 drop）
        threshold: drop 阈值
    """
    # 保留 drop 概率低的 Key
    retain_mask = drop_probs < threshold

    # Retention Rate
    retention_rate = retain_mask.sum() / len(drop_probs)

    # Argmax Hit Rate（关键指标）
    # label=0 表示会被 attend（应该保留），检查这些 Key 是否被保留
    should_retain = labels == 0
    argmax_hit_rate = (retain_mask & should_retain).sum() / should_retain.sum()

    # False Negative Rate
    # 应该保留但被错误 drop 的比例
    false_negatives = (~retain_mask & should_retain).sum()
    false_negative_rate = false_negatives / should_retain.sum()

    # Keys per Query（需要结合具体 round 计算）
    keys_per_query = retain_mask.sum()  # 简化：保留的 key 数量

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
| 2025-12-15 | 修正模型输出语义：从"保留概率"改为"drop 概率"；相应调整判断逻辑（`p < threshold` 保留）和标签定义（label=1 表示应 drop） |
