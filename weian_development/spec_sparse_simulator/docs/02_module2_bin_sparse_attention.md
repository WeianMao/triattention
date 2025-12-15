# Module 2: Bin-based Sparse Attention

## 目标

将 Key 分到不同的 bin 中，Query 只与同 bin 的 Key 做 attention，减少 attention 计算量。

---

## 执行时机

- **Key Binning**: 每 128 次解码的 round 开头
- **Query Routing + Sparse Attention**: 每次解码

---

## 算法流程

### Round 开头：Key Binning

```python
#
# ⚠️ 实现注意：以下代码使用循环仅为了表达算法逻辑清晰。
# 实际实现时应使用 tensor 操作，避免 Python 循环。
# 见下方向量化实现示例。
#
def key_binning(kv_cache, neural_net_key, num_bins=128):
    """
    Input:
        kv_cache: 当前 KV Cache 中的所有 Key（已经过 Module 1 pruning）
        neural_net_key: Key Binning 神经网络
        num_bins: bin 数量
    Output:
        bin_index: {bin_id → [K indices]}
    """
    bin_index = defaultdict(list)

    for i, K_i in enumerate(kv_cache):
        # 神经网络输出 128 维 logits，经过 softmax
        logits = neural_net_key(K_i)  # shape: (128,)
        bin_id = argmax(softmax(logits))
        bin_index[bin_id].append(i)

    return bin_index
```

**向量化实现示例**：
```python
def key_binning_vectorized(kv_cache, neural_net_key, num_bins=128):
    """
    向量化版本，实际部署使用

    kv_cache: (num_keys, head_dim)
    """
    # 批量预测
    logits = neural_net_key(kv_cache)  # (num_keys, num_bins)
    bin_assignments = logits.argmax(dim=1)  # (num_keys,)

    # 构建索引（使用 scatter 或 groupby 操作）
    # bin_index[b] = (bin_assignments == b).nonzero()
    return bin_assignments
```

### 每次解码：Query Routing + Sparse Attention

```python
#
# ⚠️ 实现注意：以下代码使用循环仅为了表达算法逻辑清晰。
# 实际实现时应使用 tensor 操作和高效索引。
#
def sparse_attention(Q, kv_cache, bin_index, neural_net_query):
    """
    Input:
        Q: 新 Query
        kv_cache: KV Cache
        bin_index: Key 的 bin 索引
        neural_net_query: Query Routing 神经网络
    Output:
        attention output
    """
    # Step 1: Query 分 bin
    logits = neural_net_query(Q)  # shape: (128,)
    bin_q = argmax(softmax(logits))

    # Step 2: 检索同 bin 的 Key
    relevant_key_indices = bin_index[bin_q]
    relevant_keys = [kv_cache[i] for i in relevant_key_indices]

    # Step 3: 只与同 bin 的 Key 做 attention
    output = attention(Q, relevant_keys)

    return output
```

**向量化实现示例**：
```python
def sparse_attention_vectorized(Q, kv_cache, bin_assignments, neural_net_query):
    """
    向量化版本

    Q: (head_dim,)
    kv_cache: (num_keys, head_dim)
    bin_assignments: (num_keys,) - 每个 Key 的 bin ID
    """
    # Query 分 bin
    logits = neural_net_query(Q.unsqueeze(0))  # (1, num_bins)
    bin_q = logits.argmax(dim=1).item()

    # 高效索引同 bin 的 Key
    mask = bin_assignments == bin_q
    relevant_keys = kv_cache[mask]  # (num_relevant, head_dim)

    # Attention
    output = attention(Q, relevant_keys)
    return output
```

---

## 神经网络结构

### Key Binning Network

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
│   Softmax                      │
│   Output: 128-dim probability  │
└─────────────────────────────────┘
    │
    ▼
    argmax → bin_id
```

### Query Routing Network

结构与 Key Binning Network **相同**，但参数**独立**。

**特点**：轻量（无 MLP，直接 Kernel → Softmax）

---

## Bin 语义

- Bin **不是**传统的位置 bin
- Bin 是神经网络**学习**出来的分类
- 神经网络输入包含位置信息（通过旋转参考系），但如何分类由学习决定
- 目标：让相关的 Q-K pair 落在同一个 bin

---

## Multi-head 处理

| 场景 | 处理方式 |
|------|----------|
| 标准 MHA | 每个 Query head 有独立的 Key/Query Binning 网络 |
| GQA | 多个 Query head 共享一个 KV head，但每个 Query head 仍有独立网络 |

**结果**：同一个 K 在不同 Query head 视角下可能被分到**不同的 bin**

---

## 边界情况处理

### 空 bin 问题

如果 Query 的 bin 中没有 Key：

**策略：Fallback 到 Full Attention + Warning**

```python
def sparse_attention_with_fallback(Q, kv_cache, bin_assignments, neural_net_query):
    """
    带空 bin fallback 的 sparse attention
    """
    logits = neural_net_query(Q.unsqueeze(0))
    bin_q = logits.argmax(dim=1).item()

    mask = bin_assignments == bin_q
    num_relevant = mask.sum().item()

    if num_relevant == 0:
        # ⚠️ WARNING: 空 bin，fallback 到 full attention
        # 这种情况应该非常罕见，如果频繁发生需要检查：
        #   1. 神经网络训练是否充分
        #   2. Bin 数量是否过多
        #   3. 数据分布是否异常
        warnings.warn(
            f"Empty bin detected! bin_id={bin_q}, "
            f"falling back to full attention. "
            f"This should be rare - investigate if frequent.",
            RuntimeWarning
        )
        relevant_keys = kv_cache  # 使用所有 Key
    else:
        relevant_keys = kv_cache[mask]

    output = attention(Q, relevant_keys)
    return output
```

> **监控要求**：生产环境需要监控空 bin 发生频率。如果频繁发生（如 > 1%），需要调查原因。

### Multi-bin Query

**当前决策**：暂不实现，只 attend 单一 bin。

待实验结果出来后再评估是否需要 multi-bin routing（soft routing）。

---

## 评估指标

| 指标 | 定义 | 目标 |
|------|------|------|
| Attention Recall | 同 bin 中包含 argmax Key 的比例 | 越高越好 |
| Computation Reduction | 1 - (avg bin size / total keys) | 越高越好 |
| Bin Balance | bin 大小的方差 | 越低越好（均匀分布） |

---

## 实验验证计划

### Phase B-1: Oracle Upper Bound

使用 oracle bin assignment（基于真实 attention pattern）：
1. 对每个 Q，将其 argmax K 分到同一个 bin
2. 评估理论最大 attention recall

### Phase B-2: 神经网络验证

1. 训练 Key/Query Binning 网络
2. 对比 oracle vs 神经网络
3. 评估 attention recall 与 bin balance

### Phase B-3: Random Baseline

对比随机 binning 作为下界

---

## 待定设计细节

- [ ] Bin 数量：固定 128？可调？
- [ ] Loss function 设计（见 04_training_and_labels.md）

### 已确定

- [x] 空 bin fallback 策略：**Fallback 到 full attention + Warning**
- [x] Multi-bin Query：**暂不实现**，待实验结果决定

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-14 | 初始化文档 |
| 2025-12-14 | 添加向量化实现注释；确定空 bin fallback 策略；Multi-bin Query 暂不实现 |
