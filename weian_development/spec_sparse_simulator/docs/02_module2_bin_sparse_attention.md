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

## Round 内新 Key 的处理

> **关键设计**：当前 round 内新生成的 Key **全部参与 attention**（Full Attention），只有**历史 Key** 需要做 Sparse Attention。

### 处理规则

在第 N 个 round 内（假设 round_start = N × 128）：
- **历史 Key**（位置 < round_start）：经过 binning，只与同 bin 的 Query 做 attention（Sparse）
- **当前 round 内新 Key**（位置 >= round_start）：**不做 binning**，与所有 Query 做 attention（Full）

### 示例

假设当前是 round 2（round_start = 256），正在解码第 356 个 token（round 内第 100 个）：

| Key 位置 | 处理方式 |
|----------|----------|
| 0 ~ 255（历史 Key） | Sparse Attention（只与同 bin 的 Query 计算） |
| 256 ~ 355（当前 round 内的 99 个 Key） | Full Attention（全部参与计算） |

### 实现

```python
def sparse_attention_with_recent_keys(Q, kv_cache, bin_assignments, neural_net_query, round_start):
    """
    混合 Sparse + Full Attention

    Args:
        Q: 当前 Query
        kv_cache: 所有 Key（包括历史 + 当前 round）
        bin_assignments: (round_start,) - 只有历史 Key 有 bin 分配
        round_start: 当前 round 的起始位置
    """
    # 1. 历史 Key：Sparse Attention
    history_keys = kv_cache[:round_start]
    logits = neural_net_query(Q.unsqueeze(0))
    bin_q = logits.argmax(dim=1).item()

    mask = bin_assignments == bin_q
    sparse_keys = history_keys[mask]

    # 2. 当前 round 内的 Key：Full Attention
    recent_keys = kv_cache[round_start:]  # 全部参与

    # 3. 合并计算 attention
    all_relevant_keys = concat([sparse_keys, recent_keys])
    output = attention(Q, all_relevant_keys)

    return output
```

### 设计理由

- **简化实现**：避免在 round 内频繁更新 bin_index
- **保证近期信息**：最近的 Key 通常更重要，full attention 保证不遗漏
- **计算开销可控**：每个 round 最多 128 个新 Key，full attention 开销有限

---

## 边界情况处理

### 空 bin 问题

如果 Query 的 bin 中没有 Key：

**策略：空 Bin Masking + 短路处理（推荐）**

> **核心思想**：在 round 开头统计空 bin，Query routing 时将空 bin 的 logits 设为 -inf，确保 Query 不会被分配到空 bin。
>
> **边界情况**：首个 round（round_start=0）或历史 Key 为空时，跳过 Sparse Attention，直接使用 Full Attention。

```python
def get_empty_bin_mask(bin_assignments, num_bins=128):
    """
    在每个 round 开头调用，统计哪些 bin 是空的

    Args:
        bin_assignments: (num_keys,) - 每个历史 Key 的 bin ID
        num_bins: bin 总数

    Returns:
        empty_mask: (num_bins,) - True 表示该 bin 为空
        all_empty: bool - True 表示所有 bin 都为空（没有历史 Key）
    """
    # 边界情况：没有历史 Key
    if len(bin_assignments) == 0:
        return torch.ones(num_bins, dtype=torch.bool), True  # 所有 bin 为空

    bin_counts = torch.zeros(num_bins)
    for bin_id in range(num_bins):
        bin_counts[bin_id] = (bin_assignments == bin_id).sum()

    empty_mask = bin_counts == 0  # (num_bins,)
    all_empty = empty_mask.all().item()

    return empty_mask, all_empty


def sparse_attention_with_empty_mask(Q, history_keys, recent_keys, bin_assignments,
                                      neural_net_query, empty_bin_mask, all_bins_empty):
    """
    使用空 bin mask 的 sparse attention，带短路处理

    Args:
        Q: 当前 Query
        history_keys: (num_history_keys, head_dim) - 历史 Key（做 Sparse）
        recent_keys: (num_recent_keys, head_dim) - 当前 round 新 Key（做 Full）
        bin_assignments: (num_history_keys,) - 历史 Key 的 bin ID
        neural_net_query: Query routing 网络
        empty_bin_mask: (num_bins,) - True 表示该 bin 为空
        all_bins_empty: bool - 是否所有 bin 都为空（首个 round）
    """
    # ========== 短路处理：首个 round 或历史 Key 为空 ==========
    if all_bins_empty or len(history_keys) == 0:
        # 没有历史 Key，只与 recent_keys 做 Full Attention
        if len(recent_keys) == 0:
            # 极端情况：完全没有 Key（不应该发生）
            return None
        return attention(Q, recent_keys)

    # ========== 正常流程：Sparse + Full ==========
    # 1. Query routing
    logits = neural_net_query(Q.unsqueeze(0))  # (1, num_bins)

    # 将空 bin 的 logits 设为 -inf，确保不会被选中
    logits = logits.masked_fill(empty_bin_mask.unsqueeze(0), float('-inf'))

    bin_q = logits.argmax(dim=1).item()

    # 2. 获取同 bin 的历史 Key（Sparse 部分）
    mask = bin_assignments == bin_q
    sparse_keys = history_keys[mask]

    # 3. 合并 Sparse + Full
    all_relevant_keys = torch.cat([sparse_keys, recent_keys], dim=0)

    output = attention(Q, all_relevant_keys)
    return output
```

### 首个 Round 处理

> **重要**：首个 round（round_start=0）时没有历史 Key，必须短路处理。

| 场景 | round_start | 历史 Key | 处理方式 |
|------|-------------|----------|----------|
| 首个 round | 0 | 空 | 跳过 Sparse，只用 round 内新 Key 做 Full Attention |
| 后续 round | > 0 | 非空 | 正常 Sparse + Full |

```python
def should_skip_sparse_attention(round_start, history_key_bins):
    """
    判断是否应该跳过 Sparse Attention

    Returns:
        True: 跳过 Sparse，直接走 Full Attention
        False: 正常执行 Sparse + Full
    """
    return round_start == 0 or len(history_key_bins) == 0
```

**优点**：
- 从源头避免空 bin 问题，无需 fallback
- 模型会自动选择有 Key 的、可能性最大的 bin
- 计算开销极小（只在 round 开头统计一次）

**备选策略：Fallback 到 Full Attention（不推荐）**

```python
# 仅作为参考，不推荐使用
def sparse_attention_with_fallback(Q, kv_cache, bin_assignments, neural_net_query):
    """
    ⚠️ 不推荐：fallback 策略会导致计算量不可预测
    """
    logits = neural_net_query(Q.unsqueeze(0))
    bin_q = logits.argmax(dim=1).item()

    mask = bin_assignments == bin_q
    num_relevant = mask.sum().item()

    if num_relevant == 0:
        warnings.warn(f"Empty bin detected! bin_id={bin_q}")
        relevant_keys = kv_cache  # Fallback 到 full attention
    else:
        relevant_keys = kv_cache[mask]

    output = attention(Q, relevant_keys)
    return output
```

> **监控要求**：即使使用 masking 策略，仍需监控 bin 分布均匀性。如果大量 bin 为空，说明 Key binning 网络需要优化（可能需要 Load Balancing Loss）。

> **训推不一致说明（允许）**：训练时数据分布可能较均匀，空 bin 较少；但推理时可能出现更多空 bin。这种不一致是**允许的**，因为空 bin masking（-inf）机制会在推理时自动处理。模型无需在训练时学习"避开空 bin"的行为。

### Multi-bin Query

**当前决策**：暂不实现，只 attend 单一 bin。

待实验结果出来后再评估是否需要 multi-bin routing（soft routing）。

---

## 评估指标

### 核心指标（最重要）

> **注意**：以下指标**仅评估历史 Key 的 Sparse Attention 部分**。Round 内新 Key 始终走 Full Attention，其计算量在实际部署时需单独计算。

| 指标 | 定义 | 目标 | Baseline (Full Attention) |
|------|------|------|---------------------------|
| **Argmax Hit Rate** | Query 仍能 attend 到原 argmax Key 的比例（即 Q 和其 argmax K 在同一 bin） | 越高越好（>99%） | 100% |
| **Keys per Query (Sparse)** | 每个 Query 参与 Sparse Attention 的平均历史 Key 数量（= 平均 bin 大小） | 越低越好 | N_history（所有历史 Key） |
| **Computation Reduction (Sparse)** | 1 - (avg bin size / num_history_keys) | 越高越好 | 0% |

> **Argmax Hit Rate 是最关键指标**：如果 Q 和其 argmax K 不在同一个 bin，Query 将无法 attend 到正确的 Key。
>
> **关于 argmax 落在当前 round 新 Key 的情况**：不存在此问题。训练和评估时 argmax 只在历史 Key 中计算（`attention[:, :round_start]`），推理时当前 round 新 Key 走 Full Attention，不涉及 binning。

### 实际计算量估算

实际部署时，每个 Query 的 attention 计算量 = **Sparse 部分 + Full 部分**：

```
Keys per Query (实际) = avg_bin_size + num_recent_keys
                      = avg_bin_size + (query_pos - round_start)
```

| 组成部分 | Key 来源 | 计算方式 |
|----------|----------|----------|
| Sparse 部分 | 历史 Key（< round_start） | 只与同 bin 的 Key 计算 |
| Full 部分 | 当前 round 新 Key（>= round_start） | 与所有新 Key 计算 |

**示例**（round_start=1024，当前解码到第 1100 个 token）：
- 历史 Key 数量：1024
- 当前 round 新 Key 数量：76（1100 - 1024）
- 假设平均 bin 大小：8
- Keys per Query (实际) = 8 + 76 = 84
- Computation Reduction (实际) = 1 - 84/1100 ≈ 92.4%

### 辅助指标

| 指标 | 定义 | 目标 |
|------|------|------|
| Bin Balance | bin 大小的方差 | 越低越好（均匀分布） |
| Empty Bin Rate | 空 bin 的比例 | 越低越好 |
| Bin Utilization | 实际使用的 bin 数量 / 总 bin 数量 | 越高越好 |

### 指标计算示例

```python
def compute_module2_metrics_sparse_only(query_bins, history_key_bins, query_to_argmax_history_key):
    """
    计算 Module 2 评估指标（仅 Sparse 部分，不含 round 内新 Key 的 Full Attention）

    ⚠️ 注意：此函数只评估历史 Key 的 Sparse Attention 效果。
    实际计算量需要额外加上 round 内新 Key 的 Full Attention 开销。

    Args:
        query_bins: (num_queries,) - 每个 Query 分配的 bin ID
        history_key_bins: (num_history_keys,) - 每个**历史** Key 分配的 bin ID
                          （不含 round 内新 Key，因为它们不做 binning）
        query_to_argmax_history_key: (num_queries,) - 每个 Query 在历史 Key 中的 argmax 索引
                                      （如果 argmax 落在 round 内新 Key，则不参与此指标计算）
    """
    num_queries = len(query_bins)
    num_history_keys = len(history_key_bins)

    # 边界情况：没有历史 Key（首个 round）
    if num_history_keys == 0:
        return {
            'argmax_hit_rate': 1.0,  # 无历史 Key 时定义为 100%
            'keys_per_query_sparse': 0,
            'computation_reduction_sparse': 1.0,
            'bin_balance_var': 0,
            'empty_bin_rate': 1.0,
        }

    # Argmax Hit Rate（关键指标）
    # 检查每个 Query 和其 argmax 历史 Key 是否在同一个 bin
    argmax_key_bins = history_key_bins[query_to_argmax_history_key]
    hits = (query_bins == argmax_key_bins).sum()
    argmax_hit_rate = hits / num_queries

    # Keys per Query - Sparse 部分（平均 bin 大小）
    bin_sizes = []
    for q_bin in query_bins:
        bin_size = (history_key_bins == q_bin).sum()
        bin_sizes.append(bin_size)
    keys_per_query_sparse = sum(bin_sizes) / num_queries

    # Computation Reduction - Sparse 部分
    computation_reduction_sparse = 1 - (keys_per_query_sparse / num_history_keys)

    # Bin Balance（方差）
    unique_bins, counts = torch.unique(history_key_bins, return_counts=True)
    bin_balance_var = counts.float().var()

    # Empty Bin Rate
    num_bins = 128
    empty_bin_rate = 1 - len(unique_bins) / num_bins

    return {
        'argmax_hit_rate': argmax_hit_rate,
        'keys_per_query_sparse': keys_per_query_sparse,
        'computation_reduction_sparse': computation_reduction_sparse,
        'bin_balance_var': bin_balance_var,
        'empty_bin_rate': empty_bin_rate,
    }


def compute_actual_computation_reduction(keys_per_query_sparse, num_history_keys,
                                          num_recent_keys, query_pos):
    """
    计算实际的计算量节省（包含 Sparse + Full 两部分）

    Args:
        keys_per_query_sparse: Sparse 部分的平均 Key 数量（从上面函数获取）
        num_history_keys: 历史 Key 数量（= round_start）
        num_recent_keys: 当前 round 内新 Key 数量（= query_pos - round_start）
        query_pos: 当前 Query 位置

    Returns:
        实际的 computation reduction
    """
    # Full Attention baseline: 所有 Key
    full_attention_keys = num_history_keys + num_recent_keys

    # Sparse Attention: 同 bin 的历史 Key + 所有新 Key
    sparse_attention_keys = keys_per_query_sparse + num_recent_keys

    actual_reduction = 1 - (sparse_attention_keys / full_attention_keys)
    return actual_reduction
```

### Baseline 对比

| 方法 | Argmax Hit Rate | Keys per Query | Computation Reduction |
|------|-----------------|----------------|----------------------|
| **Full Attention** | 100% | N | 0% |
| **Random Binning** | ~1/128 ≈ 0.78% | N/128 | ~99% |
| **Oracle Binning** | 100% | 取决于 bin 分布 | 取决于 bin 分布 |
| **Neural Network** | 目标 >99% | 目标 N/128 | 目标 ~99% |

> **Random Binning 提供下界**：如果 Q 和 K 随机分 bin，命中率约为 1/128。神经网络必须显著高于此。

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

- [x] 空 bin 处理策略：**空 Bin Masking**（Query routing 时将空 bin logits 设为 -inf）
- [x] Multi-bin Query：**暂不实现**，待实验结果决定
- [x] Round 内新 Key：**Full Attention**，只有历史 Key 做 Sparse

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-14 | 初始化文档 |
| 2025-12-14 | 添加向量化实现注释；确定空 bin fallback 策略；Multi-bin Query 暂不实现 |
| 2025-12-15 | 重构评估指标：添加 Argmax Hit Rate、Keys per Query、Computation Reduction 核心指标及 Full Attention/Random Binning baseline；添加指标计算代码 |
| 2025-12-15 | 添加 Round 内新 Key 处理说明（Full Attention）；更新空 bin 策略为 Masking（-inf）而非 Fallback |
| 2025-12-15 | 修正评估指标：明确只评估 Sparse 部分，添加实际计算量估算公式；添加首个 round/历史 Key 为空的短路处理 |
| 2025-12-15 | 添加训推不一致说明（允许）：空 bin 在训练时可能较少，推理时由 masking 机制自动处理 |
| 2025-12-15 | 澄清 argmax 落在当前 round 新 Key 的情况不存在（训练评估只看历史 Key，推理时新 Key 走 Full Attention） |
