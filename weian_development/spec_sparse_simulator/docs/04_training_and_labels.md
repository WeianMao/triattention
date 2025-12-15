# Training and Labels

## 概述

本文档描述 Module 1 和 Module 2 的：
1. 标签定义
2. Loss Function 设计
3. 训练数据构造
4. 训练策略

---

## 0. 当前阶段：Proof of Concept

> **目标**：验证算法可行性，只需要能 **overfit 一个样本**。

### 0.1 POC 设置

| 项目 | 设置 |
|------|------|
| **训练数据** | 单个 trace 的 qk.pt |
| **测试数据** | 同一个 trace（训练 = 测试） |
| **框架** | 复用 `attention_pruning_case_study_hybrid_rounds_xtrace.py` |
| **成功标准** | 能够 overfit，loss 收敛 |

### 0.2 POC 后续扩展

POC 成功后再考虑：
- [ ] 多 trace 训练
- [ ] 训练/测试分离
- [ ] 泛化性验证

---

## 1. Module 1: Key Pruning 标签

### 1.1 标签定义

> **标签语义**：label 表示 **drop 概率的目标值**（与 01_module1_key_pruning.md 一致）
> - label = 0：会被 attend → **不应 drop**
> - label = 1：不会被 attend → **应该 drop**

一个 Key K_i 的标签定义为：

```
label(K_i) = 0  iff  ∃ Q_j (j >= round_start): argmax_k Attention(Q_j, K_k) == i  (会被 attend，不 drop)
label(K_i) = 1  otherwise  (不会被 attend，drop)
```

**解释**：
- 考察范围：当前 round 开始及之后的**所有** Query（包括当前 round 内的 Query）
- 判定标准：argmax（attention 最大值）
- **只要有一个** Query 的 argmax 是 K_i，则 label = 0（不 drop）

### 1.2 从 Trace 提取标签

```python
def extract_pruning_labels(attention_trace, round_start, round_end, seq_len):
    """
    从 attention trace 提取 Key Pruning 标签

    Args:
        attention_trace: (seq_len, seq_len) attention weights
        round_start: 当前 round 开始位置
        round_end: 当前 round 结束位置
        seq_len: 总序列长度

    Returns:
        labels: (round_start,) binary labels for each key
                label=0: 会被 attend（不 drop）
                label=1: 不会被 attend（drop）
    """
    # 默认所有历史 key 都应该 drop（label=1）
    labels = ones(round_start)

    # 遍历当前 round 及之后的所有 query
    # 注意：从 round_start 开始，包括当前 round 内的 query
    for q_idx in range(round_start, seq_len):
        # 找到该 query 的 argmax key（只考虑历史 key，即 < round_start）
        attn_weights = attention_trace[q_idx, :round_start]
        argmax_key = attn_weights.argmax()

        # 这个 key 会被 attend，标记为不 drop
        labels[argmax_key] = 0

    return labels
```

### 1.3 训练时排除末尾 Key

> **重要**：训练时，**位置**在序列末尾 1k 范围内的 Key **不计算 loss**。

**原因**：
- 末尾 Key 的后续 Query 数量有限
- 一个本来重要的 Key 可能因为后续 Query 太少而一直没被 attend 到
- 这会导致标签噪声：本该保留的 Key 被错误标注为 drop（label=1）

**实现**：
```python
def compute_module1_loss(drop_probs, labels, key_positions, seq_len, exclude_tail=1000):
    """
    计算 Module 1 loss，排除位置在末尾的 Key

    Args:
        drop_probs: (num_keys,) - 模型预测的 drop 概率
        labels: (num_keys,) - 真实标签
        key_positions: (num_keys,) - 每个 Key 的位置（通常是 0, 1, 2, ..., round_start-1）
        seq_len: 序列总长度
        exclude_tail: 排除末尾多少个位置（默认 1000）
    """
    # 只对位置 < (seq_len - exclude_tail) 的 Key 计算 loss
    valid_mask = key_positions < (seq_len - exclude_tail)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0)

    loss = F.binary_cross_entropy(
        drop_probs[valid_mask],
        labels[valid_mask]
    )
    return loss
```

### 1.4 样本特点

| 特点 | 描述 |
|------|------|
| **不平衡性** | 正样本（被 attend）可能远少于负样本 |
| **时序依赖** | 越老的 Key 越可能不被 attend |
| **位置偏差** | argmax 可能集中在某些位置（如最近、最早） |

---

## 2. Module 2: Bin-based Sparse Attention 标签

### 2.1 训练目标

让相关的 Q-K pair 落在同一个 bin。

### 2.2 Ground Truth 构建规则

#### 核心定义：Group

> 如果在一次 attention 操作中，**Q_i 的最大 attention score 对应的是 K_j**，则 Q_i 和 K_j 属于同一个 **group**。

```python
def build_groups(attention_matrix):
    """
    构建 Q-K group 关系

    Args:
        attention_matrix: (num_queries, num_keys) attention weights
                          注意：传入的矩阵应该已经只包含历史 Key
                          例如：attention[round_start:round_end, :round_start]

    Returns:
        groups: dict, key_idx -> list of query_idx that attend to this key
    """
    num_queries = attention_matrix.shape[0]
    groups = defaultdict(list)

    for q_idx in range(num_queries):
        # 每个 query 只取 top-1 key
        # 注意：直接用 `:` 而非 `:q_idx`，因为传入的矩阵已经只包含历史 Key
        argmax_key = attention_matrix[q_idx, :].argmax()
        groups[argmax_key].append(q_idx)

    return groups
```

#### 关系特点

| 特点 | 描述 |
|------|------|
| **一对多** | 一个 Key 可以被多个 Query attend（多个 Q 的 argmax 是同一个 K） |
| **不会多对一** | 每个 Query 只有一个 argmax Key（取 top-1） |
| **Group = Bin** | 同一个 group 的 Q 和 K 应该落在同一个 bin |

#### 示例

```
Q_1 的 argmax → K_5
Q_2 的 argmax → K_5
Q_3 的 argmax → K_8

结果：
- Group A: {K_5, Q_1, Q_2}  ← 一个 K 对应多个 Q
- Group B: {K_8, Q_3}
```

### 2.3 Loss Function 设计

#### 符号定义

- `p_i ∈ Δ^{B-1}`: Query i 的 soft bin 分布（经过 softmax）
- `r_j ∈ Δ^{B-1}`: Key j 的 soft bin 分布（经过 softmax）
- `g(i)`: Query i 所属 group 中的所有 Key 索引
- `k(i)`: Query i 的 argmax Key 索引（每个 query 只有一个）

#### Loss 结构：两项组成

每个 Loss 都由两部分组成：
1. **拉近项**：双向交叉熵，拉近同 group 的 Q-K bin 分布
2. **推远项**：推远非 group 的 Q-K（Experiment A 和 B 不同）

```
L_total = L_attract + λ * L_repel
```

#### 共同部分：双向交叉熵（拉近同 group）

对于每个 Query i 和其 argmax Key k(i)，使用双向交叉熵拉近分布：

```
L_attract = Σ_i [ -p_i · log(r_{k(i)}) - r_{k(i)} · log(p_i) ]
          = Σ_i [ CE(p_i, r_{k(i)}) + CE(r_{k(i)}, p_i) ]
```

**实现**：
```python
def bidirectional_cross_entropy(p, log_p, r, log_r, query_to_key):
    """
    双向交叉熵：拉近同 group 的 Q-K bin 分布

    Args:
        p: (num_queries, num_bins) - query soft bin distributions (softmax)
        log_p: (num_queries, num_bins) - query log distributions (log_softmax)
        r: (num_keys, num_bins) - key soft bin distributions (softmax)
        log_r: (num_keys, num_bins) - key log distributions (log_softmax)
        query_to_key: (num_queries,) - 每个 query 的 argmax key 索引

    ⚠️ 使用 log_softmax 提高数值稳定性，避免 log(softmax + eps) 的数值问题
    """
    num_queries = p.shape[0]

    # 获取匹配的 key 分布
    r_matched = r[query_to_key]        # (num_queries, num_bins)
    log_r_matched = log_r[query_to_key]  # (num_queries, num_bins)

    # 双向交叉熵（向量化）
    # CE(p_i, r_k) = -sum(p * log_r)
    # CE(r_k, p_i) = -sum(r * log_p)
    loss = -(p * log_r_matched).sum() - (r_matched * log_p).sum()

    return loss
```

#### Experiment A: Linear Repel（推远非 group）

**目标**：最小化 query 与非 group key 发生 collision 的期望数量。

```
L_repel_linear = Σ_i Σ_{j ∉ g(i)} p_i^T r_j
```

**高效实现（直接 mask，避免重复计算）**：
```python
def linear_repel_loss(p, r, group_masks):
    """
    Args:
        p: (num_queries, num_bins)
        r: (num_keys, num_bins)
        group_masks: (num_queries, num_keys) - True if (q, k) in same group

    直接用 mask 计算非 group collision，避免 C - C_group 的重复计算
    """
    # 非 group mask
    non_group_masks = ~group_masks  # (num_queries, num_keys)

    # collision probability matrix: s_ij = p_i · r_j
    # p: (Q, B), r: (K, B) -> s: (Q, K)
    s = torch.mm(p, r.T)  # (num_queries, num_keys)

    # 只对非 group 的 (q, k) pair 求和
    loss = (s * non_group_masks.float()).sum()

    return loss
```

**Experiment A 完整 Loss**：
```
L_A = L_attract + λ * L_repel_linear
```

#### Experiment B: Log Repel（推远非 group）

**目标**：使用交叉熵形式推远非 group key。

```
L_repel_log = Σ_i Σ_{j ∉ g(i)} p_i · log(r_j)
```

> **为什么是 `+p·log(r)` 而非 `-p·log(r)`？**
>
> 这里**不是**标准交叉熵 `-p·log(r)`（用于拉近分布），而是其**负形式**。
>
> **数学直觉**：
> - `p·log(r)` 由于 log(概率) < 0，所以 `p·log(r)` 总是负数
> - 当 p 和 r **越相似**时，`p·log(r)` 越大（更接近 0）
> - 当 p 和 r **越不同**时，`p·log(r)` 越小（更负）
>
> **最小化 `p·log(r)`** 会让 `p·log(r)` 更负，即让 p 和 r 更不同（推远）。
>
> 这与 KL 散度的关系：`-p·log(r) = H(p) + KL(p||r)`，所以最小化 `p·log(r)` 等价于最大化 `KL(p||r)`。

**实现**：
```python
def log_repel_loss(p, log_r, group_masks):
    """
    交叉熵形式的推远 loss

    Args:
        p: (num_queries, num_bins) - query soft bin distributions (softmax)
        log_r: (num_keys, num_bins) - key log distributions (log_softmax)
        group_masks: (num_queries, num_keys) - True if (q, k) in same group

    注意：这里是 p · log(r)，希望 p 和非 group 的 r 分布不同
    ⚠️ 使用 log_softmax 替代 log(softmax + eps)，提高数值稳定性
    """
    non_group_masks = ~group_masks

    # 向量化实现
    # p: (Q, B), log_r: (K, B) -> ce_matrix: (Q, K)
    ce_matrix = torch.mm(p, log_r.T)  # p_i · log(r_j) for all i, j
    loss = (ce_matrix * non_group_masks.float()).sum()

    return loss
```

**Experiment B 完整 Loss**：
```
L_B = L_attract + λ * L_repel_log
```

#### 两种 Loss 对比

| 对比项 | Experiment A (Linear) | Experiment B (Log) |
|--------|----------------------|-------------------|
| 推远项形式 | `p · r` (内积) | `p · log(r)` (交叉熵) |
| 梯度特性 | 线性 | 非线性，r 小时梯度大 |
| 优化目标 | 平均 collision 数量 | 分布差异度 |
| 适用场景 | 平均性能优先 | 分布差异优先 |

#### 向量化实现（完整）

```python
def compute_loss_exp_a(p, log_p, r, log_r, query_to_key, group_masks, lambda_repel=1.0):
    """
    Experiment A: 双向交叉熵 + Linear Repel

    Args:
        p: (num_queries, num_bins) - query soft bin distributions (softmax)
        log_p: (num_queries, num_bins) - query log distributions (log_softmax)
        r: (num_keys, num_bins) - key soft bin distributions (softmax)
        log_r: (num_keys, num_bins) - key log distributions (log_softmax)
        query_to_key: (num_queries,) - 每个 query 的 argmax key 索引
        group_masks: (num_queries, num_keys) - True if (q, k) in same group
        lambda_repel: repel loss 的权重

    ⚠️ 使用 log_softmax 替代 log(softmax + eps)，提高数值稳定性
    """
    # 1. 拉近项：双向交叉熵
    r_matched = r[query_to_key]          # (num_queries, num_bins)
    log_r_matched = log_r[query_to_key]  # (num_queries, num_bins)
    attract = -(p * log_r_matched).sum() - (r_matched * log_p).sum()

    # 2. 推远项：Linear
    s = torch.mm(p, r.T)  # (num_queries, num_keys)
    repel = (s * (~group_masks).float()).sum()

    return attract + lambda_repel * repel


def compute_loss_exp_b(p, log_p, r, log_r, query_to_key, group_masks, lambda_repel=1.0):
    """
    Experiment B: 双向交叉熵 + Log Repel

    Args:
        p: (num_queries, num_bins) - query soft bin distributions (softmax)
        log_p: (num_queries, num_bins) - query log distributions (log_softmax)
        r: (num_keys, num_bins) - key soft bin distributions (softmax)
        log_r: (num_keys, num_bins) - key log distributions (log_softmax)
        query_to_key: (num_queries,) - 每个 query 的 argmax key 索引
        group_masks: (num_queries, num_keys) - True if (q, k) in same group
        lambda_repel: repel loss 的权重

    ⚠️ 使用 log_softmax 替代 log(softmax + eps)，提高数值稳定性
    """
    # 1. 拉近项：双向交叉熵
    r_matched = r[query_to_key]          # (num_queries, num_bins)
    log_r_matched = log_r[query_to_key]  # (num_queries, num_bins)
    attract = -(p * log_r_matched).sum() - (r_matched * log_p).sum()

    # 2. 推远项：Log (交叉熵形式)
    # p: (Q, B), log_r: (K, B) -> cross_entropy: (Q, K)
    ce_matrix = torch.mm(p, log_r.T)  # p_i · log(r_j) for all i, j
    repel = (ce_matrix * (~group_masks).float()).sum()

    return attract + lambda_repel * repel
```

### 2.4 待定

- [ ] Experiment A vs B 的实验对比
- [ ] λ (lambda_repel) 的取值
- [ ] Bin collapse 预防机制
- [ ] 数值稳定性处理（使用 log_softmax）
- [ ] **负载均衡 Loss**：类似 MOE（Mixture of Experts）的 load balancing loss，防止 bin 分布过于不均匀（Sanity Check 实验完成后再考虑）

---

## 3. 训练数据

### 3.1 POC 阶段数据来源

> **当前阶段**：使用单个 trace，训练和测试在同一数据上。

| 项目 | POC 设置 |
|------|----------|
| **框架** | `attention_pruning_case_study_hybrid_rounds_xtrace.py` |
| **数据** | 单个 trace 的 qk.pt |
| **训练/测试** | 同一个 trace（overfit 验证） |

### 3.2 数据处理流程

```python
def prepare_training_data(trace_dir, round_window=128):
    """
    从单个 trace 准备训练数据

    ⚠️ 实现注意：以下使用循环仅为表达逻辑，实际应使用 tensor 操作。
    """
    qk_data = torch.load(trace_dir / "qk.pt")
    Q = qk_data['q']  # (layers, heads, seq_len, head_dim)
    K = qk_data['k']

    # 计算 attention（用于提取标签）
    attention = compute_attention(Q, K)

    training_samples = []

    for round_start in range(0, seq_len, round_window):
        round_end = min(round_start + round_window, seq_len)

        # Module 1 标签
        pruning_labels = extract_pruning_labels(
            attention, round_start, round_end, seq_len
        )

        # Module 2 标签（group 关系）
        groups = build_groups(attention[round_start:round_end, :round_start])

        training_samples.append({
            'round_start': round_start,
            'keys': K[:, :, :round_start, :],
            'queries': Q[:, :, round_start:round_end, :],
            'pruning_labels': pruning_labels,
            'groups': groups,
        })

    return training_samples
```

### 3.3 参考向量（已简化）

> **简化**：不再需要预统计 Q 平均向量。直接使用**角度为 0 的向量**作为参考向量。

参考向量在每个频段上是 (1, 0)，经过 RoPE 旋转后：
- 参考角度 = RoPE 旋转角度 = `pos × ω_j`

详见 `03_neural_network_architecture.md` Section 5。

---

## 4. 训练策略

### 4.1 模块分离训练

两个模块**独立训练**：

| 阶段 | 模块 | 依赖 |
|------|------|------|
| Stage 1 | Module 1 (Key Pruning) | 无 |
| Stage 2 | Module 2 (Binning) | 可选：使用 Module 1 pruning 后的 Key |

### 4.2 训练/推理不对齐问题

> **重要**：Module 1 (Key Pruning) 存在训练和推理的不对齐。

| 阶段 | Key 处理 |
|------|----------|
| **训练** | 保留**所有**历史 Key，用于计算 loss |
| **推理** | 预测概率 < threshold 的 Key 被**丢弃**，不参与后续 attention |

**影响**：
- 训练时 ground truth 基于完整的 Key 集合计算
- 推理时 Key 集合会因 pruning 而变化
- 这种不对齐在 POC 阶段可接受，后续可能需要考虑（如 curriculum learning）

**其他部分训推对齐**：
- 128 个 token 一轮
- 每轮开头执行 pruning 和 binning
- Module 2 的 binning 训推对齐

### 4.3 Loss Function

#### Module 1: Binary Classification

```python
loss = BCEWithLogitsLoss(predictions, labels)
```

#### Module 2: 双向交叉熵 + 推远项

见 Section 2.3，完整 Loss 结构：

```
L_total = L_attract + λ * L_repel
```

| 组成部分 | 作用 |
|----------|------|
| `L_attract` | 双向交叉熵，拉近同 group 的 Q-K bin 分布 |
| `L_repel` | 推远非 group 的 Q-K（Exp A: Linear, Exp B: Log） |

两种候选方案：
- **Experiment A**: `L_attract + λ * (p · r)`（Linear Repel）
- **Experiment B**: `L_attract + λ * (p · log(r))`（Log Repel）

### 4.4 优化器

```python
optimizer = Adam(params, lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

### 4.5 POC 阶段训练规模

| 项目 | POC 设置 |
|------|----------|
| Traces 数量 | **1** |
| 训练/测试 | **同一 trace** |
| 目标 | **Overfit** |

---

## 5. Loss Function Sanity Check 实验规划

> **目的**：在完整模拟实验前，验证 Loss function 的优化行为。

### 5.1 实验设计

**核心思想**：用可学习参数直接代替神经网络输出，观察 loss 优化结果。

```python
class SanityCheckModel(nn.Module):
    """
    无输入的模拟模型，直接学习 bin 分布
    """
    def __init__(self, num_queries, num_keys, num_bins):
        super().__init__()
        # 可学习参数代替神经网络输出（logits）
        self.query_logits = nn.Parameter(torch.randn(num_queries, num_bins))
        self.key_logits = nn.Parameter(torch.randn(num_keys, num_bins))

    def forward(self):
        # 使用 log_softmax 解决数值稳定性
        # 返回 softmax 后的概率分布
        p = F.softmax(self.query_logits, dim=1)  # (num_queries, num_bins)
        r = F.softmax(self.key_logits, dim=1)    # (num_keys, num_bins)

        # 也可以返回 log_softmax 用于稳定计算
        log_p = F.log_softmax(self.query_logits, dim=1)
        log_r = F.log_softmax(self.key_logits, dim=1)

        return p, r, log_p, log_r
```

### 5.2 模拟数据

构造包含**一对一**和**一对多**关系的模拟 label：

```python
def generate_mock_data(num_queries, num_keys):
    """
    生成模拟的 group 关系

    返回：
    - query_to_key: (num_queries,) 每个 query 的 argmax key
    - group_masks: (num_queries, num_keys) True if same group
    """
    query_to_key = torch.zeros(num_queries, dtype=torch.long)
    group_masks = torch.zeros(num_queries, num_keys, dtype=torch.bool)

    # 一对一关系
    for q in range(0, num_queries // 2):
        k = q % num_keys
        query_to_key[q] = k
        group_masks[q, k] = True

    # 一对多关系：多个 query 对应同一个 key
    popular_key = 0
    for q in range(num_queries // 2, num_queries):
        query_to_key[q] = popular_key
        group_masks[q, popular_key] = True

    return query_to_key, group_masks
```

### 5.3 Loss 实现（Sanity Check 版）

```python
def sanity_check_loss_exp_a(p, r, log_p, log_r, query_to_key, group_masks, lambda_repel=1.0):
    """
    Experiment A: 双向交叉熵 + Linear Repel
    """
    # 1. 拉近项：双向交叉熵
    r_matched = r[query_to_key]  # (num_queries, num_bins)
    log_r_matched = log_r[query_to_key]
    attract = -(p * log_r_matched).sum() - (r_matched * log_p).sum()

    # 2. 推远项：Linear (p · r for non-group)
    s = torch.mm(p, r.T)  # (num_queries, num_keys)
    repel = (s * (~group_masks).float()).sum()

    return attract + lambda_repel * repel


def sanity_check_loss_exp_b(p, r, log_p, log_r, query_to_key, group_masks, lambda_repel=1.0):
    """
    Experiment B: 双向交叉熵 + Log Repel
    """
    # 1. 拉近项：双向交叉熵
    r_matched = r[query_to_key]
    log_r_matched = log_r[query_to_key]
    attract = -(p * log_r_matched).sum() - (r_matched * log_p).sum()

    # 2. 推远项：Log (p · log(r) for non-group)
    ce_matrix = torch.mm(p, log_r.T)  # (num_queries, num_keys)
    repel = (ce_matrix * (~group_masks).float()).sum()

    return attract + lambda_repel * repel
```

### 5.4 实验对比

| 实验 | Loss | 预期结果 |
|------|------|----------|
| Exp A | 双向 CE + Linear Repel | 收敛，bin 分布较均匀 |
| Exp B | 双向 CE + Log Repel | 收敛，可能有更强的分离 |
| Baseline | 仅双向 CE（无 repel） | 可能 collapse 到同一 bin |

### 5.5 观察指标

#### 训练指标
- [ ] Loss 是否收敛
- [ ] L_attract 和 L_repel 的收敛曲线
- [ ] λ (lambda_repel) 对结果的影响

#### 核心评估指标（最重要）

| 指标 | 定义 | 目标 | Baseline (Full Attention) |
|------|------|------|---------------------------|
| **Argmax Hit Rate** | Q 仍能 attend 到原 argmax K 的比例 | >99% | 100% |
| **Keys per Query** | 每个 Q 参与 attention 的平均 K 数量 | 越低越好 | N |
| **Computation Reduction** | 1 - (Keys per Query / N) | 越高越好 | 0% |

> **Argmax Hit Rate 是最关键指标**：直接衡量 sparse attention 是否能保持生成质量。
>
> **命中判定规则**：argmax 在历史 Key → 检查同 bin；argmax 在当前 round 新 Key → 直接命中（Full Attention）。

#### 辅助指标
- [ ] 最终的 bin 分布（是否均匀）
- [ ] 空 bin 数量
- [ ] Bin 利用率

#### Baseline 对比

| 方法 | Argmax Hit Rate | Keys per Query | Computation Reduction |
|------|-----------------|----------------|----------------------|
| Full Attention | 100% | N | 0% |
| Random Binning | ~0.78% | N/128 | ~99% |
| 目标（Neural） | >99% | N/128 | ~99% |

### 5.6 状态

**状态**: 待实验

---

## 6. 待定设计细节

### Module 1

（暂无待定项）

### Module 2
- [ ] Experiment A vs B 对比结果
- [ ] Bin collapse 预防机制
- [ ] 数值稳定性处理

### 通用（POC 后）
- [ ] 跨 trace 泛化性验证
- [ ] 跨 prompt/task 泛化性
- [ ] Online fine-tuning 可能性

---

## 7. 未来实验注意事项

> **重要**：以下内容在当前 POC/模拟实验阶段**暂不需要处理**，但在后续实际部署实验时必须考虑。

### 7.1 GQA 下 Key 丢弃决策规则

当多个 Query head 共享一个 KV head（GQA）时，需要定义 Key 丢弃的决策规则：

| 场景 | 决策规则 |
|------|----------|
| 所有 Query head 都认为应该 drop | **Drop** 该 Key |
| 任一 Query head 认为应该 retain | **Retain** 该 Key |

**实现**：对同一 KV head 对应的所有 Query head 的预测取 **AND**（全部认为 drop 才 drop）。

```python
def gqa_key_pruning_decision(drop_probs_all_heads, threshold):
    """
    GQA 场景下的 Key 丢弃决策

    Args:
        drop_probs_all_heads: (num_q_heads, num_keys) - 每个 Q head 对每个 K 的 drop 概率
        threshold: drop 阈值

    Returns:
        final_drop_mask: (num_keys,) - True 表示应该 drop
    """
    # 每个 Q head 的 drop 决策
    drop_decisions = drop_probs_all_heads >= threshold  # (num_q_heads, num_keys)

    # 只有所有 Q head 都认为应该 drop，才最终 drop（AND 逻辑）
    final_drop_mask = drop_decisions.all(dim=0)  # (num_keys,)

    return final_drop_mask
```

> **当前阶段**：模拟实验中每个 Query head 独立处理，暂不考虑 GQA 共享。

### 7.2 数据增强：随机偏移 Round 起始位置

**问题**：如果每次训练同一个 trace，round 的中心位置始终固定（如 64, 192, 320, ...），可能导致模型对特定位置过拟合。

**解决方案**：每次训练时，随机丢弃句子开头的 0 ~ `round_window` 个 token，使 round 的中心位置随机偏移。

```python
def augment_trace_with_offset(Q, K, round_window=128):
    """
    数据增强：随机偏移 round 起始位置

    Args:
        Q: (seq_len, head_dim) - Query 序列
        K: (seq_len, head_dim) - Key 序列
        round_window: round 大小

    Returns:
        Q_aug, K_aug: 偏移后的序列
    """
    # 随机偏移量：0 ~ round_window-1
    offset = random.randint(0, round_window - 1)

    # 丢弃开头的 offset 个 token
    Q_aug = Q[offset:]
    K_aug = K[offset:]

    return Q_aug, K_aug, offset
```

**效果**：
- 原始 round 中心：64, 192, 320, ...
- 偏移后 round 中心：(64-offset), (192-offset), (320-offset), ...
- 类似图像数据增强中的随机裁剪，提高模型对位置的泛化能力

> **当前阶段**：POC 阶段暂不实现，待 overfit 验证通过后再考虑。

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-14 | 初始化文档 |
| 2025-12-14 | 添加 POC 阶段说明；完善 Ground Truth 构建规则（一对多关系）；添加 Loss Function 设计（Linear/Log）；添加训练/推理不对齐说明；添加 Sanity Check 实验规划 |
| 2025-12-15 | 修正 Loss 设计：添加双向交叉熵拉近项；修正 Exp B 公式为 `p·log(r)`；优化实现使用直接 mask；简化参考向量说明（指向 03 文档）；添加负载均衡 Loss 待办事项 |
| 2025-12-15 | 简化 Module 1 Loss：移除 FocalLoss、正负样本重采样、时序衰减权重 |
| 2025-12-15 | 添加核心评估指标：Argmax Hit Rate、Keys per Query、Computation Reduction 及 Full Attention/Random Binning baseline |
| 2025-12-15 | 统一标签定义：与 01 文档一致（label=0 不 drop，label=1 drop）；修正时序范围为 round_start 开始；添加 Log Repel 数学直觉解释 |
| 2025-12-15 | 添加"未来实验注意事项"：GQA 下 Key 丢弃决策规则（AND 逻辑）；数据增强策略（随机偏移 round 起始位置） |
| 2025-12-15 | 添加 Module 1 训练时排除末尾 1k Key（避免标签噪声） |
| 2025-12-15 | 修复 `build_groups` 索引 bug：`:q_idx` 改为 `:`；修正排除末尾 Key 逻辑：基于位置而非 seq_len；所有 Loss 函数使用 `log_softmax` 提高数值稳定性 |
| 2025-12-15 | 添加 Argmax Hit Rate 命中判定规则：argmax 在当前 round 新 Key 直接算命中 |
