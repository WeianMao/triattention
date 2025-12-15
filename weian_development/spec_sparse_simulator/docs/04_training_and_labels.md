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

一个 Key K_i 的标签定义为：

```
label(K_i) = 1  iff  ∃ Q_j (j > round_end): argmax_k Attention(Q_j, K_k) == i
label(K_i) = 0  otherwise
```

**解释**：
- 考察范围：当前 round 之后的所有 Query
- 判定标准：argmax（attention 最大值）
- **只要有一个** future Query 的 argmax 是 K_i，则 label = 1

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
    """
    labels = zeros(round_start)

    # 遍历未来的所有 query
    for q_idx in range(round_end, seq_len):
        # 找到该 query 的 argmax key（只考虑历史 key）
        attn_weights = attention_trace[q_idx, :q_idx]
        argmax_key = attn_weights.argmax()

        if argmax_key < round_start:
            labels[argmax_key] = 1

    return labels
```

### 1.3 样本特点

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

    Returns:
        groups: dict, key_idx -> list of query_idx that attend to this key
    """
    groups = defaultdict(list)

    for q_idx in range(num_queries):
        # 每个 query 只取 top-1 key
        argmax_key = attention_matrix[q_idx, :q_idx].argmax()
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
def bidirectional_cross_entropy(p, r, query_to_key):
    """
    双向交叉熵：拉近同 group 的 Q-K bin 分布

    Args:
        p: (num_queries, num_bins) - query soft bin distributions
        r: (num_keys, num_bins) - key soft bin distributions
        query_to_key: (num_queries,) - 每个 query 的 argmax key 索引

    ⚠️ 使用 log_softmax 提高数值稳定性
    """
    loss = 0
    for i in range(num_queries):
        k_i = query_to_key[i]
        # 双向交叉熵
        loss += -(p[i] * torch.log(r[k_i] + eps)).sum()  # CE(p_i, r_k)
        loss += -(r[k_i] * torch.log(p[i] + eps)).sum()  # CE(r_k, p_i)
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

**实现**：
```python
def log_repel_loss(p, r, group_masks, eps=1e-6):
    """
    交叉熵形式的推远 loss

    注意：这里是 p · log(r)，希望 p 和非 group 的 r 分布不同
    """
    non_group_masks = ~group_masks

    loss = 0
    for i in range(num_queries):
        for j in range(num_keys):
            if non_group_masks[i, j]:
                # 交叉熵：p_i · log(r_j)
                loss += (p[i] * torch.log(r[j] + eps)).sum()

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
def compute_loss_exp_a(p, r, query_to_key, group_masks, lambda_repel=1.0, eps=1e-6):
    """
    Experiment A: 双向交叉熵 + Linear Repel
    """
    num_queries = p.shape[0]

    # 1. 拉近项：双向交叉熵
    r_matched = r[query_to_key]  # (num_queries, num_bins)
    attract = -(p * torch.log(r_matched + eps)).sum()
    attract += -(r_matched * torch.log(p + eps)).sum()

    # 2. 推远项：Linear
    s = torch.mm(p, r.T)  # (num_queries, num_keys)
    repel = (s * (~group_masks).float()).sum()

    return attract + lambda_repel * repel


def compute_loss_exp_b(p, r, query_to_key, group_masks, lambda_repel=1.0, eps=1e-6):
    """
    Experiment B: 双向交叉熵 + Log Repel
    """
    num_queries = p.shape[0]

    # 1. 拉近项：双向交叉熵
    r_matched = r[query_to_key]
    attract = -(p * torch.log(r_matched + eps)).sum()
    attract += -(r_matched * torch.log(p + eps)).sum()

    # 2. 推远项：Log (交叉熵形式)
    log_r = torch.log(r + eps)  # (num_keys, num_bins)
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
# 基础版本
loss = BCEWithLogitsLoss(predictions, labels)

# 处理不平衡（可选）
loss = FocalLoss(predictions, labels, gamma=2.0)
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

- [ ] Loss 是否收敛
- [ ] L_attract 和 L_repel 的收敛曲线
- [ ] 最终的 bin 分布（是否均匀）
- [ ] 同 group 的 Q-K 是否落在同一 bin（argmax 一致率）
- [ ] 空 bin 数量
- [ ] λ (lambda_repel) 对结果的影响

### 5.6 状态

**状态**: 待实验

---

## 6. 待定设计细节

### Module 1
- [ ] Focal Loss gamma 值
- [ ] 正负样本重采样策略
- [ ] 时序衰减权重（更老的 key 权重更低？）

### Module 2
- [ ] Experiment A vs B 对比结果
- [ ] Bin collapse 预防机制
- [ ] 数值稳定性处理

### 通用（POC 后）
- [ ] 跨 trace 泛化性验证
- [ ] 跨 prompt/task 泛化性
- [ ] Online fine-tuning 可能性

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-14 | 初始化文档 |
| 2025-12-14 | 添加 POC 阶段说明；完善 Ground Truth 构建规则（一对多关系）；添加 Loss Function 设计（Linear/Log）；添加训练/推理不对齐说明；添加 Sanity Check 实验规划 |
| 2025-12-15 | 修正 Loss 设计：添加双向交叉熵拉近项；修正 Exp B 公式为 `p·log(r)`；优化实现使用直接 mask；简化参考向量说明（指向 03 文档）；添加负载均衡 Loss 待办事项 |
