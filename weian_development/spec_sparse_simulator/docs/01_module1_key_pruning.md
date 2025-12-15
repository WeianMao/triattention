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
│   Output: logit (标量)          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Position Scaling             │
│   (Module 1 专用，见下方说明)    │
│   logit = logit × pos_weight   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Sigmoid                      │
│   Output: p ∈ [0, 1]           │
│   (p = drop 概率)               │
└─────────────────────────────────┘
```

**特点**：比 Module 2 的网络稍重（多一层 MLP + Position Scaling）

**输出语义**：
- p 接近 1 → Key 很可能不会被未来 Query attend → 应该 drop
- p 接近 0 → Key 很可能会被未来 Query attend → 应该保留

---

## Position Scaling（Module 1 专用）

> **注意**：此设计仅用于 Module 1 (Key Pruning)，Module 2 (Bin-based Sparse Attention) **不使用**位置编码。

### 设计动机

不同位置的 Key 可能有不同的"drop 倾向基线"：
- **早期 Key**（距离当前位置很远）：可能更容易被 drop
- **近期 Key**（距离当前位置较近）：通常更重要，不易被 drop

通过在 log 尺度上的可学习权重插值，可以让模型学习到位置相关的 drop 倾向。

### 核心思想

1. 在 **log 尺度**上设置若干**锚点位置**（如 1k, 10k, 100k）
2. 每个锚点有一个**可学习权重**
3. 对于任意位置 x，在锚点之间**线性插值**得到该位置的权重
4. 将权重**乘到 logit 上**（Sigmoid 之前）

### 锚点设计

| 锚点索引 | 位置 | log₁₀(位置) | 可学习权重 |
|----------|------|-------------|------------|
| 0 | 1,000 | 3 | w₀ |
| 1 | 10,000 | 4 | w₁ |
| 2 | 100,000 | 5 | w₂ |

> **可扩展**：锚点数量和位置可以根据实际序列长度调整。

### 插值规则

给定当前位置 x：

```python
def get_position_weight(x, anchor_positions=[1000, 10000, 100000], anchor_weights=learnable):
    """
    在 log 尺度上线性插值得到位置权重

    Args:
        x: 当前位置（Key 的绝对位置或相对位置，见下方讨论）
        anchor_positions: 锚点位置列表（升序）
        anchor_weights: 每个锚点对应的可学习权重

    Returns:
        weight: 该位置的插值权重
    """
    # 边界情况：小于最小锚点
    if x <= anchor_positions[0]:
        return anchor_weights[0]

    # 边界情况：大于最大锚点
    if x >= anchor_positions[-1]:
        return anchor_weights[-1]

    # 找到 x 所在的区间 [anchor_i, anchor_{i+1}]
    log_x = log10(x)
    log_anchors = [log10(p) for p in anchor_positions]

    for i in range(len(log_anchors) - 1):
        if log_anchors[i] <= log_x < log_anchors[i + 1]:
            # 在 log 尺度上线性插值
            t = (log_x - log_anchors[i]) / (log_anchors[i + 1] - log_anchors[i])
            weight = anchor_weights[i] * (1 - t) + anchor_weights[i + 1] * t
            return weight
```

### 插值示例

| 位置 x | log₁₀(x) | 插值结果 |
|--------|----------|----------|
| 500 | 2.7 | w₀（小于 1k，直接取 w₀） |
| 1,000 | 3.0 | w₀ |
| 3,162 | 3.5 | 0.5 × w₀ + 0.5 × w₁（1k 和 10k 的中点） |
| 10,000 | 4.0 | w₁ |
| 31,623 | 4.5 | 0.5 × w₁ + 0.5 × w₂ |
| 100,000 | 5.0 | w₂ |
| 200,000 | 5.3 | w₂（大于 100k，直接取 w₂） |

### 应用方式

```python
def forward_with_position_scaling(K, key_position, mlp, anchor_weights):
    """
    带位置缩放的前向传播

    Args:
        K: Key 向量
        key_position: Key 的位置（绝对位置或相对位置）
        mlp: MLP 层
        anchor_weights: 位置锚点的可学习权重
    """
    # 1. 原有流程：Kernel Encoding + MLP
    logit = mlp(kernel_encoding(K))  # 标量

    # 2. 获取位置权重
    pos_weight = get_position_weight(key_position, anchor_weights)

    # 3. 位置缩放（在 Sigmoid 之前）
    scaled_logit = logit * pos_weight

    # 4. Sigmoid 得到 drop 概率
    drop_prob = sigmoid(scaled_logit)

    return drop_prob
```

### 位置定义选项

> **待实验**：确定 `key_position` 使用哪种定义。

| 选项 | 定义 | 特点 |
|------|------|------|
| **A. 绝对位置** | `key_position = i`（Key 在序列中的位置） | 简单直接；不同 round 同一位置的 Key 权重相同 |
| **B. 相对距离** | `key_position = round_start - i`（Key 距离当前 round 的距离） | 反映"多久以前"的信息；更符合 attention 的时效性 |

**建议先用选项 B（相对距离）**，因为更符合 "越远越可能被 drop" 的直觉。

### 参数初始化与非负约束

```python
def init_position_weights(num_anchors=3):
    """
    初始化位置权重

    初始值为 1.0，表示初始时位置不影响 logit
    使用 softplus 的逆函数初始化，使得 softplus(raw) ≈ 1.0
    """
    # softplus(x) = log(1 + exp(x))
    # 要使 softplus(x) ≈ 1.0，需要 x ≈ log(exp(1) - 1) ≈ 0.5413
    init_value = math.log(math.exp(1.0) - 1)  # ≈ 0.5413
    return nn.Parameter(torch.full((num_anchors,), init_value))

def get_position_weight_with_softplus(x, anchor_positions, anchor_weights_raw):
    """
    带 softplus 非负约束的位置权重获取

    anchor_weights_raw: 原始可学习参数
    实际权重 = softplus(anchor_weights_raw)，保证非负
    """
    # 通过 softplus 保证权重非负
    anchor_weights = F.softplus(anchor_weights_raw)

    # 后续插值逻辑不变...
    return interpolated_weight
```

> **非负约束**：使用 `softplus` 确保权重始终非负。初始化时使用 `softplus` 的逆值，使初始权重 ≈ 1.0。

### 设计理由

1. **Log 尺度**：长序列中位置跨度很大（1 到 100k+），log 尺度可以更均匀地覆盖
2. **线性插值**：计算简单高效，梯度友好
3. **乘法作用**：乘在 logit 上相当于调整 Sigmoid 的"尖锐度"，权重 > 1 使预测更极端，权重 < 1 使预测更温和
4. **Module 1 专用**：因为只有 Key Pruning 需要考虑"距离越远越可能被 drop"的先验

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

## Hard Pruning 策略

> **本项目采用 Hard Pruning**：预测为 drop 的 Key **直接从 KV Cache 中删除**，同时减少显存和计算量。

| 对比项 | 传统 KV Cache 压缩 | 本项目 (Neural Hard Pruning) |
|--------|-------------------|------------------------------|
| **max_keys** | 有硬限制 | **不存在**，由神经网络动态决定 |
| **Key 处理** | 基于规则删除 | 基于神经网络预测删除 |
| **打分方式** | 手工特征（频率等） | **神经网络学习** |
| **目标** | 减少显存 + 计算量 | 减少显存 + 计算量 |

### Pruning 执行

采用 **Hard Pruning**：
- 根据 threshold 决定是否 drop
- 不存在 max_keys 限制
- 被 pruned 的 Key **直接从 KV Cache 中物理删除**

```python
# Hard Pruning: 没有 max_keys 限制，直接删除
# drop_probs: 模型预测的 drop 概率
# retain_mask: drop 概率低于阈值的 Key 被保留
retain_mask = drop_probs < threshold
# 直接删除 drop 的 Key，只保留 retain 的
retained_kv_cache = kv_cache[retain_mask]
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
- [ ] Position Scaling 的位置定义（绝对位置 vs 相对距离）
- [ ] Position Scaling 的锚点数量和位置（当前：1k, 10k, 100k）

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-14 | 初始化文档 |
| 2025-12-14 | 明确 Sparse Attention vs KV Cache 压缩区别；添加向量化实现注释 |
| 2025-12-15 | 修正 Pruning 策略：从 Soft Pruning 改为 Hard Pruning（直接从 KV Cache 中物理删除） |
| 2025-12-15 | 重构评估指标：添加 Argmax Hit Rate、Keys per Query、Computation Reduction 核心指标及 Full Attention baseline；添加指标计算代码；移除 Focal Loss |
| 2025-12-15 | 修正模型输出语义：从"保留概率"改为"drop 概率"；相应调整判断逻辑（`p < threshold` 保留）和标签定义（label=1 表示应 drop） |
| 2025-12-15 | 添加 Position Scaling 设计（Module 1 专用）：log 尺度锚点插值，乘在 Sigmoid 之前的 logit 上 |
