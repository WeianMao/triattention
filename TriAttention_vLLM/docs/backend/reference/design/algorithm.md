# 算法设计

本文档描述 TriAttention 的核心算法：打分公式和裁剪逻辑。

---

## 1. 算法概述

TriAttention 通过**频率统计打分**来评估每个 KV token 的重要性，保留最重要的 token，裁剪掉不重要的 token。

### 1.1 工作流程

```
1. 收集校准数据，统计 Query 的频率分布 → 生成 stats 文件
2. Prefill 阶段：正常写入 KV cache
3. Decode 阶段：新 token 写入 overflow pages
4. 当 overflow 满且超过 budget：
   a. 对所有 token 打分
   b. 选择 top-k 保留
   c. 执行 Fill-in-Place
   d. 释放 overflow pages
```

---

## 2. 打分公式

### 2.1 原始公式

来自 `round_pruning_utils.py:score_keys_for_round`：

$$
\text{score}(\mathbf{k}, t) = \sum_{f} \left[ A_f \cdot s_f^2 \cdot \cos\bigl((t - p) \cdot \omega_f + \phi_f\bigr) \right] + \sum_{f} \left[ E_f \cdot s_f^2 \right]
$$

### 2.2 符号定义

| 符号 | 含义 | 来源 |
|-----|------|------|
| $\mathbf{K}$ | 原始 key 向量（未旋转） | 模型输出 |
| $\mathbf{K}_{\text{rot}}$ | RoPE 旋转后的 key 向量 | 显存中存储 |
| $p$ | key 的原始位置索引 | 额外存储 |
| $t$ | 打分的目标未来位置 | 打分时指定 |
| $\omega_f$ | 第 $f$ 个频率分量的 RoPE 频率 | 模型配置 |
| $\bar{\mathbf{Q}}_{\text{mean}}$ | 统计得到的平均 query（复数形式） | stats 文件 |
| $\phi_f$ | Q 和原始 K 的相位差 | 计算得到 |
| $A_f$ | 第 $f$ 个频率分量的幅度 | 计算得到 |
| $s_f^2$ | 频率缩放因子 | stats 文件 |
| $E_f$ | 位置无关项系数 | stats 文件 |

### 2.3 幅度和相位计算

$$
\begin{aligned}
\phi_f &= \arg\bigl(\bar{\mathbf{Q}}_{\text{mean},f} \cdot \mathbf{K}_f^*\bigr) \\
A_f &= |\bar{\mathbf{Q}}_{\text{mean},f}| \cdot |\mathbf{K}_f|
\end{aligned}
$$

---

## 3. 多位置打分与聚合

### 3.1 多位置打分

对于每个 token，需要对多个未来位置打分：

```python
offsets = [0, 1, 2, ..., 15]  # 16 个未来位置
round_start = current_position  # 当前轮次起始位置

for offset in offsets:
    t = round_start + offset
    score_at_t = compute_score(k, t)
```

### 3.2 聚合策略

| 策略 | 公式 | 说明 |
|-----|------|------|
| mean | $\text{final\_score} = \frac{1}{N}\sum_{i} \text{score}(t_i)$ | 平均分 |
| max | $\text{final\_score} = \max_{i} \text{score}(t_i)$ | 最高分 |

---

## 4. 裁剪逻辑

### 4.1 触发条件

```python
def should_prune(budget_used, overflow_slots, budget_slots, divide_length):
    """
    裁剪触发条件：
    1. Overflow 满了（达到 divide_length）
    2. 且当前 KV 总量超过 budget
    """
    overflow_full = (overflow_slots >= divide_length)
    exceeds_budget = (budget_used + overflow_slots > budget_slots)
    return overflow_full and exceeds_budget
```

**重要**：如果 budget 还没满，不触发裁剪，直接合并 overflow 到 budget。

### 4.2 裁剪粒度

| 变种 | 描述 | 参数 |
|-----|------|------|
| per-head | 每个 KV head 全局独立选择 token | `pruning_mode="per_head"` |
| per-layer-per-head | 每个 (layer, head) 独立选择 | `pruning_mode="per_layer_per_head"` |
| per-layer | 同层所有 head 共享 token 选择 | `pruning_mode="per_layer"` |

### 4.3 Top-k 选择

```python
def select_tokens_to_keep(scores, budget_slots, prefill_len, protect_prefill):
    """
    选择要保留的 token

    Args:
        scores: [total_tokens] - 每个 token 的打分
        budget_slots: int - budget 槽位数
        prefill_len: int - prefill 长度
        protect_prefill: bool - 是否保护 prefill token

    Returns:
        keep_mask: [total_tokens] - True 表示保留
    """
    if protect_prefill:
        # Prefill token 不参与竞争，直接保留
        prefill_scores = scores[:prefill_len]
        decode_scores = scores[prefill_len:]

        # 从 decode 部分选择 top-k
        decode_keep_count = budget_slots - prefill_len
        _, top_indices = decode_scores.topk(decode_keep_count)

        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        keep_mask[:prefill_len] = True
        keep_mask[prefill_len + top_indices] = True
    else:
        # 所有 token 一起竞争
        _, top_indices = scores.topk(budget_slots)
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        keep_mask[top_indices] = True

    return keep_mask
```

---

## 5. Prefill 处理

### 5.1 Prefill > Budget 情况

如果 prefill 长度超过 budget：

1. Prefill 完成后立即触发裁剪
2. 将 KV 压缩到 budget 以内

```python
def on_prefill_complete(prefill_len, budget_slots, scorer):
    if prefill_len > budget_slots:
        # 立即触发裁剪
        scores = scorer.score_all_tokens()
        keep_mask = select_tokens_to_keep(scores, budget_slots, ...)
        prune_and_fill(keep_mask)
```

### 5.2 protect_prefill 参数

| 值 | 行为 |
|---|------|
| `False`（默认） | prefill token 参与裁剪竞争，可能被裁掉 |
| `True` | prefill token 被保护，不参与裁剪 |

---

## 6. Stats 文件格式

Stats 文件由校准脚本生成，包含以下内容：

| 变量 | 形状 | 说明 |
|-----|------|------|
| `Q_mean_real` | `[num_layers, num_heads, freq_count]` | 平均 query 实部 |
| `Q_mean_imag` | `[num_layers, num_heads, freq_count]` | 平均 query 虚部 |
| `freq_scale_sq` | `[num_layers, num_heads, freq_count]` | 频率缩放因子 $s_f^2$ |
| `omega` | `[freq_count]` | RoPE 频率 |
| `extra_coef` | `[num_layers, num_heads, freq_count]` | 位置无关项系数 |

其中 `freq_count = head_dim / 2`（对于 half RoPE 风格）。

---

## 7. 与优化的关系

本文档描述的是**原始算法逻辑**。实际实现中，为了性能会进行以下优化：

1. **避免 RoPE 反转**：使用 $\mathbf{K}_{\text{rot}}$ 直接计算，通过相位校正实现等价
2. **单次读取多位置打分**：K 只从显存读取一次，多个 offset 在寄存器中迭代
3. **共享三角函数表**：$\cos(t\omega)$、$\sin(t\omega)$ 预计算，所有 token 共享

详见：[optimization.md](optimization.md)

---

*文档版本：1.0*
*创建日期：2025-01-30*
