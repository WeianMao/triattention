# Discrete Top-K Loss Design (MOE-Style)

## 问题背景

### 当前训练-推理不对齐

**训练（连续）**：
```python
# 当前attraction loss
P_matched = key_probs[argmax_keys]  # (num_queries, num_bins)
match_prob = (query_bin_probs * P_matched).sum(dim=1)  # 对所有bin求和
loss = -log(match_prob)
```
- 所有bin的概率都参与loss计算
- 是一个"soft"的加权和

**推理（离散）**：
```python
# 实际推理过程
b* = argmax(query_bin_probs)  # 取Top-1 bin
top50_keys = topk(key_probs[:, b*], k=50)  # 取该bin的Top-50 keys
```
- 只用Top-1 bin
- 只取Top-50 keys

**根本矛盾**：训练优化的是"所有bin的加权期望"，但推理只用"Top-1 bin的Top-50 keys"。

---

## 新的Loss设计（MOE风格）

### 核心思想

模仿MOE的routing机制，只对**实际会被选中的bin**计算loss，并且只在**key未进入Top-K**时才优化。

### 符号定义

- $Q$: query向量
- $K_i$: 第$i$个key向量
- $B = 128$: bin数量
- $k = 50$: 每个bin取的top-k key数量
- $i^*$: ground truth argmax key

**Query网络输出**：
$$p_q(b|Q) = \text{softmax}(f_Q(Q))_b$$

**Key网络输出**：
$$p_k(i|b) = \text{softmax}(g_K(K_i, b))_i$$

### 推理过程（保持不变）

1. $b^* = \arg\max_b \, p_q(b|Q)$ — 取Top-1 bin
2. $\text{TopK}_k(b^*) = \arg\text{topk}_{i} \, p_k(i|b^*)$ — 在bin $b^*$中取Top-k keys

### 新的Loss计算

**Step 1**: 取Top-1 bin（与推理对齐）
$$b^* = \arg\max_b \, p_q(b|Q)$$

**Step 2**: 检查ground truth key是否在Top-K中
$$\text{hit} = \mathbb{1}[i^* \in \text{TopK}_k(b^*)]$$

**Step 3**: 计算Loss（只对miss的情况）
$$
\mathcal{L} =
\begin{cases}
0 & \text{if hit} \\
p_q(b^*|Q) \cdot \text{CE}(p_k(\cdot|b^*), i^*) & \text{if miss}
\end{cases}
$$

其中Cross-Entropy：
$$\text{CE}(p_k(\cdot|b^*), i^*) = -\log p_k(i^*|b^*)$$

**完整形式**：
$$\mathcal{L} = p_q(b^*|Q) \cdot (-\log p_k(i^*|b^*)) \cdot (1 - \text{hit})$$

---

## 梯度分析

### 详细推导

设 loss 为：
$$\mathcal{L} = p_q(b^*) \cdot c$$

其中 $c = -\log p_k(i^*|b^*) > 0$（CE loss，恒正）

#### Query侧梯度

设 query logits 为 $z$，则：
$$p_q(b^*) = \frac{\exp(z_{b^*})}{\sum_b \exp(z_b)}$$

对 $z_{b^*}$ 求导：
$$\frac{\partial \mathcal{L}}{\partial z_{b^*}} = c \cdot \frac{\partial p_q(b^*)}{\partial z_{b^*}} = c \cdot p_q(b^*)(1 - p_q(b^*)) > 0$$

**梯度下降更新**：
$$z_{b^*} \leftarrow z_{b^*} - \eta \cdot \frac{\partial \mathcal{L}}{\partial z_{b^*}}$$

由于梯度 > 0，所以 $z_{b^*}$ 减小 → **$p_q(b^*)$ 往下走**。

**这是正确的行为**：miss说明选错了bin，应该减少选这个bin的概率。

#### Key侧梯度

对 key logits $w$ 求导（在bin $b^*$中）：
$$\frac{\partial \mathcal{L}}{\partial w_{i^*}} = p_q(b^*) \cdot \frac{\partial}{\partial w_{i^*}}(-\log p_k(i^*|b^*))$$

$$= p_q(b^*) \cdot (p_k(i^*|b^*) - 1) < 0$$

梯度 < 0，所以 $w_{i^*}$ 增大 → **$p_k(i^*|b^*)$ 往上走**。

**这也是正确的行为**：让GT key在该bin中的概率增加。

### 梯度总结

| 网络 | miss时梯度方向 | 含义 |
|-----|---------------|-----|
| Query | $p_q(b^*)$ ↓ | "这个bin选错了，少选它" ✓ |
| Key | $p_k(i^*\|b^*)$ ↑ | "GT key在该bin排名要上升" ✓ |

---

## 实现细节（并行化）

### 核心原则

**不使用for循环**，全部用mask和batch操作实现并行计算。

### 完整实现

```python
def compute_discrete_topk_loss(key_logits, query_logits, argmax_keys, argmax_in_recent, top_k=50, eps=1e-8):
    """
    Compute Discrete Top-K Loss (MOE-style).

    Args:
        key_logits: (num_keys, num_bins) - raw logits from key network
        query_logits: (num_queries, num_bins) - raw logits from query network
        argmax_keys: (num_queries,) - ground truth argmax key indices
        argmax_in_recent: (num_queries,) - whether argmax is in recent keys (to exclude)
        top_k: number of top keys to consider as "hit"
        eps: small value for numerical stability

    Returns:
        loss: scalar tensor
    """
    # Step 0: Filter out queries where argmax is in recent keys
    valid_mask = ~argmax_in_recent
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=key_logits.device, requires_grad=True)

    valid_query_logits = query_logits[valid_mask]  # (num_valid, num_bins)
    valid_argmax_keys = argmax_keys[valid_mask]     # (num_valid,)
    num_valid = valid_query_logits.shape[0]
    num_keys, num_bins = key_logits.shape

    # Step 1: Compute probabilities
    query_probs = F.softmax(valid_query_logits, dim=-1)  # (num_valid, num_bins)
    key_log_probs = F.log_softmax(key_logits, dim=0)     # (num_keys, num_bins) - softmax over keys

    # Step 2: Get top-1 bin for each query (discrete selection, aligned with inference)
    b_star = valid_query_logits.argmax(dim=-1)  # (num_valid,)

    # Step 3: Get p_q(b*) for each query (differentiable)
    p_q_bstar = query_probs[torch.arange(num_valid, device=query_probs.device), b_star]  # (num_valid,)

    # Step 4: For each query, check if argmax_key is in top-k of bin b*
    # First, get key_logits for each query's selected bin
    # key_logits[:, b_star] -> (num_keys, num_valid)
    key_logits_selected = key_logits[:, b_star]  # (num_keys, num_valid)

    # Get top-k key indices for each query's bin
    _, topk_indices = key_logits_selected.topk(top_k, dim=0)  # (top_k, num_valid)

    # Check if argmax_key is in top-k (vectorized)
    # argmax_keys: (num_valid,) -> expand to (top_k, num_valid) for comparison
    hit_mask = (topk_indices == valid_argmax_keys.unsqueeze(0)).any(dim=0)  # (num_valid,)
    miss_mask = ~hit_mask  # (num_valid,)

    # Step 5: Compute CE loss only for miss cases
    # Get log_p_k(argmax_key | b*) for each query
    # key_log_probs: (num_keys, num_bins)
    # We need key_log_probs[argmax_keys[i], b_star[i]] for each i
    log_p_k_target = key_log_probs[valid_argmax_keys, b_star]  # (num_valid,)
    ce_loss = -log_p_k_target  # (num_valid,)

    # Step 6: Compute final loss with mask
    # loss = p_q(b*) * CE * miss_mask
    per_sample_loss = p_q_bstar * ce_loss * miss_mask.float()  # (num_valid,)

    # Mean over valid samples (or sum, depending on preference)
    loss = per_sample_loss.sum() / (miss_mask.sum() + eps)  # mean over miss samples only
    # Alternative: loss = per_sample_loss.mean()  # mean over all valid samples

    return loss
```

### 关键实现技巧

1. **避免for循环**：使用 `[:, b_star]` 索引一次性获取所有query对应bin的数据

2. **Vectorized hit检查**：
   ```python
   hit_mask = (topk_indices == valid_argmax_keys.unsqueeze(0)).any(dim=0)
   ```

3. **Mask乘法代替if-else**：
   ```python
   per_sample_loss = p_q_bstar * ce_loss * miss_mask.float()
   ```

4. **数值稳定性**：使用 `F.log_softmax` 而不是 `log(softmax(...))`

---

## 待讨论问题

### 1. Loss归一化方式

```python
# Option A: mean over miss samples only
loss = per_sample_loss.sum() / (miss_mask.sum() + eps)

# Option B: mean over all valid samples
loss = per_sample_loss.mean()
```

Option A 可能导致batch间loss scale不稳定（miss数量变化大）。

### 2. hit时完全没有梯度

- 当$i^* \in \text{TopK}_k(b^*)$时，loss=0，完全没有梯度
- 是否需要auxiliary loss来保持网络活跃？

### 3. k的选择

- 训练时用$k=50$还是更严格的$k'=45$？
- 更严格的k'可能帮助泛化

---

## 实验计划

1. 实现新的discrete top-k loss
2. 与当前attraction loss对比
3. Ablation:
   - 有/无 $p_q$ 加权
   - 不同的k值
   - Loss归一化方式
