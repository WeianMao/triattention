# Weighted Cross-Entropy Loss Design (MOE-Style)

## 核心思想

回归最基本的目标：让ground truth key在对应bin中的概率最大化。

- Key侧：在key维度做softmax，用cross entropy增大目标key的概率
- Query侧：输出每个bin的概率，作为权重
- 对所有bin计算CE loss，然后用query概率加权求和

## 符号定义

- $Q$: query向量
- $K_i$: 第$i$个key向量
- $B = 128$: bin数量
- $i^*$: ground truth argmax key

**Query网络输出**：
$$p_q(b|Q) = \text{softmax}(f_Q(Q))_b$$

**Key网络输出**：
$$p_k(i|b) = \text{softmax}(g_K(K_i, b))_i$$

## Loss公式

### Step 1: 对每个bin计算Cross-Entropy Loss

$$\text{CE}_b = -\log p_k(i^*|b)$$

这是ground truth key在bin $b$中的cross entropy loss。

### Step 2: 用Query概率加权求和

$$\mathcal{L} = \sum_{b=1}^{B} p_q(b|Q) \cdot \text{CE}_b = -\sum_{b=1}^{B} p_q(b|Q) \cdot \log p_k(i^*|b)$$

## 与之前Attraction Loss的对比

### 之前的Attraction Loss

$$\mathcal{L}_{\text{attraction}} = -\log\left(\sum_{b} p_q(b) \cdot p_k(i^*|b)\right)$$

**特点**：先计算概率的加权和，再取log

### 新的Weighted CE Loss

$$\mathcal{L}_{\text{weighted-CE}} = -\sum_{b} p_q(b) \cdot \log p_k(i^*|b)$$

**特点**：先取log（得到CE），再用概率加权求和

### 数学关系

根据Jensen不等式，对于凸函数 $-\log(\cdot)$：

$$\mathbb{E}[-\log(X)] \geq -\log(\mathbb{E}[X])$$

即：

$$\mathcal{L}_{\text{weighted-CE}} \geq \mathcal{L}_{\text{attraction}}$$

**新的loss总是大于等于旧的loss**，意味着：
- 新方法会产生更大的梯度
- 可能训练更aggressive
- 对每个bin独立优化，不允许"一个bin好就够了"的情况

## 梯度分析

### Query侧梯度

$$\frac{\partial \mathcal{L}}{\partial p_q(b)} = \text{CE}_b = -\log p_k(i^*|b)$$

通过softmax的链式法则：
- 如果bin $b$的CE很大（key排名差），会减少$p_q(b)$
- 如果bin $b$的CE很小（key排名好），会增加$p_q(b)$

**这是正确的行为**：让query学会选择那些对key友好的bin。

### Key侧梯度

$$\frac{\partial \mathcal{L}}{\partial \log p_k(i^*|b)} = -p_q(b)$$

- 每个bin都会收到梯度，但被$p_q(b)$加权
- Query认为重要的bin会收到更大的梯度

## 与MOE的类比

| MOE | 我们的设计 |
|-----|-----------|
| Router选择experts | Query选择bins |
| Expert计算loss | 每个bin计算CE loss |
| Router概率加权loss | Query概率加权CE loss |
| 所有experts都参与梯度 | 所有bins都参与梯度 |

## 实现

```python
def compute_weighted_ce_loss(key_log_probs, query_bin_probs, argmax_keys, argmax_in_recent, eps=1e-8):
    """
    Compute Weighted Cross-Entropy Loss (MOE-style).

    Loss = -sum_b(p_q[b] * log(p_k[argmax_key, b]))
         = sum_b(p_q[b] * CE_b)

    Args:
        key_log_probs: (num_keys, num_bins) - log softmax over keys
        query_bin_probs: (num_queries, num_bins) - softmax over bins
        argmax_keys: (num_queries,) - ground truth argmax key indices
        argmax_in_recent: (num_queries,) - whether argmax is in recent keys

    Returns:
        loss: scalar tensor
    """
    valid_mask = ~argmax_in_recent
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=key_log_probs.device, requires_grad=True)

    valid_query_probs = query_bin_probs[valid_mask]  # (num_valid, num_bins)
    valid_argmax_keys = argmax_keys[valid_mask]       # (num_valid,)

    # Get log_p_k(argmax_key | b) for all bins
    # key_log_probs: (num_keys, num_bins)
    # valid_argmax_keys: (num_valid,)
    # Result: (num_valid, num_bins)
    log_p_k_target = key_log_probs[valid_argmax_keys, :]  # (num_valid, num_bins)

    # CE for each bin: -log_p_k
    ce_per_bin = -log_p_k_target  # (num_valid, num_bins)

    # Weighted sum: sum_b(p_q[b] * CE_b)
    weighted_ce = (valid_query_probs * ce_per_bin).sum(dim=1)  # (num_valid,)

    # Mean over valid queries
    loss = weighted_ce.mean()

    return loss
```

## 潜在问题与讨论

### 1. 所有bin都参与优化
- **优点**：不会因为"一个bin好就够了"而忽略其他bin
- **潜在问题**：可能导致过度优化，所有bin都变得相似？

### 2. Loss scale更大
- Jensen不等式告诉我们新loss >= 旧loss
- 可能需要调整learning rate

### 3. 与推理的对齐
- 推理时只用top-1 bin
- 但训练时所有bin都参与
- 这是否会导致训练-推理不对齐？

### 4. Query侧的行为
- Query会学习避开那些让key排名差的bin
- 这是否是我们想要的？还是应该让key在所有bin都排名好？

---

## 待确认

1. 公式理解是否正确？
2. 是否需要处理数值稳定性问题？
3. 是否需要对比实验验证效果？
