# 系统架构与模块设计

## Module 1: Key Pruning (Drop KV)

### 目标

预测每个 Key 是否会被未来 Query attend，丢弃预测为"不会被 attend"的 Key。

### 算法流程

```python
# 每个 round 开头执行
def key_pruning(kv_cache, neural_net, threshold):
    drop_probs = neural_net(kv_cache)  # (num_keys,)
    retain_mask = drop_probs < threshold
    return kv_cache[retain_mask]
```

### 神经网络结构

```
K (post-RoPE) → Kernel Encoding (128-dim) → MLP (128→h→1) → Position Scaling → Sigmoid → drop 概率
```

- **输出语义**：p 接近 1 → 应该 drop；p 接近 0 → 应该保留
- **Position Scaling**：Module 1 专用，在 log 尺度锚点（1k/10k/100k）上插值，调整不同位置 Key 的 drop 倾向

### 标签定义

```
label(K_i) = 0  若存在 Q_j (j >= round_start) 使得 argmax Attention(Q_j, K) == i
label(K_i) = 1  否则
```

- 使用 **argmax** 判定，只要有一个未来 Q attend 到该 K，标签为 0
- **训练时排除末尾 1k Key**：避免标签噪声

---

## Module 2: Multi-Bin Sparse Attention

### 目标

将 Key 分到多个 bin，Query 选择一个 bin 后取 TopK 个 Key 做 attention。

### 核心设计：Softmax over Keys

与传统方案（每个 Key 只属于一个 bin）不同，**允许 Key 同时属于多个 bin**：

```python
# Key 打分：每个 Bin 在 key 维度上 softmax
logits = neural_net_key(kv_cache)  # (num_keys, num_bins)
P = softmax(logits, dim=0)         # 每列和为 1
# P[:, b] 是 bin b 对所有 key 的概率分布
# P[k, :] 不和为 1，key 可以在多个 bin 中都有高分
```

### 算法流程

**Round 开头：Key Scoring**
```python
def key_scoring(kv_cache, neural_net_key):
    logits = neural_net_key(kv_cache)      # (num_keys, num_bins)
    key_probs = softmax(logits, dim=0)     # softmax over keys
    return key_probs
```

**每次解码：Query Routing + TopK Attention**
```python
def topk_attention(Q, history_keys, recent_keys, key_probs, neural_net_query, K):
    # Query 选 bin
    bin_q = neural_net_query(Q).argmax()

    # 历史 Key: 选该 bin 的 TopK
    scores = key_probs[:, bin_q]
    topk_indices = scores.topk(K).indices #后续部署的时候需要把这个排序挪到round的开头做，提高效率
    sparse_keys = history_keys[topk_indices]

    # 当前 round 新 Key: Full Attention
    all_keys = concat([sparse_keys, recent_keys])
    return attention(Q, all_keys)
```

### 神经网络结构

```
K (post-RoPE) → Kernel Encoding (128-dim) → [无 Softmax，输出 logits]
Q (post-RoPE) → Kernel Encoding (128-dim) → Softmax → bin 概率
```

- Key 网络输出 logits，softmax 在 key 维度上做
- Query 网络输出 bin 概率（softmax 在 bin 维度）
- **无 MLP，无 Position Scaling**

### Round 内新 Key 处理

- **历史 Key**（< round_start）：TopK Sparse Attention
- **当前 round 新 Key**（>= round_start）：Full Attention

### 首个 Round 处理

当 `round_start == 0` 时，没有历史 Key，直接走 Full Attention。

---

## 待定事项

### Module 1
- [ ] MLP hidden dimension
- [ ] threshold 设定方式（固定/自适应/可学习）
- [ ] Position Scaling 锚点配置

### Module 2
- [ ] TopK 的 K 值选择（50/500/1000）
- [ ] Bin 数量（固定 128 或可调）
- [ ] 与 Module 1 的配合方式
