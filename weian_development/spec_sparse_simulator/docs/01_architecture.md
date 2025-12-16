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

## Module 2: Bin-based Sparse Attention

### 目标

将 Key 分到 128 个 bin，Query 只与同 bin 的 Key 做 attention。

### 算法流程

**Round 开头：Key Binning**
```python
def key_binning(kv_cache, neural_net_key):
    logits = neural_net_key(kv_cache)  # (num_keys, 128)
    bin_assignments = logits.argmax(dim=1)  # (num_keys,)
    return bin_assignments
```

**每次解码：Query Routing + Sparse Attention**
```python
def sparse_attention(Q, history_keys, recent_keys, bin_assignments, neural_net_query):
    # Query 分 bin
    bin_q = neural_net_query(Q).argmax()

    # 历史 Key: Sparse (同 bin)
    sparse_keys = history_keys[bin_assignments == bin_q]

    # 当前 round 新 Key: Full Attention
    all_keys = concat([sparse_keys, recent_keys])
    return attention(Q, all_keys)
```

### 神经网络结构

```
K/Q (post-RoPE) → Kernel Encoding (128-dim) → Softmax → bin 概率
```

- **无 MLP，无 Position Scaling**（与 Module 1 不同）
- Key 网络和 Query 网络参数独立

### Round 内新 Key 处理

- **历史 Key**（< round_start）：Sparse Attention
- **当前 round 新 Key**（>= round_start）：Full Attention

### 空 Bin 处理

采用 **Masking 策略**：Query routing 时将空 bin 的 logits 设为 `-inf`

```python
def get_empty_bin_mask(bin_assignments, num_bins=128):
    if len(bin_assignments) == 0:
        return torch.ones(num_bins, dtype=torch.bool), True
    bin_counts = torch.bincount(bin_assignments, minlength=num_bins)
    return bin_counts == 0, False
```

### 首个 Round 处理

当 `round_start == 0` 时，没有历史 Key，直接走 Full Attention。

---

## 待定事项

### Module 1
- [ ] MLP hidden dimension
- [ ] threshold 设定方式（固定/自适应/可学习）
- [ ] Position Scaling 锚点配置

### Module 2
- [ ] Bin 数量（固定 128 或可调）
- [ ] Multi-bin Query（暂不实现，待实验结果决定）
