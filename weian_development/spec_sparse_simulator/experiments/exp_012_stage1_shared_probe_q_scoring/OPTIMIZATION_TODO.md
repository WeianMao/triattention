# 训练脚本优化待办事项

记录时间: 2024-12-21

---

## 问题 3：所有降低效率的 for 循环

### 3.1 `compute_batch_accuracy_fast` (lines 256-268) - 最严重

**当前代码**：
```python
for i in range(top_bin_indices.shape[0]):      # O(num_queries)
    for j in range(top_bin_indices.shape[1]):  # O(top_k_bins)
        bin_scores = key_probs[:, bin_idx]
        _, topk_keys = torch.topk(bin_scores, actual_k)  # 每次都调 topk！
        if argmax_key in topk_keys:
            bin_hits += 1
            break
```

**问题**：
- 双重 Python 循环
- 每个 (query, bin) 组合都调用一次 `torch.topk`
- `argmax_key in topk_keys` 是 Python 级别的 membership check

**优化方案**：
1. 预先对所有 bins 一次性计算 topk：`topk_per_bin = torch.topk(key_probs, k, dim=0)`，得到 (k, num_bins)
2. 用 advanced indexing 批量查询每个 query 的 argmax_key 是否在对应 bin 的 topk 中
3. 完全消除 Python 循环

---

### 3.2 Round 收集循环 (lines 317-326)

**当前代码**：
```python
for round_start in range(round_window, seq_len, round_window):
    labels = extract_round_labels(attention, round_start, round_end, ...)
    rounds.append(...)
```

**问题**：
- 每个 round 单独调用 `extract_round_labels`
- 虽然 `extract_round_labels` 内部已向量化，但外层还是 Python 循环

**优化方案**：
- 可以一次性计算所有 rounds 的 labels（如果不做 lazy caching 的话）
- 或者：batch 多个 round 的 label 提取操作

**注意**：这个可能和 Label 缓存方案冲突，需要权衡

---

### 3.3 `compute_init_regularization_loss` (lines 169-173)

**当前代码**：
```python
for name, param in model.named_parameters():
    reg_loss = reg_loss + ((param - init_params[name]) ** 2).sum()
```

**问题**：Python 循环遍历参数

**优化方案**：
```python
from torch.nn.utils import parameters_to_vector

current_vec = parameters_to_vector(model.parameters())
init_vec = parameters_to_vector(init_params.values())
reg_loss = ((current_vec - init_vec) ** 2).sum()
```

---

### 3.4 `compute_gradient_norm` / `compute_parameter_norm` (lines 176-190)

**当前代码**：
```python
for param in model.parameters():
    total_norm += param.grad.data.norm(2).item() ** 2
```

**问题**：Python 循环 + 多次 `.item()` 调用导致 CPU-GPU 同步

**优化方案**：
```python
# 梯度 norm
grads = [p.grad for p in model.parameters() if p.grad is not None]
grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

# 或者用 torch.nn.utils.clip_grad_norm_ 的内部实现
```

---

### 3.5 Inner round processing loop (lines 347-398)

**当前代码**：
```python
for round_info in batch_rounds:
    historical_keys = K[:round_start]
    key_probs = model.forward_keys(historical_keys, reference_angles)
    queries = Q[query_indices]
    query_bin_probs = model.forward_queries(queries, reference_angles, empty_bin_mask)
    ...
```

**问题**：每个 round 串行处理

**优化方案**：见问题 2 的 batch 并行化方案（使用 padding + mask）

---

## 问题 4：Label 缓存

### 需求

- 第一个 epoch 计算 labels 时缓存到 `trace_data`
- 后续 epoch 检测到缓存存在就直接使用
- 不要在所有 epoch 开始前单独用 for 循环预计算

### 方案

```python
def train_on_trace_batched(model, trace_data, ...):
    ...
    # 检查是否有缓存
    if 'cached_rounds' not in trace_data:
        # 第一个 epoch：计算并缓存
        rounds = []
        for round_start in range(round_window, seq_len, round_window):
            labels = extract_round_labels(attention, round_start, round_end, ...)
            if labels is not None:
                rounds.append({
                    'round_start': round_start,
                    'round_end': round_end,
                    'labels': labels
                })
        trace_data['cached_rounds'] = rounds  # 缓存到 trace_data
    else:
        # 后续 epoch：直接用缓存
        rounds = trace_data['cached_rounds']

    # 继续训练...
```

### 显存优化

Labels 缓存内容：
- `query_indices`: (num_queries,) long tensor
- `argmax_keys`: (num_queries,) long tensor
- `argmax_in_recent`: (num_queries,) bool tensor

这些都是小 tensor（每个 round 约 128 个 query），不需要保存整个 attention 矩阵。

每个 trace 大约有 `seq_len / round_window` 个 rounds，每个 round 约 128 queries：
- 64000 / 128 = 500 rounds
- 每个 round: 128 * (8 + 8 + 1) bytes ≈ 2 KB
- 每个 trace: 500 * 2 KB = 1 MB

**总共约 7 MB**（7 个 traces），完全可以接受。

---

## 优先级

1. **问题 1 (显存优化)** - 必须先做，否则无法增大 batch size
2. **问题 2 (batch 并行化)** - 核心性能提升
3. **问题 4 (Label 缓存)** - 中等优先级
4. **问题 3.1 (accuracy 计算)** - 只在 eval 时用，优先级较低
5. **问题 3.3, 3.4 (norm 计算)** - 小优化

---

## 状态

- [ ] 问题 1：显存优化（方案 B - Q/K 留 GPU，attention 不缓存）
- [x] 问题 2：batch 内并行化（V5 已实现 forward_keys_batched/forward_queries_batched）
- [x] 问题 3.1：accuracy 计算向量化 ✅ **~10x speedup** (168s → 17s)
- [x] 问题 3.2：round 收集循环优化（extract_round_labels 内部已向量化，Python 循环开销可忽略）
- [x] 问题 3.3：regularization loss 向量化 ✅
- [x] 问题 3.4：norm 计算向量化 ✅ (避免多次 .item() 调用)
- [x] 问题 3.5：inner round 处理循环 ✅ (V5 batched forward)
- [ ] 问题 4：Label 缓存（第一个 epoch 计算时缓存，后续复用）

---

## 问题 2 详细设计（2024-12-21 更新）

### 整体架构

```
外层循环: for trace in traces:
    1. 整条 trace 的 Q, K 上 GPU
    2. 内层循环: for batch in rounds_batches:
        a. 选取 batch 数据（用索引，不复制）
        b. 并行 forward（10 个 round，10 个不同 angle，但并行计算）
        c. 计算 loss（独立计算）
```

### Step 2b: 并行 Forward 详细设计

假设 batch_size=10，rounds = [round_10, round_11, ..., round_19]

```python
# 1. 计算所有 round 的 rotated_probes (并行)
round_starts = [1280, 1408, 1536, ...]  # 10 个
ref_positions = [round_start + round_window // 2 for round_start in round_starts]
# 批量计算 rotated_probes: (batch_size, num_bins, head_dim)
rotated_probes_batch = compute_rotated_probes_batch(probes, ref_positions)

# 2. 加载 K 数据 (共享，不复制)
max_keys = max(round_starts)  # batch 内最大的 round_start
K_shared = K[:max_keys]  # (max_keys, head_dim) - 只是索引，不复制

# 3. 批量计算 key_logits
# K_shared: (max_keys, head_dim)
# rotated_probes_batch: (batch_size, num_bins, head_dim)
# 用 einsum 并行计算:
key_logits_batch = einsum('kd,rbd->rkb', K_shared, rotated_probes_batch)
# + magnitude_term + bias (需要适配)
# 结果: (batch_size, max_keys, num_bins)

# 4. 创建 mask (每个 round 不同)
# round_10: 只有 [:1280] 有效
# round_11: 只有 [:1408] 有效
# ...
mask = create_length_mask(round_starts, max_keys)  # (batch_size, max_keys)
# mask[i, j] = True if j < round_starts[i] else False

# 5. 应用 mask (无效位置填 -inf)
key_logits_masked = key_logits_batch.masked_fill(~mask.unsqueeze(-1), float('-inf'))

# 6. Softmax (对每个 round 独立计算)
key_probs_batch = F.softmax(key_logits_masked, dim=1)  # softmax over keys
# 结果: (batch_size, max_keys, num_bins)
# 无效位置的 prob 会是 ~0 (因为 softmax(-inf) → 0)
```

### Query 侧处理

```python
# 每个 round 的 queries 是不同的 Q slice
# round_10: Q[1280:1408]
# round_11: Q[1408:1536]
# ...

# 方案 A: 拼接所有 queries，记录每个 round 的起止位置
all_queries = torch.cat([Q[round_start:round_end] for ...])  # (total_queries, head_dim)
query_round_mapping = [...]  # 记录每个 query 属于哪个 round

# 用对应的 rotated_probes 计算 query_logits
# 需要根据 query_round_mapping 选择正确的 rotated_probes

# 方案 B: 如果每个 round 的 query 数量相同 (都是 round_window)
# 可以直接 batch: (batch_size, round_window, head_dim)
queries_batch = stack([Q[round_start:round_end] for ...])
# 然后并行计算
```

### 为什么 mask 用 -inf

```
softmax([1.0, 2.0, -inf, -inf]) = [0.27, 0.73, 0.0, 0.0]
```

-inf 经过 softmax 后变成 0，不会影响概率分布。这样就等价于只对有效位置做 softmax。

### Model 需要的修改

1. `forward_keys` 需要支持 batched rotated_probes:
   - 原: `(num_keys, head_dim)` → `(num_keys, num_bins)`
   - 新: `(max_keys, head_dim)` + `(batch_size, num_bins, head_dim)` → `(batch_size, max_keys, num_bins)`

2. `forward_queries` 类似

3. 或者：保持 model 不变，在 training loop 里用 einsum 手动实现 batched forward
