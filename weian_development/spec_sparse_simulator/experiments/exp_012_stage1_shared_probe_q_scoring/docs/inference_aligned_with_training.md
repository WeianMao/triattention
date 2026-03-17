# 推理对齐训练方案

## 背景

训练时的 loss 计算方式和推理时的 key 选择方式不一致。与其修改训练，不如让推理对齐训练。

## 训练时的概率计算

训练时，对于一个 query 和它的 ground truth key：

```
match_prob = Σ_b [ query_probs[b] × key_probs[gt_key, b] ]
```

其中：
- `query_probs[b]`: query 对 bin b 的概率 (softmax over 128 bins)
- `key_probs[k, b]`: key k 在 bin b 中的概率 (softmax over all keys in bin b)
- 求和遍历所有 128 个 bin

## 新的推理方案

### Step 1: 获取候选 key

对于每个 bin b，取 top-50 个 key（按 `key_probs[:, b]` 排序）。

这样得到一个 50 × 128 的矩阵，记录每个位置的 key ID：
```
key_ids[i, b] = 第 b 个 bin 中排名第 i 的 key 的 ID
```

注意：同一个 key ID 可能出现在多个 bin 中。

### Step 2: 计算联合概率矩阵

对于每个位置 (i, b)：
```
joint_prob[i, b] = query_probs[b] × key_probs[key_ids[i, b], b]
```

这给出一个 50 × 128 的联合概率矩阵。

### Step 3: 聚合每个 unique key 的概率

对于每个 unique key ID k，计算其总概率：
```
P[k] = Σ_{(i,b) where key_ids[i,b] == k} joint_prob[i, b]
```

这与训练时的 `match_prob` 计算方式完全一致。

### Step 4: 选择最终的 key

按 P[k] 排序，选择 top-K 个 key 作为最终结果。

## 数值稳定性 (LogSumExp)

为避免概率相乘导致的数值下溢，在 log 空间计算：

### Step 2 (log 空间):
```
log_joint[i, b] = log_query_probs[b] + log_key_probs[key_ids[i, b], b]
```

### Step 3 (log 空间聚合):

对于每个 unique key k：
```
log_P[k] = logsumexp({ log_joint[i, b] : key_ids[i, b] == k })
```

其中 `logsumexp(x) = log(Σ exp(x_i))`

## 伪代码

```python
def inference_aligned_with_training(query_log_probs, key_log_probs, key_ids_per_bin, top_k=50):
    """
    Args:
        query_log_probs: [128] - log softmax of query over bins
        key_log_probs: [num_keys, 128] - log softmax of keys within each bin
        key_ids_per_bin: [50, 128] - top-50 key IDs for each bin
        top_k: number of keys to return

    Returns:
        top_key_ids: [top_k] - final selected key IDs
    """
    # Step 2: 计算 log 联合概率
    # log_joint[i, b] = query_log_probs[b] + key_log_probs[key_ids_per_bin[i, b], b]
    log_joint = torch.zeros(50, 128)
    for i in range(50):
        for b in range(128):
            kid = key_ids_per_bin[i, b]
            log_joint[i, b] = query_log_probs[b] + key_log_probs[kid, b]

    # Step 3: 聚合每个 unique key 的概率
    unique_keys = torch.unique(key_ids_per_bin)
    log_P = {}
    for k in unique_keys:
        mask = (key_ids_per_bin == k)  # [50, 128] bool
        log_probs_for_k = log_joint[mask]  # 所有 k 出现位置的 log prob
        log_P[k] = torch.logsumexp(log_probs_for_k, dim=0)  # scalar

    # Step 4: 选择 top-K
    sorted_keys = sorted(log_P.keys(), key=lambda k: log_P[k], reverse=True)
    return sorted_keys[:top_k]
```

## 与原有推理的对比

| 方面 | 原有推理 | 新推理 |
|------|---------|--------|
| Bin 选择 | 硬选择 top-1 或 top-8 | 软选择，使用所有 128 个 bin 的概率 |
| Key 聚合 | 无聚合，直接取每个 bin 的 top-K | 跨 bin 聚合，同一 key 的概率累加 |
| 与训练对齐 | 不对齐 | 完全对齐 |

## 命令行参数

```
--inference-mode [original|aligned]
```

- `original`: 原有推理方式（默认）
- `aligned`: 与训练对齐的推理方式
