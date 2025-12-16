# 评估指标设计

## 核心指标

### Argmax Hit Rate（最关键）

**定义**：Query 仍能 attend 到原 argmax Key 的比例

**命中判定规则**：
- argmax 在历史 Key → 检查该 Key 是否被 retain（Module 1）或在同 bin（Module 2）
- argmax 在当前 round 新 Key → **直接算命中**（Full Attention）

| 目标 | Baseline |
|------|----------|
| >99% | 100% (Full Attention) |

### Keys per Query

**定义**：每个 Query 参与 attention 的平均 Key 数量

| Module 1 | Module 2 |
|----------|----------|
| 保留的 Key 数量 | 平均 bin 大小 |

### Computation Reduction

```
Computation Reduction = 1 - (Keys per Query / Total Keys)
```

---

## Module 1 指标

### 指标计算

```python
def compute_module1_metrics(drop_probs, labels, argmax_in_history, threshold=0.5):
    retain_mask = drop_probs < threshold

    # Retention Rate
    retention_rate = retain_mask.mean()

    # Argmax Hit Rate
    hits_recent = (~argmax_in_history).sum()  # 新 Key 直接命中
    hits_history = (retain_mask & (labels == 0)).sum()
    argmax_hit_rate = (hits_recent + hits_history) / len(argmax_in_history)

    # False Negative Rate
    should_retain = labels == 0
    fn_rate = (~retain_mask & should_retain).sum() / should_retain.sum()

    return {
        'argmax_hit_rate': argmax_hit_rate,
        'retention_rate': retention_rate,
        'false_negative_rate': fn_rate,
        'keys_per_query': retain_mask.sum(),
    }
```

### 辅助指标

| 指标 | 定义 | 目标 |
|------|------|------|
| Retention Rate | 保留 Key / 总 Key | 越低越好 |
| False Negative Rate | 错误丢弃的"会被 attend"的 Key 比例 | 越低越好 |

---

## Module 2 指标

### 指标计算

```python
def compute_module2_metrics(query_bins, history_key_bins, query_to_argmax, argmax_in_history):
    num_queries = len(query_bins)
    num_history_keys = len(history_key_bins)

    if num_history_keys == 0:
        return {'argmax_hit_rate': 1.0, ...}  # 首个 round

    # Argmax Hit Rate
    hits_recent = (~argmax_in_history).sum()
    history_query_bins = query_bins[argmax_in_history]
    argmax_key_bins = history_key_bins[query_to_argmax[argmax_in_history]]
    hits_history = (history_query_bins == argmax_key_bins).sum()
    argmax_hit_rate = (hits_recent + hits_history) / num_queries

    # Keys per Query (Sparse 部分)
    bin_sizes = [(history_key_bins == b).sum() for b in query_bins]
    keys_per_query = sum(bin_sizes) / num_queries

    # Bin Balance
    _, counts = torch.unique(history_key_bins, return_counts=True)
    bin_balance_var = counts.float().var()

    # Empty Bin Rate
    empty_bin_rate = 1 - len(torch.unique(history_key_bins)) / 128

    return {
        'argmax_hit_rate': argmax_hit_rate,
        'keys_per_query_sparse': keys_per_query,
        'computation_reduction': 1 - keys_per_query / num_history_keys,
        'bin_balance_var': bin_balance_var,
        'empty_bin_rate': empty_bin_rate,
    }
```

### 实际计算量估算

```
Keys per Query (实际) = Sparse 部分 + Full 部分
                     = avg_bin_size + (query_pos - round_start)
```

### 辅助指标

| 指标 | 定义 | 目标 |
|------|------|------|
| Bin Balance | bin 大小方差 | 越低越好 |
| Empty Bin Rate | 空 bin 比例 | 越低越好 |
| Bin Utilization | 使用的 bin / 总 bin | 越高越好 |

---

## Baseline 对比

| 方法 | Argmax Hit Rate | Keys per Query | Comp. Reduction |
|------|-----------------|----------------|-----------------|
| Full Attention | 100% | N | 0% |
| Random Binning | ~0.78% (1/128) | N/128 | ~99% |
| **目标** | >99% | N/128 | ~99% |

---

## Multi-Bin Key Assignment 指标

### TopK Hit Rate

使用 TopK 选择策略时：
- 检查 argmax Key 是否在该 bin 的 TopK 中

```python
def compute_topk_hit_rate(query_bins, key_scores, query_to_argmax, K):
    hits = 0
    for q_idx, q_bin in enumerate(query_bins):
        scores = key_scores[:, q_bin]
        topk_indices = scores.topk(K).indices
        if query_to_argmax[q_idx] in topk_indices:
            hits += 1
    return hits / len(query_bins)
```

### 实验结果参考

| 方法 | K=50 | K=500 | K=1000 |
|------|------|-------|--------|
| Attraction Loss | 100% | 100% | 100% |
| 双向交叉熵 | 67.78% | 100% | 100% |
