# 评估指标设计

## 核心指标

### Argmax Hit Rate（最关键）

**定义**：Query 仍能 attend 到原 argmax Key 的比例

**命中判定规则**：
- argmax 在历史 Key → 检查该 Key 是否被 retain（Module 1）或在 TopK 中（Module 2）
- argmax 在当前 round 新 Key → **直接算命中**（Full Attention）

| 目标 | Baseline |
|------|----------|
| >99% | 100% (Full Attention) |

### Keys per Query

**定义**：每个 Query 参与 attention 的平均 Key 数量（**含 Full Attention 部分**）

| Module 1 | Module 2 |
|----------|----------|
| 保留的 Key 数量 | TopK + num_recent_keys |

### Computation Reduction

```
Computation Reduction = 1 - (Keys per Query / Total Keys)
```

**注**：计算量需包含 Full Attention 部分（当前 round 新 Key）

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

### TopK Hit Rate（核心指标）

检查 argmax Key 是否在 Query 选择的 bin 的 TopK 中

```python
def compute_topk_hit_rate(query_bins, key_probs, query_to_argmax, argmax_in_history, K):
    """
    query_bins: (num_queries,) - Query 选择的 bin
    key_probs: (num_keys, num_bins) - softmax over keys for each bin
    query_to_argmax: (num_queries,) - 每个 query 的 argmax key 索引
    K: TopK 参数
    """
    num_queries = len(query_bins)

    # 新 Key 直接命中
    hits_recent = (~argmax_in_history).sum()

    # 历史 Key 检查是否在 TopK 中
    hits_history = 0
    for q_idx in argmax_in_history.nonzero():
        q_bin = query_bins[q_idx]
        scores = key_probs[:, q_bin]
        topk_indices = scores.topk(K).indices
        if query_to_argmax[q_idx] in topk_indices:
            hits_history += 1

    return (hits_recent + hits_history) / num_queries
```

### 实际计算量

```
Keys per Query (实际) = TopK + (query_pos - round_start)
                     = K + num_recent_keys
```

### 实验结果参考

| 方法 | K=50 | K=500 | K=1000 |
|------|------|-------|--------|
| **Attraction Loss** | 100% | 100% | 100% |
| 双向交叉熵 | 67.78% | 100% | 100% |

---

## Baseline 对比

| 方法 | Argmax Hit Rate | Keys per Query | Comp. Reduction |
|------|-----------------|----------------|-----------------|
| Full Attention | 100% | N | 0% |
| Random (K=50) | ~50/N | 50 | 高 |
| **目标** | >99% | K (50~1000) | 高 |
