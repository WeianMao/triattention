# Exp 006 深度分析报告

**分析日期**: 2025-12-17

本报告对 exp_006 (B→A 跨 trace 泛化实验) 的错误 case 和 bin 利用效率进行深度分析。

## 1. 实验背景

- **训练数据**: qid0003_trace34 (Trace B), seq_len=10,938
- **测试数据**: qid0008_trace46 (Trace A), seq_len=17,570
- **模型**: Module 2 Multi-Bin Sparse Attention (128 bins)
- **评估指标**: TopK Hit Rate (K=50)

**原始评估结果**:
| K值 | 命中率 | Bin路由命中 | Recent自动命中 | 未命中 |
|-----|--------|-------------|----------------|--------|
| 50 | 98.49% | 15,768 (95.90%) | 426 (2.59%) | 248 |

## 2. Miss Case 分析

### 2.1 错误类型分类

对 248 个错误 case 进行分类：

| 类型 | 数量 | 占比 | 定义 |
|------|------|------|------|
| **Type A (Key Network 问题)** | 103 | 41.53% | argmax key 在**所有** bin 中的 best rank >= K |
| **Type B (Query Network 问题)** | 145 | 58.47% | argmax key 在某个 bin 中 rank < K，但 Query 选错了 bin |

### 2.2 结论

- **58.47%** 的错误是 **Query Network 路由错误**：目标 key 实际上存在于某个 bin 的 TopK 中，但 Query 网络选择了错误的 bin
- **41.53%** 的错误是 **Key Network 排序问题**：目标 key 在所有 bin 中的排名都超过 50

### 2.3 Best Rank 分布

对于所有 miss case，统计 argmax key 在所有 bin 中的最佳排名：

| Best Rank 阈值 | 数量 | 占比 |
|----------------|------|------|
| < 50 | 145 | 58.47% |
| < 100 | 196 | 79.03% |
| < 200 | 219 | 88.31% |
| < 500 | 240 | 96.77% |
| < 1000 | 246 | 99.19% |

**解读**: 如果 K 提升到 100，可以额外捕获约 20% 的 miss case。

### 2.4 Rank 统计

| 指标 | Best Rank (跨所有 bin) | Selected Bin Rank |
|------|------------------------|-------------------|
| Min | 0 | 69 |
| Max | 1,102 | 12,374 |
| Mean | 92.92 | 3,850.10 |
| Median | 37.00 | 2,406.50 |

## 3. Bin 利用效率分析

### 3.1 关键发现：Query Network 严重 Bin Collapse

| 指标 | Key Network | Query Network |
|------|-------------|---------------|
| **使用的 bin 数** | 125/128 (97.7%) | **3/128 (2.3%)** |
| **有效 bin 数** (exp(entropy)) | 106.66 | **1.00** |
| **Gini 系数** | 0.32 (相对均匀) | **0.99** (极度不均) |
| **Top 5 bin 集中度** | 9.2% | **100%** |
| **Top 10 bin 集中度** | 16.6% | 100% |
| **Top 20 bin 集中度** | 29.5% | 100% |

### 3.2 Query Routing 分布

```
Bin 37:  16,436 queries (99.96%)
Bin 117:      3 queries (0.02%)
Bin 59:       3 queries (0.02%)
其他 125 个 bin: 0 queries
```

### 3.3 关键发现

**Bin 37 只有 1 个 key 被分配到它（argmax assignment），但 99.96% 的 query 都路由到这个 bin！**

这意味着模型学到的不是真正的 sparse routing，而是一种 **"全局重要 key"** 策略：

1. Key Network 把所有 key 的"重要性分数"编码到 bin 37 的列向量中
2. Query Network 几乎总是选择 bin 37
3. TopK=50 从 bin 37 的 softmax 列中选出 top 50 个"全局重要 key"

### 3.4 为什么 Hit Rate 还是 98.49%？

尽管 Query Network 完全 collapse 到单个 bin，模型仍达到高命中率：

1. **Recent keys auto-hit (2.59%)**: 当前 round 的 key 自动命中
2. **Key Network 的 bin 37 列学会了"全局排序"**: `key_probs[:, 37]` 实际上是对所有 key 的一个全局重要性排序
3. 对于大多数 query，它们需要 attend 到的 argmax key 恰好在这个"全局 TopK"中

### 3.5 Key Network 分布 (Top 10 Bins)

| Rank | Bin ID | Key Count | 占比 |
|------|--------|-----------|------|
| 1 | 44 | 361 | 2.06% |
| 2 | 20 | 350 | 2.00% |
| 3 | 101 | 310 | 1.77% |
| 4 | 96 | 310 | 1.77% |
| 5 | 125 | 283 | 1.61% |
| 6 | 124 | 278 | 1.59% |
| 7 | 67 | 271 | 1.55% |
| 8 | 49 | 252 | 1.44% |
| 9 | 3 | 251 | 1.43% |
| 10 | 89 | 244 | 1.39% |

Key Network 的分布相对健康，没有明显的 collapse。

## 4. Sample Miss Cases

| Query Idx | Argmax Key | Selected Bin | Best Bin | Rank in Selected | Best Rank | Type |
|-----------|------------|--------------|----------|------------------|-----------|------|
| 2444 | 2417 | 37 | 12 | 1,278 | 27 | B |
| 2515 | 335 | 37 | 94 | 1,364 | 3 | B |
| 2576 | 2515 | 37 | 23 | 1,714 | 7 | B |
| 2583 | 2556 | 37 | 99 | 1,620 | 28 | B |
| 2632 | 2516 | 37 | 23 | 1,212 | 3 | B |
| 2824 | 2791 | 37 | 69 | 1,399 | 68 | A |
| 2868 | 2516 | 37 | 12 | 1,313 | 2 | B |

**观察**: 几乎所有 miss case 的 selected_bin 都是 **37**，进一步证实了 Query Network collapse 的问题。

## 5. 与 MOE Bin Collapse 的对比

| 特性 | MOE Bin Collapse | 本实验 Query Network |
|------|------------------|---------------------|
| Collapse 程度 | 部分 bin 使用频率低 | **极端 collapse 到单个 bin** |
| Effective bins | 通常 30-50% | **< 1%** (只有 1.00 effective bins) |
| Gini 系数 | 0.3-0.6 | **0.99** |
| 原因 | Rich-get-richer dynamics | 可能是训练数据特性或 loss 设计问题 |

## 6. 建议的后续改进方向

### 6.1 Query Network 正则化

添加 entropy loss 或 load balancing loss 鼓励使用更多 bin：

```python
# Entropy regularization
query_bin_probs = model.forward_queries(query, reference_angles)
entropy = -(query_bin_probs * torch.log(query_bin_probs + eps)).sum(dim=-1).mean()
loss = attraction_loss - lambda_entropy * entropy
```

### 6.2 Load Balancing Loss (类似 Switch Transformer)

```python
# Load balancing loss
f_i = query_bin_probs.mean(dim=0)  # fraction of queries to each bin
P_i = query_bin_probs.sum(dim=0) / query_bin_probs.sum()  # prob mass
balance_loss = num_bins * (f_i * P_i).sum()
```

### 6.3 检查 Training Trace

分析 Trace B (qid0003_trace34) 是否本身就有某种特性导致 Query Network 的 collapse。

### 6.4 对比其他实验

检查 exp_004 (Overfit) 和 exp_005 (A→B) 是否也存在类似的 bin collapse 问题。

## 7. 分析脚本

- `analyze_miss_cases.py`: Miss case 诊断脚本
- `analyze_bin_utilization.py`: Bin 利用效率分析脚本

**输出文件**:
- `output/analysis/miss_case_analysis.json`
- `output/analysis/bin_utilization_analysis.json`

## 8. 总结

1. **主要问题**: Query Network 存在**极端的 bin collapse**，99.96% 的 query 路由到 Bin 37
2. **错误分布**: 58% Type B (Query 路由错误) + 42% Type A (Key 排序问题)
3. **模型行为**: 实际上学会了"全局 TopK"策略而非真正的 sparse routing
4. **Hit Rate 解释**: 高命中率来自于 Bin 37 的列向量作为全局重要性排序的有效性
5. **改进方向**: 需要添加 Query Network 的 entropy/load-balancing 正则化
