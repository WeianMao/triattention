# Exp 006a: Top-2 Bin Inference Variant

## 目标 (Objective)

在 exp_006 的基础上，测试推理时使用 **Top-2 bin** 而非 Top-1 (argmax) 是否能提升性能。

**关键修改**：不重新训练，直接使用 exp_006 的权重，仅修改推理代码。

## 与 exp_006 的区别

| 配置 | exp_006 (原始) | exp_006a (本实验) |
|------|----------------|-------------------|
| 训练数据 | qid0003_trace34 | 同 (不重新训练) |
| 测试数据 | qid0008_trace46 | 同 |
| 模型权重 | 训练得到 | **复用 exp_006 权重** |
| Query Bin 选择 | Top-1 (argmax) | **Top-2** |
| 每 Query Keys 数 | K | **2K** (从两个 bin 各取 K) |

## 代码修改

`evaluate.py` 中的关键修改：

```python
# 原始 (Top-1):
selected_bin = query_bin_probs.argmax(dim=-1).item()
bin_scores = key_probs[:, selected_bin]
_, topk_indices = torch.topk(bin_scores, actual_k)

# 修改后 (Top-2):
num_bins_to_select = 2
_, selected_bins = torch.topk(query_bin_probs.squeeze(0), num_bins_to_select)
all_topk_indices = set()
for bin_idx in selected_bins:
    bin_scores = key_probs[:, bin_idx]
    _, topk_indices = torch.topk(bin_scores, actual_k)
    all_topk_indices.update(topk_indices.tolist())
```

## 评估结果

### Hit Rate 对比

| K值 | Top-1 (exp_006) | Top-2 (本实验) | 提升 |
|-----|-----------------|----------------|------|
| **50** | 98.49% | **98.61%** | **+0.12%** |
| **500** | 98.63% | **98.98%** | **+0.35%** |
| **1000** | 98.81% | **99.16%** | **+0.35%** |

### Bin Hit Rate 对比

| K值 | Top-1 Bin Hit | Top-2 Bin Hit | 提升 |
|-----|---------------|---------------|------|
| 50 | 95.90% | 96.02% | +0.12% |
| 500 | 96.03% | 96.39% | +0.36% |
| 1000 | 96.22% | 96.57% | +0.35% |

### Miss Case 减少

| K值 | Top-1 Misses | Top-2 Misses | 减少数量 | 减少比例 |
|-----|--------------|--------------|----------|----------|
| 50 | 248 | 228 | 20 | 8% |
| 500 | 226 | 167 | 59 | 26% |
| 1000 | 195 | 138 | 57 | 29% |

## 结论

### 提升幅度有限

Top-2 bin 选择仅带来 **+0.12% ~ +0.35%** 的提升，原因：

1. **Query Network 极度 collapse**: exp_006 分析显示 99.96% 的 query 路由到 Bin 37
2. **第二个 bin 贡献有限**: Top-2 选的第二个 bin 几乎没有被训练使用
3. **Key Network 已学会"全局排序"**: Bin 37 的列向量已经是有效的全局重要性排序

### 关键发现

1. **简单的推理时 multi-bin 策略效果有限**: 因为训练时没有学习多 bin 路由
2. **需要在训练阶段解决 collapse**: 仅靠推理时增加 bin 数量无法充分利用多 bin 架构
3. **K 越大提升越明显**: K=1000 时减少 29% 的 miss，说明第二个 bin 确实包含一些有用的 key

### 后续建议

1. **训练时添加 entropy loss**: 鼓励 Query Network 使用多个 bin
2. **训练时使用 Top-2 loss**: 让模型学习多 bin 路由
3. **Load balancing loss**: 类似 Switch Transformer 的负载均衡

## 运行方式

```bash
# 直接评估 (使用复制过来的 exp_006 权重)
python evaluate.py
```

## 相关文档

- [exp_006 README](../exp_006_module2_reverse_cross_trace_validation/README.md) - 原始 B→A 实验
- [exp_006 ANALYSIS_REPORT](../exp_006_module2_reverse_cross_trace_validation/ANALYSIS_REPORT.md) - Bin collapse 详细分析
