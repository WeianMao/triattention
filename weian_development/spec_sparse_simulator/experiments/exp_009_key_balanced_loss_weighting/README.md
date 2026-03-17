# Exp 009: Key-Balanced Loss Weighting

## 目标 (Objective)

验证 **Key-Balanced Loss Weighting** 方法是否能改善 Module 2 训练中的长尾效应问题。

### 问题背景

在训练过程中，Query-Key Pair 在 Key 维度上存在严重的不平衡分布：
- 某些 Key 对应大量 Pair（高频 Key）
- 某些 Key 只对应少量 Pair（低频 Key）

原始 Loss 按 Pair 平均，导致高频 Key 在梯度中被过度表示，形成长尾效应。

### 解决方案

Key-Balanced Loss Weighting：
```
raw_weight_i = 1 / count_{k_i}
weight_i = raw_weight_i / sum(raw_weight_j)
loss = sum(weight_i * loss_i)
```

每个 unique Key 在训练中获得大致相等的影响力。

## 运行方式 (Usage)

```bash
# 训练 (25 epochs)
python run.py --mode train

# 评估 (在 test trace 上)
python run.py --mode evaluate
```

## 结果摘要 (Results Summary)

### 训练配置
- **训练数据**: qk.pt -> qid0003_trace34 (layer=33, head=0), seq_len=10,938
- **测试数据**: qk_test.pt -> qid0008_trace46 (layer=33, head=0), seq_len=17,570
- **模型参数**: 147,712 (Key Network: 73,856 + Query Network: 73,856)
- **训练**: 25 epochs, lr=0.001, round_window=128
- **设备**: CUDA (GPU)
- **训练时长**: ~3 分钟

### 训练曲线对比

| Epoch | exp_006 (Baseline) | exp_009 (Key-Balanced) |
|-------|-------------------|------------------------|
| 1 | 1.684 | 2.879 |
| 10 | 0.053 | 0.322 |
| 20 | 0.023 | 0.125 |
| 25 | 0.017 | 0.094 |

**注**: Key-Balanced Loss 数值较大是因为权重归一化后的计算方式不同，不能直接比较绝对值。

### 评估结果 (Cross-Trace TopK Hit Rate)

| K值 | exp_006 (Baseline) | exp_009 (Key-Balanced) | 差异 |
|-----|-------------------|------------------------|------|
| 50 | 98.49% | **98.48%** | -0.01% |
| 500 | 98.55% | **98.63%** | **+0.08%** |
| 1000 | 98.72% | **98.82%** | **+0.10%** |

### Bin 路由命中率对比

| K值 | exp_006 Bin Hit | exp_009 Bin Hit | 差异 |
|-----|-----------------|-----------------|------|
| 50 | 95.89% | 95.89% | 0.00% |
| 500 | 95.96% | **96.03%** | **+0.07%** |
| 1000 | 96.13% | **96.23%** | **+0.10%** |

### 详细命中统计

| 指标 | exp_006 | exp_009 |
|------|---------|---------|
| 总查询数 | 16,442 | 16,442 |
| Recent 命中 | 426 (2.59%) | 426 (2.59%) |
| K=50 Bin 命中 | 15,767 | 15,766 (-1) |
| K=500 Bin 命中 | 15,778 | **15,790 (+12)** |
| K=1000 Bin 命中 | 15,805 | **15,822 (+17)** |

## 结论 (Conclusion)

### 主要发现

1. **整体命中率略有提升**：K=500 和 K=1000 时，Key-Balanced Loss 带来 0.08-0.10% 的提升
2. **K=50 基本持平**：在最严格的 TopK 设置下，两种方法表现相当
3. **Bin 路由改善**：K=500/1000 时 Bin 命中数分别增加 12/17 个

### 分析

1. **效果验证**：Key-Balanced Loss 确实对中等 K 值（500, 1000）有轻微正向效果
2. **改善幅度有限**：提升约 0.1%，说明原始方法已经较为均衡
3. **可能原因**：
   - 原始数据中 Key 分布可能本身不是极端长尾
   - 模型容量足够学习所有 Key 的模式
   - 25 epochs 训练可能未充分体现差异

### 建议

1. **继续使用**：Key-Balanced Loss 不会降低性能，且在某些情况下有提升
2. **进一步验证**：可在更长 trace 或更极端长尾分布数据上验证
3. **统计分析**：可添加 per-key hit rate 分析，验证低频 Key 是否确实改善

## 与其他实验对比

| 实验 | 训练->测试 | Loss 函数 | K=50 命中率 | K=1000 命中率 |
|------|----------|----------|-------------|---------------|
| exp_006 | B->A | Original Mean | 98.49% | 98.72% |
| **exp_009** | B->A | **Key-Balanced** | 98.48% | **98.82%** |

## 相关文档 (Related Docs)

- [exp_009_design.md](../exp_009_key_balanced_loss_weighting_design.md) - 设计文档
- [exp_006 README](../exp_006_module2_reverse_cross_trace_validation/README.md) - Baseline 实验
