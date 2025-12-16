# exp_001_module1_oracle

## 目标

验证 Module 1 Key Pruning 的 Oracle Upper Bound：使用真实的 attention pattern 作为 oracle，评估理论上的最大压缩率。

## 方法

1. 加载 qk.pt trace 数据
2. 计算完整的 attention matrix
3. 提取 ground-truth labels：label=0 表示 Key 会被未来 Query attend（保留），label=1 表示不会被 attend（丢弃）
4. 使用 oracle labels 作为预测（完美预测）
5. 计算评估指标：Argmax Hit Rate, Retention Rate, Keys per Query, Computation Reduction

## 运行方式

```bash
cd weian_development/spec_sparse_simulator/experiments/exp_001_module1_oracle

# 运行 oracle 实验
python run.py --trace <path_to_trace_dir>

# 示例
python run.py --trace /path/to/qid0000_trace00
```

## 结果摘要

### 实验配置
- **Trace**: qid0003_trace34 (seq_len=10938)
- **Round Window**: 128 tokens
- **Heads**: 10 个采样 heads (hybrid_sample_heads_lowret_top10.json)

### 主要指标

| 指标 | 值 |
|------|-----|
| **Oracle Argmax Hit Rate** | 100.00% (符合理论预期) |
| **平均 Retention Rate** | 6.01% |
| **最低 Retention Rate** | 0.07% (L32H27) |
| **最高 Retention Rate** | 25.25% (L17H25) |
| **平均 Computation Reduction** | 93.99% |
| **最大 Computation Reduction** | 99.93% |

### 每个 Head 的详细结果

| Head | Mean Retention Rate | Mean Argmax Hit Rate |
|------|---------------------|---------------------|
| L33H0 | 0.15% | 100% |
| L34H6 | 0.22% | 100% |
| L9H19 | 15.55% | 100% |
| L24H0 | 1.70% | 100% |
| L31H30 | 4.02% | 100% |
| L24H14 | 10.38% | 100% |
| L3H7 | 1.66% | 100% |
| L24H11 | 1.11% | 100% |
| L32H27 | 0.07% | 100% |
| L17H25 | 25.25% | 100% |

### 可视化

- `output/figures/retention_rate_per_head.png` - 每个 head 的 retention rate
- `output/figures/argmax_hit_rate_per_head.png` - 每个 head 的 argmax hit rate
- `output/figures/oracle_summary.png` - 汇总对比图

## 结论

1. **Oracle 验证成功**: 100% Argmax Hit Rate 证明了基于 argmax 的标签定义是正确的
2. **高压缩潜力**: 理论上可以平均减少 94% 的 Key 计算，某些 head 可达 99.93%
3. **Head 差异显著**: 不同 head 的压缩率差异很大（0.07% ~ 25.25%），说明不同 head 的 attention pattern 特性不同
4. **浅层 vs 深层**: 深层 heads (L32, L33, L34) 普遍有更低的 retention rate，可能因为它们更专注于少量关键位置

## 相关文档

- [01_module1_key_pruning.md](../../docs/01_module1_key_pruning.md) - Module 1 设计文档
- [04_training_and_labels.md](../../docs/04_training_and_labels.md) - 标签定义和训练策略
- [05_experiment_conventions.md](../../docs/05_experiment_conventions.md) - 实验规范
