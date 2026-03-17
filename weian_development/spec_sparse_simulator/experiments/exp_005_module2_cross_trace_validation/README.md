# Exp 005: Module 2 Cross-Trace Validation

## 目标 (Objective)

验证 Module 2 Multi-Bin Sparse Attention 的**泛化能力**：
- 训练在一个 trace 上 (qid0008_trace46)
- 测试在另一个 trace 上 (qid0003_trace34)
- 目的：评估模型是否能在未见过的 trace 上保持高 TopK Hit Rate

## 与 exp_004 的区别

| 配置 | exp_004 (Overfit) | exp_005 (Cross-trace) |
|------|-------------------|----------------------|
| 训练数据 | qid0008_trace46 | qid0008_trace46 |
| 测试数据 | qid0008_trace46 | **qid0003_trace34** |
| Epochs | 100 | **25** |
| 目的 | 验证过拟合学习能力 | **验证泛化能力** |

## 运行方式 (Usage)

```bash
# 训练 (25 epochs)
python run.py --mode train

# 评估 (在 test trace 上)
python run.py --mode evaluate
```

## 结果摘要 (Results Summary)

### 训练配置
- **训练数据**: qk.pt → qid0008_trace46 (layer=33, head=0), seq_len=17,570
- **测试数据**: qk_test.pt → qid0003_trace34 (layer=33, head=0), seq_len=10,938
- **模型参数**: 147,712 (Key Network: 73,856 + Query Network: 73,856)
- **训练**: 25 epochs, lr=0.001, round_window=128
- **设备**: CUDA (GPU 1)
- **训练时长**: ~13 分钟

### 训练曲线
| Epoch | Loss |
|-------|------|
| 1 | 1.241 |
| 10 | 0.105 |
| 20 | 0.058 |
| 25 | 0.048 |

### 评估结果 (Cross-Trace TopK Hit Rate)

| K值 | 命中率 | Bin路由命中 | Recent自动命中 | 未命中 |
|-----|--------|-------------|----------------|--------|
| 50 | **99.15%** | 9,601 (97.87%) | 126 (1.28%) | 83 |
| 500 | **99.47%** | 9,632 (98.19%) | 126 (1.28%) | 52 |
| 1000 | **99.51%** | 9,636 (98.23%) | 126 (1.28%) | 48 |

总查询数: 9,810

## 结果对比 (Comparison with exp_004)

### TopK Hit Rate 对比

| K值 | exp_004 (Overfit) | exp_005 (Cross-trace) | 泛化差距 |
|-----|-------------------|----------------------|----------|
| 50 | 99.77% | 99.15% | **-0.62%** |
| 500 | 99.78% | 99.47% | **-0.31%** |
| 1000 | 99.79% | 99.51% | **-0.28%** |

### Bin路由命中率对比

| 指标 | exp_004 (Overfit) | exp_005 (Cross-trace) | 泛化差距 |
|------|-------------------|----------------------|----------|
| Bin Hit Rate | 97.18-97.20% | 97.87-98.23% | **+0.7-1.0%** |

## 结论 (Conclusion)

### 泛化验证成功

1. **极小泛化差距**: 仅 0.3-0.6% 的命中率下降，模型泛化能力优秀
2. **Bin路由甚至提升**: Cross-trace 的 Bin 路由命中率反而更高 (+0.7-1.0%)，可能是因为 qid0003 trace 更容易路由
3. **25 epochs 足够**: 只用 25% 训练时间（25 vs 100 epochs）即达到相近性能

### 关键发现

1. **Position-Aware 特征的泛化性**: KernelEncodingLayer 学习的位置相关特征可以跨 trace 泛化
2. **K=50 仍然足够**: 与 exp_004 一致，K=50 已接近最佳效果
3. **模型不依赖特定 trace 的 Q-K 对**: 学习的是通用的位置-attention 模式

### 后续方向

1. 在更多 trace 对上验证泛化性的一致性
2. 分析不同 trace 的分布差异对命中率的影响
3. 尝试多 trace 联合训练以进一步提升泛化能力

## 相关文档 (Related Docs)

- [exp_004 README](../exp_004_module2_multibin_sparse_attention/README.md) - Overfit 验证实验
- [01_architecture.md](../../docs/01_architecture.md) - Module 2 架构设计
- [02_neural_network.md](../../docs/02_neural_network.md) - 网络结构
