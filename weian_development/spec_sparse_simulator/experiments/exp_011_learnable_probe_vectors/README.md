# Exp 006: Module 2 Reverse Cross-Trace Validation

## 目标 (Objective)

验证 Module 2 Multi-Bin Sparse Attention 的**双向泛化能力**：
- 训练在 trace B 上 (qid0003_trace34)
- 测试在 trace A 上 (qid0008_trace46)
- 目的：评估模型是否能在**反向**跨 trace 场景下保持高 TopK Hit Rate

这是 exp_005 的**逆向验证**实验，exp_005 是 A→B，本实验是 B→A。

## 与 exp_004/exp_005 的区别

| 配置 | exp_004 (Overfit) | exp_005 (A→B) | exp_006 (B→A) |
|------|-------------------|---------------|---------------|
| 训练数据 | qid0008_trace46 (A) | qid0008_trace46 (A) | **qid0003_trace34 (B)** |
| 测试数据 | qid0008_trace46 (A) | qid0003_trace34 (B) | **qid0008_trace46 (A)** |
| Epochs | 100 | 25 | 25 |
| 目的 | 验证过拟合学习能力 | 验证泛化能力 A→B | **验证泛化能力 B→A** |

## 运行方式 (Usage)

```bash
# 训练 (25 epochs)
python run.py --mode train

# 评估 (在 test trace 上)
python run.py --mode evaluate
```

## 结果摘要 (Results Summary)

### 训练配置
- **训练数据**: qk.pt → qid0003_trace34 (layer=33, head=0), seq_len=10,938
- **测试数据**: qk_test.pt → qid0008_trace46 (layer=33, head=0), seq_len=17,570
- **模型参数**: 147,712 (Key Network: 73,856 + Query Network: 73,856)
- **训练**: 25 epochs, lr=0.001, round_window=128
- **设备**: CUDA (GPU)
- **训练时长**: ~3 分钟

### 训练曲线
| Epoch | Loss |
|-------|------|
| 1 | 1.684 |
| 10 | 0.053 |
| 20 | 0.023 |
| 25 | 0.017 |

### 评估结果 (Reverse Cross-Trace TopK Hit Rate)

| K值 | 命中率 | Bin路由命中 | Recent自动命中 | 未命中 |
|-----|--------|-------------|----------------|--------|
| 50 | **98.49%** | 15,768 (95.90%) | 426 (2.59%) | 248 |
| 500 | **98.63%** | 15,790 (96.03%) | 426 (2.59%) | 226 |
| 1000 | **98.81%** | 15,821 (96.22%) | 426 (2.59%) | 195 |

总查询数: 16,442

## 结果对比 (Comparison with exp_004 & exp_005)

### TopK Hit Rate 对比

| K值 | exp_004 (Overfit) | exp_005 (A→B) | exp_006 (B→A) | A→B vs B→A 差距 |
|-----|-------------------|---------------|---------------|-----------------|
| 50 | 99.77% | 99.15% | **98.49%** | **-0.66%** |
| 500 | 99.78% | 99.47% | **98.63%** | **-0.84%** |
| 1000 | 99.79% | 99.51% | **98.81%** | **-0.70%** |

### Bin路由命中率对比

| 指标 | exp_004 (Overfit) | exp_005 (A→B) | exp_006 (B→A) |
|------|-------------------|---------------|---------------|
| Bin Hit Rate | 97.18-97.20% | 97.87-98.23% | **95.90-96.22%** |

### 训练 Trace 特征对比

| Trace | QID | Seq Length | 用途 |
|-------|-----|------------|------|
| A | qid0008_trace46 | 17,570 | exp_004/005 训练, exp_006 测试 |
| B | qid0003_trace34 | 10,938 | exp_005 测试, exp_006 训练 |

**关键观察**: Trace A 长度 (17,570) > Trace B 长度 (10,938)，exp_006 用较短 trace 训练、较长 trace 测试。

## 结论 (Conclusion)

### 双向泛化验证成功

1. **高命中率保持**: 即使反向训练 (B→A)，仍达到 98.5%+ 命中率
2. **略低于 A→B**: 命中率比 exp_005 低 ~0.7%，可能原因：
   - Trace B 较短，训练样本较少
   - Trace A 较长，测试场景更复杂
3. **Bin 路由稍弱**: 95.9% vs exp_005 的 97.9%，差距约 2%

### 关键发现

1. **Position-Aware 特征具有双向泛化性**: 无论从 A→B 还是 B→A，模型都能有效泛化
2. **训练数据量影响**: 较短 trace 训练导致略低性能，但影响有限
3. **K=50 仍然足够**: 与之前实验一致，K=50 已接近最佳效果
4. **模型不依赖训练方向**: 学习的是通用的位置-attention 模式

### 综合对比总结

| 实验 | 训练→测试 | K=50 命中率 | 结论 |
|------|----------|-------------|------|
| exp_004 | A→A | 99.77% | 基准 (过拟合上限) |
| exp_005 | A→B | 99.15% | 泛化成功 (-0.62%) |
| exp_006 | B→A | 98.49% | **双向泛化成功 (-1.28%)** |

**最终结论**: Module 2 Multi-Bin Sparse Attention 模型具备**双向跨 trace 泛化能力**，在不同训练方向下都能保持 >98% 的 TopK 命中率。

## 相关文档 (Related Docs)

- [exp_004 README](../exp_004_module2_multibin_sparse_attention/README.md) - Overfit 验证实验
- [exp_005 README](../exp_005_module2_cross_trace_validation/README.md) - Cross-trace A→B 验证实验
- [01_architecture.md](../../docs/01_architecture.md) - Module 2 架构设计
- [02_neural_network.md](../../docs/02_neural_network.md) - 网络结构
