# Exp 004: Module 2 Multi-Bin Sparse Attention POC

## 目标 (Objective)

验证 Module 2 Position-Aware Attention Router 的可行性：
- 使用 Multi-Bin Key Assignment 将历史 Keys 分配到多个 bins
- Query 选择一个 bin，从中选取 TopK 个 Keys 进行 attention
- 目标：TopK Hit Rate = 100% (K=50/500/1000)

## 方法 (Method)

### 网络架构
- **Key Network**: KernelEncodingLayer → logits → softmax(dim=0) → key_probs
- **Query Network**: KernelEncodingLayer → logits → softmax(dim=-1) → bin_probs
- **无 MLP，无 PositionScaling**（与 Module 1 不同）

### 损失函数
- **Attraction Loss**: `-log(sum_bins(p_q[b] * P[argmax_key, b])).mean()`
- 鼓励 Query 选择的 bin 包含其 argmax Key

### 评估指标
- **TopK Hit Rate**: Query 的 argmax Key 是否在选中 bin 的 TopK 中
- Recent keys (当前 round) 自动算命中

## 运行方式 (Usage)

```bash
# 训练
python run.py --mode train

# 评估
python run.py --mode evaluate
```

## 结果摘要 (Results Summary)

### 训练配置
- **数据**: qk.pt (layer=33, head=0), seq_len=17,570
- **模型参数**: 147,712 (Key Network: 73,856 + Query Network: 73,856)
- **训练**: 100 epochs, lr=0.001, round_window=128
- **设备**: CUDA (GPU 1)
- **训练时长**: ~53 分钟

### 训练曲线
| Epoch | Loss |
|-------|------|
| 1 | 1.241 |
| 10 | 0.105 |
| 50 | 0.026 |
| 100 | 0.013 |

### 评估结果 (TopK Hit Rate)

| K值 | 命中率 | Bin路由命中 | Recent自动命中 | 未命中 |
|-----|--------|-------------|----------------|--------|
| 50 | **99.77%** | 15,979 (97.18%) | 426 (2.59%) | 37 |
| 500 | **99.78%** | 15,980 (97.19%) | 426 (2.59%) | 36 |
| 1000 | **99.79%** | 15,982 (97.20%) | 426 (2.59%) | 34 |

总查询数: 16,442

## 结论 (Conclusion)

### POC 验证成功

1. **高命中率**: 达到 99.8%，接近 100% 目标
2. **K值不敏感**: K=50 和 K=1000 命中率几乎相同（只差3个查询），说明 argmax key 在所选 bin 内排名很靠前（基本在 Top 50 以内）
3. **压缩效果显著**: 每个 Query 只需计算 K + round_window = 50 + 128 = 178 个 key 的 attention，相比原始 ~16,000 个 key，**压缩约 90 倍**

### 关键发现

- Bin 路由准确率: 15,979 / 16,016 = **99.77%**（排除 recent keys 后）
- 模型不仅能路由到正确的 bin，而且 argmax key 在 bin 内的概率排名很高
- **K=50 已足够**，无需更大 K 值

### 后续方向

1. 在更多 head 上验证泛化性
2. 分析 argmax key 在 bin 内的排名分布
3. 与 Module 1 (Neural Pruning) 结合测试端到端效果

## 相关文档 (Related Docs)

- [01_architecture.md](../../docs/01_architecture.md) - Module 2 架构设计
- [02_neural_network.md](../../docs/02_neural_network.md) - 网络结构
- [03_loss_and_training.md](../../docs/03_loss_and_training.md) - Attraction Loss
- [04_evaluation_metrics.md](../../docs/04_evaluation_metrics.md) - TopK Hit Rate
