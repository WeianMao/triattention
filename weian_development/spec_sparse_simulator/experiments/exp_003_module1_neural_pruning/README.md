# Experiment 003: Module 1 Neural Network Key Pruning

## 目标
训练神经网络预测Key重要性分数(drop概率)，验证稀疏推理性能。使用Kernel Encoding + MLP + Position Scaling实现Module 1算法，在单trace上overfit验证模型架构的有效性。

## 方法
- **神经网络架构**:
  - Kernel Encoding Layer: 64频段 x 3个von Mises kernels = 128维输出
  - MLP Layer: 128 -> hidden -> 1
  - Position Scaling Layer: log尺度锚点插值 (1k, 10k, 100k)
  - Output: Sigmoid得到drop概率
- **训练**:
  - Loss: Binary Cross Entropy
  - 标签: 基于argmax的drop标签 (排除末尾1k Key)
  - 模式: POC单trace overfit验证
- **评估**:
  - 指标: Argmax Hit Rate (>99%), Keys per Query, Computation Reduction
  - 方法: 复用参考实现的compute_pooled_attention推理方式

## 运行方式

### 方式1: 使用train.py直接训练 (IMPL-005已完成)
```bash
# 训练模型 (保存checkpoint到output/checkpoints/)
python train.py

# 查看训练日志
tail -f output/logs/train.log
```

### 方式2: 使用run.py完整流程 (IMPL-007待实现)
```bash
# 训练
python run.py --mode train --config config.yaml

# 评估
python run.py --mode evaluate --config config.yaml --checkpoint output/checkpoints/best_model.pt

# 完整流程 (训练 + 评估)
python run.py --config config.yaml
```

## 结果摘要

### 训练结果
- Final Loss: 0.000216 (100 epochs)
- Model Parameters: 41,156

### 评估结果 (threshold=0.5)
| 指标 | 值 |
|------|-----|
| Argmax Hit Rate | 98.86% |
| Keys per Query | 70.65 |
| Retention Rate (historical) | 0.07% |
| Computation Reduction | 99.20% |

### Threshold Sweep 分析

通过调整 threshold 可以达到更高的 hit rate：

| Threshold | Hit Rate | Keys/Query | Retention | Comp Reduction |
|-----------|----------|------------|-----------|----------------|
| 0.50 | 98.86% | 70.6 | 0.071% | 99.20% |
| 0.90 | 99.13% | 71.4 | 0.079% | 99.19% |
| **0.993** | **99.52%** | **72.5** | **0.092%** | **99.17%** |
| 0.999 | 99.69% | 74.2 | 0.111% | 99.15% |

### 关键发现

1. **评估方法修正**: 原评估未正确保留当前 round 内的 keys，导致 hit rate 被低估
   - 约 3.26% 的 queries 的 argmax 落在当前 round（非 self token）
   - 修正后 Module 1 只预测历史 keys，当前 round 使用 full attention

2. **达到 99.5% Hit Rate 的代价**:
   - Threshold: 0.5 → 0.993
   - Keys/Query: 70.6 → 72.5 (增加 ~3%)
   - Computation Reduction: 99.20% → 99.17% (基本不变)

3. **模型有效性**: 在 threshold=0.5 已达到 98.86% hit rate，验证了神经网络架构的有效性

## 结论
Module 1 神经网络架构验证成功。通过调整 threshold 至 0.993，可达到 99.52% Argmax Hit Rate，同时保持 99.17% 的计算量减少。模型成功学习了 Key 重要性预测，在 POC 单 trace overfit 场景下表现良好。
## 相关文档
- [01_module1_key_pruning.md](../../docs/01_module1_key_pruning.md)
- [03_neural_network_architecture.md](../../docs/03_neural_network_architecture.md)
- [04_training_and_labels.md](../../docs/04_training_and_labels.md)
- [05_experiment_conventions.md](../../docs/05_experiment_conventions.md)

## 默认配置
- Head索引: `hybrid_sample_heads_lowret_top10.json` (10个heads)
- Round window: 128
- 参考实现: `attention_pruning_case_study_hybrid_rounds_xtrace.py`
