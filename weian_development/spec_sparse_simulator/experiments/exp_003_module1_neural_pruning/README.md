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
- Argmax Hit Rate: 96.32%
- Keys per Query: 7.70
- Retention Rate: 0.07%
- Computation Reduction: 99.91%
## 结论
Module 1神经网络架构需要进一步调优。当前Argmax Hit Rate为96.32%，未达到99%目标。建议调整超参数或增加训练轮数。
## 相关文档
- [01_module1_key_pruning.md](../../docs/01_module1_key_pruning.md)
- [03_neural_network_architecture.md](../../docs/03_neural_network_architecture.md)
- [04_training_and_labels.md](../../docs/04_training_and_labels.md)
- [05_experiment_conventions.md](../../docs/05_experiment_conventions.md)

## 默认配置
- Head索引: `hybrid_sample_heads_lowret_top10.json` (10个heads)
- Round window: 128
- 参考实现: `attention_pruning_case_study_hybrid_rounds_xtrace.py`
