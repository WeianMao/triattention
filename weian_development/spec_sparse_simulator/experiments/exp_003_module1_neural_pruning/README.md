# Experiment 003: Module 1 神经网络 Key 剪枝

## 目标
训练神经网络预测 Key 重要性分数（drop 概率），实现稀疏注意力推理。使用 Kernel Encoding + MLP + Position Scaling 实现 Module 1 算法。

## 最优配置（实验验证）

经过系统性实验验证，最优配置为：
- **num_kernels**: 1（从 3 减少到 1，参数量减少 60%）
- **mlp_hidden_dim**: 64（保持不变）
- **kappa_init**: 2.5（单核初始化的关键参数）

```python
Module1KeyPruningNetwork(
    num_kernels=1,
    mlp_hidden=64,
    kappa_init=2.5
)
```

**性能指标**:
- 参数量: 16,580（vs 原始 41,156，减少 60%）
- 命中率: 99.50%（满足 >= 99.5% 目标）
- Keys/Query: 71.84（略优于基线 72.48）

## 神经网络架构

```
Input: Key embeddings (head_dim=128)
    ↓
Kernel Encoding Layer: 64 频段 x 1 个 von Mises kernel
    ↓
MLP Layer: 64 -> 64 -> 1
    ↓
Position Scaling Layer: log 尺度锚点插值 (1k, 10k, 100k)
    ↓
Output: Sigmoid → drop 概率 [0, 1]
```

## 运行方式

### 1. 基础训练（使用默认最优配置）
```bash
cd weian_development/spec_sparse_simulator/experiments/exp_003_module1_neural_pruning

# 训练模型
python train.py

# 查看训练日志
tail -f output/logs/train.log
```

### 2. 剪枝实验（测试不同配置）
```bash
# 使用 run_pruning_experiment.py 进行参数扫描

# 基线实验（使用已训练的 checkpoint）
python run_pruning_experiment.py \
    --checkpoint output/checkpoints/final_model.pt \
    --experiment-name baseline

# 测试 num_kernels=1（推荐配置）
python run_pruning_experiment.py \
    --num-kernels 1 \
    --kappa-init 2.5 \
    --experiment-name exp_kernels_1

# 测试 MLP 维度缩减
python run_pruning_experiment.py \
    --mlp-hidden-dim 32 \
    --experiment-name exp_mlp_h32

# 组合实验
python run_pruning_experiment.py \
    --num-kernels 1 \
    --mlp-hidden-dim 32 \
    --kappa-init 2.5 \
    --experiment-name exp_k1_h32
```

### 3. 评估模式
```bash
python run.py --mode evaluate \
    --config config.yaml \
    --checkpoint output/checkpoints/final_model.pt
```

## 实验结果汇总

| 配置 | 参数量 | 变化 | 命中率 | Keys/Query | 状态 |
|------|--------|------|--------|------------|------|
| **num_kernels=1, h=64** | **16,580** | **-60%** | **99.50%** | **71.84** | **推荐** |
| num_kernels=1, h=32 | 14,468 | -65% | 99.50% | 81.08 | 可选 |
| num_kernels=3, h=64 | 41,156 | 基线 | 99.51% | 72.48 | 原始 |
| num_kernels=3, h=32 | 39,044 | -5% | 99.51% | 72.57 | 收益小 |
| avg_pooling | 36,998 | -10% | 100% | 8785 | 失败 |

详细分析见 `output/PRUNING_EXPERIMENT_REPORT.md`

## 关键发现

1. **核数量缩减非常有效**: 单个 von Mises 核（配合 kappa=2.5 初始化）足以捕获必要的频率模式
2. **MLP 层不可或缺**: 平均池化完全失败；MLP 隐藏层维度影响剪枝质量
3. **最佳配置**: `num_kernels=1, mlp_hidden=64` 实现参数/性能的最优权衡
4. **剪枝效果**: 平均每个 query 只需保留约 72 个 keys 即可达到 99.5% 命中率

## 文件结构

```
exp_003_module1_neural_pruning/
├── config.yaml                 # 配置文件（已更新为最优设置）
├── model.py                    # 神经网络模型定义
├── train.py                    # 训练脚本
├── run.py                      # 完整流程脚本
├── evaluate.py                 # 评估函数
├── run_pruning_experiment.py   # 剪枝实验脚本（参数扫描）
├── data/
│   └── qk.pt -> (symlink)      # 训练数据
├── output/
│   ├── checkpoints/            # 模型 checkpoint（gitignore）
│   ├── logs/                   # 训练日志（gitignore）
│   ├── pruning_experiments/    # 实验结果 JSON（gitignore）
│   └── PRUNING_EXPERIMENT_REPORT.md  # 实验报告
└── README.md                   # 本文档
```

## 相关文档

- [Module 1 Key Pruning 设计文档](../../docs/01_module1_key_pruning.md)
- [神经网络架构设计](../../docs/03_neural_network_architecture.md)
- [训练与标签生成](../../docs/04_training_and_labels.md)
