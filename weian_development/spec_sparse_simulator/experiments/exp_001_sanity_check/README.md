# Experiment 001: Loss Function Sanity Check

## 目标

验证 Module 2 Bin-based Sparse Attention 的 Loss Function 能否正确优化 bin 分布。

使用可学习参数直接代替神经网络输出，观察 loss 优化结果。

## 方法

1. **SanityCheckModel**: 无输入的模拟模型，直接学习 bin 分布（nn.Parameter）
2. **Mock Data**: 构造包含一对一和一对多关系的模拟 Q-K group 关系
3. **Loss Functions**:
   - **Exp A**: 双向交叉熵 + Linear Repel
   - **Exp B**: 双向交叉熵 + Log Repel
   - **Baseline**: 仅双向交叉熵（无 repel）

## 运行方式

```bash
cd weian_development/spec_sparse_simulator/experiments/exp_001_sanity_check
python run.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --num_queries | 64 | Query 数量 |
| --num_keys | 32 | Key 数量 |
| --num_bins | 128 | Bin 数量 |
| --epochs | 1000 | 训练轮数 |
| --lr | 0.01 | 学习率 |
| --lambda_repel | 1.0 | Repel loss 权重 |
| --seed | 42 | 随机种子 |
| --force_regenerate | false | 强制重新生成 mock data |

### Mock Data 持久化

Mock data（虚拟的 Q-K ground truth 关系）会保存到硬盘：
- **路径**: `output/mock_data/mock_data_q{num_queries}_k{num_keys}.pt`
- **特性**: 所有实验共享同一份 mock data，确保实验可重复性
- **重新生成**: 使用 `--force_regenerate` 参数

## 预期结果

| 实验 | 预期结果 |
|------|----------|
| Exp A (Linear Repel) | 收敛，bin 分布较均匀 |
| Exp B (Log Repel) | 收敛，可能有更强的分离 |
| Baseline (无 Repel) | 可能 collapse 到同一 bin |

## 结果摘要 (256 queries × 256 keys)

| 实验 | Hit Rate | Keys/Query | Comp. Red. | Bin Util. | 结论 |
|------|----------|------------|------------|-----------|------|
| **Exp A (Linear Repel)** | **1.0000** | **1.27** | 99.50% | 75.0% | ✅ 最佳：完美命中率 + 高稀疏度 |
| Exp B (Log Repel) | 0.5039 | 1.00 | 99.61% | 28.1% | ❌ 失败：Log Repel 导致优化不稳定 |
| Baseline (No Repel) | **1.0000** | 3.00 | 98.83% | 85.9% | ✅ 收敛但 Keys/Query 较高 |

### 关键发现

1. **Exp A (Linear Repel) 表现最佳**：
   - 100% Argmax Hit Rate（所有 Q-K 对正确分到同一 bin）
   - Keys per Query = 1.27（接近最优稀疏度）
   - Bin Utilization = 75%（使用了 96/128 个 bin）
   - Loss 稳定收敛

2. **Exp B (Log Repel) 存在问题**：
   - Hit Rate 仅 50.39%（接近随机）
   - Loss 为负且持续下降（-2M+），说明 Log Repel 项主导了优化方向
   - Bin Utilization 仅 28.1%（严重坍缩到少数 bin）
   - **建议**：Log Repel 需要更小的 λ 或不同的正则化策略

3. **Baseline（无 Repel）未发生坍缩**：
   - 与预期不同，Baseline 也达到了 100% Hit Rate
   - 但 Keys per Query = 3.0，高于 Exp A 的 1.27
   - Bin Utilization = 85.9%（bin 分布较均匀但稀疏度不够）
   - 说明双向交叉熵足以学习正确的 bin 分配，但缺少 repel 项导致每个 bin 包含过多 keys

## 结论

1. **Linear Repel (Exp A) 是可行的 Loss 设计**：能够正确优化 bin 分布，达到 100% Argmax Hit Rate
2. **Log Repel (Exp B) 需要调整**：当前 λ=1.0 下，Log Repel 项主导优化，导致 Q-K 分离而非聚合
3. **Baseline 意外成功**：双向交叉熵本身足够强，但 repel 项可以进一步优化稀疏度（Keys per Query 从 3.0 降到 1.27）
4. **无 Bin Collapse 问题**：Exp A 和 Baseline 都没有出现严重的 bin 坍缩

## 相关文档

- [04_training_and_labels.md](../../docs/04_training_and_labels.md) - Loss 设计与 Sanity Check 规划
- [02_module2_bin_sparse_attention.md](../../docs/02_module2_bin_sparse_attention.md) - Module 2 设计
