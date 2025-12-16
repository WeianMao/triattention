# Experiment 001: Loss Function Sanity Check

## 目标

验证 Module 2 Bin-based Sparse Attention 的 Loss Function 能否正确优化 bin 分布。

使用可学习参数直接代替神经网络输出，观察 loss 优化结果。

## 方法

1. **SanityCheckModel**: 无输入的模拟模型，直接学习 bin 分布（nn.Parameter）
2. **Mock Data**: 构造包含一对一和二对一关系的模拟 Q-K group 关系
3. **Loss Functions**:
   - **Exp A**: 双向交叉熵 + Linear Repel
   - **Exp B**: 双向交叉熵 + Log Repel
   - **Baseline**: 仅双向交叉熵（无 repel）

## 运行方式

```bash
cd weian_development/spec_sparse_simulator/experiments/exp_001_sanity_check
python run.py --config config.yaml
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --num_queries | 6000 | Query 数量 |
| --num_keys | 6000 | Key 数量 |
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

### Mock Data 分布 (6000 queries × 6000 keys)

| 关系类型 | Query 范围 | Key 范围 | 数量 |
|----------|------------|----------|------|
| 一对一 | 0~2999 | 0~2999 | 3000 pairs |
| 二对一 | 3000~5999 | 3000~4499 | 3000 queries → 1500 keys |
| 未使用 | - | 4500~5999 | 1500 keys (25%) |

---

## 关键指标说明

### Keys/Query

每个 Query 平均需要计算 attention 的 Key 数量：
- 找到每个 Query 的 argmax bin
- 统计该 bin 里有多少 Keys
- 对所有 Query 求平均

**理论下限** = num_keys / num_bins = 6000 / 128 ≈ **46.875**

越低越好（在保证 Hit Rate 的前提下）。

### Computation Reduction

计算量减少比例：`1 - (Keys/Query / num_keys)`

---

## 归一化的重要性

### 问题：未归一化时 Repel 项主导

对于 6000×6000 的数据：
- **Attract pairs**: 6000 个（每个 Query 对应 1 个 Key）
- **Repel pairs**: 6000 × 5999 ≈ **36M** 个

比例约 1:6000，未归一化时 Repel 项会完全主导优化。

### 解决方案：Per-Query 归一化

每个 Query 的 loss 分别除以其正/负样本数量：

```python
# 拉近项归一化
attract_per_query = ... / num_pos_per_query

# 推远项归一化
repel_per_query = ... / num_neg_per_query
```

### 归一化前后对比 (6000×6000, λ=1.0)

| 状态 | Exp A Hit Rate | Exp B Hit Rate | Baseline Hit Rate |
|------|----------------|----------------|-------------------|
| 未归一化 | 0% | 0% | 100% |
| **归一化后** | **100%** | **100%** | **100%** |

---

## 实验结果 (6000×6000, 归一化版本)

### 基础对比 (λ=1.0)

| 实验 | Hit Rate | Keys/Query | Comp. Red. | Bin Util. |
|------|----------|------------|------------|-----------|
| **Exp A (Linear Repel)** | **1.0000** | 42.52 | 99.29% | 100% |
| Exp B (Log Repel) | 1.0000 | 48.04 | 99.20% | 100% |
| Baseline | 1.0000 | 47.95 | 99.20% | 100% |

### Exp A: 不同 λ 值的影响

| λ | Hit Rate | Keys/Query | Attract | Repel |
|---|----------|------------|---------|-------|
| 1.0 | 1.0000 | 42.52 | 368.80 | 41.55 |
| 2.0 | 1.0000 | 42.11 | 368.76 | 41.14 |
| 5.0 | 1.0000 | **39.10** | 368.85 | 38.14 |
| 10.0 | 1.0000 | **37.68** | 368.75 | 36.73 |
| 20.0 | 1.0000 | **37.66** | 368.69 | 36.70 |
| 50.0 | 1.0000 | 37.79 | 368.20 | 36.83 |

**结论**: 增加 λ 可以将 Keys/Query 从 42.52 降到 37.66（降低 11.4%），且保持 100% Hit Rate。

### Exp B: 不同 λ 值的影响

| λ | Hit Rate | Keys/Query | Repel |
|---|----------|------------|-------|
| 1.0 | 1.0000 | 48.04 | -151,769 |
| 2.0 | 0.9997 | 53.50 | -161,906 |
| 5.0 | 0.7093 | 239.44 | -180,722 |
| 10.0 | 0.2827 | 84.62 | -190,997 |
| 20.0 | 0.0000 | 0.01 | -224,734 |

**问题**: Log Repel 值为负数，增加 λ 反而让模型更激进地推远，导致 Hit Rate 急剧下降。

---

## Loss Function 数学表达式

### Exp A: Linear Repel

**拉近项**:
$$L_{attract} = \sum_q \frac{1}{|N_{pos}(q)|} \left[ -\sum_b p_q(b) \log r_{k_q}(b) - \sum_b r_{k_q}(b) \log p_q(b) \right]$$

**推远项**:
$$L_{repel} = \sum_q \frac{1}{|N_{neg}(q)|} \sum_{k \notin group(q)} \sum_b p_q(b) \cdot r_k(b)$$

Repel 值为**正数**（0~1 之间的概率乘积），优化方向正确。

### Exp B: Log Repel

**推远项**:
$$L_{repel} = \sum_q \frac{1}{|N_{neg}(q)|} \sum_{k \notin group(q)} \sum_b p_q(b) \cdot \log r_k(b)$$

因为 $\log r_k(b) < 0$，Repel 值为**负数**，最小化 loss 会让模型更激进地推远 Q 和 K，导致优化方向错误。

---

## 结论

1. **归一化是必要的**：未归一化时 Repel 项主导优化，导致 Exp A/B 完全失败
2. **Linear Repel (Exp A) 是可行的 Loss 设计**：
   - 100% Hit Rate
   - 增加 λ 可以降低 Keys/Query（从 42.52 到 37.66）
   - 推荐 λ=10~20
3. **Log Repel (Exp B) 不可用**：
   - Repel 项为负数，优化方向错误
   - 增加 λ 会导致 Hit Rate 急剧下降
   - 如需使用，应改为 $-\sum p \log r$（取负号）
4. **Baseline 表现良好**：双向交叉熵足以学习正确的 bin 分配，但缺少 repel 项导致 Keys/Query 较高
5. **Keys/Query 理论下限**：受限于 num_bins，当前 128 bins 的理论下限约为 47

---

## 相关文档

- [04_training_and_labels.md](../../docs/04_training_and_labels.md) - Loss 设计与 Sanity Check 规划
- [02_module2_bin_sparse_attention.md](../../docs/02_module2_bin_sparse_attention.md) - Module 2 设计
