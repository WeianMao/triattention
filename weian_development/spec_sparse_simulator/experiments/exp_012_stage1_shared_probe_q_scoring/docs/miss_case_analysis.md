# Miss Case Analysis (2024-12-22)

## 实验目的

分析训练集上的 miss cases，诊断模型性能瓶颈的根本原因。

## 实验设置

- **Model**: `best_model_multi_trace_v5.pt` (epoch 10)
- **Trace**: qid0003_trace34 (seq_len=10938)
- **Head**: layer=17, head=25
- **K**: 50
- **round_window**: 128
- **exclude_tail**: 1000

## Miss Case 分类

| Type | 描述 |
|------|------|
| **Type A (Key Network Issue)** | argmax key 在**所有** bins 中排名都 >= K，即 key network 没有把这个 key 排好 |
| **Type B (Query Network Issue)** | argmax key 在**某个** bin 中排名 < K，但 query 选错了 bin |

## 实验结果

### 训练集整体统计 (qid0003_trace34)

| 统计项 | 数值 | 百分比 |
|-------|------|--------|
| Total queries | 9810 | 100% |
| Recent hits | 2000 | 20.39% |
| Bin hits | 2159 | 22.01% |
| Misses | 5651 | 57.60% |
| **Overall accuracy** | 4159 | **42.40%** |

### Miss Case 分类

| Type | 数量 | 占 Misses 百分比 |
|------|------|------------------|
| Type A (Key Network) | 3101 | 54.9% |
| Type B (Query Network) | 2550 | 45.1% |

## 关键发现

### 1. 两边都有问题

- **Key Network** 贡献了 ~55% 的 misses
- **Query Network** 贡献了 ~45% 的 misses
- 问题不是单一的，需要同时改进两边

### 2. Key Network 问题

对于 Type A misses，argmax key 在所有 128 个 bins 中的最佳排名都 >= 50。这意味着：
- Key network 的 scoring 函数没有学好
- 可能是 probe vectors 没有很好地覆盖 key 空间
- 或者 scoring 函数的表达能力不足

### 3. Query Network 问题

对于 Type B misses，argmax key 在某个 bin 中排名很好 (< 50)，但 query 选择了错误的 bin。这意味着：
- Query network 的 bin selection 没有学好
- Query 没有学会正确路由到包含 argmax key 的 bin

## 与之前 Loss 实验的关联

### Discrete Top-K Loss (已 archive)

- 只在 miss 时计算 loss
- 结果：K=50 hit rate 从 60.83% 降到 ~37%
- 原因：Loss 太 sparse，模型忘记正确 patterns

### Weighted CE Loss (已 archive)

- 用 query probability 加权所有 bins 的 CE loss
- 结果：K=50 hit rate 从 60.83% 降到 44.21%
- 原因：过度优化所有 bins，与推理 (top-1 bin only) 不对齐

## 可能的改进方向

### 针对 Key Network (Type A)

1. **增加 probe 数量或改进初始化** - 让 probes 更好地覆盖 key 空间
2. **改进 key scoring 函数** - 增加表达能力
3. **使用 margin-based loss** - 让 argmax key 和其他 keys 有更大的 margin

### 针对 Query Network (Type B)

1. **Load Balancing Loss** - 防止所有 query 选择同一个 bin
2. **Entropy Regularization** - 增加 bin selection 的 diversity
3. **Contrastive Learning** - 让 query 学会正确路由

### 联合改进

1. **Multi-Bin Inference** - 不改训练，推理时查多个 bins
2. **Auxiliary Ranking Loss** - 在 top-1 bin 中显式优化 ranking
3. **Temperature Annealing** - 逐渐从 soft 过渡到 hard selection

## 下一步计划

1. 检查是否存在 Query Network mode collapse（所有 query 选同一个 bin）
2. 分析 Key Network 的 probe coverage
3. 尝试 Load Balancing Loss 防止 mode collapse
