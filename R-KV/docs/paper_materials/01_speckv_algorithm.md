# SpecKV Algorithm

## 1. 核心思想

SpecKV利用**RoPE位置编码的频率域特性**来预测token的重要性。核心洞察是：RoPE将位置信息编码为频率空间中的旋转，通过分析这些频率分量的统计特性，可以预测未来query对历史key的注意力权重。

### 与R-KV的区别

| 方面 | SpecKV | R-KV |
|------|--------|------|
| 评分信号 | 预计算的频率统计量 | 实时注意力分数 + 相似度 |
| 计算时机 | 离线统计 + 在线评分 | 完全在线 |
| 数学基础 | RoPE频率域分析 | 注意力模式 + 余弦相似度 |

## 2. 数学基础

### 2.1 RoPE的复数表示

对于head维度为d的key向量，RoPE后的key可以表示为：
```
k_rotated = k * cos(ω * pos) + rotate_half(k) * sin(ω * pos)
```

逆RoPE变换可以恢复原始key：
```python
k_unrot = k_rotated * cos(ω * pos) - rotate_half(k_rotated) * sin(ω * pos)
```

### 2.2 复相关统计量

将逆RoPE后的Q/K表示为复数形式：
```
q_tilde = q_x + i*q_y = |Q| * e^(i*θ_q)
k_tilde = k_x + i*k_y = |K| * e^(i*θ_k)
```

复相关定义为：
```
C_f = E[q_tilde * conj(k_tilde)] = E[|Q||K| cos(θ_q - θ_k)] + i*E[|Q||K| sin(θ_q - θ_k)]
```

**为什么用复相关而不是简单平均相位**：简单平均相位隐含"幅值与相位差独立"的假设，但实际上近邻配对幅值大、相位集中，远距配对幅值小、相位像噪声。复相关同时记录幅值和相位差的耦合信息。

### 2.3 重构注意力曲线

从复相关可以重构不同相对距离Δ下的预期点积：
```
Recon(Δ) = Σ_f [Re(C_f) * cos(ω_f * Δ) - Im(C_f) * sin(ω_f * Δ)]
```

## 3. 算法流程

### 3.1 离线统计收集

1. 在训练/校准数据上运行模型，收集Q/K对
2. 对每个sampled head，计算：
   - `q_mean_complex`: Q的复数均值
   - `q_abs_mean`: |Q|的均值
3. 保存统计文件到 `stats/*.pt`

### 3.2 在线评分与裁剪

1. **初始化**: 加载统计文件，构建RoPE表

2. **Prefill阶段**: 保留所有prefill token（不压缩问题）

3. **Decode阶段**（每生成divide_length个token触发一次）:
   ```python
   for each key at position p:
       # 逆RoPE恢复原始key
       k_unrot = invert_rope(key, cos_table, sin_table)

       # 计算频率统计
       amp, phi, extra = compute_frequency_statistics_from_means(
           q_mean_complex, q_abs_mean, k_unrot
       )

       # 评分：预测未来token对该key的注意力
       score = score_keys_for_round(
           key_positions, round_start, amp, phi, omega, extra, offsets
       )
   ```

4. **Token选择**:
   - 聚合多个sampled head的分数
   - 使用union-based选择或per-head独立选择
   - 保留top-k个token

### 3.3 Per-Head独立裁剪模式

`per_head_pruning=True`时，每个KV head独立选择要保留的token：

```python
for kv_head_idx in range(num_kv_heads):
    # 聚合该KV head对应的所有sampled attention head的分数
    group_scores = per_head_scores[indices_for_this_kv_head]
    aggregated = group_scores.max(dim=0).values

    # 独立top-k选择
    keep_indices[kv_head_idx] = aggregated.topk(keep_count).indices
```

**优势**: 不同head可以保留不同位置的token，保持每个head的特异性。

## 4. 关键参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `kv_budget` | KV cache最大容量 | 2048 |
| `divide_length` | 压缩触发间隔 | 128 |
| `sparse_round_window` | 评分聚合窗口 | 32 |
| `sparse_score_aggregation` | 分数聚合方式 | "mean" |
| `sparse_normalize_scores` | 是否z-score标准化 | True |
| `per_head_pruning` | 是否per-head独立裁剪 | True |
| `include_prefill_in_budget` | prefill是否计入budget | True |

## 5. 核心代码位置

- **Pruner类**: `sparse_round_pruner_prefill_keep.py` - `SparseRoundPruner`
- **评分函数**: `round_pruning_utils.py` - `score_keys_for_round()`
- **逆RoPE**: `round_pruning_utils.py` - `invert_rope()`
- **频率统计**: `round_pruning_utils.py` - `compute_frequency_statistics_from_means()`
