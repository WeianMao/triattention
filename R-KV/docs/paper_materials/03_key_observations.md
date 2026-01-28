# Key Observations and Design Motivation

本文档记录在算法开发过程中的关键观察发现和设计motivation。

## 1. RoPE位置偏移发现

**来源**: `R-KV/docs/positional_bias_discovery.md`

### 观察现象

在分析一个bug时发现：将RoPE位置编码进行偏移，使prefill(question)token在相对位置上显得更"近"，可以**提升约8%的性能**。

### 机制解释

在AIME数学推理场景中：
- Prefill (问题): 约150 tokens
- Decode (推理): 5,000 - 30,000 tokens

正常情况下，当decode到第10,000个token时：
- 问题的相对距离 = 10,000 + 150 = 10,150
- RoPE注意力权重随距离衰减 → 问题被"遗忘"

使用位置偏移后：
- 问题的相对距离接近0
- 问题保持在"最近"位置 → 持续获得高注意力权重

### 设计启示

1. **长推理需要持续参考问题**: 数学推理链需要反复检查问题条件
2. **位置衰减是问题**: 生成越长，位置衰减问题越严重
3. **Prefill保护的必要性**: SpecKV保留prefill token的设计是正确的

---

## 2. 频率域分析的优势

**来源**: `weian_development/attention_qk_analysis/docs/summary_head_analysis.md`

### 为什么不直接用平均相位？

简单平均`cos(θ_q - θ_k)`隐含"幅值与相位差独立"的假设。但实际上：
- 近邻配对：幅值大、相位集中
- 远距配对：幅值小、相位像噪声

简单平均会被大量远距配对拉偏，使统计量失效。

### 复相关的优势

复相关`C_f = E[q_tilde * conj(k_tilde)]`同时记录幅值和相位差：
- `|C_f|`: 频段贡献强度（幅值×相位综合效果）
- `arg(C_f)`: 主导方向，近似"平均相位差"
- 可直接用于重构注意力曲线

### 逆RoPE的必要性

如果不先逆RoPE，会把RoPE的基波相位和模型学到的相位混在一起，不利于分辨模型的真实行为。逆RoPE后再计算复相关效果更好。

### Yarn RoPE缩放的影响

DeepSeek-R1-Qwen使用Yarn RoPE，cos/sin在生成时会乘以全局`attention_scaling`。如果逆RoPE时忽略它，会让"还原"的向量系统性放大，导致统计量失真。**必须在逆RoPE前先除以该缩放**。

---

## 3. Per-Head独立裁剪

**来源**: `R-KV/docs/perhead_aligned_gap.md`, `sparse_round_pruner_prefill_keep.py`

### 问题背景

最初的实现使用全局统一的token选择，所有KV head保留相同的token。

### 观察

不同注意力head关注不同的信息：
- 有的head关注语法结构
- 有的head关注语义关系
- 有的head关注位置信息

全局选择会丢失这种多样性。

### Per-Head设计

每个KV head独立选择要保留的token：

1. **分组**: 将sampled attention heads按KV head分组
2. **聚合**: 每组内计算max分数
3. **独立选择**: 每个KV head独立top-k

### 实现细节

```python
# 分组：sampled heads按(layer, kv_head)分组
for i, (layer, attn_head) in enumerate(sampled_heads):
    kv_head = attn_head // num_key_value_groups
    kv_head_groups[(layer, kv_head)].append(i)

# 每个KV head独立选择
for kv_head_idx in range(num_kv_heads):
    # 聚合该head的分数
    layer_max_scores = []
    for layer_idx in layers:
        group_scores = head_matrix[kv_head_groups[(layer_idx, kv_head_idx)]]
        layer_max_scores.append(group_scores.max(dim=0).values)

    # 跨层平均
    aggregated = torch.stack(layer_max_scores).mean(dim=0)

    # 独立top-k
    keep_indices[kv_head_idx] = aggregated.topk(keep_count).indices
```

---

## 4. 与R-KV对齐的设计选择

**来源**: 实验脚本参数分析

### 对齐参数

为了与R-KV公平对比，SpecKV采用了以下对齐设计：

1. **`--include-prefill-in-budget`**: Prefill token计入budget，与R-KV行为一致
2. **`--rkv-style-compression`**: 使用R-KV风格的压缩触发机制
3. **`--rkv-style-slack-trigger`**: 触发时机对齐
4. **`--divide-length 128`**: 压缩间隔对齐R-KV

### 评分标准化

`--sparse-normalize-scores`: 对每个head的分数进行z-score标准化，确保不同head的分数可比。

---

## 5. 统计文件的跨数据集使用

**来源**: 配置文件`aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml`

### 设计

```yaml
# AIME24测试使用AIME25的统计数据
sparse_stats_path: R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/...
```

### 原因

使用不同数据集的统计数据避免数据泄露。统计量捕获的是模型的一般注意力模式，而不是特定问题的信息。

---

## 6. 高频分量的作用

**来源**: `sparse_round_pruner_prefill_keep.py` - `disable_top_n_high_freq`参数

### 观察

RoPE的高频分量（omega[0:n]）对应短距离的位置信息。在某些情况下，可能需要消融这些高频分量来研究其作用。

### 参数设计

```python
disable_top_n_high_freq: int = 0  # 默认使用所有频率分量
```

当设置大于0时，在position-dependent scoring中屏蔽最高的n个频率分量。
