# R-KV SpeckV 性能下降原因分析

## 问题背景

算法从 LazyEviction 迁移到 R-KV 后，效果明显下降。本文档列出所有可能导致此问题的原因。

---

## 1. Prompt 模板对齐检查

### 检查结果：✅ 已对齐

**原始 R-KV `run_math.py` 使用的 Prompt:**
```python
# R-KV/HuggingFace/run_math.py:46
prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"
```

**SpeckV 使用的 Prompt:**
```python
# R-KV/weian_development/speckv/prompt_utils.py:8-12
PROMPT_TEMPLATE = (
    "You are given a math problem.\n\nProblem: {question}\n\n "
    "You need to solve the problem step by step. First, you need to provide the chain-of-thought, "
    "then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"
)
```

**结论：** Prompt 模板与原始 R-KV 完全一致，不是问题原因。

---

## 2. RoPE 对齐检查 - ✅ 已验证无影响

### 问题描述

代码中有 RoPE 对齐检查，但因为 `transformers>=4.54` 架构变更，对于 Qwen2 和 Llama3 模型**检查逻辑被跳过**。

### 但实际验证结果：无影响

虽然检查被跳过，但 `build_rotary()` 通过 `AutoConfig` 正确构建了与模型一致的 rotary embedding：

```
测试 Qwen2 (DeepSeek-R1-Distill-Qwen-7B):
  Model inv_freq[:5]:  [1.0000, 0.8660, 0.7499, 0.6494, 0.5623]
  Pruner inv_freq[:5]: [1.0000, 0.8660, 0.7499, 0.6494, 0.5623]
  Max diff: 0.00e+00
  torch.allclose(atol=1e-5): True ✅

测试 Llama3 (DeepSeek-R1-Distill-Llama-8B):
  Max diff: 0.00e+00
  attention_scaling: Model=1.0, Pruner=1.0 ✅
  torch.allclose(atol=1e-5): True ✅
```

**结论：** RoPE 参数完全对齐，检查被跳过不会导致实验结果偏差。

**注意：** 虽然建议修复检查逻辑以避免 WARNING 输出，但这不是导致性能问题的原因。

---

## 3. Cache 类型兼容性检查

### 检查结果：✅ 无信息丢失

**测试代码：**
```python
from transformers.cache_utils import DynamicCache
import torch

cache = DynamicCache()
k1 = torch.randn(1, 4, 10, 64)
v1 = torch.randn(1, 4, 10, 64)
cache.update(k1, v1, 0)

# 转换为 tuple
legacy = cache.to_legacy_cache()

# 转换回 DynamicCache
cache2 = DynamicCache.from_legacy_cache(legacy)
legacy2 = cache2.to_legacy_cache()

# 验证
>>> torch.allclose(legacy[0][0], legacy2[0][0])
True
>>> torch.allclose(legacy[0][1], legacy2[0][1])
True
```

**结论：** `DynamicCache` 和 `tuple` 之间的转换是无损的，不是问题原因。

---

## 4. position_ids 处理检查

### 代码分析

**`speckv_forward` 中的 position 处理 (`rkv_speckv_generate.py:139-182`):**

```python
if past_key_values is not None and input_ids is not None:
    bsz, step = input_ids.shape

    # Absolute positions for rotary (RoPE 使用绝对位置)
    start_pos = state.pruner.absolute_position
    abs_positions = torch.arange(start_pos, start_pos + step, ...)
    position_ids_override = abs_positions  # 例如: [100]

    # Relative positions for cache placement (cache 使用相对位置)
    current_cache_len = past_key_values.get_seq_length()  # 例如: 50 (pruning 后)
    rel_positions = torch.arange(current_cache_len, current_cache_len + step, ...)
    cache_position_override = rel_positions  # 例如: [50]

    attention_mask_override = None  # 使用 causal mask
else:
    cache_position_override = None  # prefill 阶段不覆盖
```

### Qwen2Model 如何使用这些参数

```python
# Qwen2Model.forward 中:
if cache_position is None:
    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, ...)

if position_ids is None:
    position_ids = cache_position.unsqueeze(0)

# RoPE 使用 position_ids
position_embeddings = self.rotary_emb(hidden_states, position_ids)

# Attention 中 cache.update() 使用 cache_position
# 但 DynamicCache.update() 实际上不使用 cache_position，只是简单 append
```

### 潜在问题

**场景：** 假设 prefill 了 100 个 token，pruning 后 cache 只剩 50 个 token，生成第 101 个 token

| 变量 | 值 | 用途 |
|------|-----|------|
| `pruner.absolute_position` | 100 | 记录实际生成到第几个 token |
| `position_ids_override` | [100] | RoPE 编码使用 |
| `current_cache_len` | 50 | cache 中实际的 KV 数量 |
| `cache_position_override` | [50] | 新 token 写入 cache 的位置 |

**问题点：**

1. **DynamicCache 不使用 cache_position** - 它只是简单 append，所以 `cache_position_override` 实际上没有生效
2. **attention mask 被设为 None** - 依赖自动生成的 causal mask

让我验证 attention mask 生成是否正确：

```python
# create_causal_mask 使用:
# - cache_position: 用于确定 query 的位置
# - past_key_values.get_seq_length(): 用于确定 KV 的长度

# 如果 cache_position = [50] 但 past_key_values 长度也是 50
# 那么 mask 应该是正确的 (query at 50, attending to 0-50)
```

### 检查结果：⚠️ 需要进一步验证

从代码逻辑来看，position_ids 处理**看起来是正确的**：
- RoPE 使用绝对位置 (`position_ids_override`)
- Cache 简单 append (DynamicCache 不使用 `cache_position`)
- Attention mask 使用 causal mask

但建议添加 debug logging 来验证运行时的实际值：

```python
# 在 speckv_forward 中添加:
print(f"[DEBUG] absolute_position={state.pruner.absolute_position}, "
      f"cache_len={current_cache_len}, "
      f"position_ids={position_ids_override}, "
      f"cache_position={cache_position_override}")
```

---

## 5. 核心算法架构差异分析 ⚠️ 关键发现

### R-KV vs SpeckV 架构对比

通过深度代码审查，发现 **R-KV 和 SpeckV 在算法层面存在根本性差异**：

| 特性 | R-KV (R1KV) | SpeckV |
|------|------------|--------|
| **压缩位置** | Attention forward 内部 (per-layer) | Model forward 外部 (post-hoc) |
| **Query 来源** | **实时 query** (最近 window_size 个) | **预计算校准数据** (q_mean_complex) |
| **评分方式** | attention scores + similarity | 频率统计 + RoPE 反演 |
| **触发时机** | 每 divide_length 步 | 每 round_window 个 token |
| **压缩粒度** | Per-layer 独立压缩 | 跨 sampled_heads 聚合 |

### R-KV 核心逻辑 (`R-KV/HuggingFace/rkv/compression/r1_kv.py`)

```python
def update_kv(self, key_states, query_states, value_states):
    # 1. 使用实时 query 计算 attention scores
    attn_weights = compute_attention_scores(query_states, key_states)

    # 2. 对最近 window_size 个 query 的 attention 做 softmax + mean
    attn_weights_sum = softmax(attn_weights[:, :, -window_size:, :-window_size]).mean(dim=-2)

    # 3. Max pooling 平滑
    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=7, ...)

    # 4. 计算 key 之间的相似度 (去重)
    similarity_cos = cal_similarity(key_states, ...)

    # 5. 组合评分
    final_score = attn_cache * mix_lambda - similarity_cos * (1 - mix_lambda)

    # 6. TopK 选择
    indices = final_score.topk(budget - window_size).indices
```

### SpeckV 核心逻辑 (`sparse_round_pruner_prefill_keep.py`)

```python
def _compute_head_scores(self, past_key_values, key_positions, ...):
    for layer, head in self.sampled_heads:
        # 1. 获取 key 并进行 RoPE 反演
        k_values = past_key_values[layer][0][0, kv_head]
        k_unrot = invert_rope(k_values, cos_table, sin_table, ...)

        # 2. 使用预计算的 q_mean_complex 计算频率统计
        amp, phi, extra = compute_frequency_statistics_from_means(
            stats.q_mean_complex,  # 校准数据
            stats.q_abs_mean,      # 校准数据
            k_unrot,
        )

        # 3. 预测未来 attention scores
        head_scores = score_keys_for_round(
            key_indices, round_start, amp, phi, omega, extra, offsets, ...
        )

    # 4. 跨 head 聚合 (max-pooling 或 rank-based)
    combined = head_matrix.max(dim=0).values
```

### 关键差异分析

1. **Query 来源差异**：
   - R-KV 使用**实时生成的 query** 来评估 key 的重要性
   - SpeckV 使用**校准阶段预计算的 query 统计量** (`q_mean_complex`)

2. **评分方向**：
   - R-KV 评估 "这个 key 对当前/最近的 query 有多重要"
   - SpeckV 评估 "这个 key 在未来 round 中可能有多重要"

3. **Per-Layer vs Cross-Layer**：
   - R-KV 在每个 Attention layer 内部独立做压缩决策
   - SpeckV 在 model forward 完成后，基于 sampled_heads 做跨层决策

4. **压缩后的处理**：
   - R-KV 直接修改 `past_key_value.key_cache[layer_idx]`
   - SpeckV 返回新的 tuple cache，依赖 HF generate 的下一步 forward

### 可能导致性能差异的原因

1. **校准数据泛化能力**：SpeckV 的 `q_mean_complex` 是在特定数据集上计算的，可能无法很好地泛化到不同的推理过程

2. **Sampled Heads 限制**：SpeckV 只使用部分 head 做评分，而 R-KV 在每个 layer 的所有 head 都做评分

3. **时间维度差异**：
   - R-KV 每步都根据最新 query 调整压缩决策
   - SpeckV 的评分基于静态的校准统计量

---

## 总结

### 确认无问题的项目

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Prompt 模板对齐 | ✅ | 与原始 R-KV 完全一致 |
| Cache 转换 | ✅ | DynamicCache ↔ tuple 无损 |
| RoPE 参数对齐 | ✅ | inv_freq 和 attention_scaling 完全匹配 |
| Model 配置对齐 | ✅ | speckv/rkv/fullkv 的 Qwen 配置使用相同模型 |

### 已排除的问题

| 问题 | 状态 | 说明 |
|------|------|------|
| RoPE 对齐检查被跳过 | ✅ 已验证无影响 | 虽然检查被跳过，但实际 `inv_freq` 和 `attention_scaling` 完全对齐 (max diff: 0.00e+00) |

### 核心发现：算法架构差异

SpeckV 和 R-KV 在算法层面存在根本性差异：
- **R-KV 使用实时 query** 计算 attention scores
- **SpeckV 使用预计算的校准数据** 预测未来 attention

这不是 "bug"，而是**设计差异**。SpeckV 的目标是用预计算的统计量来近似 R-KV 的实时评分，但这种近似可能在某些场景下效果不佳。

### 建议的后续调查方向

1. **校准数据质量检查**：
   - 检查 `q_mean_complex` 的统计特性
   - 对比校准数据 vs 实际推理时的 query 分布

2. **Sampled Heads 覆盖率**：
   - 检查哪些 heads 被采样，是否遗漏了重要的 attention patterns

3. **评分函数对比实验**：
   - 在同一个 forward 中同时计算 R-KV 和 SpeckV 的评分
   - 比较两者的 correlation

4. **消融实验**：
   - 尝试在 SpeckV 中加入实时 query 信息
   - 或者尝试在 R-KV 中使用类似的预计算机制
