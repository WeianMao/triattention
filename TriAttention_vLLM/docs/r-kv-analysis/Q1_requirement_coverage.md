# Q1: 需求覆盖对比分析

对比 R-KV/vLLM 官方实现与 TriAttention 规划，分析他们考虑到但我们可能遗漏的需求。

> **快速查阅**：关键结论已汇总到 [../project/key_decisions.md](../project/key_decisions.md)

---

## 1. 开发阶段定义

| 阶段 | 目标 | 框架 | Batch Size | 说明 |
|-----|------|------|------------|------|
| **阶段 0** | 快速验证 | 在 R-KV 框架内 | = 1 | 最简单方式实现，复用 R-KV 代码 |
| **阶段 1** | 高效无侵入版本 | TriAttention 独立 | > 1 | Triton kernel，从头开发 |
| **阶段 2** | 高级功能 | TriAttention 独立 | > 1 | 内存触发、CUDA Graph 等 |

---

## 2. 需求覆盖矩阵

| 需求 | R-KV/vLLM | 阶段 0 | 阶段 1 | 阶段 2 |
|-----|-----------|-------|-------|-------|
| 基于频率统计的打分 | ❌（用 attention） | ✅ | ✅ | ✅ |
| 冗余感知（相似度过滤） | ✅ R1KV | ❌ | ❌ | ❌ |
| Per-head 独立裁剪 | ✅ | ✅ | ✅ | ✅ |
| Per-layer 裁剪 | ❌ | ✅ | ✅ | ✅ |
| Recent KV 保护 | ✅ | ✅ | ✅ | ✅ |
| Prefill 保护 | ✅ | ✅ | ✅ | ✅ |
| RoPE 位置追踪 | ✅ | ✅ | ✅ | ✅ |
| **Batch Size > 1** | ❌ | ❌ | ✅ | ✅ |
| CUDA Graph | ❌ | ❌ | ❌ | ⏸️ |
| 内存触发压缩 | ❌ | ❌ | ❌ | ⏸️ |
| **Triton 级别效率** | ❌ | ❌ | ✅ | ✅ |

---

## 3. R-KV 考虑到但需评估的需求

### 3.1 RoPE 配置一致性验证 ⚠️ **重要**

**R-KV 实现**（`rkv_speckv_generate.py` L119-145）：
```python
# CRITICAL SAFETY CHECK: Verify that the Pruner's internal Rotary Embedding
# matches the Model's live one.
# If these do not match, the pruner will use WRONG frequencies. GIGO.
```

**问题**：不同模型的 RoPE 配置字段名可能不同（`attn_factor` vs `factor`），如果打分器使用的频率与模型不一致，会导致打分完全错误。

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | 复用 R-KV 的检查逻辑 |
| 阶段 1 | 实现独立的 RoPE 一致性验证 |

---

### 3.2 多问题场景的状态重置

**R-KV 实现**：
```python
if is_empty_cache and state.attached:
    state.pruner = SparseRoundPruner(state.config)  # Fresh pruner
    state.attached = False
```

**问题**：连续处理多个问题时，需要正确重置压缩器状态。

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | 复用 R-KV 的状态管理 |
| 阶段 1 | 设计清晰的状态重置接口 |

---

### 3.3 Union-based Token Selection（多头多样性）

**R-KV 实现**：使用 union-based 选择确保每个 head 的重要 token 都有代表。

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | 不实现（简单 top-k） |
| 阶段 1 | 可选实现 |

---

### 3.4 Tie-breaking（分数相同时的处理）

**R-KV 实际做法**：⚠️ **没有任何处理**！直接使用 `topk()`，依赖 PyTorch 默认行为。

**验证结果**：经过代码审查，R-KV 并未实现 noise injection，之前的分析有误。

**我们的做法**：

| 阶段 | 处理方式 | 说明 |
|-----|---------|------|
| 阶段 0 | 怎么方便怎么来 | 打平时取前 N 个，或随机取，**不加噪声** |
| 阶段 1 | 确定性选择 | 打平时按位置顺序取前 N 个，**不加噪声** |

**原则**：不使用加噪声的方式解决 tie-breaking，这种做法不够优雅。与 R-KV 实际行为一致。

---

### 3.5 FP32 TopK 精度选项

**R-KV 实现**：
```python
fp32_topk: bool = False  # 在 topk 前转换为 fp32 避免精度损失
```

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | 可选实现 |
| 阶段 1 | Triton kernel 内部处理精度 |

---

### ~~3.6 Window-based Query Cache~~（不适用）

**R-KV 需要**：缓存最近的 Query 用于打分。

**我们不需要**：SpeckV 基于预计算的 Q 频率统计打分，不依赖实时 Query。

**结论**：只需保护最近的 KV，不需要 Query cache。

---

### 3.7 GQA（Grouped Query Attention）处理 ✅ **已正确处理**

**R-KV 实现**（`HuggingFace/rkv/utils.py` L8-39）：
```python
query_group_size = q_heads // kv_heads
if query_group_size == 1:  # MHA case
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
else:  # GQA case - reshape and pool over groups
    query_states.view(batch_size, kv_heads, query_group_size, q_len, head_dim)
    attn_weights = attn_weights.mean(dim=2)  # Pool over groups
```

**支持的模型**：
- Qwen2.5: 28 query heads, 4 KV heads (7:1)
- Qwen3: 32 query heads, 8 KV heads (4:1)

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | 复用 R-KV 的 GQA 处理（对于 SpeckV 不需要，因为不用 attention 打分） |
| 阶段 1 | 如需要，参考实现 |

---

## 4. 验证发现的新问题

### 4.1 Batch Size > 1 是**静默失败** ⚠️ **关键风险**

**R-KV 实现问题**（`HuggingFace/rkv/utils.py` L48）：
```python
def cal_similarity(key_states, ...):
    k = key_states[0]  # 硬编码 batch index！
```

**问题**：当 batch_size > 1 时，代码不会报错，而是**静默地只使用第一个 batch 的数据**。

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | R-KV 框架本身只支持 batch=1，无需处理 |
| 阶段 1 | **必须**显式支持 batch>1，或在 batch>1 时抛出明确错误 |

---

### 4.2 长序列 OOM 风险 ⚠️ **重要**

**R-KV 实现问题**（`HuggingFace/rkv/utils.py` L52-54）：
```python
k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
# 创建 [num_heads, seq_len, seq_len] 矩阵 - O(n²) 内存！
```

**问题**：对于 seq_len=100K，单个 head 需要 ~80GB 内存。

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | **不使用相似度计算**（SpeckV 不需要）—— 无此问题 |
| 阶段 1 | 不适用（SpeckV 不做相似度计算） |

---

### 4.3 R-KV 代码版本不一致 ⚠️

**发现**：`rkv/compression/r1_kv.py` 与 `HuggingFace/rkv/compression/r1_kv.py` 功能不同：

| 功能 | `rkv/` 版本 | `HuggingFace/` 版本 |
|-----|------------|-------------------|
| `protect_prefill` | ❌ 无 | ✅ 有 |
| `attach_prefill_length()` | ❌ 无 | ✅ 有 |
| 代码行数 | ~200 行 | ~300 行 |

**影响**：使用不同版本的 R-KV 可能导致行为差异。

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | 使用 HuggingFace 版本（功能更完整） |
| 阶段 1 | 独立实现，不受版本影响 |

---

### 4.4 无 TP/PP 支持

**R-KV 实现**：所有压缩操作假设单 GPU 模型。

**缺失功能**：
- 无 TP 跨 rank 的 indices 同步
- 无 PP stage 间的 cache 更新通信
- 无分布式 topk 选择

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | 不需要（单 GPU 验证） |
| 阶段 1 | 可选实现 |
| 阶段 2 | 根据需求考虑 |

---

### 4.5 RoPE 检查位置说明

**验证发现**：RoPE 一致性检查位于 **SpeckV 集成层**（`weian_development/speckv/round_pruning_utils.py`），而不是核心 R1KV 类。

**函数**：`verify_rotary_alignment()` 在 `rkv_speckv_generate.py` 中被调用。

| 阶段 | 处理方式 |
|-----|---------|
| 阶段 0 | 复用 SpeckV 集成层的检查 |
| 阶段 1 | 独立实现，放在初始化阶段 |

---

## 5. 我们规划了但 R-KV 没有的需求

| 需求 | 阶段 | 说明 |
|-----|------|------|
| **Batch Size > 1** | 阶段 1 | R-KV 不支持（静默失败），我们阶段 1 必须支持 |
| 内存触发压缩 | 阶段 2 | 在 preemption 之前先尝试压缩 |
| Per-layer 裁剪 | 阶段 0+ | 同层所有 head 共享 token 选择 |
| vLLM 0.15+ 无侵入集成 | 阶段 1 | 更好的兼容性 |
| **Triton Kernel 优化** | 阶段 1 | 2-3x 性能提升 |
| 频率统计打分 | 阶段 0+ | 更稳定，不依赖实时 Query |
| **无 O(n²) 内存问题** | 阶段 0+ | SpeckV 不做相似度计算，避免长序列 OOM |

---

## 6. 各阶段需求清单

### 阶段 0（R-KV 框架内）

**必须**：
- [ ] 基于频率统计的打分（SpeckV 核心）
- [ ] Per-head / Per-layer 裁剪模式
- [ ] RoPE 位置追踪（复用 R-KV）
- [ ] Prefill 保护选项

**可选**：
- [ ] RoPE 一致性检查（复用 R-KV）
- [ ] FP32 TopK

**不需要**：
- Batch Size > 1
- Triton 优化
- 状态重置（R-KV 框架已有）

---

### 阶段 1（独立高效版本）

**必须**：
- [ ] Triton kernel 实现打分、TopK、Gather
- [ ] **Batch Size > 1 支持**
- [ ] RoPE 一致性检查
- [ ] 状态重置接口
- [ ] Per-head / Per-layer 裁剪模式
- [ ] Prefill 保护

**可选**：
- [ ] Union-based 选择
- [ ] 多种聚合策略（mean/max）

---

### 阶段 2（高级功能）

**计划**：
- [ ] 内存触发压缩
- [ ] CUDA Graph 兼容
- [ ] 更灵活的 Budget 策略
- [ ] TP/PP 支持（可选）

---

## 7. 总结

| 类别 | 数量 | 说明 |
|-----|------|------|
| R-KV 有我们需要的 | 5 项 | RoPE 检查、状态重置、FP32 TopK、Union 选择、GQA 处理 |
| 我们有 R-KV 没有的 | 5 项 | Batch>1、Triton、内存触发、频率打分、无 O(n²) 问题 |
| 验证发现的 R-KV 问题 | 4 项 | 静默 batch 失败、长序列 OOM、代码版本不一致、无 TP/PP |
| 不适用我们的 | 2 项 | Query cache、Noise injection |

**关键差异**：
1. **阶段 1 必须支持 Batch Size > 1**（R-KV 静默失败，不会报错）
2. **阶段 1 必须达到 Triton 级别效率**（R-KV 无法达到，已验证）
3. **不使用 Noise injection**（打平时确定性选择，与 R-KV 实际行为一致）
4. **SpeckV 无 O(n²) 内存问题**（不做相似度计算）

**验证确认**：
- R-KV 确实没有 Triton/CUDA kernel（全部 native PyTorch）
- R-KV 确实没有实现 noise injection（之前分析有误，已修正）
- R-KV 的 batch_size=1 限制是静默的（使用 `key_states[0]`）

---

*创建日期：2025-01-31*
*更新日期：2025-01-31（添加阶段划分，移除不适用需求，添加验证发现的新问题）*
