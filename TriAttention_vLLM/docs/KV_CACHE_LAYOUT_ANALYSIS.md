# KV Cache 布局分析与裁剪策略

本文档分析 vLLM 的 KV cache 布局，以及裁剪 KV cache 时可能遇到的问题和解决方案。

---

## 1. vLLM KV Cache 当前假设

### 1.1 连续性假设

vLLM 的 PagedAttention 有以下**隐式假设**：

| 假设 | 说明 |
|-----|------|
| **Page 内连续** | 一个 block 内的 token 位置是连续的：`[i, i+1, i+2, ..., i+block_size-1]` |
| **Page 间有序** | block_table 中的 block 按序列顺序排列 |
| **位置可推算** | 第 `i` 个 token 的位置就是 `i`（从 0 开始） |

### 1.2 Attention 计算中的位置使用

```python
# vLLM attention kernel 伪代码
def attention_kernel(query, key_cache, value_cache, block_table, seq_len):
    for block_idx in range(num_blocks):
        block_id = block_table[block_idx]
        k_block = key_cache[block_id]   # [block_size, num_heads, head_dim]
        v_block = value_cache[block_id]

        # 注意：这里 k_block 已经是 RoPE 旋转后的！
        # RoPE 在写入 KV cache 时就已经应用了

        # Attention 计算
        attn_scores = query @ k_block.T  # 直接点积，不再考虑位置
        attn_output += softmax(attn_scores) @ v_block
```

### 1.3 关键观察

**RoPE 在写入时已经应用**：
- Key 写入 KV cache 时：`k_cache[slot] = apply_rope(k, position)`
- Attention 计算时：直接用旋转后的 key，不再需要位置信息

**这意味着**：
- KV cache 中存储的是 **已旋转的** key
- Attention kernel **不关心** token 的原始位置
- 只要 key 在写入时用了正确的位置做 RoPE，后续计算就是正确的

---

## 2. 裁剪后的连续性问题

### 2.1 场景分析

假设原始序列有 10 个 token（位置 0-9），裁剪后保留位置 [0, 1, 5, 7, 9]：

**裁剪前**：
```
Page 0: [pos=0, pos=1, pos=2, pos=3]  (block_size=4)
Page 1: [pos=4, pos=5, pos=6, pos=7]
Page 2: [pos=8, pos=9, -, -]
```

**裁剪后（如果不整理）**：
```
Page 0: [pos=0, pos=1, -, -]   # 位置 2,3 被裁剪，留下空洞
Page 1: [-, pos=5, -, pos=7]  # 位置 4,6 被裁剪
Page 2: [-, pos=9, -, -]      # 位置 8 被裁剪
```

### 2.2 问题 1：Attention 计算是否受影响？

**答案：不受影响**（如果正确处理）

因为：
1. KV cache 中的 key 已经是用正确位置旋转过的
2. Attention kernel 只做 `Q @ K^T`，不关心位置
3. 只要读取到正确的 K/V，计算就是对的

**但是**：需要有办法告诉 kernel 哪些 slot 是有效的，哪些是空洞。

### 2.3 问题 2：空洞如何处理？

**方案 A：Mask 方式**（保持原位置，标记空洞）

```python
# 维护一个 valid_mask
valid_mask[block_id, slot_offset] = True/False

# Attention kernel 中
attn_scores = query @ k_block.T
attn_scores = attn_scores.masked_fill(~valid_mask, -inf)  # mask 掉无效位置
```

**优点**：
- 不需要移动数据
- 实现简单

**缺点**：
- 浪费显存（空洞占用空间）
- 需要修改 attention kernel
- 空洞越多，效率越低

**方案 B：Compaction 方式**（压缩，消除空洞）

```python
# 裁剪后重新整理
# 把保留的 token 压缩到连续空间
# 释放空出来的 block

# 裁剪前：[pos=0, pos=1, -, -, -, pos=5, -, pos=7, -, pos=9]
# 压缩后：[pos=0, pos=1, pos=5, pos=7, pos=9, -, -, -, -, -]
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         有效数据连续存储
```

**优点**：
- 不浪费显存
- 不需要修改 attention kernel
- 可以释放空 block

**缺点**：
- 需要数据移动（额外开销）
- 需要额外存储原始位置（position_indices）

### 2.4 问题 3：位置信息丢失？

**裁剪 + Compaction 后，原始位置信息丢失了**：

```
压缩后的 KV cache：[k_0, k_1, k_5, k_7, k_9]  # 存储位置 0,1,2,3,4
原始位置：         [0,   1,   5,   7,   9  ]  # 需要额外记录
```

**这就是为什么需要 `position_indices`**：
- 压缩后的存储位置 ≠ 原始序列位置
- TriAttention 打分需要知道原始位置来计算 `(t - p)`

---

## 3. 推荐策略：Compaction + Position Indices

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    KV Cache (Compacted)                 │
├─────────────────────────────────────────────────────────┤
│ Block 0: [k_0, k_1, k_5, k_7]                           │
│ Block 1: [k_9, -, -, -]                                 │
│ Block 2: [-, -, -, -]  ← 可以释放                        │
├─────────────────────────────────────────────────────────┤
│              Position Indices (同步维护)                 │
├─────────────────────────────────────────────────────────┤
│ Block 0: [0, 1, 5, 7]   ← 原始位置                       │
│ Block 1: [9, -, -, -]                                   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Compaction 流程

```python
def compact_kv_cache(kv_cache, position_indices, keep_mask):
    """
    裁剪后压缩 KV cache

    Args:
        kv_cache: [num_blocks, block_size, num_heads, head_dim]
        position_indices: [num_blocks, block_size]
        keep_mask: [total_tokens] - True 表示保留

    Returns:
        compacted_kv_cache, compacted_position_indices, new_num_tokens
    """
    # 1. 收集要保留的 token
    keep_indices = keep_mask.nonzero()

    # 2. 按顺序重新排列（保持原始顺序）
    # 注意：保留的 token 之间的相对顺序不变

    # 3. 写入新位置
    new_slot = 0
    for old_slot in keep_indices:
        old_block = old_slot // block_size
        old_offset = old_slot % block_size
        new_block = new_slot // block_size
        new_offset = new_slot % block_size

        # 移动 KV
        kv_cache[new_block, new_offset] = kv_cache[old_block, old_offset]
        # 移动 position
        position_indices[new_block, new_offset] = position_indices[old_block, old_offset]

        new_slot += 1

    # 4. 更新 block_table，释放空 block
    new_num_blocks = (new_slot + block_size - 1) // block_size
    # 释放 block new_num_blocks 之后的 block

    return new_num_tokens
```

### 3.3 为什么保持顺序？

**裁剪后的 token 之间的相对顺序必须保持不变**：

```
原始：    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
保留：    [0, 1,       5,    7,    9]
压缩后：  [0, 1, 5, 7, 9]  ← 顺序不变
```

**原因**：
1. Causal attention 依赖顺序（token 只能 attend 到之前的 token）
2. 如果顺序乱了，attention mask 会出错

---

## 4. Attention Kernel 兼容性分析

### 4.1 vLLM Flash Attention

```python
# vllm/v1/attention/backends/flash_attn.py
def forward(query, key_cache, value_cache, ...):
    # Flash attention 接收：
    # - query: [num_tokens, num_heads, head_dim]
    # - key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    # - block_table: [num_seqs, max_num_blocks]
    # - seq_lens: [num_seqs]

    # 它通过 block_table 和 seq_lens 知道每个序列有多少有效 token
    # 不关心 token 的原始位置
```

**兼容性**：✅ 兼容

只要：
1. `seq_lens` 更新为压缩后的长度
2. `block_table` 正确反映压缩后的 block 布局
3. KV cache 中的数据是正确的（已旋转的 key）

### 4.2 vLLM Triton Attention

```python
# vllm/v1/attention/backends/triton_attn.py
# 类似的接口，通过 block_table 和 seq_lens 访问 KV
```

**兼容性**：✅ 兼容

### 4.3 需要修改的地方

| 组件 | 是否需要修改 | 说明 |
|-----|------------|------|
| Attention kernel | ❌ 不需要 | 只要 block_table 和 seq_lens 正确 |
| Block allocator | ⚠️ 可能需要 | 支持释放部分 block |
| Scheduler | ⚠️ 可能需要 | 感知压缩后的 seq_len |
| Model runner | ✅ 需要 | 触发压缩、更新 metadata |

---

## 5. 潜在 Bug 风险

### 5.1 风险 1：RoPE 位置错误

**场景**：新 token decode 时，RoPE 用了错误的位置

```python
# 压缩前：seq_len = 10，新 token 位置 = 10
# 压缩后：有效 token = 5，但新 token 位置应该还是 10（不是 5）
```

**解决**：
- 维护 `actual_seq_len`（真实序列长度，包括被裁剪的）
- 新 token 的 RoPE 位置 = `actual_seq_len`，不是压缩后的长度

### 5.2 风险 2：Attention mask 错误

**场景**：Causal mask 基于压缩后的位置

```python
# 压缩后的存储位置：[0, 1, 2, 3, 4]
# 原始位置：         [0, 1, 5, 7, 9]
#
# 如果新 token 位置 = 10，它应该能 attend 到所有保留的 token
# Causal mask 基于原始位置：10 > 0, 1, 5, 7, 9 ✓
```

**解决**：
- Causal mask 不需要特殊处理（只要顺序正确）
- Flash attention 的 causal mask 基于存储顺序，压缩后顺序保持不变，所以没问题

### 5.3 风险 3：Block 释放时机错误

**场景**：释放了还在使用的 block

**解决**：
- 压缩完成后再释放
- 使用引用计数或锁

### 5.4 风险 4：并发访问

**场景**：压缩过程中有其他操作访问 KV cache

**解决**：
- 压缩在 attention 计算之外进行
- 适当的同步机制

---

## 6. 推荐实现方案

### 6.1 数据结构

```python
class TriAttentionKVManager:
    def __init__(self, ...):
        # 原有 vLLM 结构
        self.kv_cache = ...          # [num_blocks, 2, block_size, num_heads, head_dim]
        self.block_table = ...       # [max_num_reqs, max_num_blocks]

        # TriAttention 额外结构
        self.position_indices = ...   # [num_blocks, block_size]
        self.actual_seq_lens = ...    # [max_num_reqs] - 真实序列长度
        self.cached_seq_lens = ...    # [max_num_reqs] - 压缩后的 KV 数量
```

### 6.2 关键操作

| 操作 | 更新内容 |
|-----|---------|
| Prefill | kv_cache, position_indices, actual_seq_lens, cached_seq_lens |
| Decode | kv_cache, position_indices, actual_seq_lens++, cached_seq_lens++ |
| Prune | 不变（只标记要删除的） |
| Compact | kv_cache 重排, position_indices 重排, cached_seq_lens 更新, 释放空 block |

### 6.3 Compaction 触发时机

```python
# 选项 1：每次裁剪后立即 compact
# 优点：简单，显存效率高
# 缺点：频繁数据移动

# 选项 2：累积到一定程度再 compact
# 优点：减少数据移动
# 缺点：临时浪费显存

# 选项 3：显存紧张时 compact
# 优点：按需压缩
# 缺点：可能在关键时刻增加延迟

# 推荐：选项 1（每次裁剪后立即 compact）
# 原因：裁剪本身就是为了省显存，立即 compact 才能释放 block
```

---

## 7. 待确认问题

1. vLLM 的 block allocator 是否支持部分释放 block？
2. CUDA Graph 模式下，KV cache 布局变化是否会导致问题？
3. Speculative decoding 是否有额外的假设？
4. Prefix caching 是否受影响？

---

## 8. 替代方案：Fill-in-Place（原地填充）

### 8.1 方案描述

**你的方案**：
- 每 128 轮压缩一次
- Budget 2048，触发时从 2048+128 压回 2048
- 裁剪掉的 128 个位置形成空洞
- 后续 decode 的新 token 填入空洞
- 不做 compaction，Page 内 KV 变成乱序

**示例**：
```
触发压缩前：[k0, k1, k2, k3, ..., k2175]  (2048+128 个)
裁剪后：    [k0, k1, _, k3, k4, _, k6, ...]  (保留 2048 个，128 个空洞)
新 decode： [k0, k1, k2176, k3, k4, k2177, k6, ...]  (新 token 填入空洞)
                    ^^^^^        ^^^^^
                    乱序了！
```

### 8.2 Decode 阶段 Causal Mask 分析

**Flash Attention 的 causal mask 实现**：
- `causal=True` 时，对于 query 位置 `i` 和 key 位置 `j`：`mask[i][j] = 1 if i >= j else 0`
- 这里的 `i` 和 `j` 是**存储索引**，不是原始序列位置

**Decode 阶段的特点**：
- Query 只有 1 个 token（最新生成的）
- Query 的"位置"是 `seq_len`（存储的 token 数量）
- 所有 KV cache 中的 token 存储索引都 < `seq_len`
- 因此 causal mask 全是 1，**所有 key 都会被 attend**

**结论**：✅ **Decode 阶段 causal mask 没问题**

因为：
1. Decode 时 query 永远是"最新的"
2. 所有 KV cache 都是"之前的"（不管存储顺序如何）
3. Causal mask 不会 mask 掉任何 key

### 8.3 Attention 计算正确性

**Attention 计算本质**：
```
output = softmax(Q @ K^T / sqrt(d)) @ V
```

**乱序影响分析**：
- `Q @ K^T`：点积，不关心 K 的顺序
- `softmax`：对 attention scores 归一化，不关心顺序
- `@ V`：加权求和，只要 K 和 V 的对应关系正确，结果就正确

**结论**：✅ **Attention 计算正确**

KV 乱序存储只是重新排列了矩阵的行，不影响数学结果。

### 8.4 Prefill 阶段的限制

**Prefill 时 causal mask 很重要**：
- Query 有多个 token：`[q0, q1, q2, ..., q_prefill_len]`
- 每个 query 只能 attend 到位置 <= 它的 key
- `q5` 只能 attend 到 `[k0, k1, k2, k3, k4, k5]`

**如果 Prefill 时 KV 乱序**：
- 假设存储顺序是 `[k0, k5, k2, k3, k1, k4]`
- Flash Attention 会错误地认为：
  - `k5` 的位置是 1
  - `k1` 的位置是 4
- Causal mask 会出错

**解决方案**：**Prefill 时不触发压缩**

只要保证：
1. Prefill 阶段 KV 顺序写入
2. 压缩只在 decode 阶段触发
3. 压缩后新 decode 的 token 填入空洞

### 8.5 方案可行性总结

| 场景 | 是否有问题 | 原因 |
|-----|----------|------|
| Decode causal mask | ✅ 没问题 | Query 永远是最新的，所有 key 都被 attend |
| Attention 计算 | ✅ 没问题 | 点积和加权求和不关心顺序 |
| Prefill causal mask | ⚠️ 需要注意 | 必须保证 prefill 时 KV 顺序存储 |
| RoPE | ✅ 没问题 | RoPE 在写入时已应用，用的是正确的原始位置 |

### 8.6 实现要点

```python
class TriAttentionKVManager:
    def __init__(self, budget, divide_length=128):
        self.budget = budget
        self.divide_length = divide_length
        self.free_slots = []  # 空洞列表

    def should_prune(self, current_len):
        # 只在 decode 阶段、超过 budget 时触发
        return (
            current_len > self.budget and
            not self.is_prefill_phase and
            current_len % self.divide_length == 0
        )

    def prune(self, scores, keep_count):
        # 选择要裁剪的 token
        _, prune_indices = scores.topk(len(scores) - keep_count, largest=False)

        # 标记空洞（不移动数据）
        for idx in prune_indices:
            self.free_slots.append(idx)

        # 清空被裁剪位置的 KV（可选，用于调试）
        # self.kv_cache[prune_indices] = 0

    def write_new_kv(self, k, v, position):
        if self.free_slots:
            # 填入空洞
            slot = self.free_slots.pop(0)
        else:
            # 追加到末尾
            slot = self.current_len
            self.current_len += 1

        self.kv_cache[slot] = (k, v)
        self.position_indices[slot] = position

    def get_valid_mask(self):
        # 返回哪些 slot 是有效的
        mask = torch.ones(self.max_len, dtype=torch.bool)
        for slot in self.free_slots:
            mask[slot] = False
        return mask
```

### 8.7 与 Compaction 方案对比

| 方面 | Fill-in-Place | Compaction |
|-----|---------------|------------|
| 数据移动 | ❌ 无 | ✅ 需要移动 |
| 显存利用 | ⚠️ 有空洞 | ✅ 紧凑 |
| Block 释放 | ❌ 不能释放 | ✅ 可以释放 |
| 实现复杂度 | ✅ 简单 | ⚠️ 较复杂 |
| Valid mask | ⚠️ 需要维护 | ✅ 不需要 |

### 8.8 Fill-in-Place 的限制

1. **不能释放 block**：空洞分散在各个 block 中，无法释放整个 block
2. **需要 valid mask**：attention kernel 需要知道哪些 slot 是空洞
3. **碎片化**：长期运行后空洞分布可能很分散

**适用场景**：
- 短序列（空洞不多）
- 不需要释放 block 回收显存的场景
- 追求实现简单的场景

---

## 9. 推荐：混合方案

结合两种方案的优点：

1. **日常 decode**：使用 Fill-in-Place，新 token 填入空洞
2. **定期整理**：当空洞超过阈值时（如 50%），触发一次 Compaction
3. **显存紧张时**：触发 Compaction 释放 block

```python
def maybe_compact(self):
    fragmentation = len(self.free_slots) / self.current_len
    if fragmentation > 0.5:  # 空洞超过 50%
        self.do_compaction()
```

---

*文档版本：1.1*
*创建日期：2025-01-30*
*更新：添加 Fill-in-Place 方案分析*
