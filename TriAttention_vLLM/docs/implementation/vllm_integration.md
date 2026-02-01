# vLLM PagedAttention 集成

本文档分析 vLLM 的 KV cache 布局，以及 TriAttention 如何与之集成。

---

## 1. vLLM PagedAttention 机制

### 1.1 KV Cache 布局

```python
# vLLM KV cache shape
kv_cache.shape = (num_blocks, 2, block_size, num_kv_heads, head_size)
#                 ^           ^  ^           ^             ^
#                 物理block数  K/V block内槽位 KV head数    head维度
```

| 维度 | 说明 | 典型值 |
|-----|------|-------|
| `num_blocks` | 物理 block 总数（显存池） | 取决于显存 |
| `2` | K 和 V 分开存储 | 固定 |
| `block_size` | 每个 block 的槽位数 | 16 |
| `num_kv_heads` | KV head 数量 | 4~32 |
| `head_size` | head 维度 | 128 |

### 1.2 Block Table

```python
block_table.shape = (max_num_reqs, max_num_blocks_per_req)
# block_table[req_id, logical_block_idx] = physical_block_id
```

每个 request 维护一个 block_table，将逻辑 block 索引映射到物理 block ID。

### 1.3 Slot Mapping

```python
# 将 token 的逻辑位置映射到物理槽位
physical_slot = block_table[logical_block_idx] * block_size + offset_in_block

# 例如：token 位置 35，block_size=16
logical_block_idx = 35 // 16 = 2
offset_in_block = 35 % 16 = 3
physical_slot = block_table[2] * 16 + 3
```

### 1.4 关键观察

**RoPE 在写入时已经应用**：
- Key 写入 KV cache 时：`k_cache[slot] = apply_rope(k, position)`
- Attention 计算时：直接用旋转后的 key，不再需要位置信息

---

## 2. vLLM 的连续性假设

### 2.1 隐式假设

| 假设 | 说明 |
|-----|------|
| Page 内连续 | block 内的 token 位置是连续的 |
| Page 间有序 | block_table 中的 block 按序列顺序排列 |
| 位置可推算 | 第 `i` 个 token 的位置就是 `i` |

### 2.2 裁剪后的问题

一旦做了 KV cache 压缩，连续性假设不成立：

```
裁剪前：[pos=0, pos=1, pos=2, pos=3, pos=4, pos=5, ...]
裁剪后：[pos=0, pos=1, pos=5, pos=7, pos=9, ...]  ← 不连续！
```

**解决方案**：使用 `position_indices` 显式存储每个 token 的原始位置。

---

## 3. Decode 阶段 Causal Mask 分析

### 3.1 Flash Attention 的 causal mask

- `causal=True` 时，对于 query 位置 `i` 和 key 位置 `j`：`mask[i][j] = 1 if i >= j else 0`
- 这里的 `i` 和 `j` 是**存储索引**，不是原始序列位置

### 3.2 Decode 阶段的特点

- Query 只有 1 个 token（最新生成的）
- Query 的"位置"是 `seq_len`（存储的 token 数量）
- 所有 KV cache 中的 token 存储索引都 < `seq_len`
- 因此 causal mask 全是 1，**所有 key 都会被 attend**

### 3.3 结论

**Decode 阶段 causal mask 没问题**：
1. Query 永远是"最新的"
2. 所有 KV cache 都是"之前的"（不管存储顺序如何）
3. Causal mask 不会 mask 掉任何 key

---

## 4. Attention 计算正确性

### 4.1 Attention 计算本质

```
output = softmax(Q @ K^T / sqrt(d)) @ V
```

### 4.2 乱序影响分析

- `Q @ K^T`：点积，不关心 K 的顺序
- `softmax`：对 attention scores 归一化，不关心顺序
- `@ V`：加权求和，只要 K 和 V 的对应关系正确，结果就正确

### 4.3 结论

**Attention 计算正确**：KV 乱序存储只是重新排列了矩阵的行，不影响数学结果。

---

## 5. Prefill 阶段的限制

### 5.1 Prefill 时 causal mask 很重要

- Query 有多个 token：`[q0, q1, q2, ..., q_prefill_len]`
- 每个 query 只能 attend 到位置 <= 它的 key
- `q5` 只能 attend 到 `[k0, k1, k2, k3, k4, k5]`

### 5.2 如果 Prefill 时 KV 乱序

假设存储顺序是 `[k0, k5, k2, k3, k1, k4]`：
- Flash Attention 会错误地认为 `k5` 的位置是 1
- Causal mask 会出错

### 5.3 解决方案

**Prefill 时不触发压缩**（Phase 1）：
1. Prefill 阶段 KV 顺序写入
2. 压缩只在 decode 阶段触发
3. **Prefill > budget 的处理放到 Phase 2**

---

## 6. Attention Kernel 兼容性

### 6.1 vLLM Flash Attention

```python
# vllm/v1/attention/backends/flash_attn.py
def forward(query, key_cache, value_cache, block_table, seq_lens, ...):
    # 通过 block_table 和 seq_lens 知道每个序列有多少有效 token
    # 不关心 token 的原始位置
```

**兼容性**：✅ 只要 `seq_lens` 和 `block_table` 正确反映压缩后的状态。

### 6.2 vLLM Triton Attention

类似的接口，通过 block_table 和 seq_lens 访问 KV。

**兼容性**：✅

### 6.3 需要修改的地方

| 组件 | 是否需要修改 | 说明 |
|-----|------------|------|
| Attention kernel | ❌ 不需要 | 只要 block_table 和 seq_lens 正确 |
| Block allocator | ⚠️ 可能需要 | 支持释放 overflow pages |
| Model runner | ✅ 需要 | 触发压缩、更新 metadata |

---

## 7. Block Manager 集成

### 7.1 与 vLLM Block Manager 集成

```python
class FillInPlaceBlockManager:
    def __init__(self, vllm_block_manager, budget_pages):
        self.vllm_bm = vllm_block_manager
        self.budget_pages = budget_pages

    def allocate_overflow_page(self):
        """分配一个 overflow page"""
        return self.vllm_bm.allocate_block()

    def release_overflow_pages(self, block_ids):
        """释放 overflow pages 回 free pool"""
        for block_id in block_ids:
            self.vllm_bm.free_block(block_id)

    def prune_and_release(self, state, budget_keep_mask, overflow_keep_mask):
        """执行 Fill-in-Place 并释放 overflow pages"""
        state.prune_and_fill(budget_keep_mask, overflow_keep_mask, ...)

        # 释放 overflow pages
        overflow_block_ids = state.overflow_block_table.tolist()
        self.release_overflow_pages(overflow_block_ids)

        return len(overflow_block_ids)
```

---

## 8. 潜在风险与缓解

### 8.1 风险 1：RoPE 位置错误

**场景**：新 token decode 时，RoPE 用了错误的位置

```python
# 压缩前：seq_len = 10，新 token 位置 = 10
# 压缩后：有效 token = 5，但新 token 位置应该还是 10（不是 5）
```

**解决**：维护 `actual_seq_len`（真实序列长度），新 token 的 RoPE 位置 = `actual_seq_len`。

### 8.2 风险 2：Block 释放时机错误

**场景**：释放了还在使用的 block

**解决**：压缩完成后再释放，使用引用计数或锁。

### 8.3 风险 3：并发访问

**场景**：压缩过程中有其他操作访问 KV cache

**解决**：压缩在 attention 计算之外进行，适当的同步机制。

### 8.4 风险 4：CUDA Graph 不兼容

**场景**：KV cache 布局变化导致 CUDA Graph 失效

**解决**：提供 eager 模式回退。

---

## 9. 待确认问题

1. vLLM 的 block allocator 是否支持部分释放 block？
2. CUDA Graph 模式下，KV cache 布局变化是否会导致问题？
3. Speculative decoding 是否有额外的假设？
4. Prefix caching 是否受影响？
5. 与 vLLM 其他特性的兼容性（Chunked prefill?）

---

*文档版本：1.0*
*创建日期：2025-01-30*
