# TriAttention 额外存储需求

本文档列出 TriAttention 算法相比原始 vLLM 需要额外存储的变量。

---

## 1. vLLM 已有的变量

### 1.1 InputBatch 中的位置信息

| 变量 | 形状 | 说明 |
|-----|------|------|
| `positions` | `[num_tokens]` | 当前 batch 中每个 token 的位置 |
| `seq_lens` | `[num_reqs]` | 每个 request 的序列长度 |
| `query_start_loc` | `[num_reqs + 1]` | 每个 request 在 batch 中的起始位置 |

### 1.2 BlockTables 中的 KV 管理信息

| 变量 | 形状 | 说明 |
|-----|------|------|
| `block_tables` | `[max_num_reqs, max_num_blocks]` | 每个 request 使用的 block ID 列表 |
| `slot_mappings` | `[num_kv_cache_groups, max_num_tokens]` | token 位置 → KV cache slot 映射 |
| `num_blocks` | `[num_kv_cache_groups, max_num_reqs]` | 每个 request 使用的 block 数量 |

### 1.3 关键观察

**vLLM 不存储 KV cache 中每个 token 的原始位置**。

vLLM 假设 KV cache 是按序列顺序连续存储的，位置可以通过以下公式推算：

```python
# 对于 request r 的第 i 个 token
position = i  # 假设连续存储
slot_id = block_table[r][i // block_size] * block_size + i % block_size
```

**但是**，一旦做了 KV cache 压缩（移除了某些 token），这个假设就不成立了。

---

## 2. TriAttention 额外存储的变量

### 2.1 Per-Token 变量（随 KV cache 存储）

| 变量 | 形状 | 数据类型 | 说明 | 必需性 |
|-----|------|---------|------|--------|
| `position_indices` | `[num_cached_tokens]` | bf16/int32 | 每个 KV 的原始序列位置 $p$ | **必需** |

**存储方式**：与 KV cache 一起存储，每个 token 一个位置值。

**数据类型选择**：
- 支持 bf16 的 GPU（A100, H100 等）：使用 bf16
- 不支持 bf16 的 GPU（V100 等）：回退到 int32

### 2.2 Per-Request 变量（TriAttentionState）

| 变量 | 形状 | 数据类型 | 说明 | 条件 |
|-----|------|---------|------|------|
| `prefill_len` | `[1]` | int32 | Prefill 长度 | 仅当 `protect_prefill=True` 时维护 |
| `current_budget_used` | `[1]` | int32 | 当前已使用的 KV 数量 | 始终 |
| `last_prune_step` | `[1]` | int32 | 上次裁剪的 step | 始终 |

**注意**：`protect_prefill` 默认为 `False`，此时不需要维护 `prefill_len`。

### 2.3 共享三角函数表（每轮打分动态计算）

| 变量 | 形状 | 数据类型 | 说明 |
|-----|------|---------|------|
| `trig_cos` | `[num_offsets, freq_count]` | bf16/fp16 | $\cos(t \cdot \omega)$ 表 |
| `trig_sin` | `[num_offsets, freq_count]` | bf16/fp16 | $\sin(t \cdot \omega)$ 表 |

**注意**：每轮打分开始时根据当前 `round_start` 和 `offsets` 计算，所有 token 共享。

### 2.4 全局变量（从 stats 文件加载，所有 request 共享）

| 变量 | 形状 | 数据类型 | 说明 |
|-----|------|---------|------|
| `Q_mean_real` | `[num_layers, num_heads, freq_count]` | bf16/fp16 | 平均 query 实部 |
| `Q_mean_imag` | `[num_layers, num_heads, freq_count]` | bf16/fp16 | 平均 query 虚部 |
| `freq_scale_sq` | `[num_layers, num_heads, freq_count]` | bf16/fp16 | 频率缩放因子 $s_f^2$ |
| `omega` | `[freq_count]` | fp32 | RoPE 频率 |
| `extra_coef` | `[num_layers, num_heads, freq_count]` | bf16/fp16 | 位置无关项系数 |

**注意**：这些是从 stats 文件预加载的，不随 KV cache 变化。

---

## 3. 显存开销估算

### 3.1 Per-Token 开销

假设配置：
- `num_cached_tokens = 8192`
- `num_kv_heads = 8`

| 变量 | 计算 | 显存 |
|-----|------|------|
| `position_indices` (bf16) | 8192 × 2 bytes | **16 KB** |
| `position_indices` (int32) | 8192 × 4 bytes | **32 KB** |

### 3.2 共享三角函数表开销

假设配置：
- `num_offsets = 16`
- `freq_count = 64`

| 变量 | 计算 | 显存 |
|-----|------|------|
| `trig_cos` | 16 × 64 × 2 bytes | 2 KB |
| `trig_sin` | 16 × 64 × 2 bytes | 2 KB |
| **合计** | | **4 KB** |

### 3.3 全局 Stats 开销

假设配置：
- `num_layers = 32`
- `num_heads = 32`
- `freq_count = 64`（head_dim / 2）

| 变量 | 计算 | 显存 |
|-----|------|------|
| `Q_mean_real` | 32 × 32 × 64 × 2 bytes | 128 KB |
| `Q_mean_imag` | 32 × 32 × 64 × 2 bytes | 128 KB |
| `freq_scale_sq` | 32 × 32 × 64 × 2 bytes | 128 KB |
| `omega` | 64 × 4 bytes | 0.25 KB |
| `extra_coef` | 32 × 32 × 64 × 2 bytes | 128 KB |
| **合计** | | **~512 KB** |

### 3.4 典型配置开销估算

**模型配置**（以 Qwen2.5-7B 为例）：
- `num_layers = 28`
- `num_kv_heads = 4`（GQA）
- `head_dim = 128`
- `freq_count = 64`（head_dim / 2）

**Budget 4K（4096 tokens）**：

| 组件 | 计算 | 显存 |
|-----|------|------|
| `position_indices` (bf16) | 4096 × 2 bytes | **8 KB** |
| `trig_cos/sin` | 16 × 64 × 2 × 2 bytes | **4 KB** |
| 全局 Stats | 28 × 4 × 64 × 5 × 2 bytes | **~70 KB** |
| **TriAttention 额外总计** | | **~82 KB** |
| KV cache 本身 | 4096 × 28 × 4 × 128 × 2 × 2 bytes | **~230 MB** |
| **额外开销占比** | 82 KB / 230 MB | **< 0.04%** |

**Budget 8K（8192 tokens）**：

| 组件 | 计算 | 显存 |
|-----|------|------|
| `position_indices` (bf16) | 8192 × 2 bytes | **16 KB** |
| `trig_cos/sin` | 16 × 64 × 2 × 2 bytes | **4 KB** |
| 全局 Stats | 28 × 4 × 64 × 5 × 2 bytes | **~70 KB** |
| **TriAttention 额外总计** | | **~90 KB** |
| KV cache 本身 | 8192 × 28 × 4 × 128 × 2 × 2 bytes | **~460 MB** |
| **额外开销占比** | 90 KB / 460 MB | **< 0.02%** |

### 3.5 总结

| Budget | TriAttention 额外开销 | KV cache 本身 | 占比 |
|--------|---------------------|--------------|------|
| 4K | ~82 KB | ~230 MB | < 0.04% |
| 8K | ~90 KB | ~460 MB | < 0.02% |

**结论**：TriAttention 的额外显存开销可忽略不计。

---

## 4. 与 vLLM 现有变量的关系

### 4.1 能否复用 vLLM 的 `positions`？

**不能**。`positions` 是当前 batch 的 token 位置，不是 KV cache 中 token 的位置。

```python
# vLLM 的 positions：当前 batch 中 token 的位置
# 例如：decode 阶段，positions = [seq_len] 表示新 token 的位置

# 我们需要的：KV cache 中每个 token 的原始位置
# 例如：KV cache 存了位置 [0, 1, 5, 10, 15] 的 token（经过裁剪后）
```

### 4.2 能否从 `slot_mapping` 推算位置？

**不能**。`slot_mapping` 只是存储位置（哪个 block 的哪个 slot），不是序列位置。

```python
# slot_id = block_number * block_size + offset_in_block
# 这是 KV cache 的存储地址，不是原始序列位置
```

### 4.3 能否从 `block_table` 推算位置？

**压缩前可以，压缩后不行**。

```python
# 压缩前：位置 i 的 token 存在 block_table[i // block_size] 的第 i % block_size 个 slot
# 压缩后：位置不连续，无法推算
```

---

## 5. 实现建议

### 5.1 position_indices 存储位置

**方案 A**：与 KV cache 一起存储（推荐）

```python
# 扩展 KV cache 结构
# 原来：kv_cache[layer] = (k_cache, v_cache)
# 现在：kv_cache[layer] = (k_cache, v_cache, position_indices)

# position_indices 形状：[num_blocks, block_size]
# 与 k_cache/v_cache 的前两维对齐
```

**方案 B**：单独的 tensor

```python
# 为每个 request 维护一个 position_indices tensor
# position_indices[req_id] = [p_0, p_1, ..., p_n]
```

### 5.2 更新时机

| 事件 | position_indices 操作 |
|-----|----------------------|
| Prefill | 初始化为 `[0, 1, 2, ..., prefill_len-1]` |
| Decode | 追加新位置 `append(current_seq_len)` |
| Prune | 移除被裁剪 token 的位置，保留的 token 位置不变 |

---

## 6. ⚠️ 重要：PagedAttention Page 组织

**所有额外存储的变量都必须按 PagedAttention 的 page（block）方式组织**，以保持与 vLLM KV cache 管理的一致性。

### 6.1 需要按 Page 组织的变量

| 变量 | 原始形状 | Page 组织后形状 | 说明 |
|-----|---------|----------------|------|
| `position_indices` | `[num_tokens]` | `[num_blocks, block_size]` | 与 KV cache block 对齐 |
| `trig_cos` | `[num_offsets, freq_count]` | 可选，较小可不分页 | 共享表 |
| `trig_sin` | `[num_offsets, freq_count]` | 可选，较小可不分页 | 共享表 |

### 6.2 Page 组织的好处

1. **与 KV cache 同步**：裁剪 KV 时，对应的 position_indices 自动跟随
2. **block 回收**：释放 KV block 时，对应的 position_indices block 一起释放
3. **内存对齐**：利用 vLLM 的 block allocator，减少碎片
4. **Kernel 访问模式一致**：使用相同的 block_table 索引

### 6.3 实现建议

```python
# position_indices 与 KV cache 使用相同的 block 布局
class TriAttentionKVCache:
    def __init__(self, num_blocks, block_size, ...):
        # KV cache: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.zeros(...)

        # position_indices: [num_blocks, block_size]
        # 与 kv_cache 的前两维对齐
        self.position_indices = torch.zeros(
            num_blocks, block_size,
            dtype=torch.bfloat16 if bf16_supported else torch.int32
        )

    def write_kv(self, block_id, slot_offset, k, v, position):
        self.kv_cache[block_id, 0, slot_offset] = k
        self.kv_cache[block_id, 1, slot_offset] = v
        self.position_indices[block_id, slot_offset] = position

    def prune_and_compact(self, keep_mask):
        # 裁剪时，kv_cache 和 position_indices 一起处理
        ...
```

### 6.4 全局 Stats 的处理

全局 stats（`Q_mean_real`、`Q_mean_imag`、`freq_scale_sq` 等）**不需要按 page 组织**：
- 它们是只读的，模型加载时一次性分配
- 不随 KV cache 变化
- 所有 request 共享

### 6.5 共享三角函数表的处理

`trig_cos` 和 `trig_sin` 表：
- 大小固定：`[num_offsets, freq_count]`（通常 16 × 64 = 4 KB）
- 每轮打分动态计算
- **不需要按 page 组织**（太小，不值得）
- 可以放在共享内存或寄存器中

---

*文档版本：1.1*
*创建日期：2025-01-30*
*更新：添加 trig table 和 page 组织要求*
