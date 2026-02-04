# 数据结构设计

本文档详细说明 TriAttention 需要额外存储的数据结构。

> **⚠️ 重要更新 (2026-02-03)**:
> - `position_indices` 已废弃，不再用于 Triton kernel 和核心算法
> - 保留在 API 中仅为向后兼容，实际实现中使用内部位置追踪
> - 新实现请参考 `triattention/state.py` 中的 `CompressionState` 类

---

## 1. 与 vLLM 现有变量的关系

### 1.1 vLLM 已有的变量

| 变量 | 形状 | 说明 |
|-----|------|------|
| `positions` | `[num_tokens]` | 当前 batch 中每个 token 的位置 |
| `seq_lens` | `[num_reqs]` | 每个 request 的序列长度 |
| `block_tables` | `[max_num_reqs, max_num_blocks]` | 每个 request 的 block ID 列表 |
| `slot_mappings` | `[num_kv_cache_groups, max_num_tokens]` | token 位置 → KV cache slot 映射 |

### 1.2 关键观察

**vLLM 不存储 KV cache 中每个 token 的原始位置**。

vLLM 假设 KV cache 是按序列顺序连续存储的，位置可以推算：

```python
position = i  # 第 i 个 token 的位置就是 i
```

但是，一旦做了 KV cache 压缩（移除了某些 token），这个假设就不成立了。

### 1.3 为什么不能复用 vLLM 的变量

| 变量 | 能否复用 | 原因 |
|-----|---------|------|
| `positions` | 否 | 这是当前 batch 的 token 位置，不是 KV cache 中的位置 |
| `slot_mapping` | 否 | 这是存储地址，不是序列位置 |
| `block_table` | 部分 | 压缩前可以推算位置，压缩后不行 |

---

## 2. TriAttention 额外存储的变量

### 2.1 Per-Token 变量（已废弃）

> **⚠️ 已废弃 (Deprecated)**: `position_indices` 不再用于实际实现。

| 变量 | 形状 | 数据类型 | 说明 | 状态 |
|-----|------|---------|------|------|
| `position_indices` | `[num_cached_tokens]` | bf16/int32 | 每个 KV 的原始序列位置 | **已废弃** |

**废弃原因**：
- Triton kernel 不再需要显式的 position_indices 参数
- 位置追踪通过 `CompressionState` 内部管理更高效
- 减少了显存开销和数据传输

**当前实现**：
位置信息通过 `triattention/state.py` 中的 `CompressionState` 类管理：
```python
class CompressionState:
    """Per-request compression state tracking."""
    current_cache_len: int  # 当前 cache 长度
    last_compress_trigger: int  # 上次压缩触发位置
    # ... 其他状态
```

**向后兼容**：
部分 API 仍保留 `position_indices` 参数以保持接口兼容性，但实际不使用。

### 2.2 Per-Request 变量

| 变量 | 形状 | 数据类型 | 说明 | 条件 |
|-----|------|---------|------|------|
| `prefill_len` | `[1]` | int32 | Prefill 长度 | 仅当 `protect_prefill=True` |
| `current_budget_used` | `[1]` | int32 | 当前已使用的 KV 数量 | 始终 |
| `last_prune_step` | `[1]` | int32 | 上次裁剪的 step | 始终 |

### 2.3 共享三角函数表

| 变量 | 形状 | 数据类型 | 说明 |
|-----|------|---------|------|
| `trig_cos` | `[num_offsets, freq_count]` | bf16/fp16 | $\cos(t \cdot \omega)$ 表 |
| `trig_sin` | `[num_offsets, freq_count]` | bf16/fp16 | $\sin(t \cdot \omega)$ 表 |

每轮打分开始时根据当前 `round_start` 和 `offsets` 动态计算，所有 token 共享。

### 2.4 全局 Stats（从 stats 文件加载）

| 变量 | 形状 | 数据类型 | 说明 |
|-----|------|---------|------|
| `Q_mean_real` | `[num_layers, num_heads, freq_count]` | bf16/fp16 | 平均 query 实部 |
| `Q_mean_imag` | `[num_layers, num_heads, freq_count]` | bf16/fp16 | 平均 query 虚部 |
| `freq_scale_sq` | `[num_layers, num_heads, freq_count]` | bf16/fp16 | 频率缩放因子 $s_f^2$ |
| `omega` | `[freq_count]` | fp32 | RoPE 频率 |
| `extra_coef` | `[num_layers, num_heads, freq_count]` | bf16/fp16 | 位置无关项系数 |

这些是从 stats 文件预加载的，不随 KV cache 变化，所有 request 共享。

---

## 3. 显存开销估算

### 3.1 典型配置（Qwen2.5-7B）

- `num_layers = 28`
- `num_kv_heads = 4`（GQA）
- `head_dim = 128`
- `freq_count = 64`（head_dim / 2）

### 3.2 Budget 4K（4096 tokens）

| 组件 | 计算 | 显存 |
|-----|------|------|
| `position_indices` (bf16) | 4096 × 2 bytes | **8 KB** |
| `trig_cos/sin` | 16 × 64 × 2 × 2 bytes | **4 KB** |
| 全局 Stats | 28 × 4 × 64 × 5 × 2 bytes | **~70 KB** |
| **TriAttention 额外总计** | | **~82 KB** |
| KV cache 本身 | 4096 × 28 × 4 × 128 × 2 × 2 bytes | **~230 MB** |
| **额外开销占比** | 82 KB / 230 MB | **< 0.04%** |

### 3.3 Budget 8K（8192 tokens）

| 组件 | 计算 | 显存 |
|-----|------|------|
| `position_indices` (bf16) | 8192 × 2 bytes | **16 KB** |
| `trig_cos/sin` | 16 × 64 × 2 × 2 bytes | **4 KB** |
| 全局 Stats | 28 × 4 × 64 × 5 × 2 bytes | **~70 KB** |
| **TriAttention 额外总计** | | **~90 KB** |
| KV cache 本身 | 8192 × 28 × 4 × 128 × 2 × 2 bytes | **~460 MB** |
| **额外开销占比** | 90 KB / 460 MB | **< 0.02%** |

### 3.4 总结

| Budget | TriAttention 额外开销 | KV cache 本身 | 占比 |
|--------|---------------------|--------------|------|
| 4K | ~82 KB | ~230 MB | < 0.04% |
| 8K | ~90 KB | ~460 MB | < 0.02% |

**结论**：TriAttention 的额外显存开销可忽略不计。

---

## 4. PagedAttention Page 组织

**所有额外存储的变量都必须按 PagedAttention 的 page（block）方式组织**。

### 4.1 需要按 Page 组织的变量

> **⚠️ 注意**: `position_indices` 已废弃，以下内容仅供参考。

| 变量 | 原始形状 | Page 组织后形状 | 状态 |
|-----|---------|----------------|------|
| `position_indices` | `[num_tokens]` | `[num_blocks, block_size]` | **已废弃** |

### 4.2 Page 组织的好处

1. **与 KV cache 同步**：裁剪 KV 时，对应的 position_indices 自动跟随
2. **block 回收**：释放 KV block 时，对应的 position_indices block 一起释放
3. **内存对齐**：利用 vLLM 的 block allocator，减少碎片
4. **Kernel 访问模式一致**：使用相同的 block_table 索引

### 4.3 实现示例（已废弃）

> **⚠️ 以下代码仅供历史参考，当前实现不使用 position_indices**

```python
# DEPRECATED: 旧的 position_indices 实现方式
class TriAttentionKVCache:
    def __init__(self, num_blocks, block_size, num_kv_heads, head_dim, bf16_supported):
        # KV cache: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.zeros(
            num_blocks, 2, block_size, num_kv_heads, head_dim,
            dtype=torch.bfloat16
        )

        # position_indices: [num_blocks, block_size] (已废弃)
        # 与 kv_cache 的前两维对齐
        self.position_indices = torch.zeros(
            num_blocks, block_size,
            dtype=torch.int32
        )

    def write_kv(self, block_id, slot_offset, k, v, position):
        self.kv_cache[block_id, 0, slot_offset] = k
        self.kv_cache[block_id, 1, slot_offset] = v
        self.position_indices[block_id, slot_offset] = position
```

**当前推荐实现**:
参考 `triattention/state.py` 和 `triattention/compressor.py` 中的实际实现。

### 4.4 不需要按 Page 组织的变量

| 变量 | 原因 |
|-----|------|
| 全局 Stats | 只读，模型加载时一次性分配，所有 request 共享 |
| `trig_cos/sin` | 太小（约 4 KB），每轮动态计算 |

---

## 5. Position Indices 更新时机（已废弃）

> **⚠️ 已废弃**: 以下内容仅供历史参考

| 事件 | position_indices 操作（已废弃） |
|-----|----------------------|
| Prefill | 初始化为 `[0, 1, 2, ..., prefill_len-1]` |
| Decode | 追加新位置 `append(current_seq_len)` |
| Prune | 移除被裁剪 token 的位置，保留的 token 位置不变 |

**当前实现**: 位置信息通过 `CompressionState` 内部管理，不需要显式存储。

---

*文档版本：1.2*
*创建日期：2025-01-30*
*更新历史：*
- *2025-01-31：position_indices 类型对齐 R-KV*
- *2026-02-03：标记 position_indices 为废弃，更新为内部状态管理*
