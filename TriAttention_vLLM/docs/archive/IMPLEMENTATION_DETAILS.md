# TriAttention 实现细节

本文档详细说明关键实现细节，特别是 KV 裁剪后如何维护 PagedAttention 的 page 结构。

---

## 1. vLLM PagedAttention 机制

### 1.1 KV Cache 布局

```python
# vLLM KV cache shape
kv_cache.shape = (num_blocks, 2, block_size, num_kv_heads, head_size)
#                 ^           ^  ^           ^             ^
#                 物理block数  K/V block内槽位 KV head数    head维度
```

- `num_blocks`：物理 block 总数（显存池）
- `2`：K 和 V 分开存储
- `block_size`：每个 block 的槽位数（通常 16）
- `num_kv_heads`：KV head 数量
- `head_size`：head 维度（通常 128）

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

---

## 2. Fill-in-Place 策略

### 2.1 核心思路

- **Budget pages**：固定大小（如 1024 pages），存放压缩后保留的 KV
- **Overflow pages**：临时分配，存放新 decode 出的 token
- **裁剪时**：budget pages 里被删的 token 空出槽位，overflow pages 里幸存的 token **填入**这些空槽
- **裁剪后**：**释放 overflow pages**

### 2.2 工作流程

```
===== 初始状态 =====

Budget pages (固定 1024 pages，容纳 16384 tokens):
Page 0:    [T0, T1, T2, ..., T15]
Page 1:    [T16, T17, T18, ..., T31]
...
Page 1023: [T16368, T16369, ..., T16383]

position_indices: [0, 1, 2, 3, ..., 16383]  # 每个槽位记录原始位置

===== Decode 阶段 =====

新 token 写入 Overflow pages (临时):
Page 1024: [T16384, T16385, T16386, ...]  <- 新 decode 的 token
Page 1025: [T16400, T16401, ...]
...
Page 1031: [T16496, T16497, ..., T16511]  <- 128 个新 token

overflow_position_indices: [16384, 16385, ..., 16511]

===== 裁剪触发（每 128 tokens）=====

1. 对所有 16384 + 128 = 16512 个 token 打分
2. 选出 16384 个保留（塞满 budget pages）

假设打分结果：
- Budget pages 中删除 120 个 token（空出 120 个槽位）
- Overflow pages 中保留 120 个 token（删除 8 个）

===== 执行 Fill-in-Place =====

Step 1: 找出 budget 中被删的槽位
  free_slots = [0, 3, 17, 45, ...]   # 共 120 个空槽

Step 2: 找出 overflow 中幸存的 token
  survivors = [
    (slot=0, pos=16385, k=..., v=...),
    (slot=2, pos=16387, k=..., v=...),
    ...
  ]  # 共 120 个

Step 3: Fill-in（把 survivors 填入 free_slots）
  budget_kv[0]  <- overflow_kv[0]   (T16385)
  budget_kv[3]  <- overflow_kv[2]   (T16387)
  budget_kv[17] <- overflow_kv[5]   (T16390)
  ...

  position_indices[0]  = 16385
  position_indices[3]  = 16387
  position_indices[17] = 16390
  ...

Step 4: 释放 overflow pages
  Page 1024~1031 归还 free pool
```

### 2.3 数据移动量分析

**只移动 overflow 幸存者**，不移动 budget 中保留的 token：

| 配置 | 数据移动量 |
|-----|-----------|
| budget=16K, divide=128, head_dim=128, bf16 | 128 × 128 × 2 = **32 KB** |
| budget=8K, divide=64, head_dim=128, bf16 | 64 × 128 × 2 = **16 KB** |

### 2.4 Causal Mask 安全性

```
Decode 阶段，query = 最新 token (位置 N)
KV cache 包含位置 [p1, p2, ..., pk]，都 < N

Flash Attention causal mask: attend if query_pos >= key_pos
由于 N > max(p1, p2, ..., pk)，所有 K 都会被 attend ✓

物理存储顺序无关紧要，只有逻辑位置（通过 position_indices 记录）影响 RoPE
```

---

## 3. 数据结构

### 3.1 FillInPlaceState

```python
@dataclass
class FillInPlaceState:
    """Fill-in-Place 策略的状态"""

    # === Budget Pages（固定大小）===
    budget_slots: int                          # Budget 槽位总数（固定）
    budget_position_indices: torch.Tensor      # [budget_slots], dtype=bf16/int32
    budget_block_table: torch.Tensor           # Budget pages 的 block table

    # === Overflow Pages（临时）===
    overflow_slots: int                        # 当前 overflow 槽位数
    overflow_position_indices: torch.Tensor    # [max_overflow], dtype=bf16/int32
    overflow_block_table: torch.Tensor         # Overflow pages 的 block table

    def decode_write(self, position: int, k: torch.Tensor, v: torch.Tensor):
        """Decode 阶段写入新 token 到 overflow pages"""
        slot = self.overflow_slots
        self.overflow_position_indices[slot] = position
        # 写入 overflow KV cache...
        self.overflow_slots += 1

    def prune_and_fill(
        self,
        budget_keep_mask: torch.Tensor,      # [budget_slots] bool
        overflow_keep_mask: torch.Tensor,    # [overflow_slots] bool
        budget_kv: Tuple[torch.Tensor, ...], # budget 的 K, V cache
        overflow_kv: Tuple[torch.Tensor, ...], # overflow 的 K, V
    ):
        """
        执行 Fill-in-Place 裁剪

        1. Budget pages 中被删的槽位变成 free_slots
        2. Overflow pages 中幸存的 token 填入 free_slots
        3. 更新 budget_position_indices
        4. 释放 overflow pages
        """
        # Step 1: 找出 budget 中被删的槽位
        free_slots = torch.where(~budget_keep_mask)[0]

        # Step 2: 找出 overflow 中幸存的 token
        survivor_indices = torch.where(overflow_keep_mask)[0]

        # Step 3: Fill-in（调用 Triton kernel）
        fill_in_place_kernel[grid](
            budget_kv[0], budget_kv[1],
            self.budget_position_indices,
            overflow_kv[0][survivor_indices],
            overflow_kv[1][survivor_indices],
            self.overflow_position_indices[survivor_indices],
            free_slots,
            len(survivor_indices),
            ...
        )

        # Step 4: 释放 overflow pages
        self.release_overflow_pages()
        self.overflow_slots = 0
```

### 3.2 position_indices 维护

`position_indices` 记录每个槽位存储的 token 在**原始序列中的位置**，与 KV cache **同步维护**：

| | KV cache | position_indices |
|--|----------|------------------|
| Budget | `budget_kv` (固定大小) | `budget_position_indices` (固定大小) |
| Overflow | `overflow_kv` (临时) | `overflow_position_indices` (临时) |

#### Decode 时

```python
# 写入 overflow
overflow_kv[slot] = (k, v)
overflow_position_indices[slot] = current_seq_position
```

#### Fill-in 时

```python
# 把 overflow 幸存者填入 budget 空槽（KV 和 position 一起拷贝）
budget_kv[dst_slot] = overflow_kv[src_slot]
budget_position_indices[dst_slot] = overflow_position_indices[src_slot]
```

#### 多轮后

经过多轮裁剪，`budget_position_indices` 不再连续，但不影响正确性：

```
budget_position_indices = [16385, 1, 32769, 16389, 4, 49153, ...]
                           ^      ^  ^      ^      ^  ^
                           来自   原始 来自   来自  原始 来自
                           第1轮      第2轮  第1轮      第3轮
```

---

## 4. Triton Kernel 实现

```python
@triton.jit
def fill_in_place_kernel(
    # Budget KV cache
    budget_k_cache,           # [budget_slots, num_heads, head_dim]
    budget_v_cache,
    budget_position_indices,  # [budget_slots]
    # Overflow survivors（已筛选）
    survivor_k,               # [num_survivors, num_heads, head_dim]
    survivor_v,
    survivor_positions,       # [num_survivors]
    # 映射
    free_slots,               # [num_survivors] - budget 中的空槽位
    # 参数
    num_survivors,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """把 overflow 幸存者填入 budget 空槽"""
    pid = tl.program_id(0)
    if pid >= num_survivors:
        return

    # 源：第 pid 个幸存者
    src_idx = pid

    # 目标：budget 中的空槽
    dst_slot = tl.load(free_slots + pid)

    # 拷贝 position_indices
    pos = tl.load(survivor_positions + src_idx)
    tl.store(budget_position_indices + dst_slot, pos)

    # 拷贝 K, V（向量化）
    for h in range(num_heads):
        for d in range(0, head_dim, BLOCK_SIZE):
            offsets = d + tl.arange(0, BLOCK_SIZE)
            mask = offsets < head_dim

            # K
            k_val = tl.load(
                survivor_k + src_idx * num_heads * head_dim + h * head_dim + offsets,
                mask=mask
            )
            tl.store(
                budget_k_cache + dst_slot * num_heads * head_dim + h * head_dim + offsets,
                k_val,
                mask=mask
            )

            # V
            v_val = tl.load(
                survivor_v + src_idx * num_heads * head_dim + h * head_dim + offsets,
                mask=mask
            )
            tl.store(
                budget_v_cache + dst_slot * num_heads * head_dim + h * head_dim + offsets,
                v_val,
                mask=mask
            )
```

---

## 5. Block 回收机制

### 5.1 Block 生命周期

```
Prefill:
  分配 budget pages -> 写入 prefill KV

Decode:
  分配 overflow pages -> 写入新 decode KV

裁剪触发:
  打分 -> 选择保留 -> Fill-in -> 释放 overflow pages
                                    |
                                    v
                              回收到 free pool
```

### 5.2 与 vLLM Block Manager 集成

```python
class FillInPlaceBlockManager:
    def __init__(self, vllm_block_manager, budget_pages: int):
        self.vllm_bm = vllm_block_manager
        self.budget_pages = budget_pages

    def allocate_overflow_page(self) -> int:
        """分配一个 overflow page"""
        return self.vllm_bm.allocate_block()

    def release_overflow_pages(self, block_ids: List[int]):
        """释放 overflow pages 回 free pool"""
        for block_id in block_ids:
            self.vllm_bm.free_block(block_id)

    def prune_and_release(
        self,
        state: FillInPlaceState,
        budget_keep_mask: torch.Tensor,
        overflow_keep_mask: torch.Tensor,
    ) -> int:
        """执行 Fill-in-Place 并释放 overflow pages"""
        # 执行 Fill-in-Place
        state.prune_and_fill(budget_keep_mask, overflow_keep_mask, ...)

        # 释放 overflow pages
        overflow_block_ids = state.overflow_block_table.tolist()
        self.release_overflow_pages(overflow_block_ids)

        return len(overflow_block_ids)
```

---

## 6. 裁剪触发机制

### 6.1 触发条件

```python
def should_prune(budget_used: int, overflow_slots: int, budget_slots: int, divide_length: int) -> bool:
    """
    裁剪触发条件：
    1. Overflow 满了（达到 divide_length）
    2. 且当前 KV 总量超过 budget
    """
    overflow_full = (overflow_slots >= divide_length)
    exceeds_budget = (budget_used + overflow_slots > budget_slots)
    return overflow_full and exceeds_budget
```

**注意**：如果 budget 还没满（`budget_used + overflow_slots <= budget_slots`），不触发裁剪，直接把 overflow 合并进 budget。

### 6.2 处理流程

```python
def on_overflow_full(state: FillInPlaceState, scorer: TriAttentionScorer):
    """Overflow 满时的处理"""
    total_tokens = state.budget_used + state.overflow_slots

    if total_tokens <= state.budget_slots:
        # Budget 还有空间，直接合并，不裁剪
        state.merge_overflow_into_budget()
    else:
        # 超出 budget，需要裁剪
        state.prune_and_fill(scorer)
```

---

## 7. 效率分析

### 7.1 操作复杂度

| 操作 | 复杂度 | 频率 |
|-----|-------|-----|
| 打分 | O((budget + overflow) × freq_count × num_offsets) | 每 divide_length 步 |
| Top-k | O((budget + overflow) × log(budget)) | 每 divide_length 步 |
| Fill-in | O(overflow × num_heads × head_dim) | 每 divide_length 步 |

### 7.2 关键优势

1. **最小数据移动**：只移动 overflow 幸存者，budget 中保留的 token 不动
2. **可释放 overflow pages**：每次裁剪后 overflow pages 归还 free pool
3. **无碎片累积**：每次裁剪后 budget pages 始终是满的

---

*文档版本：2.0*
*创建日期：2025-01-30*
*更新：重写为仅 Fill-in-Place 策略*
