# Fill-in-Place 策略

本文档详细说明 KV 裁剪后的 Fill-in-Place 策略。

---

## 1. 核心思路

- **Budget pages**：固定大小（如 1024 pages），存放压缩后保留的 KV
- **Overflow pages**：临时分配，存放新 decode 的 token
- **裁剪时**：budget 中被删的 token 空出槽位，overflow 中幸存的 token **填入** 这些空槽
- **裁剪后**：**释放 overflow pages**

---

## 2. 工作流程图解

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
Page 1024: [T16384, T16385, T16386, ...]  ← 新 decode 的 token
Page 1025: [T16400, T16401, ...]
...
Page 1031: [T16496, T16497, ..., T16511]  ← 128 个新 token

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
  budget_kv[0]  ← overflow_kv[0]   (T16385)
  budget_kv[3]  ← overflow_kv[2]   (T16387)
  budget_kv[17] ← overflow_kv[5]   (T16390)
  ...

  position_indices[0]  = 16385
  position_indices[3]  = 16387
  position_indices[17] = 16390
  ...

Step 4: 释放 overflow pages
  Page 1024~1031 归还 free pool
```

---

## 3. 数据移动量分析

**只移动 overflow 幸存者**，不移动 budget 中保留的 token：

| 配置 | 数据移动量 |
|-----|-----------|
| budget=16K, divide=128, head_dim=128, bf16 | 128 × 128 × 2 = **32 KB** |
| budget=8K, divide=64, head_dim=128, bf16 | 64 × 128 × 2 = **16 KB** |

这个数据量非常小，对性能影响可忽略。

---

## 4. Position Indices 维护

`position_indices` 记录每个槽位存储的 token 在**原始序列中的位置**，与 KV cache **同步维护**：

| | KV cache | position_indices |
|--|----------|------------------|
| Budget | `budget_kv` (固定大小) | `budget_position_indices` (固定大小) |
| Overflow | `overflow_kv` (临时) | `overflow_position_indices` (临时) |

### 4.1 Decode 时

```python
# 写入 overflow
overflow_kv[slot] = (k, v)
overflow_position_indices[slot] = current_seq_position
```

### 4.2 Fill-in 时

```python
# 把 overflow 幸存者填入 budget 空槽（KV 和 position 一起拷贝）
budget_kv[dst_slot] = overflow_kv[src_slot]
budget_position_indices[dst_slot] = overflow_position_indices[src_slot]
```

### 4.3 多轮后的 Position Indices

经过多轮裁剪，`budget_position_indices` 不再连续，但不影响正确性：

```
budget_position_indices = [16385, 1, 32769, 16389, 4, 49153, ...]
                           ^      ^  ^      ^      ^  ^
                           来自   原始 来自   来自  原始 来自
                           第1轮      第2轮  第1轮      第3轮
```

---

## 5. Causal Mask 安全性

```
Decode 阶段，query = 最新 token (位置 N)
KV cache 包含位置 [p1, p2, ..., pk]，都 < N

Flash Attention causal mask: attend if query_pos >= key_pos
由于 N > max(p1, p2, ..., pk)，所有 K 都会被 attend ✓

物理存储顺序无关紧要，只有逻辑位置（通过 position_indices 记录）影响 RoPE
```

---

## 6. 裁剪触发机制

### 6.1 触发条件

```python
def should_prune(budget_used, overflow_slots, budget_slots, divide_length):
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
def on_overflow_full(state, scorer):
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

## 7. 伪代码实现

### 7.1 FillInPlaceState

```python
@dataclass
class FillInPlaceState:
    """Fill-in-Place 策略的状态"""

    # === Budget Pages（固定大小）===
    budget_slots: int                          # Budget 槽位总数（固定）
    budget_position_indices: torch.Tensor      # [budget_slots]
    budget_block_table: torch.Tensor           # Budget pages 的 block table

    # === Overflow Pages（临时）===
    overflow_slots: int                        # 当前 overflow 槽位数
    overflow_position_indices: torch.Tensor    # [max_overflow]
    overflow_block_table: torch.Tensor         # Overflow pages 的 block table

    def decode_write(self, position, k, v):
        """Decode 阶段写入新 token 到 overflow pages"""
        slot = self.overflow_slots
        self.overflow_position_indices[slot] = position
        # 写入 overflow KV cache...
        self.overflow_slots += 1

    def prune_and_fill(self, budget_keep_mask, overflow_keep_mask, budget_kv, overflow_kv):
        """
        执行 Fill-in-Place 裁剪
        """
        # Step 1: 找出 budget 中被删的槽位
        free_slots = torch.where(~budget_keep_mask)[0]

        # Step 2: 找出 overflow 中幸存的 token
        survivor_indices = torch.where(overflow_keep_mask)[0]

        # Step 3: Fill-in（调用 Triton kernel）
        fill_in_place_kernel(
            budget_kv, self.budget_position_indices,
            overflow_kv, self.overflow_position_indices,
            free_slots, survivor_indices
        )

        # Step 4: 释放 overflow pages
        self.release_overflow_pages()
        self.overflow_slots = 0
```

### 7.2 Triton Kernel

```python
@triton.jit
def fill_in_place_kernel(
    budget_k_cache, budget_v_cache, budget_position_indices,
    survivor_k, survivor_v, survivor_positions,
    free_slots,
    num_survivors, num_heads, head_dim,
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
            k_val = tl.load(survivor_k + src_idx * num_heads * head_dim + h * head_dim + offsets, mask=mask)
            tl.store(budget_k_cache + dst_slot * num_heads * head_dim + h * head_dim + offsets, k_val, mask=mask)

            # V
            v_val = tl.load(survivor_v + src_idx * num_heads * head_dim + h * head_dim + offsets, mask=mask)
            tl.store(budget_v_cache + dst_slot * num_heads * head_dim + h * head_dim + offsets, v_val, mask=mask)
```

---

## 8. Block 生命周期

```
Prefill:
  分配 budget pages → 写入 prefill KV

Decode:
  分配 overflow pages → 写入新 decode KV

裁剪触发:
  打分 → 选择保留 → Fill-in → 释放 overflow pages
                                    ↓
                              回收到 free pool
```

---

## 9. 关键优势

1. **最小数据移动**：只移动 overflow 幸存者，budget 中保留的 token 不动
2. **可释放 overflow pages**：每次裁剪后 overflow pages 归还 free pool
3. **无碎片累积**：每次裁剪后 budget pages 始终是满的

---

*文档版本：1.0*
*创建日期：2025-01-30*
