# Phase 0 设计补充说明（vLLM 集成）

基于 R-KV/vLLM 代码分析的技术细节和设计决策。

---

## 1. R-KV/vLLM 压缩架构详解

### 1.1 R1KV 在 vLLM v1 中的集成方式

**文件**：`R-KV/vLLM/vllm/v1/attention/backends/flash_attn.py`

```python
class FlashAttentionImpl:
    def __init__(self, ...):
        # 行 431：初始化压缩器
        self.kvcompressor = R1KV(budget=VLLM_V1_R_KV_BUDGET)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, ...):
        # 行 547-588：压缩逻辑
        for i in range(attn_metadata.num_reqs):
            if attn_metadata.seq_lens[i] < VLLM_V1_R_KV_BUDGET + VLLM_V1_R_KV_BUFFER:
                continue  # 序列太短，不压缩

            # 提取当前序列的 KV cache
            current_key_cache = key_cache.view(...)[occupied_slot_mapping[...], ...]
            current_value_cache = value_cache.view(...)[occupied_slot_mapping[...], ...]

            # 调用压缩器
            compressed_key, compressed_value = self.kvcompressor.update_kv(
                current_key_cache,
                current_query,
                current_value_cache,
            )

            # 回写压缩后的 KV
            key_cache.view(...)[...] = compressed_key.transpose(0, 1)
            value_cache.view(...)[...] = compressed_value.transpose(0, 1)

            # 记录压缩统计
            num_dropped = current_kv_len - compressed_kv_len
            attn_metadata.num_dropped_tokens_list[i] = num_dropped
```

### 1.2 触发条件

```python
# vLLM/vllm/envs.py
VLLM_V1_R_KV_BUDGET: int = 64   # 保留的 token 数
VLLM_V1_R_KV_BUFFER: int = 64   # 触发压缩的缓冲区

# 触发条件
if seq_len >= VLLM_V1_R_KV_BUDGET + VLLM_V1_R_KV_BUFFER:
    # 执行压缩
```

**含义**：当序列长度 >= 128 (64+64) 时触发压缩，压缩后保留 64 个 token。

### 1.3 KV Cache 布局

vLLM 使用 PagedAttention，KV cache 是分块存储的：

```python
# kv_cache 形状：[num_blocks, block_size, num_heads, head_dim]
# 通过 occupied_slot_mapping 访问特定序列的 token

# 提取当前序列的 KV
current_key = key_cache.view(num_blocks * block_size, num_heads, head_dim)
current_key = current_key[occupied_slot_mapping[start:end], ...]
# 结果形状：[seq_len, num_heads, head_dim]

# 需要转置为 R1KV 期望的格式
current_key = current_key.transpose(0, 1).unsqueeze(0)
# 结果形状：[1, num_heads, seq_len, head_dim]
```

---

## 2. R1KV 接口分析

### 2.1 update_kv() 接口

**文件**：`R-KV/HuggingFace/rkv/compression/r1_kv.py`
> 注意：vLLM 通过 `from rkv.modeling import R1KV` 导入，而 `rkv.modeling` 再从 `rkv.compression` 导入。

```python
def update_kv(
    self,
    key_states: torch.Tensor,      # [batch, num_kv_heads, seq_len, head_dim]
    query_states: torch.Tensor,    # [batch, num_heads, seq_len, head_dim]
    value_states: torch.Tensor,    # [batch, num_kv_heads, seq_len, head_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回压缩后的 (key_states, value_states)
    输出形状：[batch, num_kv_heads, compressed_len, head_dim]
    其中 compressed_len <= self.budget
    """
```

### 2.2 R1KV 算法核心

```python
# 1. 计算注意力权重
attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
attn_cache = attn_weights.mean(dim=-2)  # [batch, heads, seq_len]

# 2. 计算 token 相似度（用于去重）
similarity = cal_similarity(key_states)

# 3. 混合评分
final_score = attn_cache * mix_lambda - similarity * (1 - mix_lambda)

# 4. TopK 选择
keep_indices = final_score.topk(budget - window_size, dim=-1).indices

# 5. 保留最近 window_size 个 token
k_past = key_states.gather(dim=2, index=keep_indices)
k_cur = key_states[:, :, -window_size:, :]
key_states = torch.cat([k_past, k_cur], dim=2)
```

---

## 3. SpeckV vLLM 适配要点

### 3.1 与 R1KV 的关键区别

| 方面 | R1KV | SpeckV |
|-----|------|--------|
| 评分依据 | attention + similarity | 频率统计 |
| 是否需要 query | 是 | 否（使用预计算统计） |
| 额外输入 | 无 | stats 文件 |
| 窗口保留 | 是（window_size） | 可选 |
| RoPE 处理 | 无 | 需要 RoPE 反演 |

### 3.2 SpeckV 需要适配的地方

1. **Stats 加载**
   ```python
   def __init__(self, budget, stats_path, ...):
       self.stats = torch.load(stats_path)
       # stats 包含：q_mean_complex, freq_scale_sq, sampled_heads 等
   ```

2. **避免 RoPE 反演**（重要优化）
   - **不需要**对 key 做 RoPE 反演
   - 直接用存储的 $K_{rot}$ 计算，通过相位校正等价
   - 详见 `docs/design/optimization.md` 第 2 节
   ```python
   # 优化后：直接用 K_rot 计算
   Re = q_r * k_r + q_i * k_i
   Im = q_i * k_r - q_r * k_i
   A_coef = freq_scale_sq * Re  # 无需三角函数
   B_coef = freq_scale_sq * Im
   ```

3. **忽略 query_states**
   - SpeckV 不使用 query 打分，但接口需要保持兼容
   ```python
   def update_kv(self, key_states, query_states, value_states):
       # query_states 被忽略，保持接口兼容
       scores = self._compute_scores(key_states)  # 不使用 query
       ...
   ```

### 3.3 位置信息解决方案

> 详细设计见 `docs/implementation/data_structures.md` 和 `docs/implementation/fill_in_place.md`

**问题**：SpeckV 打分需要知道每个 token 的绝对位置，但 KV cache 压缩后 token 变成乱序。

**解决方案**：维护 `position_indices` tensor

```python
# 与 KV cache 同步存储，形状与 PagedAttention 对齐
position_indices: torch.Tensor  # [num_blocks, block_size]

# 每个 token 一个位置值，记录其原始序列位置
# 例如压缩后：position_indices = [0, 1, 16385, 4, 32769, ...]
#                                  ↑  ↑  ↑      ↑  ↑
#                                  原始位置（可能乱序）
```

**更新时机**：
| 事件 | position_indices 操作 |
|-----|----------------------|
| Prefill | 初始化为 `[0, 1, 2, ..., prefill_len-1]` |
| Decode | 追加新位置 `append(current_seq_len)` |
| Prune（回填） | 保留的 token 位置不变，新位置填入空槽 |

**显存开销**：极小（< 0.04% of KV cache），详见 `docs/implementation/data_structures.md` 第 3 节

### 3.4 打分函数优化

> 详细设计见 `docs/design/optimization.md`

Phase 0 应实现以下优化（PyTorch 版本），为 Phase 1 Triton 实现打基础：

**优化 1：避免 RoPE 反演**（已在 3.2 说明）

**优化 2：单次读取多位置打分**
```python
# 只加载 K 一次，在寄存器中迭代所有 offset
k = load_from_memory()  # 只读取一次
A_coef, B_coef = compute_coefficients(k)

for offset in offsets:  # 在寄存器中迭代，无显存访问
    score[offset] = dot(A_coef, C[offset]) - dot(B_coef, S[offset]) + extra_term
```

**优化 3：共享三角函数查找表（LUT）**
```python
# 每轮打分开始时预计算（所有 token 共享）
C = cos(offsets * omega)  # [num_offsets, freq_count]
S = sin(offsets * omega)  # [num_offsets, freq_count]

# 显存占用仅 ~4 KB
```

**批量矩阵形式**（GPU 友好）：
$$\mathbf{Scores} = \mathbf{A}_{all} \cdot \mathbf{C}^T - \mathbf{B}_{all} \cdot \mathbf{S}^T + \mathbf{E}$$

---

## 4. 关键设计决策（Review 反馈）

> 本节针对 review 中指出的关键问题给出明确决策。

### 4.1 模块路径问题

**问题**：文档中 import 路径不一致。

**决策**：统一使用 `rkv.modeling` 作为**唯一导入入口**。

```
# 1. 文件位置
R-KV/HuggingFace/rkv/compression/speckv_vllm.py  ← SpeckV 实现

# 2. 导出链（内部实现，用户不直接用）
R-KV/HuggingFace/rkv/compression/__init__.py:
    from .speckv_vllm import SpeckVvLLM

# 3. 统一导入入口（flash_attn.py 使用这个）
R-KV/HuggingFace/rkv/modeling.py:
    from .compression import R1KV, SpeckVvLLM

# 4. flash_attn.py 中的 import（唯一正确写法）
from rkv.modeling import R1KV, SpeckVvLLM
```

**禁止**：直接 `from rkv.compression.speckv_vllm import SpeckVvLLM`（绕过 modeling.py）

### 4.2 状态管理：全局 position_indices + occupied_slot_mapping

**问题**：SpeckV 需要位置信息，但 `kvcompressor` 是全局单例。

**决策**：使用**全局 `position_indices` 张量**，与 KV cache 同布局，通过 `occupied_slot_mapping` 实现 per-request 隔离。

> **重要**：不使用 `req_id` 字典，因为 flash_attn.py 中没有 `request_ids` 可用。
> vLLM 的 `occupied_slot_mapping` 已提供原生的 per-request 隔离机制。

```python
# position_indices 存储布局：与 KV cache 完全对齐
# 形状：[num_blocks, block_size]（PagedAttention 布局）
# 隔离机制：通过 occupied_slot_mapping 索引，天然 per-request 隔离

# 谁维护：flash_attn.py（不是 SpeckV 压缩器）
# 何时更新：
#   - Prefill: position_indices[slots] = torch.arange(prefill_len)
#   - Decode: position_indices[new_slot] = current_seq_len
#   - Prune: position_indices[slots] = position_indices[slots][keep_indices]

# 清理策略：position_indices 无需显式清理，prefill_lens 需要在 slot 复用时重置
# - 当 request 完成时，其 block 被 vLLM 回收
# - 新 request 分配时，会重新初始化对应 slot 的 position_indices
# - 若 slot 被复用，必须覆盖 prefill_lens[slot_key]，避免复用旧 prefill_len

# SpeckV 压缩器设计原则：
# 1. update_kv() 接口与 R1KV 完全一致，只返回 (K, V)
# 2. 保留索引通过 get_last_keep_indices() 暴露
# 3. layer_idx 采用 lazy init（首次 forward 时初始化）
# 4. position_indices=None 时，fallback 到 torch.arange(seq_len)

class SpeckVvLLM:
    _shared_stats = None  # 类级别缓存

    def __init__(self, budget, stats_path, ...):
        self.budget = budget
        self.stats_path = stats_path
        self.layer_idx = None  # lazy init
        self._initialized = False
        self._last_keep_indices = None  # 本次压缩保留的索引

    def _lazy_init(self, layer_idx: int):
        """首次 forward 时调用"""
        if self._initialized:
            return
        self.layer_idx = layer_idx
        full_stats = self._load_shared_stats(self.stats_path)
        self.q_mean = full_stats["q_mean_complex"][layer_idx]
        self._initialized = True

    def update_kv(self, key_states, query_states, value_states, position_indices=None):
        """接口与 R1KV 一致，只返回 (K, V)"""
        self._last_keep_indices = None
        if key_states.shape[2] <= self.budget:
            return key_states, value_states

        # Fallback: position_indices 为空时用序列位置
        if position_indices is None:
            position_indices = torch.arange(key_states.shape[2], device=key_states.device)

        scores = self._compute_scores(key_states, position_indices)
        keep_indices = self._select_tokens(scores)
        self._last_keep_indices = keep_indices  # 保存供外部获取

        return self._gather(key_states, keep_indices), self._gather(value_states, keep_indices)

    def get_last_keep_indices(self):
        """供 FlashAttentionImpl 更新 position_indices"""
        return self._last_keep_indices
```

**为什么不用 per-request 字典**：
- vLLM 的 position_indices 已经按 request 隔离（通过 `occupied_slot_mapping`）
- 全局张量 + occupied_slot_mapping 天然实现了 per-request 隔离
- 避免压缩器内部维护状态，简化实现

### 4.3 layer_idx 传递（Lazy Init 方案）

**问题**：SpeckV stats 是按层采样的，需要知道当前层索引。但 vLLM 的 `FlashAttentionImpl.__init__` 不一定能拿到 `layer_idx`。

**决策**：采用 **Lazy Init**，在首次 `forward` 时从 `layer.layer_idx` 获取。

**实现方式**：
```python
# flash_attn.py

class FlashAttentionImpl:
    def __init__(self, ...):
        # 不在构造时传入 layer_idx
        if VLLM_COMPRESSION_ALGO == "speckv":
            self.kvcompressor = SpeckVvLLM(
                budget=VLLM_V1_R_KV_BUDGET,
                stats_path=VLLM_SPECKV_STATS_PATH,
                # 注意：不传 layer_idx
            )
        else:
            self.kvcompressor = R1KV(budget=VLLM_V1_R_KV_BUDGET)

    def forward(self, layer, ...):
        # Lazy init：首次 forward 时初始化 layer_idx
        if isinstance(self.kvcompressor, SpeckVvLLM):
            self.kvcompressor._lazy_init(layer.layer_idx)
        ...

# SpeckVvLLM 的 lazy init
class SpeckVvLLM:
    _shared_stats = None  # 类级别缓存（所有层共享）

    def __init__(self, budget, stats_path, ...):
        self.budget = budget
        self.stats_path = stats_path
        self.layer_idx = None  # 延迟初始化
        self._initialized = False

    def _lazy_init(self, layer_idx: int):
        if self._initialized:
            return
        self.layer_idx = layer_idx
        full_stats = self._load_shared_stats(self.stats_path)
        self.q_mean = full_stats["q_mean_complex"][layer_idx]
        self.freq_scale_sq = full_stats["freq_scale_sq"][layer_idx]
        self._initialized = True

    @classmethod
    def _load_shared_stats(cls, stats_path):
        if cls._shared_stats is None:
            cls._shared_stats = torch.load(stats_path)
        return cls._shared_stats
```

**优点**：
- 不依赖 `FlashAttentionImpl.__init__` 的参数（vLLM 不一定传 layer_idx）
- Stats 文件只加载一次（类级别缓存）
- 每层压缩器只初始化一次（`_initialized` 保护）

### 4.4 position_indices 在 vLLM 中的维护（occupied_slot_mapping 方案）

**问题**：HF 路径靠 `cache_positions` 跟踪位置，vLLM 没有现成的位置追踪。

**决策**：
1. `position_indices` 使用**全局张量**，形状 `[num_blocks, block_size]`，与 KV cache 对齐
2. 通过 `occupied_slot_mapping` 索引，实现 per-request 隔离（**无需 `req_id`**）
3. **绝不**作为 `update_kv()` 返回值
4. 压缩后用 `get_last_keep_indices()` 获取保留索引，更新位置表

> **为什么不用 req_id 字典**：flash_attn.py 中没有 `request_ids` 可用，
> 而 `occupied_slot_mapping` 是 vLLM 原生的 per-request 隔离机制。

```python
# flash_attn.py 修改

class FlashAttentionImpl:
    def __init__(self, ...):
        if VLLM_COMPRESSION_ALGO == "speckv":
            self.kvcompressor = SpeckVvLLM(budget=..., stats_path=...)
        else:
            self.kvcompressor = R1KV(budget=...)

        # SpeckV 专用：全局 position_indices（与 KV cache 同布局）
        # 形状：[num_blocks, block_size]
        self.position_indices = None  # 惰性初始化

        # prefill_len 记录：用 (layer_idx, slot_start) 作为 key
        self.prefill_lens = {}

    def _init_position_indices(self, num_blocks, block_size, device):
        """惰性初始化 position_indices"""
        if self.position_indices is None:
            self.position_indices = torch.full(
                (num_blocks, block_size), -1, dtype=torch.long, device=device
            )

    def forward(self, layer, ...):
        # Lazy init
        if isinstance(self.kvcompressor, SpeckVvLLM):
            self.kvcompressor._lazy_init(layer.layer_idx)
            self._init_position_indices(num_blocks, block_size, device)

        # 压缩循环（与现有 R1KV 代码结构一致）
        for i in range(attn_metadata.num_reqs):
            seq_len = seq_lens[i]

            if seq_len < BUDGET + BUFFER:
                continue

            # 通过 occupied_slot_mapping 获取当前序列的 slots
            slots = occupied_slot_mapping[seq_starts_ends_indices[i]:seq_starts_ends_indices[i+1]]
            slot_key = slots[0].item()  # 用起始 slot 作为序列标识

            # 首次压缩：初始化 position_indices
            if slot_key not in self.prefill_lens:
                self.prefill_lens[slot_key] = seq_len
                # 写入全局 position_indices
                self._write_positions(slots, torch.arange(seq_len, device=device))

            # 提取当前序列的 position_indices
            current_positions = self._read_positions(slots)

            # 压缩（接口与 R1KV 完全一致）
            compressed_k, compressed_v = self.kvcompressor.update_kv(
                current_key, current_query, current_value,
                position_indices=current_positions,  # 打分用
            )

            # 用保留索引更新 position_indices
            if isinstance(self.kvcompressor, SpeckVvLLM):
                keep_indices = self.kvcompressor.get_last_keep_indices()
                if keep_indices is not None:
                    kept_positions = current_positions[keep_indices]
                    # 压缩后 slots 数量减少，需配合 vLLM block 管理
                    # （这里简化，实际需要与 block 回收逻辑配合）

            # 回写 KV cache
            key_cache[...] = compressed_k
            value_cache[...] = compressed_v

    def _read_positions(self, slots):
        """从全局 position_indices 读取"""
        positions = torch.empty(len(slots), dtype=torch.long, device=self.position_indices.device)
        for j, slot in enumerate(slots):
            block_idx = slot // self.block_size
            offset = slot % self.block_size
            positions[j] = self.position_indices[block_idx, offset]
        return positions

    def _write_positions(self, slots, positions):
        """写入全局 position_indices"""
        for j, slot in enumerate(slots):
            block_idx = slot // self.block_size
            offset = slot % self.block_size
            self.position_indices[block_idx, offset] = positions[j]
```

**清理策略**：
- `position_indices`：全局张量，block 被回收后下次使用时会被覆盖，无需显式清理
- `prefill_lens`：使用 `slot_key` 作为 key，**必须在 slot 复用时重置**
  - **可选方案 A**：在 scheduler 通知 request 完成时清理（需要 hook）
  - **可选方案 B**：惰性清理，当 `prefill_lens` 大小超过阈值时清理不活跃的 key
  - **Phase 0 最小实现**：当检测到该 slot 尚未初始化（如 `position_indices` 为 -1）时，覆盖 `prefill_lens[slot_key]` 并重写位置表

**关键点**：
- `update_kv()` 只返回 `(K, V)`，与 R1KV 接口完全一致
- 位置表更新逻辑在 `FlashAttentionImpl` 中，不在压缩器内
- 保留索引通过 `get_last_keep_indices()` 只读获取
- **无需 `request_ids`**，使用 `occupied_slot_mapping` 的起始 slot 作为序列标识

### 4.5 Prefill 处理规则与 prefill_len 来源

**问题**：HF 路径有 `include_prefill_in_budget` / `allow_prefill_compression`，vLLM 路径未明确 prefill_len 从哪里获取。

**决策**：Phase 0 默认 **prefill 不参与压缩**（保护前缀）。

| 配置 | 值 | 说明 |
|-----|-----|------|
| `protect_prefill` | `True`（默认） | Prefill token 不参与裁剪 |
| `prefill_budget` | 与 `budget` 分开 | Prefill 有独立配额（可选） |

**prefill_len 在 vLLM 中的来源**：

> **隔离原则**：Phase 0 不修改 `attn_metadata` 结构，避免影响其他模块。

```python
# 方案 1：从 attn_metadata 获取（不采用）
# vLLM v1 的 FlashAttentionMetadata 只有 batch 级别信息：
#   - num_prefill_tokens: int  # 当前 batch 的 prefill token 总数
#   - num_decode_tokens: int   # 当前 batch 的 decode token 总数
# 无法直接获取 per-request prefill_len，且添加新字段违反隔离原则

# 方案 2（推荐）：首次压缩时记录
# 首次触发压缩时（seq_len >= budget + buffer），当前 seq_len 就是 prefill_len
# 将其存储在 FlashAttentionImpl 的实例变量中（不修改 attn_metadata）
# 其中 seq_starts_ends_indices 由 seq_lens 的前缀和得到（长度 = num_reqs + 1）
class FlashAttentionImpl:
    def __init__(self, ...):
        self.prefill_lens = {}  # slot_key -> prefill_len，本地维护

    def forward(self, ...):
        for i in range(attn_metadata.num_reqs):
            slots = occupied_slot_mapping[seq_starts_ends_indices[i]:seq_starts_ends_indices[i+1]]
            slot_key = slots[0].item()
            if slot_key not in self.prefill_lens:
                # 首次压缩：记录 prefill_len
                self.prefill_lens[slot_key] = seq_lens[i]

            prefill_len = self.prefill_lens[slot_key]
            ...

# 方案 3（备选）：使用 position_indices 推断
# prefill 阶段的 token 位置是连续的 [0, 1, 2, ..., prefill_len-1]
# 找到位置 0 所在的索引，其之后连续位置的 token 就是 prefill 部分
# 优点：无需额外存储；缺点：逻辑较复杂
```

**推荐方案**：使用方案 2（本地记录），简单且不侵入现有结构。

**实现方式**（SpeckVvLLM.update_kv）：
```python
def update_kv(self, key_states, ..., position_indices, prefill_len=0):
    """
    prefill_len 由调用方（flash_attn.py）从本地 prefill_lens dict 获取后传入，
    不依赖 attn_metadata 结构变更。
    """
    if prefill_len > 0:
        # 分离 prefill 和 decode 部分
        prefill_k = key_states[:, :, :prefill_len, :]
        decode_k = key_states[:, :, prefill_len:, :]

        # 只对 decode 部分打分和裁剪
        scores = self._compute_scores(decode_k, position_indices[prefill_len:])
        keep_indices = self._select_tokens(scores, budget - prefill_len)

        # 拼接：prefill（全保留）+ decode（裁剪后）
        compressed_k = torch.cat([prefill_k, decode_k.gather(..., keep_indices)], dim=2)
```

**与 HF 基线对比**：确保使用相同的 prefill 策略，否则结果不可比。

### 4.6 per_head 模式限制（跨层聚合）

**问题**：vLLM 按层调用 `update_kv`，无法实现"全局 per_head"（跨层聚合打分）。

**决策**：Phase 0 仅支持 **per_layer_perhead**（每层独立、每 head 独立）。

| 模式 | HF 路径 | vLLM Phase 0 | 原因 |
|-----|---------|-------------|------|
| `per_head` (跨层) | ✓ | ✗ | 需要所有层的 KV 才能聚合打分 |
| `per_layer` | ✓ | ✗ | 需要同层所有 head 聚合 |
| `per_layer_perhead` | ✓ | ✓ | 每层每 head 独立，无跨层依赖 |

**配置映射**：
```python
# 环境变量
VLLM_SPECKV_PRUNING_MODE = "per_layer_perhead"  # 唯一支持的模式

# 如果用户指定其他模式，警告并 fallback
if pruning_mode != "per_layer_perhead":
    logger.warning(f"vLLM Phase 0 只支持 per_layer_perhead，忽略 {pruning_mode}")
    pruning_mode = "per_layer_perhead"
```

### 4.7 Stats 校验与 Model Config

**问题**：Stats 需要与模型配置匹配（RoPE/num_heads/dtype/prompt），但文档只有 `stats_path`。

**决策**：添加 model config 校验，stats 必须包含元数据。

**Stats 文件结构**：
```python
stats = {
    # 核心数据
    "q_mean_complex": ...,
    "freq_scale_sq": ...,
    "sampled_heads": ...,

    # 元数据（用于校验）
    "metadata": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "rope_theta": 1000000.0,
        "rope_scaling": None,
        "dtype": "bfloat16",
        "prompt_template": "plain",  # plain / chat
    }
}
```

**校验逻辑**：
```python
def _load_stats(self, stats_path, model_config=None):
    stats = torch.load(stats_path)

    if model_config is not None:
        meta = stats.get("metadata", {})

        # 必须匹配的项
        assert meta.get("num_kv_heads") == model_config.num_kv_heads
        assert meta.get("head_dim") == model_config.head_dim

        # 警告但不阻止的项
        if meta.get("rope_theta") != model_config.rope_theta:
            logger.warning(f"Stats rope_theta 不匹配: {meta.get('rope_theta')} vs {model_config.rope_theta}")

    return stats
```

**环境变量补充**：
```python
VLLM_SPECKV_MODEL_CONFIG = os.getenv("VLLM_SPECKV_MODEL_CONFIG", None)  # 可选：model config 路径
```

### 4.8 优化策略：正确性优先

**问题**：文档要求 Phase 0 直接实现"优化版（无 RoPE 反演 + LUT）"，但这可能导致与 HF 基线结果偏离，难以验证正确性。

**决策**：**两阶段实现**，先正确性后优化。

```
Phase 0a: 等价实现（正确性验证）
    ├── 直接移植 HF 路径的打分逻辑
    ├── 与 HF 基线做 bit-exact 对比
    └── 确认准确率差异 < 0.1%

Phase 0b: 优化实现（可选，在正确性验证通过后）
    ├── 应用"避免 RoPE 反演"优化
    ├── 应用"共享三角函数表（LUT）"优化
    └── 再次与 HF 基线对比，确认等价性
```

**验证步骤**：
```python
def test_equivalence_with_hf():
    """Phase 0a 完成后必须通过此测试"""
    # 准备相同输入
    key = torch.randn(1, 4, 1000, 128)
    value = torch.randn(1, 4, 1000, 128)
    position_indices = torch.arange(1000)

    # HF 路径（基准）
    hf_compressor = SpeckVRKVStyle(...)
    hf_key, hf_value = hf_compressor.compress(key, value)

    # vLLM 路径（待验证）
    vllm_compressor = SpeckVvLLM(...)
    vllm_key, vllm_value = vllm_compressor.update_kv(key, None, value, position_indices)

    # bit-exact 对比（相同随机种子下）
    assert torch.allclose(hf_key, vllm_key, rtol=1e-5, atol=1e-5)
    assert torch.allclose(hf_value, vllm_value, rtol=1e-5, atol=1e-5)
```

**文档中"优化"章节（3.4 打分函数优化）的定位**：
- 描述的是**目标优化方案**（Phase 0b 或 Phase 1）
- Phase 0a 可以先**不实现**这些优化，直接用 HF 逻辑
- 优化是**可选的性能改进**，不是 Phase 0 的硬性要求

---

## 5. GQA 处理

### 5.1 背景

Qwen2.5-7B 使用 GQA：
- `num_attention_heads` = 28
- `num_key_value_heads` = 4
- `num_key_value_groups` = 7

### 5.2 vLLM 中的 GQA 处理

```python
# flash_attn.py 中，KV cache 使用 num_kv_heads
key_states.shape = [batch, num_kv_heads, seq_len, head_dim]  # [1, 4, seq, 128]

# query 使用 num_attention_heads
query_states.shape = [batch, num_heads, seq_len, head_dim]   # [1, 28, seq, 128]
```

### 5.3 SpeckV 的 GQA 适配

SpeckV 的 `sampled_heads` 是 attention head 索引，需要映射到 KV heads：

```python
# sampled_heads: [(layer, attn_head), ...]
# 需要映射到 kv_head

def map_to_kv_head(attn_head, num_kv_groups):
    return attn_head // num_kv_groups

# 例如：attn_head=14 -> kv_head=14//7=2
```

---

## 6. 环境变量设计

### 6.1 新增环境变量

| 变量名 | 默认值 | 说明 |
|-------|--------|------|
| `VLLM_COMPRESSION_ALGO` | `"r1kv"` | 压缩算法选择 |
| `VLLM_SPECKV_STATS_PATH` | `None` | SpeckV stats 文件路径 |
| `VLLM_SPECKV_PRUNING_MODE` | `"per_head"` | pruning 模式 |

### 6.2 在 envs.py 中添加

```python
# vLLM/vllm/envs.py

VLLM_COMPRESSION_ALGO: str = "r1kv"
VLLM_SPECKV_STATS_PATH: Optional[str] = None
VLLM_SPECKV_PRUNING_MODE: str = "per_head"

environment_variables: Dict[str, Callable[[], Any]] = {
    ...
    "VLLM_COMPRESSION_ALGO": lambda: os.getenv("VLLM_COMPRESSION_ALGO", "r1kv"),
    "VLLM_SPECKV_STATS_PATH": lambda: os.getenv("VLLM_SPECKV_STATS_PATH", None),
    "VLLM_SPECKV_PRUNING_MODE": lambda: os.getenv("VLLM_SPECKV_PRUNING_MODE", "per_head"),
}
```

---

## 7. 测试策略

### 7.1 单元测试优先级

1. **接口兼容性**：`update_kv()` 输入输出格式正确
2. **边界条件**：`seq_len <= budget` 时直接返回
3. **压缩比**：输出长度 <= budget
4. **数值正确性**：与 HF 路径对比

### 7.2 与 HF 路径对比验证

```python
def test_consistency_with_hf():
    """验证 vLLM 路径与 HF 路径结果一致"""
    # 准备相同的输入
    key = torch.randn(1, 4, 1000, 128)
    value = torch.randn(1, 4, 1000, 128)

    # HF 路径
    hf_compressor = SpeckVRKVStyle(...)
    hf_key, hf_value = hf_compressor.compress(key, value)

    # vLLM 路径
    vllm_compressor = SpeckVvLLM(...)
    vllm_key, vllm_value = vllm_compressor.update_kv(key, None, value)

    # 对比
    assert torch.allclose(hf_key, vllm_key, rtol=1e-4)
    assert torch.allclose(hf_value, vllm_value, rtol=1e-4)
```

---

## 8. 已知限制

### 8.1 Phase 0 限制

| 限制 | 原因 | Phase 1 解决方案 |
|-----|------|-----------------|
| 无 Triton 优化 | Phase 0 重点是正确性 | Phase 1 实现 Triton kernel |
| 仅支持 per_layer_perhead 模式 | vLLM 按层调用，无法跨层聚合 | Phase 1 重新设计接口 |

### 8.2 与 HF 路径的功能差异

| 功能 | HF 路径 | vLLM Phase 0 | 说明 |
|-----|--------|--------------|------|
| per_head pruning (跨层) | ✓ | ✗ | 需要全局聚合，见 4.5 节 |
| per_layer pruning | ✓ | ✗ | 需要同层所有 head 聚合 |
| per_layer_perhead | ✓ | ✓ | **Phase 0 唯一支持的模式** |
| 位置追踪 | ✓ | ✓ | 通过 position_indices，见 4.3 节 |
| prefill 保护 | ✓ | ✓ | 默认保护，见 4.4 节 |
| 打分优化（LUT） | ✗ | ✓ | Phase 0 即实现 |
| Stats 校验 | ✓ | ✓ | 见 4.6 节 |

---

## 9. 文件修改检查清单

### 9.1 新建文件

- [ ] `R-KV/HuggingFace/rkv/compression/speckv_vllm.py` - SpeckV vLLM 接口实现
- [ ] `R-KV/vLLM/tests/test_speckv_vllm.py` - 单元测试（放在 vLLM 测试目录）

### 9.2 修改文件

- [ ] `R-KV/HuggingFace/rkv/compression/__init__.py` - 导出 SpeckVvLLM
- [ ] `R-KV/HuggingFace/rkv/modeling.py` - 添加 SpeckVvLLM import（保持与 R1KV 一致的 import 方式）
- [ ] `R-KV/vLLM/vllm/v1/attention/backends/flash_attn.py` - 算法选择分支 + position_indices 更新
- [ ] `R-KV/vLLM/vllm/envs.py` - 环境变量

### 9.3 关键数据结构

- [ ] `position_indices`: `[num_blocks, block_size]` 与 KV cache 对齐
- [ ] `trig_cos/sin`: `[num_offsets, freq_count]` 共享三角表（~4 KB）
- [ ] Stats 从文件加载：`q_mean_complex`, `freq_scale_sq`, `omega` 等

### 9.4 验证默认行为不变

```bash
# 不设置 VLLM_COMPRESSION_ALGO，确认使用 R1KV
python -c "from vllm.envs import VLLM_COMPRESSION_ALGO; print(VLLM_COMPRESSION_ALGO)"
# 应输出: r1kv
```

---

*创建日期：2025-01-31*
*更新日期：2025-01-31（重写为 vLLM 集成路径）*
