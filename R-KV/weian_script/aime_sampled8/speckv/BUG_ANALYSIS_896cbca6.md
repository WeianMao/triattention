# Bug 分析：896cbca6 SpeckV 状态重置问题

## 1. 背景

Commit `896cbca60167cef6d0d122f13813a71a4fdbf636` 修复了 SpeckV 的一个 pruner 状态重置时机问题。

```
fix(align budget): refresh pruner states before prefilling
```

**修改内容**：将 pruner 状态重置逻辑从 `orig_forward` **之后**移到**之前**。

## 2. 修改前后对比

### 修改前的代码流程

```python
def speckv_forward(...):
    # 1. 计算 position_ids_override（使用可能过时的 state.pruner.absolute_position）
    if past_key_values is not None and input_ids is not None:
        start_pos = state.pruner.absolute_position  # ← 可能是旧值
        position_ids_override = torch.arange(start_pos, start_pos + step, ...)

    # 2. 调用原始 forward
    outputs = orig_forward(..., position_ids=position_ids_override, ...)

    # 3. 检查是否需要返回
    if getattr(outputs, "past_key_values", None) is None:
        return outputs

    # 4. 【修改前的位置】状态重置检查
    is_empty_cache = True
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            if past_key_values.get_seq_length() > 0:
                is_empty_cache = False

    if is_empty_cache and state.attached:
        state.pruner = SparseRoundPruner(state.config)  # 重置
        state.attached = False
        state.initial_prefix_length = None

    # 5. 继续处理 KV cache...
```

### 修改后的代码流程

```python
def speckv_forward(...):
    # 1. 【修改后的位置】先检查并重置状态
    is_empty_cache = True
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            if past_key_values.get_seq_length() > 0:
                is_empty_cache = False

    if is_empty_cache and state.attached:
        state.pruner = SparseRoundPruner(state.config)  # 重置
        state.attached = False
        state.initial_prefix_length = None

    # 2. 现在再计算 position_ids_override（使用新的 absolute_position=0）
    if past_key_values is not None and input_ids is not None:
        start_pos = state.pruner.absolute_position  # ← 现在是正确的值
        position_ids_override = torch.arange(start_pos, start_pos + step, ...)

    # 3. 调用原始 forward
    outputs = orig_forward(...)

    # 4. 继续处理...
```

## 3. Bug 触发条件

当连续处理多个问题时（如 AIME 数据集有多道题），第 n 题（n≥2）开始时：

1. `past_key_values` = 空 DynamicCache 或 None
2. `state.attached = True`（来自上一题）
3. `state.pruner` 的各项状态保留上一题结束时的值

## 4. 初步分析：位置偏移对打分函数的影响

### 4.1 SpeckV 打分机制

SpeckV 使用 `score_keys_for_round` 函数预测"未来位置的 query 对当前 key 的关注度"：

```python
# round_pruning_utils.py
def score_keys_for_round(key_indices, round_start, amp, phi, omega, offsets, ...):
    base_delta = round_start - key_indices
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)
    phase = delta_grid * omega + phi
    score = cos(phase)
```

其中 `offsets` 由 `build_geometric_offsets(max_length)` 生成：
- `max_length=65536` → `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]`（17个元素）

### 4.2 位置偏移的等效变换

假设第一题生成长度为 L₁，第二题 prefill 时：

- **修改前**：`position_ids_override = [L₁, L₁+1, ...]`，KV cache 用这些位置编码
- **然后重置**：`cache_positions = [0, 1, ...]`，`absolute_position = P₂`

由于 `invert_rope` 使用 `cache_positions` 而非实际编码位置，导致 `phi` 有额外偏移 `-L₁·ω`。

**等效变换**：
```
offsets_effective = offsets - L₁
                  = [1, 2, 4, ..., 65536] - L₁
```

例如 L₁ = 3000 时：
- 原本预测位置：[1, 2, 4, 8, ..., 65536]
- 实际预测位置：[-2999, -2998, -2996, ..., 62536]

### 4.3 位置偏移只影响 Prefill 部分

| 阶段 | 实际编码位置 | cache_positions | 偏移 |
|------|-------------|-----------------|------|
| Prefill | [L₁, L₁+1, ...] | [0, 1, ...] | L₁ |
| Decode | [P₂, P₂+1, ...] | [P₂, P₂+1, ...] | 0 |

因为 decode 阶段的 `position_ids_override` 是从重置后的 `absolute_position`（= P₂）开始的。

### 4.4 消融实验：offset_max_length

基于上述分析，怀疑 `offset_max_length=65536` 过大导致问题。进行了消融实验：

| 实验 | offset_max_length | 元素数 |
|------|-------------------|--------|
| 基线 | 65536 | 17 |
| offset32k | 32768 | 16 |
| offset16k | 16384 | 15 |
| offset8k | 8192 | 14 |
| offset4k | 4096 | 13 |
| offset2k | 2048 | 12 |
| offset1k | 1024 | 11 |
| offset512 | 512 | 10 |
| offset256 | 256 | 9 |
| offset128 | 128 | 8 |

**实验结果**：offset_max_length 的变化对结果没有显著影响，说明问题不在这里。

实验脚本位置：`R-KV/weian_script/aime_sampled8/speckv/aime24/offset_ablation/`

## 5. 深入分析：Pruner 状态未重置的真正影响

错误版本的关键在于**没有重置 pruner 状态**，即没有执行 `SparseRoundPruner.__init__`。

### 5.1 受影响的状态变量

在 `SparseRoundPruner.__init__` 中初始化的关键变量：

```python
# sparse_round_pruner_prefill_keep.py
def __init__(self, config: SparsePruningConfig) -> None:
    ...
    self.cache_positions: List[int] = []      # 记录每个 cache 位置的绝对位置
    self.absolute_position: int = 0           # 当前的绝对位置
    self.tokens_in_round: int = 0             # 当前 round 内的 token 数
    self.prefix_length: int = 0               # prefill 的长度
```

**未重置时**：这些变量在两次 request 之间保持不变，继承上一题结束时的值。

### 5.2 可能的影响路径

#### 影响 1：cache_positions 被加了一个 base

`cache_positions` 记录的是每个 KV cache 位置对应的绝对位置。

在 `_compute_head_scores` 中：
```python
cos_table, sin_table = self._rotary_for_positions(key_positions)  # key_positions = cache_positions
k_unrot = invert_rope(k_values, cos_table, sin_table, ...)
```

如果 `cache_positions` 有偏移，`invert_rope` 会使用错误的位置来逆转 RoPE，导致 `k_unrot` 不正确。

**待确认**：这个绝对位置偏移是否会影响最终评分？（从理论上分析，由于 RoPE 的相对位置不变性，可能影响有限）

#### 影响 2：tokens_in_round 触发错误的 prune

`tokens_in_round` 用于判断是否开始下一个 round：

```python
def should_start_next_round(self) -> bool:
    return self.tokens_in_round >= self.round_window
```

如果 `tokens_in_round` 没有重置，第二题开始时可能立即触发 `start_next_round`。

但在 `rkv_aligned_budget` 模式下：
```python
# Skip round-based pruning when using R-KV aligned budget mode
if not state.pruner.rkv_aligned_budget:
    while state.pruner.should_start_next_round():
        pkv_tuple = state.pruner.start_next_round(pkv_tuple)
```

**使用 `--rkv-aligned-budget` 时**，round-based pruning 被跳过，所以 `tokens_in_round` 的影响可能有限。

#### 影响 3：prefix_length 错误（最可能的问题）

`prefix_length` 用于多个关键计算：

**1. 动态 cache size 计算**：
```python
@property
def _dynamic_cache_size(self) -> int:
    if self.config.include_prefill_in_budget:
        return len(self.cache_positions)
    return max(0, len(self.cache_positions) - self.prefix_length)
```

**2. 压缩目标计算**：
```python
def ensure_capacity(self, past_key_values):
    keep_capacity = self.max_keys if self.rkv_aligned_budget else max(0, self.max_keys - self.round_window)
    if self.allow_prefill_compression:
        prune_target = keep_capacity
    elif self.config.include_prefill_in_budget:
        prune_target = max(0, keep_capacity - self.prefix_length)  # ← 使用 prefix_length
    else:
        prune_target = keep_capacity
```

**3. Prefill token 保护**：
```python
def _prune_to_size(self, past_key_values, keep_count, *, dynamic_only=False):
    ...
    if dynamic_only:
        if self.allow_prefill_compression:
            ...
        else:
            # Original behavior: prefill tokens are always preserved
            prefix_count = min(self.prefix_length, candidate_count)  # ← 使用 prefix_length
            ...
```

**问题场景**：
- 假设第一题 prefill 长度 P₁ = 500
- 第二题 prefill 长度 P₂ = 300
- 未重置时，`prefix_length` 仍为 500
- 导致：
  1. `_dynamic_cache_size` 计算错误
  2. `prune_target` 可能变成负数或不合理的值
  3. 可能尝试"保护"不存在的 prefill tokens

## 6. 总结

### Bug 的核心问题

修改前的代码在第二题及之后的 prefilling 阶段：
1. 使用旧的 `absolute_position` 计算 `position_ids`
2. 在 `orig_forward` 执行后才重置 pruner 状态
3. 导致多个状态变量（`cache_positions`, `absolute_position`, `tokens_in_round`, `prefix_length`）在两次 request 之间没有正确重置

### 最可能的影响路径

1. **`prefix_length` 错误**：影响 budget 判断和 cache size 计算，可能导致压缩行为异常
2. **`cache_positions` 偏移**：影响 `invert_rope` 的位置计算，但由于 RoPE 相对位置不变性，影响可能有限
3. **`tokens_in_round` 未重置**：在 `rkv_aligned_budget` 模式下影响有限

### 待进一步验证

1. `prefix_length` 错误具体如何影响压缩决策
2. 为什么错误版本反而"提点"了（可能是某种意外的正则化效果？）

## 7. 相关文件

- 修改的文件：`R-KV/weian_development/speckv/rkv_speckv_generate.py`
- Pruner 实现：`R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py`
- 消融实验脚本：`R-KV/weian_script/aime_sampled8/speckv/aime24/offset_ablation/`
- 基准实验脚本：`R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_budget.sh`
