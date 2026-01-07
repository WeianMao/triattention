# Bug 分析：896cbca6 SpeckV 状态重置问题

## 1. 背景

Commit `896cbca60167cef6d0d122f13813a71a4fdbf636` 修复了 SpeckV 的一个 pruner 状态重置时机问题。

```
fix(align budget): refresh pruner states before prefilling
```

**修改内容**：将 pruner 状态重置逻辑从 `orig_forward` **之后**移到**之前**。

**关键发现**：这个 bug 反而提升了性能，需要分析原因。

## 2. Bug 的根本原因

### 为什么状态不会被重置？

修改前的代码在 `orig_forward` **之后**检查是否需要重置：

```python
def speckv_forward(..., past_key_values, ...):
    # 1. 用旧的 absolute_position 计算 position_ids
    start_pos = state.pruner.absolute_position  # ← 旧值
    position_ids_override = torch.arange(start_pos, start_pos + step, ...)

    # 2. 调用原始 forward
    outputs = orig_forward(..., past_key_values=past_key_values, ...)
    # ⚠️ 关键：orig_forward 会就地修改 past_key_values！
    # 即使输入时是空的 DynamicCache，执行后也有内容了

    # 3. 检查是否需要重置
    is_empty_cache = True
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            if past_key_values.get_seq_length() > 0:  # ← 现在是 True！
                is_empty_cache = False

    # 4. 因为 is_empty_cache=False，永远不会重置！
    if is_empty_cache and state.attached:
        state.pruner = SparseRoundPruner(state.config)  # 永远不会执行
```

**核心问题**：`orig_forward` 会就地修改 `past_key_values`（DynamicCache），所以检查时 cache 已经不为空了。

## 3. 状态变量在多道题之间的演变

### 初始状态
```
attached = False
absolute_position = 0
cache_positions = []
prefix_length = 0
```

### 第 1 题
```
[Prefill P₁=500]
- 新 DynamicCache (空)
- attached=False → 进入 attach_initial_cache（唯一一次正确初始化）
- absolute_position = 500
- cache_positions = [0, 1, ..., 499]
- prefix_length = 500
- attached = True

[Decode 生成 2000 tokens，多次压缩]
- absolute_position = 2500
- cache_positions 长度 ≈ 1024 (压缩后)
- prefix_length = 500 (不变)
```

### 第 2 题
```
[Prefill P₂=400]
- 新 DynamicCache (空)
- attached=True (没重置!)
- position_ids = [2500, 2501, ..., 2899] (用旧的 absolute_position)
- orig_forward 后 cache 有 400 token
- is_empty_cache 检查：cache.get_seq_length()=400 > 0 → 不重置
- 进入 else 分支 (attached=True)

状态更新逻辑：
- seq_len = 400 (新 cache 实际长度)
- cached_len = 1024 (旧 cache_positions 长度)
- cached_len > seq_len → cache_positions = cache_positions[-400:]
  (取第 1 题的最后 400 个位置!)
- absolute_position 在此分支不更新，保持 2500

[Decode]
- 每个新 token：cache_positions.append(absolute_position)
- absolute_position 递增
```

### 第 3 题及之后
```
- prefix_length = 500 (永远是第 1 题的值!)
- absolute_position = 累积值 (越来越大)
- cache_positions = 前面题目残留的混合
```

## 4. 各变量的实际影响

### 4.1 `absolute_position` - 影响 position_ids

**作用**：计算 RoPE 的 position_ids

**对模型推理的影响**：❌ **无影响**

RoPE 是相对位置编码，只要 query 和 key 的相对位置不变，attention 结果就一样：
```
正确版本: key 用 [0,1,2...], query 用 [100]  → 相对位置 [100,99,98...]
Bug 版本: key 用 [2500,2501,...], query 用 [2600] → 相对位置 [100,99,98...]
结果相同！
```

### 4.2 `cache_positions` - 影响 SpeckV 打分

**作用**：记录每个 cache 位置对应的绝对位置，用于 `invert_rope`

**对打分的影响**：✅ **严重错误**

```python
# _compute_head_scores 中
key_positions = torch.tensor(self.cache_positions, ...)
cos_table, sin_table = self._rotary_for_positions(key_positions)
k_unrot = invert_rope(k_values, cos_table, sin_table, ...)
```

Bug 情况下：
- KV cache 中的 key 用位置 `[2500, 2501, ...]` 编码
- 但 `cache_positions` 记录的是第 1 题残留的位置 `[1850, 1900, ...]`
- `invert_rope` 用错误位置逆转 → 得到垃圾 `k_unrot`
- 打分基本是随机的

### 4.3 `prefix_length` - 影响压缩决策

**作用**（在 `include_prefill_in_budget=True`, `allow_prefill_compression=False` 配置下）：

1. **压缩目标计算**：`prune_target = budget - prefix_length`
2. **Prefill 保护**：前 `prefix_length` 个 token 不参与打分竞争

**Bug 情况下的影响**：

| 题目 | 实际 prefill | prefix_length | 后果 |
|------|-------------|---------------|------|
| 第 1 题 | 500 | 500 | 正确 |
| 第 2 题 | 400 | 500 | 压缩更激进 (target = budget-500 而非 budget-400) |
| 第 3 题 | 600 | 500 | 保护不足 (只保护 500 而非 600) |

**对 KV cache 用量的影响**：不会导致使用更多 cache，反而可能更激进地压缩。

### 4.4 `tokens_in_round` - 影响 round-based pruning

在 `rkv_aligned_budget` 模式下，round-based pruning 被跳过，所以 **影响有限**。

## 5. 消融实验

### 5.1 offset_max_length 消融

**假设**：Bug 导致位置偏移，可能影响 geometric offsets 的有效范围

**实验**：调整 `offset_max_length` 从 65536 到 128

**结果**：无显著影响

**结论**：问题不在 offset 范围

实验脚本：`R-KV/weian_script/aime_sampled8/speckv/aime24/offset_ablation/`

### 5.2 高频消融

**假设**：Bug 导致高频成分的相位随机化，如果高频有害，显式 disable 应该也能提点

**实验**：添加 `--disable-top-n-high-freq` 参数，mask top-n 高频在位置相关项中的贡献

**结果**：无显著影响

**结论**：问题不在高频成分

实验脚本：`R-KV/weian_script/aime_sampled8/speckv/aime24/high_freq_ablation/`

## 6. 为什么 Bug 能提点？

### 已排除的假设
- ❌ offset 范围问题
- ❌ 高频成分有害

### 待验证的假设

1. **随机打分反而更均匀**：精确打分可能过度偏向某些 token，随机选择反而保留更多样化的信息

2. **更激进的压缩**：`prefix_length` 偏大导致 `prune_target` 偏小，更激进的压缩可能迫使模型聚焦于真正重要的 token

3. **某种正则化效果**：错误的打分相当于给 token 选择加了噪声

## 7. 相关文件

- 修改的文件：`R-KV/weian_development/speckv/rkv_speckv_generate.py`
- Pruner 实现：`R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py`
- 打分函数：`R-KV/weian_development/speckv/round_pruning_utils.py`
- offset 消融脚本：`R-KV/weian_script/aime_sampled8/speckv/aime24/offset_ablation/`
- 高频消融脚本：`R-KV/weian_script/aime_sampled8/speckv/aime24/high_freq_ablation/`
- 基准实验脚本：`R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_budget.sh`
