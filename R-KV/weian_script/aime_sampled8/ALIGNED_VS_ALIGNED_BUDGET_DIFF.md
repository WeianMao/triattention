# aligned vs aligned_budget 实现差异分析

本文档详细分析 `run_speckv_aime24_qwen_norm_aligned.sh` 和 `run_speckv_aime24_qwen_norm_aligned_budget.sh` 两个脚本的实现差异，解释为什么它们产生不同的实验结果。

## 脚本配置对比

| 参数 | `aligned_budget.sh` | `aligned.sh` |
|------|---------------------|--------------|
| `--include-prefill-in-budget` | ✓ | ✓ |
| `--rkv-aligned-budget` | ✓ | ✗ |
| `--rkv-style-compression` | ✗ | ✓ |
| `--divide-length 128` | ✓ | ✓ |
| 实现文件 | `rkv_speckv_generate.py` → `SparseRoundPruner` | `speckv_rkv_style.py` → `SpeckVRKVStyle` |

---

## 差异汇总

| 优先级 | 问题 | 严重程度 | 类型 |
|--------|------|----------|------|
| **1** | Token 选择算法不同 | **严重** | 算法差异 |
| **2** | 缺少噪声注入 | **严重** | Bug |
| **3** | 归一化范围不同 | **中等** | 算法差异 |
| **4** | 分数计算范围不同 | **低** | 实现差异 |
| **5** | 压缩触发条件不同 | **低** | 实现差异（实测等价） |

---

## 问题 1：Token 选择算法不同（最严重）

### 问题描述

两个实现使用了**完全不同的 token 选择算法**：

**aligned (`speckv_rkv_style.py`) - 简单 Top-K：**
```python
# speckv_rkv_style.py:213-215
decode_scores = combined[prefix_length:]  # 取 decode 部分的 max-pooled 分数
k = min(decode_budget, decode_scores.shape[0])
decode_topk = torch.topk(decode_scores, k=k, largest=True).indices  # 直接选 top-k
```

**aligned_budget (`SparseRoundPruner`) - Union-Based 选择：**
```python
# sparse_round_pruner_prefill_keep.py:424-451
# 步骤 1: 每个 head 独立选择 top-k
per_head_quota = min(keep_count, candidate_count)
union_mask = torch.zeros(candidate_count, device=combined.device, dtype=torch.bool)
for head_scores in per_head_scores:
    head_k = min(per_head_quota, head_scores.numel())
    top_idx = torch.topk(head_scores, k=head_k, largest=True).indices
    union_mask[top_idx] = True  # 加入并集

# 步骤 2: 从并集中选择最终结果
union_indices = torch.nonzero(union_mask, as_tuple=False).view(-1)
if union_indices.numel() >= keep_count:
    # 从并集中按 combined 分数选 top-k
    subset_scores = combined.index_select(0, union_indices)
    top_subset = torch.topk(subset_scores, k=keep_count, largest=True).indices
    return union_indices.index_select(0, torch.sort(top_subset).values)

# 步骤 3: 如果并集不够，从剩余 token 补充
remaining = keep_count - union_indices.numel()
if remaining > 0:
    residual_scores = combined.clone()
    residual_scores[union_mask] = float("-inf")
    extra_indices = torch.topk(residual_scores, k=remaining, largest=True).indices
    union_indices = torch.cat([union_indices, extra_indices])
```

### 算法差异详解

| 步骤 | aligned (简单 Top-K) | aligned_budget (Union-Based) |
|------|---------------------|------------------------------|
| 1 | 对每个 token 取所有 head 的最大分数 | 每个 head 独立选出自己的 top-k token |
| 2 | 直接选分数最高的 k 个 token | 把所有 head 的选择合并成并集 |
| 3 | 完成 | 从并集中按 combined 分数选 top-k |
| 4 | - | 如果并集不够，从剩余 token 补充 |

### 影响分析

Union-Based 算法的设计目的是**确保每个 head 认为重要的 token 都有机会被保留**。

**举例说明差异：**

假设有 2 个 head，3 个 token，budget = 1：

| Token | Head 0 分数 | Head 1 分数 | Max (combined) |
|-------|------------|------------|----------------|
| A | 10.0 | 1.0 | 10.0 |
| B | 1.0 | 9.0 | 9.0 |
| C | 5.0 | 5.0 | 5.0 |

- **简单 Top-K**：选 combined 最高的 → 选 **A**
- **Union-Based**：
  - Head 0 选 top-1 → A
  - Head 1 选 top-1 → B
  - 并集 = {A, B}
  - 从并集中按 combined 选 top-1 → 选 **A**

在这个例子中结果相同，但当 budget 更大、head 更多时，Union-Based 会保留更多"单个 head 认为重要但 combined 不是最高"的 token。

### 为什么这是最严重的问题

1. **算法本质不同**：不是实现细节差异，而是选择策略完全不同
2. **影响每次压缩**：每次触发压缩时都会产生不同的保留集合
3. **累积效应**：差异会随着生成长度增加而累积

---

## 问题 2：缺少噪声注入（Bug）

### 问题描述

`speckv_rkv_style.py` 创建了随机数生成器但**从未使用**：

```python
# speckv_rkv_style.py:137-143 - 创建 generator
self.generator: torch.Generator | None = None
if config.seed is not None:
    if config.device.type == "cuda":
        self.generator = torch.Generator(device=config.device)
    else:
        self.generator = torch.Generator()
    self.generator.manual_seed(int(config.seed))

# 但是在 compute_keep_indices 中从未使用！
```

而 `SparseRoundPruner` 正确使用了噪声注入：

```python
# sparse_round_pruner_prefill_keep.py:526-532
if self.generator is not None and head_matrix.numel() > 0:
    noise = torch.rand(
        head_matrix.shape,
        device=head_matrix.device,
        generator=self.generator,
    ) * 1e-6
    head_matrix = head_matrix + noise
```

### 为什么需要噪声注入

1. **打破平局**：当多个 token 分数完全相同时，`torch.topk` 的选择是不确定的
2. **可复现性**：通过 seeded generator 确保相同种子产生相同结果
3. **避免系统性偏差**：没有噪声时，平局可能总是选择相同位置的 token

### 影响分析

当 token 分数非常接近时：
- `aligned`：选择结果可能不稳定或有系统性偏差
- `aligned_budget`：通过噪声确保确定性选择

### 修复建议

在 `speckv_rkv_style.py` 的 `compute_keep_indices` 方法中，第 203 行之后添加：

```python
# 添加噪声打破平局（与 SparseRoundPruner 对齐）
if self.generator is not None and head_matrix.numel() > 0:
    noise = torch.rand(
        head_matrix.shape,
        device=head_matrix.device,
        generator=self.generator,
    ) * 1e-6
    head_matrix = head_matrix + noise
```

---

## 问题 3：归一化范围不同

### 问题描述

**aligned (`speckv_rkv_style.py`) - 在所有 token 上归一化：**

```python
# speckv_rkv_style.py:145-206
def compute_keep_indices(self, pkv_tuple, prefix_length=0):
    kv_cache_len = pkv_tuple[0][0].shape[-2]  # 总 cache 长度
    key_positions = torch.tensor(
        self.cache_positions[:kv_cache_len], ...  # 所有位置
    )

    # 对所有 token 计算分数
    all_head_scores = []
    for layer_idx, (key_states, _) in enumerate(pkv_tuple):
        layer_scores = self._compute_layer_head_scores(key_states, key_positions, layer_idx)
        all_head_scores.append(layer_scores)

    head_matrix = torch.cat(all_head_scores, dim=0)  # [heads, prefill+decode]

    # 在所有 token 上归一化
    if self.normalize_scores:
        mean = head_matrix.mean(dim=1, keepdim=True)  # 均值包含 prefill
        std = head_matrix.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        head_matrix = (head_matrix - mean) / std
```

**aligned_budget (`SparseRoundPruner`) - 仅在 decode token 上归一化：**

```python
# sparse_round_pruner_prefill_keep.py:267-284
# 在 _prune_to_size 中，只传入 decode 位置
dynamic_positions = key_positions[prefix_count:]  # 仅 decode 位置
dynamic_keep_rel = self._select_keep_indices(
    past_key_values,
    dynamic_positions,  # <- 只有 decode token
    keep_count,
    start_index=prefix_count,
)

# sparse_round_pruner_prefill_keep.py:521-525
# _compute_head_scores 中的归一化只涉及传入的 token
if self.normalize_scores and head_matrix.numel() > 0:
    mean = head_matrix.mean(dim=1, keepdim=True)  # 均值只包含 decode
    std = head_matrix.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    head_matrix = (head_matrix - mean) / std
```

### 影响分析

假设：
- Prefill token 分数分布：均值 = 5.0，标准差 = 1.0
- Decode token 分数分布：均值 = -1.0，标准差 = 2.0

**aligned 的归一化（所有 token）：**
- 全局均值 ≈ (5.0 × 150 + (-1.0) × 2000) / 2150 ≈ -0.58
- Decode token 的 z-score 会被 prefill 的高分数影响

**aligned_budget 的归一化（仅 decode）：**
- 均值 = -1.0
- Decode token 的 z-score 只反映 decode 内部的相对重要性

### 实测结果

通过测试脚本 `test_normalization_scope_diff.py` 验证：
- 即使 prefill 分数分布差异很大，top-k 选择重叠率仍达 **99%+**
- 平均排名差异约 **18 位**
- 结论：**影响存在但相对较小**

---

## 问题 4：分数计算范围不同

### 问题描述

| 实现 | key_positions 包含 | 分数计算对象 |
|------|-------------------|-------------|
| aligned | 所有 token (prefill + decode) | 所有 token |
| aligned_budget | 仅 decode token | 仅 decode token |

### 代码位置

**aligned：**
```python
# speckv_rkv_style.py:171-175
key_positions = torch.tensor(
    self.cache_positions[:kv_cache_len],  # 所有位置
    device=self.config.device,
    dtype=torch.long
)
```

**aligned_budget：**
```python
# sparse_round_pruner_prefill_keep.py:277-278
dynamic_positions = key_positions[prefix_count:]  # 仅 decode 位置
```

### 影响分析

分数计算本身是独立的（每个 token 的分数只取决于该 token 的 key 值和位置），所以这个差异的直接影响较小。主要影响在于：

1. **计算开销**：aligned 会多计算 prefill token 的分数（虽然最终不参与选择）
2. **归一化**：与问题 3 相关，影响归一化的统计量

---

## 问题 5：压缩触发条件不同

### 问题描述

**aligned (`speckv_rkv_style.py`)：**
```python
# speckv_rkv_style.py:487-489
should_compress = (comp.absolute_position % comp.divide_length == 0) if is_decode_step else False

if effective_size >= comp.budget and should_compress:
    # 触发压缩
```
触发条件：`cache >= budget` **且** `absolute_position % 128 == 0`

**aligned_budget (`rkv_speckv_generate.py`)：**
```python
# rkv_speckv_generate.py:261-265
trigger_threshold = (state.pruner.max_keys + state.pruner.divide_length
                     if state.pruner.rkv_aligned_budget
                     else state.pruner.max_keys)
if state.pruner._dynamic_cache_size >= trigger_threshold:
    # 触发压缩
```
触发条件：`cache >= budget + divide_length`（即 `cache >= 2176`）

### 实测结果

通过测试脚本 `test_compression_trigger_diff.py` 验证：

```
First 5 compression events:

aligned:
  Step 2025: abs_pos= 2176, cache 2176 -> 2048

aligned_budget:
  Step 2025: abs_pos= 2176, cache 2176 -> 2048

Compression position comparison:
  Overlap: 24/24 (100%)
```

**结论：两种触发条件在实际运行中完全等价**，都是在 cache 达到 2176 时触发压缩。

---

## 总结

### 最关键的差异

1. **Token 选择算法**：简单 top-k vs union-based，这是算法设计层面的本质差异
2. **噪声注入缺失**：这是一个 bug，应该修复

### 次要差异

3. **归一化范围**：影响存在但实测较小
4. **分数计算范围**：主要影响归一化
5. **压缩触发条件**：实测等价，无影响

### 修复优先级

1. **高**：修复噪声注入缺失
2. **中**：决定是否需要对齐选择算法（如果需要完全等价）
3. **低**：归一化范围对齐（可选）

---

## 相关文件

### 实现文件
- `R-KV/weian_development/speckv/speckv_rkv_style.py` - aligned 实现
- `R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py` - aligned_budget 核心
- `R-KV/weian_development/speckv/rkv_speckv_generate.py` - aligned_budget 入口

### 测试脚本
- `R-KV/weian_development/tests/test_normalization_scope_diff.py` - 归一化差异测试
- `R-KV/weian_development/tests/test_compression_trigger_diff.py` - 压缩触发测试
- `R-KV/weian_development/tests/test_selection_algorithm_diff.py` - 选择算法测试

### 脚本文件
- `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned.sh`
- `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_budget.sh`
