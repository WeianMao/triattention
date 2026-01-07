# Bug 896cbca6 变量影响分析

## 概述

Commit `896cbca6` 的 bug 导致 pruner 状态在多道题之间不会被重置。本文档逐一分析每个受影响变量对流程的具体影响。

**根本原因**：`orig_forward` 就地修改 `past_key_values`，导致 `is_empty_cache` 检查永远为 False，状态重置逻辑永远不执行。

---

## 受影响变量列表

| 变量 | 正常行为 | Bug 行为 |
|------|----------|----------|
| `absolute_position` | 每道题重置为 0 | 跨题目累积递增 |
| `cache_positions` | 每道题重新初始化 | 保留前面题目的残留位置 |
| `prefix_length` | 每道题设为当前 prefill 长度 | 永远是第一道题的值 |
| `tokens_in_round` | 每道题重置为 0 | 不重置 |
| `attached` | 每道题重置为 False | 第一道题后永远是 True |

---

## 1. `absolute_position`

**定义位置**：`SparseRoundPruner` 类

**正常值**：每道题从 0 开始，随 decode 递增

**Bug 值**：跨题目累积，第 N 道题开始时可能是几千甚至上万

### 使用位置

#### 1.1 计算 position_ids (RoPE)

```python
# rkv_speckv_generate.py:177-189
start_pos = state.pruner.absolute_position  # ← Bug: 累积值而非 0
abs_positions = torch.arange(
    start_pos,
    start_pos + step,
    device=input_ids.device,
    dtype=torch.long,
)
position_ids_override = abs_positions
```

#### 1.2 作为打分函数的 round_start

```python
# sparse_round_pruner_prefill_keep.py:536-538
head_scores = score_keys_for_round(
    key_indices=key_positions,
    round_start=self.absolute_position,  # ← Bug: 累积值
    ...
)
```

```python
# round_pruning_utils.py:288-289
base_delta = round_start - key_indices.to(device=amp.device, dtype=torch.float32)
delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)
```

### 影响分析

**对模型推理的影响**：❌ **无影响**

RoPE 是相对位置编码。只要 query 和 key 的相对位置不变，attention 结果就相同：

```
正确版本: key 用 [0,1,2...], query 用 [100]  → 相对位置 [100,99,98...]
Bug 版本: key 用 [2500,2501,...], query 用 [2600] → 相对位置 [100,99,98...]
结果相同！
```

**对打分函数的影响**：⚠️ **间接影响，但相比 invert_rope 错误是次要的**

#### base_delta 是什么？

```python
# round_pruning_utils.py
base_delta = round_start - key_indices
#            ↑                ↑
#            absolute_position    cache_positions
#            (当前 query 位置)    (每个 key 的位置)
```

`base_delta[i]` = 当前 query 离第 i 个 key 有多远（相对距离）

#### base_delta 用来干什么？

SpeckV 打分公式需要预测：**未来的 query 会有多关注这个 key？**

```python
# 简化版打分公式
phase = base_delta * omega + phi      # omega 是 RoPE 频率，phi 是 query-key 相位差
score = amp * cos(phase)              # amp 是 query-key 幅度乘积
```

这个公式的物理意义：
- `base_delta` 越大 = 这个 key 离当前位置越远
- `cos(base_delta * omega)` 会随距离振荡
- SpeckV 用这个来预测 attention 分数

#### 正常情况 vs Bug 情况的详细对比

假设：第 1 题 prefill 500 + decode 2000，第 2 题 prefill 400 + decode 500

```
================== 正常情况（每道题重置状态）==================

【第 2 题 Prefill 完成后】

kv_cache 内容：
    kv_cache[0] = 第2题第1个token的key，用位置 0 做 RoPE 编码
    kv_cache[1] = 第2题第2个token的key，用位置 1 做 RoPE 编码
    ...
    kv_cache[399] = 第2题第400个token的key，用位置 399 做 RoPE 编码

状态变量：
    cache_positions = [0, 1, 2, ..., 399]     ← 记录每个 slot 的编码位置
    absolute_position = 400                    ← 下一个 token 该用的位置

一致性检查：
    kv_cache[i] 用位置 i 编码  ←→  cache_positions[i] = i  ✓ 完全一致！

【第 2 题 Decode 到第 500 个 token 时】

kv_cache 内容：
    kv_cache[0~399] = prefill 的 key，用位置 0~399 编码
    kv_cache[400~899] = decode 的 key，用位置 400~899 编码

状态变量：
    cache_positions = [0, 1, ..., 399, 400, 401, ..., 899]
    absolute_position = 900

打分时计算 base_delta：
    base_delta[0] = absolute_position - cache_positions[0] = 900 - 0 = 900
    含义：kv_cache[0] 是 900 个位置之前的 token ✓ 正确！


================== Bug 情况（状态不重置）==================

【第 2 题 Prefill 完成后】

kv_cache 内容（新的 cache，但用累积位置编码）：
    kv_cache[0] = 第2题第1个token的key，用位置 2500 做 RoPE 编码
    kv_cache[1] = 第2题第2个token的key，用位置 2501 做 RoPE 编码
    ...
    kv_cache[399] = 第2题第400个token的key，用位置 2899 做 RoPE 编码

    为什么用 2500 开始？因为 absolute_position = 2500（第1题遗留）

状态变量（被错误处理）：
    cache_positions = [2100, 2105, 2110, ..., 2495]  ← 第1题最后400个残留位置！
    absolute_position = 2500                          ← 没更新（走了截断分支）

一致性检查：
    kv_cache[0] 用位置 2500 编码  ←→  cache_positions[0] = 2100  ✗ 不一致！
    差了 400！

【第 2 题 Decode 到第 500 个 token 时】

kv_cache 内容：
    kv_cache[0~399] = prefill 的 key，用位置 2500~2899 编码
    kv_cache[400~899] = decode 的 key，用位置 2900~3399 编码

状态变量：
    cache_positions = [2100, 2105, ..., 2495, 2900, 2901, ..., 3399]
                       ↑____ 第1题残留 ____↑  ↑___ 新追加（正确）___↑
    absolute_position = 3400

打分时计算 base_delta：
    base_delta[0] = absolute_position - cache_positions[0] = 3400 - 2100 = 1300

    但实际上 kv_cache[0] 是用位置 2500 编码的
    正确的 base_delta[0] 应该是 3400 - 2500 = 900

    错了 400！（= 2500 - 2100）
```

**总结**：

| | 正常情况 | Bug 情况 |
|---|---|---|
| `kv_cache[0]` 编码位置 | 0 | 2500 |
| `cache_positions[0]` 记录 | 0 | 2100（第1题残留）|
| 是否一致 | ✓ | ✗ 差了 400 |
| `base_delta[0]` 计算 | 正确 | 错了 400 |

#### 两个错误会抵消吗？

**直觉**：如果 `absolute_position` 和 `cache_positions` 都多了相同的偏移量，那 `base_delta = absolute_position - cache_positions` 差值应该不变？

**答案**：不会抵消。原因如下：

```
base_delta = absolute_position - cache_positions

正常情况（第2题 Decode 到第500个token）：
    absolute_position = 900                      (从0开始，prefill 400 + decode 500)
    cache_positions = [0, 1, 2, ..., 899]        (正确记录)
    kv_cache 编码位置 = [0, 1, 2, ..., 899]      (与记录一致)

    base_delta[0] = 900 - 0 = 900  ✓

Bug 情况（第2题 Decode 到第500个token）：
    absolute_position = 3000                     (累积值：2500 + 500)
    cache_positions = [2100, 2105, ..., 2495, 2500, 2501, ..., 2999]
                       ↑____第1题残留____↑   ↑____新追加____↑
    kv_cache 编码位置 = [2500, 2501, ..., 2899, 2500, 2501, ..., 2999]
                        ↑____prefill____↑     ↑____decode____↑

    对于 kv_cache[0]（prefill 第1个token）：
        实际编码位置 = 2500
        cache_positions[0] = 2100（第1题残留）
        absolute_position = 3000

        base_delta[0] = 3000 - 2100 = 900   ← Bug 计算
        正确 base_delta = 3000 - 2500 = 500 ← 应该是这个

        错了 400！(= 2500 - 2100)
```

**为什么不能抵消**：

| 错误类型 | 错误量 |
|---------|-------|
| `absolute_position` | 多了 2500（第1题的累积） |
| `cache_positions[0]` | 是 2100 而非 2500（第1题残留，不是简单的偏移） |

关键：`cache_positions` 的错误**不是简单的"+2500 偏移"**，而是**被第1题的残留值覆盖了**。

```
如果 cache_positions 也是简单累积（都+2500）：
    cache_positions[0] = 0 + 2500 = 2500
    base_delta = 3000 - 2500 = 500  ✓ 这样就抵消了

但实际是：
    cache_positions[0] = 2100（第1题 decode 阶段某个被保留的位置）
    base_delta = 3000 - 2100 = 900  ✗ 差了 400
```

**结论**：两个错误不能抵消，因为它们的错误方式不同：
- `absolute_position`：线性累积
- `cache_positions`：被截断成第1题的残留值（随机位置，不是线性偏移）

---

## Bug 的等效效果：数学推导

让我用复数形式严格推导 bug 对打分的影响。

### RoPE 的复数表示

```
RoPE 编码（对于频率分量 ω）：
    key_rotated = original_key × e^{i × p × ω}
    其中 p = 编码位置

invert_rope（用位置 p' 逆转）：
    key_recovered = key_rotated × e^{-i × p' × ω}
                  = original_key × e^{i × p × ω} × e^{-i × p' × ω}
                  = original_key × e^{i × (p - p') × ω}
                  = original_key × e^{i × Δ × ω}

    其中 Δ = p - p' = 实际编码位置 - cache_positions记录的位置
```

**关键结论**：如果 p ≠ p'，恢复的 key 会多一个 e^{i×Δ×ω} 的相位旋转。

### Bug 对各个量的影响

对于 cache slot j：
- p_j = 实际编码位置（如 2500）
- p'_j = cache_positions[j]（如 2100，第1题残留）
- Δ_j = p_j - p'_j（如 400）

```
1. key_recovered：
   错误 key_recovered = original_key × e^{i × Δ_j × ω}
   （多了 Δ_j × ω 的相位旋转）

2. amp = |q_mean| × |key_recovered|：
   |key_recovered| = |original_key × e^{i × Δ_j × ω}| = |original_key|
   ✓ amp 不变！（复数的模不受相位影响）

3. phi = angle(q_mean × conj(key_recovered))：
   错误 phi = angle(q_mean × conj(original_key) × e^{-i × Δ_j × ω})
            = 正确phi - Δ_j × ω
   ✗ phi 被偏移了 -Δ_j × ω

4. extra = (|q|_abs_mean - |q_mean|) × |key_recovered|：
   ✓ extra 不变！（也是基于模）

5. base_delta：
   正确 base_delta = 正确absolute_position - p_j
   错误 base_delta = 错误absolute_position - p'_j

   巧合的是，由于 absolute_position 的累积方式，错误 base_delta ≈ 正确 base_delta
   （具体分析见上文）
```

### 最终 phase 的变化

```
phase = base_delta × ω + phi

正确 phase = 正确base_delta × ω + 正确phi
错误 phase = 正确base_delta × ω + (正确phi - Δ_j × ω)
           = 正确phase - Δ_j × ω
```

**Bug 的等效效果**：phase 被偏移了 -Δ_j × ω

### score 的变化

```
score = Σ(amp × freq_scale² × cos(phase)) + Σ(extra × freq_scale²)

正确 score = Σ(amp × scale × cos(正确phase)) + extra_term
错误 score = Σ(amp × scale × cos(正确phase - Δ_j × ω)) + extra_term
                                    ↑
                          这里 cos 的参数被偏移了
```

### Δ_j 的分布

```
Δ_j = p_j - p'_j = 实际编码位置 - cache_positions[j]

对于第2题的 cache：
    slot 0: Δ_0 = 2500 - 2100 = 400
    slot 1: Δ_1 = 2501 - 2105 = 396
    slot 2: Δ_2 = 2502 - 2110 = 392
    slot 3: Δ_3 = 2503 - 2115 = 388
    ...

Δ_j 的分布取决于第1题压缩时保留了哪些位置！
如果第1题保留的是 [2100, 2105, 2110, ...]（每隔5个），那么 Δ 会线性变化。
如果保留的是 [2100, 2150, 2180, ...]（不规则），那么 Δ 分布更复杂。
```

---

## 核心结论：Bug 等效于什么？

```
Bug 等效于：给每个 token 的打分加一个 phase 偏移

score_j = Σ(amp_j × scale × cos(正确phase_j - Δ_j × ω)) + extra_term
                                        ↑
                              phase 偏移量 = Δ_j × ω

其中：
- amp 不变 ✓
- extra 不变 ✓
- 只有 phase 被偏移 ✗

偏移量 Δ_j = 实际编码位置 - 第1题残留的 cache_positions[j]
```

### 这不是"完全随机"的打分！

```
❌ 之前的理解：invert_rope 产生"垃圾"数据，打分完全随机

✓ 实际情况：
  1. amp 和 extra 都是正确的（只依赖模，不依赖相位）
  2. 只有 phase 被偏移了 Δ_j × ω
  3. Δ_j 的分布取决于第1题的压缩历史
  4. 打分是"有结构的扰动"，不是完全随机

等效于：cos(phase) → cos(phase - Δ_j × ω)
```

### 对不同频率分量的影响

```
高频分量（ω 大）：偏移量 Δ_j × ω 大，cos 变化剧烈
低频分量（ω 小）：偏移量 Δ_j × ω 小，cos 变化平缓

如果 Δ_j = 400，对于：
- 最低频 ω_63 ≈ 0.0001：偏移 400 × 0.0001 = 0.04 弧度 ≈ 2°
- 最高频 ω_0 ≈ 1：偏移 400 × 1 = 400 弧度 ≈ 很多圈，等效随机

所以：高频分量的打分贡献变成"随机"，低频分量基本不变
```

---

## 修正之前的结论

| 之前的理解 | 修正后的理解 |
|-----------|-------------|
| invert_rope 产生"垃圾"数据 | invert_rope 产生的是 original_key × e^{iΔω}，相位偏移而非垃圾 |
| amp, phi, extra 都是错的 | amp 和 extra 正确，只有 phi 被偏移 |
| 打分完全随机 | 打分是"phase 偏移后的结果"，高频接近随机，低频基本不变 |
| absolute_position 错误有中等影响 | absolute_position 错误和 cache_positions 错误部分抵消，base_delta 基本正确 |

---

## 为什么高频消融实验没效果？

### 实验回顾

我们尝试用 `--disable-top-n-high-freq` 参数显式 mask 高频分量，假设：
- Bug 让高频变成随机噪声
- 如果高频有害，显式 disable 应该也能提点

但实验结果显示：**没有显著效果**

### 原因分析

```
Bug 的效果：
    score = Σ(amp × scale × cos(phase - Δ × ω)) + extra

    高频（ω 大）：cos(phase - Δ × ω) ≈ 随机值（因为 Δ × ω 很大）
    低频（ω 小）：cos(phase - Δ × ω) ≈ cos(phase)（基本不变）

高频消融的效果：
    score = Σ(amp × scale × cos(phase))，但 scale[高频] = 0

    相当于：直接去掉高频分量的贡献
```

### 两者不等价！

```
Bug 效果：高频贡献 = amp × scale × cos(随机值) ≈ amp × scale × 随机数
消融效果：高频贡献 = 0

区别：
- Bug：高频仍有贡献，但是随机的（可能正可能负）
- 消融：高频完全没有贡献

如果 amp[高频] 很大，Bug 情况下高频的随机贡献可能主导了分数
消融情况下这部分贡献直接消失了

两种情况对分数排序的影响不同！
```

### 更本质的原因

```
Bug 情况下：
    score_j ∝ Σ_i (amp_j[i] × cos(phase_j[i] - Δ_j × ω[i]))

    关键：Δ_j 对每个 token j 是不同的！

    token A: Δ_A = 400，高频 cos 偏移 400 × ω
    token B: Δ_B = 396，高频 cos 偏移 396 × ω
    token C: Δ_C = 350，高频 cos 偏移 350 × ω

    虽然都是"随机"，但每个 token 的随机种子（Δ_j）不同
    这引入了一种"基于 Δ_j 的伪随机排序"

消融情况下：
    score_j ∝ Σ_{低频} (amp_j[i] × cos(phase_j[i]))

    完全去掉了高频，排序只基于低频
    这和 Bug 的"伪随机排序"是不同的机制
```

### 结论

**高频消融实验无法复现 Bug 的效果**，因为：

1. Bug 不是简单地"让高频变成随机噪声"
2. Bug 是"让高频的 phase 偏移一个依赖于 token 位置的量 Δ_j × ω"
3. 这个 Δ_j 依赖于第1题的压缩历史，引入了一种"结构化的扰动"
4. 直接 disable 高频和这种"结构化扰动"是完全不同的操作

---

## Δ 的实际数值分析

通过分析 AIME24 数据集的输出结果，我们可以精确计算 Δ 的大小。

### 数据集统计

```
AIME24 数据集 (30道题):
  Prefill 长度: min=95, max=427, mean=156.4 tokens
  Decode 长度:  min=2844, max=32639, mean=16196.1 tokens
```

### Δ 的来源分析

```
第1题结束时:
  - absolute_position = P1 + D1 (prefill + decode)
  - cache_positions = [0, 1, ..., P1-1, 一些被保留的decode位置]
                       └─ 前缀保护 ─┘  └─ 经过压缩采样 ─┘

第2题 prefill P2 个 token:
  - 实际编码位置 = [P1+D1, P1+D1+1, ..., P1+D1+P2-1]
  - cache_positions 被截断成最后 P2 个

如果压缩后 cache_positions 末尾是近似连续的:
  cache_positions[-P2:] ≈ [P1+D1-P2, P1+D1-P2+1, ..., P1+D1-1]

则:
  Δ[j] = (P1+D1+j) - (P1+D1-P2+j) = P2
```

### 验证结果

| 题目 | Δ (实测) | prefill_len |
|------|---------|-------------|
| 1    | 160     | 159         |
| 2    | 129     | 129         |
| 3    | 134     | 134         |
| 4    | 118     | 118         |
| 5    | 202     | 202         |

**关键发现**：Δ ≈ 当前题目的 prefill 长度，而**不是**前一题的 decode 长度，也**不会累加**！

### 统计结果

```
AIME24 数据集 (30道题):
  Δ 最小值: 95 tokens
  Δ 最大值: 427 tokens
  Δ 平均值: 156 tokens
  Δ 中位数: 144 tokens
```

### 对相位的影响

对于典型的 Δ ≈ 160 tokens，RoPE base=10000，head_dim=128：

| 频率索引 | ω | 相位偏移(度) | 绕圈数 |
|---------|---|------------|-------|
| 0 (最高频) | 1.0 | -166° | 25.5 |
| 8 | 0.316 | 28° | 8.1 |
| 16 | 0.1 | -161° | 2.6 |
| 32 | 0.01 | 92° | 0.3 |
| 63 (最低频) | 0.00012 | 1° | 0.003 |

**结论**：
- 高频分量（28/64 个）：绕圈 > 0.5，相位接近随机
- 低频分量（25/64 个）：绕圈 < 0.1，相位基本不变

---

## 一句话总结

**Bug 等效于**：在 SpeckV 打分公式中，给每个 token 的 phase 加一个偏移量 Δ × ω，其中 **Δ ≈ 当前题目的 prefill 长度**（约 100-400 tokens）。

这导致：
- amp 和 extra 不变
- 高频 phase 被大幅偏移（接近随机）
- 低频 phase 基本不变
- 打分变成"低频主导 + 高频随机扰动"的混合

---

## 2. `cache_positions`

**定义位置**：`SparseRoundPruner` 类

**正常值**：每道题初始化为 `[0, 1, ..., prefill_len-1]`，decode 时追加新位置

**Bug 值**：保留前面题目的残留位置，被截断或混合

### 使用位置

#### 2.1 生成 invert_rope 的 cos/sin table（核心影响）

```python
# sparse_round_pruner_prefill_keep.py:275-277
key_positions = torch.tensor(
    self.cache_positions, device=self.config.device, dtype=torch.long
)

# sparse_round_pruner_prefill_keep.py:501
cos_table, sin_table = self._rotary_for_positions(key_positions)

# sparse_round_pruner_prefill_keep.py:523-529
k_unrot = invert_rope(
    k_values,
    cos_table,  # ← Bug: 基于错误位置生成
    sin_table,  # ← Bug: 基于错误位置生成
    self.attention_scale,
    style=self.rope_style,
)
```

#### 2.2 作为打分函数的 key_indices

```python
# sparse_round_pruner_prefill_keep.py:536-537
head_scores = score_keys_for_round(
    key_indices=key_positions,  # ← Bug: 错误的位置
    ...
)
```

#### 2.3 计算 _dynamic_cache_size

```python
# sparse_round_pruner_prefill_keep.py:184-189
@property
def _dynamic_cache_size(self) -> int:
    if self.config.include_prefill_in_budget:
        return len(self.cache_positions)  # ← 只影响长度，不影响正确性
    return max(0, len(self.cache_positions) - self.prefix_length)
```

### 影响分析

**对 SpeckV 打分的影响**：✅ **严重错误 - 这是 bug 最核心的影响**

```python
# invert_rope 的工作原理：
# 1. KV cache 中的 key 用位置 [p₁, p₂, ...] 经过 RoPE 编码
# 2. invert_rope 需要用相同位置的 cos/sin 来逆转 RoPE，恢复原始 key
# 3. 如果位置不匹配，恢复出来的就是垃圾数据

# Bug 情况下：
# - KV cache 中的 key 实际用位置 [2500, 2501, ...] 编码
# - 但 cache_positions 记录的是第 1 题残留的位置 [1850, 1900, ...]
# - _rotary_for_positions 生成错误位置的 cos/sin
# - invert_rope 用错误的 cos/sin 逆转 → 得到垃圾 k_unrot
# - 后续基于 k_unrot 的打分基本是随机的
```

**对 _dynamic_cache_size 的影响**：⚠️ **轻微**

只影响 `len(cache_positions)` 的计算，不影响打分正确性，只可能导致压缩时机略有偏差。

---

## 3. `prefix_length`

**定义位置**：`SparseRoundPruner` 类

**正常值**：当前题目的 prefill token 数量

**Bug 值**：永远是第一道题的 prefill 长度

### 使用位置

#### 3.1 计算 _dynamic_cache_size（非默认分支）

```python
# sparse_round_pruner_prefill_keep.py:184-189
@property
def _dynamic_cache_size(self) -> int:
    if self.config.include_prefill_in_budget:
        return len(self.cache_positions)  # ← 不受 prefix_length 影响
    return max(0, len(self.cache_positions) - self.prefix_length)
```

#### 3.2 计算 prune_target

```python
# sparse_round_pruner_prefill_keep.py:221-228
if self.allow_prefill_compression:
    prune_target = keep_capacity
elif self.config.include_prefill_in_budget:
    prune_target = max(0, keep_capacity - self.prefix_length)  # ← Bug: 第一题的值
else:
    prune_target = keep_capacity
```

#### 3.3 Prefill token 保护

```python
# sparse_round_pruner_prefill_keep.py:302-306
prefix_count = min(self.prefix_length, candidate_count)  # ← Bug: 第一题的值
dynamic_count = max(0, candidate_count - prefix_count)
keep_count = max(0, min(keep_count, dynamic_count))
prefix_indices = candidate_indices[:prefix_count]  # 保护前 prefix_count 个 token
```

### 影响分析

**对模型推理的影响**：❌ **无直接影响**

`prefix_length` 不参与模型的 forward 计算，只影响 SpeckV 的压缩决策。

**对压缩行为的影响**：⚠️ **有影响但可能不大**

| 题目 | 实际 prefill | prefix_length (Bug) | 后果 |
|------|-------------|---------------------|------|
| 第 1 题 | 500 | 500 | 正确 |
| 第 2 题 | 400 | 500 | prune_target 偏小 100，压缩更激进；保护了 500 个 token 而非 400 |
| 第 3 题 | 600 | 500 | prune_target 偏大 100，压缩更宽松；只保护 500 而非 600 |

**结论**：`prefix_length` 错误会导致：
1. 压缩目标偏移（可能更激进或更宽松）
2. Prefill 保护范围不准确

但这些都是**二阶影响**，核心问题是 `cache_positions` 导致的打分错误。

---

## 4. `tokens_in_round`

**定义位置**：`SparseRoundPruner` 类

**正常值**：每道题从 0 开始，每轮压缩后重置

**Bug 值**：不重置（但在 `rkv_aligned_budget` 模式下 round-based pruning 被跳过）

### 使用位置

#### 4.1 判断是否开始新一轮

```python
# sparse_round_pruner_prefill_keep.py:240-241
def should_start_next_round(self) -> bool:
    return self.tokens_in_round >= self.round_window
```

#### 4.2 在 generate 中的调用（被跳过）

```python
# rkv_speckv_generate.py:274-277
# Skip round-based pruning when using R-KV aligned budget mode
if not state.pruner.rkv_aligned_budget:  # ← 在 rkv_aligned_budget 模式下跳过
    while state.pruner.should_start_next_round():
        pkv_tuple = state.pruner.start_next_round(pkv_tuple)
```

### 影响分析

**对模型推理的影响**：❌ **无影响**

在 `rkv_aligned_budget=True` 模式下，round-based pruning 逻辑被完全跳过。`tokens_in_round` 虽然不正确，但没有任何代码使用它。

---

## 5. `attached`

**定义位置**：`_SpeckVState` 类

**正常值**：每道题开始时为 False，attach 后变为 True

**Bug 值**：第一道题后永远是 True

### 使用位置

#### 5.1 检查是否需要重置

```python
# rkv_speckv_generate.py:167-170
if is_empty_cache and state.attached:  # ← Bug: is_empty_cache 永远 False
    state.pruner = SparseRoundPruner(state.config)
    state.attached = False
    state.initial_prefix_length = None
```

#### 5.2 决定走哪个分支

```python
# rkv_speckv_generate.py:253-272
if not state.attached:
    # 第一次 attach - 只在第一道题执行
    state.pruner.attach_initial_cache(pkv_tuple)
    state.initial_prefix_length = state.pruner.prefix_length
    pkv_tuple = state.pruner.enforce_max_limit(pkv_tuple)
    state.attached = True
else:
    # 后续更新 - 第 2 道题开始永远走这里
    seq_len = pkv_tuple[0][0].shape[2]
    cached_len = len(state.pruner.cache_positions)
    if cached_len < seq_len:
        # 追加新位置
        ...
    elif cached_len > seq_len:
        # 截断 cache_positions
        state.pruner.cache_positions = state.pruner.cache_positions[-seq_len:]
```

### 影响分析

**对模型推理的影响**：❌ **无直接影响**

`attached` 只是一个流程控制标志，不参与任何数学计算。

**对状态管理的影响**：✅ **这是 bug 的触发点**

因为 `attached=True` 且 `is_empty_cache=False`（orig_forward 就地修改导致），第 2 道题开始永远走 else 分支，导致：
- `attach_initial_cache` 不被调用 → 所有状态不被正确初始化
- `cache_positions` 被错误截断而非重新初始化

---

## 伪代码：核心算法逻辑

```python
# ============================================================
# 变量定义
# ============================================================

absolute_position = 0      # 当前生成到第几个 token（从 0 开始计数）
cache_positions = []       # 列表：cache_positions[i] = 第 i 个 cache slot 存的是哪个位置的 token
prefix_length = 0          # prefill 阶段有多少个 token（这些 token 不参与压缩竞争）

# ============================================================
# 第一阶段：Prefill（一次性处理整个 prompt）
# ============================================================

def prefill(prompt_tokens):
    """处理 prompt，生成初始 KV cache"""
    n = len(prompt_tokens)  # 假设 prompt 有 n 个 token

    # 1. 计算 RoPE position_ids = [0, 1, 2, ..., n-1]
    position_ids = [0, 1, 2, ..., n-1]

    # 2. 对每个 token 做 RoPE 编码并存入 KV cache
    for i in range(n):
        key[i] = RoPE_encode(token_embedding[i], position=i)
        # 实际存的是: key[i] = original_key[i] * cos(i*ω) + rotate(original_key[i]) * sin(i*ω)

    # 3. 初始化状态变量
    absolute_position = n           # 下一个要生成的 token 位置是 n
    cache_positions = [0, 1, ..., n-1]  # cache slot i 存的是位置 i 的 token
    prefix_length = n               # 记住 prefill 有 n 个 token

# ============================================================
# 第二阶段：Decode（逐个生成 token）
# ============================================================

def decode_one_token():
    """生成一个新 token"""

    # 1. 当前 query 的位置就是 absolute_position
    query_position = absolute_position

    # 2. 用这个位置做 RoPE 编码
    query = RoPE_encode(query_embedding, position=query_position)

    # 3. 计算 attention（RoPE 保证了相对位置正确）
    #    attention(query, key) 的结果只依赖于 query_position - key_position（相对位置）
    output = attention(query, kv_cache)

    # 4. 把新 token 加入 cache
    new_key = RoPE_encode(new_embedding, position=query_position)
    kv_cache.append(new_key)

    # 5. 更新状态
    cache_positions.append(absolute_position)  # 新 slot 存的是位置 absolute_position
    absolute_position += 1                      # 位置 +1

# ============================================================
# 第三阶段：SpeckV 打分与压缩
# ============================================================

def speckv_score_and_compress(kv_cache, budget):
    """当 cache 太大时，用 SpeckV 算法选择保留哪些 token"""

    # 核心问题：cache 里的 key 是 RoPE 编码后的，我们需要恢复原始 key 来打分

    scores = []
    for i in range(len(kv_cache)):
        # 1. 取出 cache 中第 i 个位置的 key（已经被 RoPE 编码过了）
        encoded_key = kv_cache[i]

        # 2. 【关键】用 cache_positions[i] 生成对应的 cos/sin
        #    这个位置必须和当初编码时用的位置一致！
        pos = cache_positions[i]
        cos_table = cos(pos * ω)
        sin_table = sin(pos * ω)

        # 3. 逆转 RoPE，恢复原始 key
        #    如果 pos 错误 → cos/sin 错误 → 恢复出来的是垃圾
        original_key = invert_RoPE(encoded_key, cos_table, sin_table)

        # 4. 用预先统计的 query 均值和恢复的 key 计算打分
        #    score = 预测这个 key 在未来会有多重要
        score = compute_score(
            original_key,           # 恢复的原始 key
            query_mean_stats,       # 预先统计的 query 分布
            query_position=absolute_position,  # 当前 query 位置
            key_position=pos                   # 这个 key 的位置
        )
        scores.append(score)

    # 5. 按分数排序，保留 top-k（同时保护 prefill token）
    protected = cache[:prefix_length]          # 前 prefix_length 个不参与竞争
    candidates = cache[prefix_length:]         # 只有 decode token 参与竞争
    keep_indices = topk(scores[prefix_length:], k=budget-prefix_length)

    # 6. 压缩 cache 和 cache_positions
    new_cache = protected + [candidates[i] for i in keep_indices]
    cache_positions = cache_positions[:prefix_length] + [cache_positions[prefix_length+i] for i in keep_indices]

# ============================================================
# Bug 的影响：cache_positions 的精确演变
# ============================================================

"""
============ 第 1 题（正常执行）============

Prefill 阶段（500 tokens）：
    - attach_initial_cache() 被调用
    - cache_positions = [0, 1, 2, ..., 499]
    - absolute_position = 500
    - prefix_length = 500
    - KV cache 中的 key 用位置 [0, 1, ..., 499] 编码 ✓

Decode 阶段（生成 2000 tokens，假设 budget=1024）：
    - 每生成一个 token:
        cache_positions.append(absolute_position)  # 追加 500, 501, 502, ...
        absolute_position += 1

    - 假设在位置 1200 触发压缩（cache 满了）：
        压缩前: cache_positions = [0,1,...,499, 500,501,...,1199]  (长度 1200)
        SpeckV 打分后保留 top-1024，假设丢弃了一些中间位置
        压缩后: cache_positions = [0,1,...,499, 502,510,520,...,1199]  (长度 1024)
        注意：前 500 个（prefill）被保护，只有 decode 部分参与竞争

第 1 题结束时的状态：
    absolute_position = 2500  (500 prefill + 2000 decode)
    cache_positions = [0,1,...,499, 还有524个被保留的decode位置]  (长度约 1024)
    prefix_length = 500
    KV cache 中的 key 用位置 [0,1,...,499, 那524个位置] 编码 ✓
    此时 cache_positions 和实际编码位置是一致的 ✓

============ 第 2 题开始（Bug 触发）============

新题目 Prefill（400 tokens）：
    1. transformers.generate() 创建新的空 DynamicCache
    2. speckv_forward 被调用
    3. 【关键】position_ids 计算：
       start_pos = absolute_position = 2500  (第1题遗留!)
       position_ids = [2500, 2501, ..., 2899]

    4. orig_forward 执行：
       - 用 position_ids = [2500, 2501, ..., 2899] 对新 prompt 做 RoPE 编码
       - KV cache 现在有 400 个 token，用位置 [2500, 2501, ..., 2899] 编码
       - 【副作用】DynamicCache 被就地修改，不再为空！

    5. is_empty_cache 检查：
       cache.get_seq_length() = 400 > 0  →  is_empty_cache = False
       不会重置状态！

    6. 走 else 分支：
       seq_len = 400  (新 cache 的实际长度)
       cached_len = 1024  (第1题遗留的 cache_positions 长度)

       因为 cached_len > seq_len，执行截断：
       cache_positions = cache_positions[-400:]

       结果：cache_positions = [第1题最后400个被保留的位置]
             比如 = [2100, 2105, 2110, ..., 2495]  (第1题 decode 阶段的某些位置)

第 2 题 Prefill 后的状态：
    absolute_position = 2500  (没变，因为走的是 cached_len > seq_len 分支)
    cache_positions = [2100, 2105, 2110, ..., 2495]  (第1题的残留位置！)
    prefix_length = 500  (还是第1题的值！)

    但 KV cache 中实际存的是：
        第2题的 prompt，用位置 [2500, 2501, ..., 2899] 编码

    不匹配！
    cache_positions[0] = 2100，但 kv_cache[0] 是用位置 2500 编码的
    cache_positions[1] = 2105，但 kv_cache[1] 是用位置 2501 编码的
    ...

============ 第 2 题 Decode 阶段 ============

第一次触发 SpeckV 打分时：
    for i in range(len(kv_cache)):
        pos = cache_positions[i]          # 比如 2100（错的！）
        cos_table = cos(pos * ω)          # 用 2100 生成 cos
        sin_table = sin(pos * ω)          # 用 2100 生成 sin

        encoded_key = kv_cache[i]         # 实际是用位置 2500 编码的

        # 逆转 RoPE：
        # 正确做法：用 2500 的 cos/sin 逆转
        # 实际做法：用 2100 的 cos/sin 逆转
        # 结果：垃圾数据
        original_key = invert_RoPE(encoded_key, cos_table, sin_table)  # 错！

============ 总结：错位的本质 ============

                     cache_positions 记录的    KV cache 实际编码位置
第1题 Prefill:       [0, 1, ..., 499]          [0, 1, ..., 499]           ✓ 一致
第1题 Decode:        [0,...,499, 502,510,...]  [0,...,499, 502,510,...]   ✓ 一致
第2题 Prefill:       [2100, 2105, ...]         [2500, 2501, ...]          ✗ 完全错位！
第2题 Decode:        [2100, 2105, ..., 2500, 2501, ...]  [2500, ..., 2900, ...]  ✗ 继续错位

错位原因：
    1. cache_positions 被截断成第1题的残留值
    2. 但 absolute_position 继续累积，新 token 用累积位置编码
    3. 新 token 追加到 cache_positions 时用的是累积的 absolute_position
    4. 所以 prefill 部分完全错位，decode 部分追加的是对的，但整体混乱
"""
```

---

## 总结

### 对 decode 结果有明显影响的变量

| 变量 | 影响程度 | 影响类型 |
|------|----------|----------|
| `cache_positions` | ✅ **严重** | 导致 `invert_rope` 产生垃圾 `k_unrot`，打分基本随机 |
| `absolute_position` | ⚠️ 中等 | 与 `cache_positions` 配合，导致 `base_delta` 错误 |
| `prefix_length` | ⚠️ 轻微 | 压缩目标和保护范围偏移 |

### 对 decode 结果无影响的变量

| 变量 | 原因 |
|------|------|
| `tokens_in_round` | `rkv_aligned_budget` 模式下不被使用 |
| `attached` | 只是流程控制标志 |

### 核心结论

**Bug 的核心影响是 `cache_positions` 错误**，导致：

1. `invert_rope` 用错误位置的 cos/sin 去逆转 key
2. 恢复出来的 `k_unrot` 是垃圾数据
3. 基于 `k_unrot` 计算的 `amp`, `phi`, `extra` 都是错误的
4. 最终 `score_keys_for_round` 的打分基本是**随机的**

模型推理本身不受影响（RoPE 相对位置正确），但 SpeckV 的 token 选择变成了随机选择。
