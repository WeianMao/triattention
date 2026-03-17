# SpecKV 打分函数吞吐量优化方案

## 1. 背景与目标

### 1.1 项目背景

我们正在进行 KV cache 压缩算法的对比实验。需要对比的两个算法：

| 算法 | 启动脚本 | 核心实现 |
|------|----------|----------|
| **SpecKV**（我们的算法） | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh` | `R-KV/weian_development/speckv/speckv_rkv_style.py` |
| **R-KV**（对比算法） | `R-KV/weian_script/aime_sampled8/rkv/aime24/run_rkv_aime24_qwen.sh` | `R-KV/rkv/compression/r1_kv.py` |

### 1.2 对比维度

- **指标**：同一 GPU 型号上的 decoding 吞吐量（tokens/sec）
- **公平性约束**：只能修改**打分函数**和 **KV cache 裁剪**部分的算法和代码实现，不能改变算法的数学等价性

### 1.3 优化目标

在保持算法输出完全等价的前提下，通过优化计算流程来提高 SpecKV 的吞吐量。

---

## 2. 当前实现分析

### 2.1 SpecKV 打分流程概述

SpecKV 的核心思想是基于**频率统计**来预测哪些 KV cache token 在未来会被 attention 关注。

**打分发生的位置**：`speckv_rkv_style.py` 中的 `_compute_layer_head_scores()` 方法，调用 `round_pruning_utils.py` 中的工具函数。

**当前打分流程**（每个 sampled head）：

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 获取 RoPE 的 cos/sin 表                                       │
│    cos, sin = self.rotary(base, key_positions)                  │
│    [调用 transformers 的 RotaryEmbedding]                        │
├─────────────────────────────────────────────────────────────────┤
│ 2. 逆 RoPE 操作 (invert_rope)                                    │
│    k_unrot = invert_rope(k_rotated, cos, sin, scale)            │
│    [包含：3次除法 + rotate_half() + 乘法减法]                     │
├─────────────────────────────────────────────────────────────────┤
│ 3. 计算频率统计 (compute_frequency_statistics_from_means)        │
│    phi = angle(q_mean * conj(k_unrot))                          │
│    amp = |q_mean| * |k_unrot|                                   │
├─────────────────────────────────────────────────────────────────┤
│ 4. 计算分数 (score_keys_for_round)                               │
│    delta = round_start - key_indices                            │
│    phase = (delta + offset) * omega + phi                       │
│    score = sum(amp * cos(phase))  [昂贵的 cos 计算！]            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键代码位置

| 功能 | 文件 | 行号 | 函数名 |
|------|------|------|--------|
| 打分主入口 | `speckv_rkv_style.py` | 699-796 | `_compute_layer_head_scores()` |
| 逆 RoPE | `round_pruning_utils.py` | 51-83 | `invert_rope()` |
| 频率统计 | `round_pruning_utils.py` | 256-270 | `compute_frequency_statistics_from_means()` |
| 分数计算 | `round_pruning_utils.py` | 273-317 | `score_keys_for_round()` |

### 2.3 当前实现的性能瓶颈

#### 瓶颈 1：逆 RoPE 操作（最大瓶颈）

**代码位置**：`round_pruning_utils.py:51-83`

```python
def invert_rope(rotated, cos, sin, scale, *, style="half"):
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t           # 除法 1
    cos_unit = cos / scale_t           # 除法 2
    sin_unit = sin / scale_t           # 除法 3
    # ...
    return base * cos_unit - rotate_half(base, style=style) * sin_unit
```

**问题**：
- 每个 head、每次打分都要执行完整的逆 RoPE
- 包含 3 次张量除法 + `rotate_half()` 旋转操作 + 乘法减法
- 在调用 `invert_rope` 之前，还需要调用 `self.rotary(base, positions)` 获取 cos/sin 表

#### 瓶颈 2：大量 cos 计算

**代码位置**：`round_pruning_utils.py:300`

```python
cos_phase = torch.cos(phase)  # phase shape: [seq_len, num_offsets, freq_count]
```

**问题**：
- `phase` 张量维度为 `[seq_len, num_offsets, freq_count]`
- 例如 seq_len=2048, num_offsets=16, freq_count=64 → 2M+ 个 cos 计算
- GPU 计算三角函数相对较慢

#### 瓶颈 3：多 head 独立循环

**代码位置**：`speckv_rkv_style.py:749-791`

```python
for layer, head in layer_heads:  # 逐个 head 循环
    k_values = key_states[0, kv_head].index_select(0, gather_indices)
    # ... 计算分数
```

**问题**：
- 每个 sampled head 独立循环处理
- GQA 架构下，多个 attention head 映射到同一个 KV head，但仍分别读取 key
- 没有利用 batch 并行

---

## 3. 优化方案

### 3.1 方案一：消除逆 RoPE 操作（核心优化，必做）

#### 3.1.1 数学推导

**RoPE 的复数表示**：

对于位置 $p$ 的 key 向量，RoPE 对第 $i$ 个频率分量的作用是：
$$k_{\text{rotated}}^{(i)} = k_{\text{original}}^{(i)} \cdot e^{j \omega_i p}$$

其中 $\omega_i$ 是第 $i$ 个频率，$j$ 是虚数单位。

**逆 RoPE**：
$$k_{\text{original}}^{(i)} = k_{\text{rotated}}^{(i)} \cdot e^{-j \omega_i p}$$

**当前打分公式中的 phi**：

当前实现先做逆 RoPE，然后计算：
$$\phi_{\text{original}} = \angle(q_{\text{mean}} \cdot \overline{k_{\text{original}}})$$

其中 $\overline{(\cdot)}$ 表示复共轭，$\angle(\cdot)$ 表示取辐角。

**如果直接使用 $k_{\text{rotated}}$（不做逆 RoPE）**：

$$\phi_{\text{direct}} = \angle(q_{\text{mean}} \cdot \overline{k_{\text{rotated}}})$$

由于 $k_{\text{rotated}} = k_{\text{original}} \cdot e^{j \omega p_k}$（$p_k$ 是 key 的位置），有：

$$\phi_{\text{direct}} = \angle(q_{\text{mean}} \cdot \overline{k_{\text{original}}} \cdot e^{-j \omega p_k})$$
$$= \angle(q_{\text{mean}} \cdot \overline{k_{\text{original}}}) - \omega p_k$$
$$= \phi_{\text{original}} - \omega p_k$$

**代入 phase 公式**：

当前 phase 公式：
$$\text{phase} = (\text{round\_start} - p_k + \text{offset}) \cdot \omega + \phi_{\text{original}}$$

将 $\phi_{\text{original}} = \phi_{\text{direct}} + \omega p_k$ 代入：

$$\text{phase} = (\text{round\_start} - p_k + \text{offset}) \cdot \omega + \phi_{\text{direct}} + \omega p_k$$
$$= \text{round\_start} \cdot \omega - p_k \cdot \omega + \text{offset} \cdot \omega + \phi_{\text{direct}} + \omega p_k$$
$$= (\text{round\_start} + \text{offset}) \cdot \omega + \phi_{\text{direct}}$$

#### 3.1.2 关键发现

$$\boxed{\text{phase}_{\text{new}} = (\text{round\_start} + \text{offset}) \cdot \omega + \phi_{\text{direct}}}$$

**$p_k$（key 的位置）从 phase 公式中完全消失了！**

这意味着：
1. **不需要逆 RoPE 操作**：直接用 KV cache 中的 `k_rotated` 计算 `phi_direct`
2. **不需要调用 `self.rotary()`**：不用获取 cos/sin 表
3. **phase 计算简化**：不再需要 `key_indices`

#### 3.1.3 代码修改

**修改文件**：`round_pruning_utils.py`

**修改 1**：新增函数 `compute_frequency_statistics_from_rotated_keys()`

```python
def compute_frequency_statistics_from_rotated_keys(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_rotated: torch.Tensor,  # 直接使用 rotated key，不做逆 RoPE
    *,
    style: str = "half",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    与 compute_frequency_statistics_from_means 功能相同，
    但输入是 rotated key（KV cache 中的原始 key），不是 unrotated key。

    返回的 phi 是 phi_direct = angle(q_mean * conj(k_rotated))
    """
    k_complex = to_complex_pairs(k_rotated, style=style)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)

    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi_direct = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs

    return amp, phi_direct, extra
```

**修改 2**：新增函数 `score_keys_for_round_no_rope_inversion()`

```python
def score_keys_for_round_no_rope_inversion(
    round_start: int,
    amp: torch.Tensor,
    phi_direct: torch.Tensor,  # 来自 rotated key 的 phi
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
    freq_scale_sq: torch.Tensor,
    disable_top_n_high_freq: int = 0,
) -> torch.Tensor:
    """
    无逆 RoPE 版本的打分函数。

    phase = (round_start + offset) * omega + phi_direct

    注意：key_indices 参数被移除了，因为新公式不需要它。
    """
    if amp.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    seq_len = amp.shape[0]

    # 新公式：pos_delta 只依赖 round_start 和 offsets
    pos_delta = round_start + offsets  # [num_offsets]

    freq_scale_sq = freq_scale_sq.to(device=amp.device, dtype=torch.float32)

    # phase = pos_delta * omega + phi_direct
    # pos_delta: [num_offsets], omega: [freq_count], phi_direct: [seq_len, freq_count]
    # phase: [seq_len, num_offsets, freq_count]
    phase = pos_delta.view(1, -1, 1) * omega.view(1, 1, -1) + phi_direct.unsqueeze(1)

    cos_phase = torch.cos(phase)

    # High-frequency ablation
    if disable_top_n_high_freq > 0:
        position_scale = freq_scale_sq.clone()
        position_scale[:disable_top_n_high_freq] = 0
        scale = position_scale.view(1, 1, -1)
    else:
        scale = freq_scale_sq.view(1, 1, -1)

    base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)
    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1, keepdim=True)
    combined = base_scores + additive

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values
```

**修改文件**：`speckv_rkv_style.py`

**修改 3**：修改 `_compute_layer_head_scores()` 方法

将：
```python
# 获取 RoPE cos/sin 表
cos, sin = self.rotary(base, key_positions.unsqueeze(0))

# 逆 RoPE
k_unrot = invert_rope(k_values, cos_table, sin_table, self.attention_scale, style=self.rope_style)

# 计算频率统计（使用 unrotated key）
amp, phi, extra = compute_frequency_statistics_from_means(
    stats.q_mean_complex, stats.q_abs_mean, k_unrot, style=self.rope_style
)

# 打分（需要 key_indices）
head_scores = score_keys_for_round(
    key_indices=head_key_positions,
    round_start=self.absolute_position,
    ...
)
```

改为：
```python
# 直接使用 rotated key（无需获取 cos/sin，无需逆 RoPE）
amp, phi_direct, extra = compute_frequency_statistics_from_rotated_keys(
    stats.q_mean_complex, stats.q_abs_mean, k_values, style=self.rope_style
)

# 打分（不需要 key_indices）
head_scores = score_keys_for_round_no_rope_inversion(
    round_start=self.absolute_position,
    amp=amp,
    phi_direct=phi_direct,
    ...
)
```

---

### 3.2 方案二：预计算 cos/sin 偏置（在方案一基础上）

#### 3.2.1 原理

方案一之后，phase 公式变为：
$$\text{phase} = (\text{round\_start} + \text{offset}) \cdot \omega + \phi_{\text{direct}}$$

其中：
- $\text{offset} \cdot \omega$ 是**常量**（offsets 在初始化时就确定了）
- $\text{round\_start}$ 是压缩时的当前位置（变化的，但是标量）
- $\phi_{\text{direct}}$ 是每个 key 不同的（来自 key 的频率特征）

使用三角恒等式：
$$\cos(A + B) = \cos A \cdot \cos B - \sin A \cdot \sin B$$

令 $A = (\text{round\_start} + \text{offset}) \cdot \omega$，$B = \phi_{\text{direct}}$。

#### 3.2.2 代码修改

**修改文件**：`round_pruning_utils.py` 或 `speckv_rkv_style.py`

**修改 1**：在 `SpeckVRKVStyle.__init__()` 中预计算

```python
class SpeckVRKVStyle:
    def __init__(self, config: SpeckVRKVStyleConfig) -> None:
        # ... 现有初始化代码 ...

        # 预计算 offset * omega 的 cos/sin
        # offsets: [num_offsets], omega: [freq_count]
        offset_omega = self.offsets.unsqueeze(1) * self.omega.unsqueeze(0)  # [num_offsets, freq_count]
        self.cos_offset_omega = torch.cos(offset_omega)  # 缓存
        self.sin_offset_omega = torch.sin(offset_omega)  # 缓存
```

**修改 2**：新增优化版打分函数

```python
def score_keys_for_round_optimized(
    round_start: int,
    amp: torch.Tensor,
    phi_direct: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    cos_offset_omega: torch.Tensor,  # 预计算的 cos(offset * omega)
    sin_offset_omega: torch.Tensor,  # 预计算的 sin(offset * omega)
    aggregation: str,
    freq_scale_sq: torch.Tensor,
    disable_top_n_high_freq: int = 0,
) -> torch.Tensor:
    """
    使用预计算的 cos/sin 表进行打分。

    phase = (round_start + offset) * omega + phi_direct
          = round_start * omega + offset * omega + phi_direct

    令 A = round_start * omega + offset * omega (position-dependent, key-independent)
        B = phi_direct (key-dependent)

    cos(phase) = cos(A + B) = cos(A)*cos(B) - sin(A)*sin(B)
    """
    if amp.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    seq_len = amp.shape[0]
    freq_scale_sq = freq_scale_sq.to(device=amp.device, dtype=torch.float32)

    # round_start * omega: [freq_count]
    round_phase = round_start * omega
    cos_round = torch.cos(round_phase)
    sin_round = torch.sin(round_phase)

    # cos(round_phase + offset_omega) 使用三角恒等式
    # cos_offset_omega: [num_offsets, freq_count]
    # sin_offset_omega: [num_offsets, freq_count]
    cos_A = cos_round.unsqueeze(0) * cos_offset_omega - sin_round.unsqueeze(0) * sin_offset_omega
    sin_A = sin_round.unsqueeze(0) * cos_offset_omega + cos_round.unsqueeze(0) * sin_offset_omega
    # cos_A, sin_A: [num_offsets, freq_count]

    # phi_direct: [seq_len, freq_count]
    cos_B = torch.cos(phi_direct)
    sin_B = torch.sin(phi_direct)

    # cos(A + B) = cos(A)*cos(B) - sin(A)*sin(B)
    # cos_A: [num_offsets, freq_count] -> [1, num_offsets, freq_count]
    # cos_B: [seq_len, freq_count] -> [seq_len, 1, freq_count]
    cos_phase = cos_A.unsqueeze(0) * cos_B.unsqueeze(1) - sin_A.unsqueeze(0) * sin_B.unsqueeze(1)
    # cos_phase: [seq_len, num_offsets, freq_count]

    # 后续计算与原实现相同
    if disable_top_n_high_freq > 0:
        position_scale = freq_scale_sq.clone()
        position_scale[:disable_top_n_high_freq] = 0
        scale = position_scale.view(1, 1, -1)
    else:
        scale = freq_scale_sq.view(1, 1, -1)

    base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)
    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1, keepdim=True)
    combined = base_scores + additive

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values
```

#### 3.2.3 收益分析

| 计算项 | 优化前 | 优化后 |
|--------|--------|--------|
| `offset * omega` 的 cos/sin | 每次打分都算 | 初始化时算一次 |
| `round_start * omega` | 每次打分 | 每次打分（但只有 freq_count 个） |
| 大张量 cos | `seq_len * num_offsets * freq_count` | `seq_len * freq_count`（phi_direct） |

---

### 3.3 方案三：按 KV head 分组减少内存读取

#### 3.3.1 问题分析

当前实现中，GQA 架构下多个 attention head 映射到同一个 KV head：

```python
# 当前实现：每个 attention head 独立读取 key
for layer, head in layer_heads:
    kv_head = head // self.num_key_value_groups  # 多个 head 映射到同一个 kv_head
    k_values = key_states[0, kv_head].index_select(0, gather_indices)  # 重复读取！
    # ...
```

例如 Qwen-7B 有 28 个 attention heads，4 个 KV heads，每个 KV head 被 7 个 attention heads 共享。当前实现会读取同一个 KV head 7 次。

#### 3.3.2 代码修改

**修改文件**：`speckv_rkv_style.py`

**修改 `_compute_layer_head_scores()` 方法**：

```python
def _compute_layer_head_scores(
    self,
    key_states: torch.Tensor,
    key_positions: torch.Tensor,
    layer_idx: int,
    start_index: int = 0,
    positions_per_kv_head: Optional[List[torch.Tensor]] = None,
) -> Optional[torch.Tensor]:
    """优化版：按 KV head 分组，减少重复读取"""

    layer_heads = [(l, h) for l, h in self.sampled_heads if l == layer_idx]
    if not layer_heads:
        return None

    seq_len = key_positions.shape[0]
    gather_indices = torch.arange(seq_len, device=self.config.device, dtype=torch.long) + start_index

    # 按 KV head 分组
    kv_head_to_attn_heads: Dict[int, List[Tuple[int, int]]] = {}
    for layer, head in layer_heads:
        kv_head = head // max(1, self.num_key_value_groups)
        if kv_head not in kv_head_to_attn_heads:
            kv_head_to_attn_heads[kv_head] = []
        kv_head_to_attn_heads[kv_head].append((layer, head))

    per_head_scores: List[torch.Tensor] = []

    # 每个 KV head 只读取一次
    for kv_head, attn_heads in kv_head_to_attn_heads.items():
        # 读取这个 KV head 的 key（只读一次）
        k_values = key_states[0, kv_head].index_select(0, gather_indices)
        k_values = k_values.to(device=self.config.device, dtype=self.config.dtype)

        # 对映射到这个 KV head 的所有 attention heads 计算分数
        for layer, head in attn_heads:
            stats = self.head_stats[(layer, head)]

            # 使用优化后的打分函数（无逆 RoPE）
            amp, phi_direct, extra = compute_frequency_statistics_from_rotated_keys(
                stats.q_mean_complex, stats.q_abs_mean, k_values, style=self.rope_style
            )

            head_scores = score_keys_for_round_optimized(
                round_start=self.absolute_position,
                amp=amp,
                phi_direct=phi_direct,
                omega=self.omega,
                extra=extra,
                cos_offset_omega=self.cos_offset_omega,
                sin_offset_omega=self.sin_offset_omega,
                aggregation=self.score_aggregation,
                freq_scale_sq=self.freq_scale_sq,
                disable_top_n_high_freq=self.disable_top_n_high_freq,
            )
            per_head_scores.append(head_scores)

    if not per_head_scores:
        return None

    return torch.stack(per_head_scores, dim=0)
```

#### 3.3.3 收益分析

| 模型 | attention heads | KV heads | 优化前读取次数 | 优化后读取次数 |
|------|----------------|----------|---------------|---------------|
| Qwen-7B | 28 | 4 | 28 | 4 |
| Llama-3-8B | 32 | 8 | 32 | 8 |

---

## 4. 验证方案

### 4.1 数学等价性验证

在实施任何优化之前，必须验证优化后的打分结果与原实现完全一致。

**创建测试文件**：`R-KV/weian_development/tests/test_optimization_equivalence.py`

```python
"""
验证打分函数优化的数学等价性。

运行方法：
    cd /data/rbg/users/weian/project/rl/dc
    conda activate rkv
    python -m pytest R-KV/weian_development/tests/test_optimization_equivalence.py -v
"""
import torch
import pytest
from pathlib import Path

# 导入原实现
from weian_development.speckv.round_pruning_utils import (
    invert_rope,
    compute_frequency_statistics_from_means,
    score_keys_for_round,
    to_complex_pairs,
    build_rotary,
    build_geometric_offsets,
    compute_frequency_scaling,
)

# 假设新实现在同一文件或新文件中
# from weian_development.speckv.round_pruning_utils_optimized import (
#     compute_frequency_statistics_from_rotated_keys,
#     score_keys_for_round_no_rope_inversion,
# )


class TestNoRopeInversionEquivalence:
    """验证消除逆 RoPE 后的数学等价性"""

    @pytest.fixture
    def setup_test_data(self):
        """准备测试数据"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        # 模拟参数
        seq_len = 128
        head_dim = 128
        freq_count = head_dim // 2
        num_offsets = 16
        round_start = 1000

        # 生成随机 key（模拟 KV cache 中的 rotated key）
        k_rotated = torch.randn(seq_len, head_dim, device=device, dtype=dtype)

        # 生成 key 的位置
        key_indices = torch.arange(seq_len, device=device, dtype=torch.long)

        # 生成随机 q_mean（模拟统计量）
        q_mean_complex = torch.complex(
            torch.randn(freq_count, device=device, dtype=dtype),
            torch.randn(freq_count, device=device, dtype=dtype),
        )
        q_abs_mean = torch.abs(q_mean_complex) + torch.rand(freq_count, device=device, dtype=dtype) * 0.1

        # 构建 rotary embedding（使用真实配置）
        model_path = Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B")
        rotary = build_rotary(device, model_path, dtype)

        # 获取 omega 和 offsets
        omega = rotary.inv_freq.to(device=device, dtype=torch.float32)[:freq_count]
        offsets = build_geometric_offsets(65536, device)
        freq_scale = compute_frequency_scaling(rotary, head_dim, dtype, device)
        freq_scale_sq = freq_scale.pow(2)

        # 获取 cos/sin 表
        base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
        cos_table, sin_table = rotary(base, key_indices.unsqueeze(0))
        cos_table = cos_table[0]
        sin_table = sin_table[0]

        attention_scale = float(getattr(rotary, "attention_scaling", 1.0))
        rope_style = getattr(rotary, "_rope_style", "half")

        return {
            "k_rotated": k_rotated,
            "key_indices": key_indices,
            "q_mean_complex": q_mean_complex,
            "q_abs_mean": q_abs_mean,
            "cos_table": cos_table,
            "sin_table": sin_table,
            "attention_scale": attention_scale,
            "rope_style": rope_style,
            "omega": omega,
            "offsets": offsets,
            "freq_scale_sq": freq_scale_sq,
            "round_start": round_start,
            "device": device,
        }

    def test_phase_equivalence(self, setup_test_data):
        """验证新旧 phase 公式的等价性"""
        data = setup_test_data

        # ===== 原实现：使用逆 RoPE =====
        k_unrot = invert_rope(
            data["k_rotated"],
            data["cos_table"],
            data["sin_table"],
            data["attention_scale"],
            style=data["rope_style"],
        )

        amp_orig, phi_orig, extra_orig = compute_frequency_statistics_from_means(
            data["q_mean_complex"],
            data["q_abs_mean"],
            k_unrot,
            style=data["rope_style"],
        )

        # 原始 phase: (round_start - key_indices + offset) * omega + phi_orig
        base_delta = data["round_start"] - data["key_indices"].to(dtype=torch.float32)
        delta_grid = base_delta.unsqueeze(1) + data["offsets"].unsqueeze(0)
        phase_orig = delta_grid.unsqueeze(2) * data["omega"].view(1, 1, -1) + phi_orig.unsqueeze(1)

        # ===== 新实现：不做逆 RoPE =====
        # 直接用 k_rotated 计算 phi_direct
        k_complex_direct = to_complex_pairs(data["k_rotated"], style=data["rope_style"])
        relative_direct = data["q_mean_complex"].unsqueeze(0) * torch.conj(k_complex_direct)
        phi_direct = torch.atan2(relative_direct.imag, relative_direct.real)

        # 新 phase: (round_start + offset) * omega + phi_direct
        pos_delta = data["round_start"] + data["offsets"]
        phase_new = pos_delta.view(1, -1, 1) * data["omega"].view(1, 1, -1) + phi_direct.unsqueeze(1)

        # ===== 验证 cos(phase) 相等 =====
        cos_phase_orig = torch.cos(phase_orig)
        cos_phase_new = torch.cos(phase_new)

        # 允许小的浮点误差
        assert torch.allclose(cos_phase_orig, cos_phase_new, atol=1e-5, rtol=1e-4), \
            f"cos(phase) mismatch! Max diff: {(cos_phase_orig - cos_phase_new).abs().max().item()}"

        print(f"Phase equivalence verified! Max diff: {(cos_phase_orig - cos_phase_new).abs().max().item()}")

    def test_score_equivalence(self, setup_test_data):
        """验证最终分数的等价性"""
        data = setup_test_data

        # ===== 原实现 =====
        k_unrot = invert_rope(
            data["k_rotated"],
            data["cos_table"],
            data["sin_table"],
            data["attention_scale"],
            style=data["rope_style"],
        )

        amp_orig, phi_orig, extra_orig = compute_frequency_statistics_from_means(
            data["q_mean_complex"],
            data["q_abs_mean"],
            k_unrot,
            style=data["rope_style"],
        )

        scores_orig = score_keys_for_round(
            key_indices=data["key_indices"],
            round_start=data["round_start"],
            amp=amp_orig,
            phi=phi_orig,
            omega=data["omega"],
            extra=extra_orig,
            offsets=data["offsets"],
            aggregation="mean",
            freq_scale_sq=data["freq_scale_sq"],
        )

        # ===== 新实现（手动实现，等待正式代码） =====
        k_complex_direct = to_complex_pairs(data["k_rotated"], style=data["rope_style"])
        q_mean_abs = torch.abs(data["q_mean_complex"])
        k_abs = torch.abs(k_complex_direct)

        relative_direct = data["q_mean_complex"].unsqueeze(0) * torch.conj(k_complex_direct)
        phi_direct = torch.atan2(relative_direct.imag, relative_direct.real)
        amp_new = q_mean_abs.unsqueeze(0) * k_abs
        extra_new = (data["q_abs_mean"] - q_mean_abs).unsqueeze(0) * k_abs

        # 新打分公式
        pos_delta = data["round_start"] + data["offsets"]
        phase_new = pos_delta.view(1, -1, 1) * data["omega"].view(1, 1, -1) + phi_direct.unsqueeze(1)
        cos_phase_new = torch.cos(phase_new)

        scale = data["freq_scale_sq"].view(1, 1, -1)
        base_scores_new = (amp_new.unsqueeze(1) * scale * cos_phase_new).sum(dim=2)
        additive_new = (extra_new * data["freq_scale_sq"].view(1, -1)).sum(dim=1, keepdim=True)
        combined_new = base_scores_new + additive_new
        scores_new = combined_new.mean(dim=1)

        # ===== 验证分数相等 =====
        assert torch.allclose(scores_orig, scores_new, atol=1e-4, rtol=1e-3), \
            f"Score mismatch! Max diff: {(scores_orig - scores_new).abs().max().item()}"

        print(f"Score equivalence verified! Max diff: {(scores_orig - scores_new).abs().max().item()}")


class TestPrecomputedCosEquivalence:
    """验证预计算 cos/sin 偏置的等价性"""

    @pytest.fixture
    def setup_test_data(self):
        """准备测试数据"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        freq_count = 64
        num_offsets = 16
        seq_len = 128
        round_start = 1000

        omega = torch.rand(freq_count, device=device) * 0.1
        offsets = build_geometric_offsets(65536, device)[:num_offsets]
        phi_direct = torch.randn(seq_len, freq_count, device=device)

        return {
            "omega": omega,
            "offsets": offsets,
            "phi_direct": phi_direct,
            "round_start": round_start,
            "device": device,
        }

    def test_precomputed_cos_equivalence(self, setup_test_data):
        """验证预计算三角函数的等价性"""
        data = setup_test_data

        # ===== 原实现：直接计算 cos(phase) =====
        pos_delta = data["round_start"] + data["offsets"]
        phase = pos_delta.view(1, -1, 1) * data["omega"].view(1, 1, -1) + data["phi_direct"].unsqueeze(1)
        cos_phase_direct = torch.cos(phase)

        # ===== 新实现：使用预计算 + 三角恒等式 =====
        # 预计算 offset * omega 的 cos/sin
        offset_omega = data["offsets"].unsqueeze(1) * data["omega"].unsqueeze(0)
        cos_offset_omega = torch.cos(offset_omega)
        sin_offset_omega = torch.sin(offset_omega)

        # round_start * omega
        round_phase = data["round_start"] * data["omega"]
        cos_round = torch.cos(round_phase)
        sin_round = torch.sin(round_phase)

        # cos(A) where A = round_phase + offset_omega
        cos_A = cos_round.unsqueeze(0) * cos_offset_omega - sin_round.unsqueeze(0) * sin_offset_omega
        sin_A = sin_round.unsqueeze(0) * cos_offset_omega + cos_round.unsqueeze(0) * sin_offset_omega

        # cos(B), sin(B) where B = phi_direct
        cos_B = torch.cos(data["phi_direct"])
        sin_B = torch.sin(data["phi_direct"])

        # cos(A + B) = cos(A)*cos(B) - sin(A)*sin(B)
        cos_phase_precomputed = cos_A.unsqueeze(0) * cos_B.unsqueeze(1) - sin_A.unsqueeze(0) * sin_B.unsqueeze(1)

        # ===== 验证 =====
        assert torch.allclose(cos_phase_direct, cos_phase_precomputed, atol=1e-5), \
            f"Precomputed cos mismatch! Max diff: {(cos_phase_direct - cos_phase_precomputed).abs().max().item()}"

        print(f"Precomputed cos equivalence verified! Max diff: {(cos_phase_direct - cos_phase_precomputed).abs().max().item()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 4.2 端到端验证

在单元测试通过后，运行端到端测试验证优化后的输出与原实现一致。

```bash
# 1. 运行原实现获取 baseline 结果
cd /data/rbg/users/weian/project/rl/dc
conda activate rkv

# 使用小数据集快速验证
python R-KV/weian_development/rkv_sharded_runner.py \
    --dataset_path R-KV/HuggingFace/data/aime24.jsonl \
    --model_path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B \
    --output_dir /tmp/speckv_baseline \
    --method speckv \
    --kv_budget 2048 \
    --max_length 8192 \
    --num_samples 1 \
    --shard_id 0 \
    --total_shards 1 \
    --question_indices 0

# 2. 运行优化后实现
python R-KV/weian_development/rkv_sharded_runner.py \
    --dataset_path R-KV/HuggingFace/data/aime24.jsonl \
    --model_path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B \
    --output_dir /tmp/speckv_optimized \
    --method speckv \
    --kv_budget 2048 \
    --max_length 8192 \
    --num_samples 1 \
    --shard_id 0 \
    --total_shards 1 \
    --question_indices 0 \
    --use_optimized_scoring  # 需要新增此参数

# 3. 对比输出
python -c "
import json
baseline = json.load(open('/tmp/speckv_baseline/shard_0.jsonl'))
optimized = json.load(open('/tmp/speckv_optimized/shard_0.jsonl'))
assert baseline == optimized, 'Output mismatch!'
print('End-to-end equivalence verified!')
"
```

### 4.3 吞吐量测试

验证优化后吞吐量确实提升。

```bash
# 使用 PyTorch Profiler 或简单计时
python -c "
import torch
import time
from weian_development.speckv.round_pruning_utils import (
    invert_rope, score_keys_for_round, compute_frequency_statistics_from_means
)

device = torch.device('cuda')
seq_len, head_dim = 2048, 128
k_rotated = torch.randn(seq_len, head_dim, device=device)
# ... 准备其他数据 ...

# 原实现计时
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    k_unrot = invert_rope(k_rotated, cos_table, sin_table, scale)
    amp, phi, extra = compute_frequency_statistics_from_means(q_mean, q_abs, k_unrot)
    scores = score_keys_for_round(key_indices, round_start, amp, phi, omega, extra, offsets, 'mean', freq_scale_sq)
torch.cuda.synchronize()
print(f'Original: {(time.time() - start) * 10:.2f} ms/iter')

# 优化实现计时
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    amp, phi_direct, extra = compute_frequency_statistics_from_rotated_keys(q_mean, q_abs, k_rotated)
    scores = score_keys_for_round_optimized(round_start, amp, phi_direct, omega, extra, cos_offset_omega, sin_offset_omega, 'mean', freq_scale_sq)
torch.cuda.synchronize()
print(f'Optimized: {(time.time() - start) * 10:.2f} ms/iter')
"
```

---

## 5. 实施计划

### 5.1 优先级

| 优先级 | 优化项 | 预期收益 | 实施复杂度 | 依赖 |
|--------|--------|----------|------------|------|
| **P0** | 消除逆 RoPE | 高（去掉整个 invert_rope + rotary 调用） | 中 | 无 |
| **P1** | 预计算 cos(offset·ω) | 中（减少 cos 计算量） | 低 | P0 |
| **P2** | 按 KV head 分组 | 中（GQA 下减少内存读取） | 低 | 无 |

### 5.2 实施步骤

1. **Step 1**：编写并运行 `test_optimization_equivalence.py`，验证数学推导正确
2. **Step 2**：实现 `compute_frequency_statistics_from_rotated_keys()` 和 `score_keys_for_round_no_rope_inversion()`
3. **Step 3**：再次运行测试验证实现正确
4. **Step 4**：实现预计算 cos/sin 偏置
5. **Step 5**：实现按 KV head 分组
6. **Step 6**：端到端验证
7. **Step 7**：吞吐量测试

---

## 6. 附录

### 6.1 关键文件路径

| 用途 | 路径 |
|------|------|
| SpecKV 主实现 | `R-KV/weian_development/speckv/speckv_rkv_style.py` |
| 打分工具函数 | `R-KV/weian_development/speckv/round_pruning_utils.py` |
| 启动脚本 | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh` |
| 配置文件 | `R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml` |
| R-KV 对比实现 | `R-KV/rkv/compression/r1_kv.py` |

### 6.2 conda 环境

```bash
conda activate rkv
# Python 3.10, torch 2.3.1+cu121, transformers 4.48.1
```

### 6.3 联系人

如有问题请联系 weian。
