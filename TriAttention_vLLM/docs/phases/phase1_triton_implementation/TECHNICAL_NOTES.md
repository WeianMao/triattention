# Phase 1 技术可行性评估

基于 Triton API 分析和性能建模的技术可行性评估。

---

## 1. 关键结论

| 项目 | 状态 | 说明 |
|-----|------|------|
| **技术可行性** | ✓ 可行 | 需要 workaround |
| **性能预期调整** | ⚠️ 需修正 | 原 2-3x → 实际 1.3-1.7x |
| **Triton API 限制** | ⚠️ 需 workaround | 无 atan2、topk、complex |

---

## 2. Triton API 限制与解决方案

### 2.1 无 `tl.math.atan2`

**影响**：SpeckV 打分公式需要计算相位角 φ = atan2(Im, Re)

**解决方案**：RoPE 反演实际上是**纯矩阵操作**，不需要 atan2！

原始代码（`round_pruning_utils.py` L83-84）：
```python
def invert_rope(rotated, cos, sin, scale, style="half"):
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t

    # 反演公式（纯矩阵操作）
    return base * cos_unit - rotate_half(base, style=style) * sin_unit
```

**Triton 实现**：
```python
@triton.jit
def invert_rope_triton(
    rotated_ptr,  # [seq_len, head_dim]
    cos_ptr,      # [seq_len, freq_count]
    sin_ptr,      # [seq_len, freq_count]
    output_ptr,
    scale,
    seq_len,
    head_dim,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < seq_len

    d_offs = tl.arange(0, BLOCK_D)
    half_dim = head_dim // 2

    # 加载 rotated 和 cos/sin
    rotated = tl.load(rotated_ptr + n_offs[:, None] * head_dim + d_offs, mask=n_mask[:, None])
    cos = tl.load(cos_ptr + n_offs[:, None] * half_dim + d_offs % half_dim, mask=n_mask[:, None])
    sin = tl.load(sin_ptr + n_offs[:, None] * half_dim + d_offs % half_dim, mask=n_mask[:, None])

    # RoPE 反演（纯算术）
    base = rotated / scale
    cos_unit = cos / scale
    sin_unit = sin / scale

    # rotate_half: [first_half, second_half] -> [-second_half, first_half]
    is_first_half = d_offs < half_dim
    base_rotated = tl.where(is_first_half, -base, base)  # 简化实现

    output = base * cos_unit - base_rotated * sin_unit
    tl.store(output_ptr + n_offs[:, None] * head_dim + d_offs, output, mask=n_mask[:, None])
```

### 2.2 无 `tl.topk`

**影响**：选择 top-k token 需要高效实现

**解决方案**：使用 `tl.sort()` + 索引追踪

**方案 A：简单排序（适合小 k）**
```python
@triton.jit
def topk_sort(scores_ptr, indices_ptr, k, seq_len, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_len

    # 加载 scores
    scores = tl.load(scores_ptr + offs, mask=mask, other=-float('inf'))

    # 创建索引
    indices = offs

    # 排序（Triton 的 sort 是 bitonic sort）
    sorted_scores, sorted_indices = tl.sort(scores, indices, descending=True)

    # 取前 k 个
    k_mask = offs < k
    tl.store(indices_ptr + offs, sorted_indices, mask=k_mask)
```

**方案 B：阈值过滤（适合大 seq_len）**
```python
# 1. 采样估计第 k 大的阈值
# 2. 并行过滤 > threshold
# 3. 处理边界情况
```

**推荐**：混合方案
```python
def topk_hybrid(scores, k):
    if scores.shape[-1] < 4096:
        return torch.topk(scores, k, dim=-1).indices  # PyTorch
    else:
        return triton_block_topk(scores, k)  # Triton
```

### 2.3 无 `tl.complex`

**影响**：频率统计计算涉及复数

**解决方案**：手动分解为实部和虚部

```python
# 原始 PyTorch（complex）
k_complex = torch.view_as_complex(k_unrot.view(..., 2))
q_mean_complex = stats.q_mean_complex
amp = torch.abs(q_mean_complex) * torch.abs(k_complex)
phi = torch.angle(q_mean_complex * k_complex.conj())

# Triton 等效（real/imag 分离）
@triton.jit
def compute_amp_phi(
    k_real_ptr, k_imag_ptr,
    q_real_ptr, q_imag_ptr,
    amp_ptr, phi_real_ptr, phi_imag_ptr,  # phi 用 (cos, sin) 表示
    ...
):
    # |q| = sqrt(q_real² + q_imag²)
    q_abs = tl.sqrt(q_real * q_real + q_imag * q_imag + 1e-8)
    k_abs = tl.sqrt(k_real * k_real + k_imag * k_imag + 1e-8)

    # amp = |q| * |k|
    amp = q_abs * k_abs

    # q * conj(k) = (q_r + j*q_i) * (k_r - j*k_i)
    #             = (q_r*k_r + q_i*k_i) + j*(q_i*k_r - q_r*k_i)
    prod_real = q_real * k_real + q_imag * k_imag
    prod_imag = q_imag * k_real - q_real * k_imag

    # 不需要 atan2，因为后续只需要 cos(phase)
    # cos(delta*omega + phi) 可以用 cos/sin 加法公式展开
    prod_norm = tl.sqrt(prod_real * prod_real + prod_imag * prod_imag + 1e-8)
    phi_cos = prod_real / prod_norm  # cos(phi)
    phi_sin = prod_imag / prod_norm  # sin(phi)
```

**关键洞察**：后续打分只需要 `cos(phase)`，可以用角度加法公式：
```
cos(delta*omega + phi) = cos(delta*omega)*cos(phi) - sin(delta*omega)*sin(phi)
```
因此不需要显式计算 phi 的数值，只需要 (cos(phi), sin(phi))。

---

## 3. 性能预期修正

### 3.1 原始估计（错误）

| 假设 | 预期加速 |
|-----|---------|
| K/V 压缩 8192 → 256 (32x) | 2-3x |
| 融合 kernel 减少内存往返 | 额外 1.5x |

### 3.2 实际分析

**Attention 计算的内存带宽分析**：

| 组件 | 大小 | 压缩后 | 占比 |
|-----|------|--------|------|
| Q | [batch, heads, 1, dim] | 不变 | ~40% |
| K | [batch, heads, 8192, dim] | [batch, heads, 256, dim] | ~25% |
| V | [batch, heads, 8192, dim] | [batch, heads, 256, dim] | ~25% |
| Output | [batch, heads, 1, dim] | 不变 | ~10% |

**K/V 压缩只影响 50% 的内存带宽**，Q 和 Output 不变。

**实际加速计算**：
```
原始带宽占比: Q(40%) + K(25%) + V(25%) + O(10%) = 100%
压缩后带宽: Q(40%) + K(25%/32) + V(25%/32) + O(10%) ≈ 51.6%

理论加速: 100% / 51.6% ≈ 1.94x

考虑 overhead (TopK + 打分 ~0.4ms):
实际加速: 1.3-1.7x
```

### 3.3 修正后的性能目标

| 场景 | 基线 | 目标 | 加速比 |
|-----|------|------|-------|
| seq_len=8K, budget=256 | 2.5ms | 1.5-1.9ms | **1.3-1.7x** |
| seq_len=16K, budget=512 | 5.0ms | 3.0-3.5ms | **1.4-1.7x** |
| seq_len=32K, budget=1024 | 10.0ms | 6.0-7.0ms | **1.4-1.7x** |

### 3.4 加速的主要来源

1. **K/V 带宽减少**：~1.5-2x
2. **融合 kernel**：减少 launch overhead ~0.1ms
3. **避免中间结果**：减少内存分配

**不是来源**：
- ❌ 计算量减少（attention 仍是 O(N) 因为 Q 未变）
- ❌ 并行度提升（已经充分并行）

---

## 4. 推荐实现策略

### 4.1 两阶段 Kernel 策略

**Stage 1: 打分 + TopK**（~0.3-0.4ms）
- 输入：K [batch, heads, seq_len, dim]
- 操作：RoPE 反演 → 频率打分 → TopK
- 输出：indices [batch, heads, k]

**Stage 2: Gather + Attention**（~1.1-1.5ms）
- 输入：K, V, indices, Q
- 操作：Gather K/V → Standard Flash Attention
- 输出：Attention output

### 4.2 为什么不融合所有操作

1. **打分需要完整 K**：无法流水线化
2. **TopK 有全局依赖**：需要看完所有 token
3. **Flash Attention 已高度优化**：复用比重写更好

### 4.3 代码结构

```python
class TriAttentionCompressor:
    def compress(self, key_states, value_states, positions):
        # Stage 1: Triton 打分 + TopK
        scores = scoring_kernel(key_states, self.stats, self.omega, ...)
        indices = topk_kernel(scores, k=self.config.budget)

        # Stage 2: PyTorch Gather（或 Triton Gather）
        k_compressed = key_states.gather(dim=2, index=indices)
        v_compressed = value_states.gather(dim=2, index=indices)

        return k_compressed, v_compressed, positions.gather(-1, indices)
```

---

## 5. 风险评估矩阵

| 风险 | 概率 | 影响 | 缓解措施 | 残余风险 |
|-----|------|------|---------|---------|
| Triton TopK 效率不足 | 中 | 中 | 混合策略 | 低 |
| 复数运算误差累积 | 低 | 高 | FP32 中间计算 | 低 |
| 内存布局不优 | 中 | 中 | 预 Gather 或转置 | 低 |
| Per-head warp 发散 | 中 | 低 | 批量多 head | 低 |
| 性能不达 1.3x | 低 | 高 | 调优或降级 PyTorch | 中 |

---

## 6. 开发建议

### 6.1 分阶段实现

1. **Week 1**：PyTorch 参考实现 + 打分 Triton kernel
2. **Week 2**：TopK Triton kernel + 融合测试
3. **Week 3**：集成 + 性能调优

### 6.2 测试优先级

1. **正确性**：与 PyTorch 实现数值对比
2. **性能**：端到端 benchmark
3. **边界情况**：小 seq_len、大 batch

### 6.3 备选方案

如果 Triton 性能不达预期：
- 使用 `torch.compile` 优化 PyTorch 实现
- 仅 TopK + Gather 使用 Triton，打分用 PyTorch
- 考虑 CUDA kernel（复杂度更高）

---

## 7. 参考实现

### 7.1 可参考的 Triton Kernels

| Kernel | 位置 | 参考价值 |
|--------|------|---------|
| Flash Attention | `vLLM/vllm/attention/ops/triton_flash_attention.py` | 内存布局、数值稳定性 |
| Decode Attention | `SGLang/srt/layers/attention/triton_ops/decode_attention.py` | 双阶段设计 |
| Logits Softcap | `SGLang/srt/layers/logits_processor.py` | 简单 Triton 模式 |

### 7.2 TopK 参考（CUDA）

| 实现 | 位置 | 说明 |
|-----|------|------|
| vLLM MOE TopK | `vLLM/csrc/moe/topk_softmax_kernels.cu` | CUDA 高效实现 |
| SGLang TopK | `SGLang/srt/layers/moe/topk.py` | 调用 vLLM ops |

---

*创建日期：2025-01-31*
