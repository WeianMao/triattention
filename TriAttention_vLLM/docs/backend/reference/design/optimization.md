# 优化设计

本文档描述 TriAttention 打分计算的三个核心优化。

---

## 1. 优化目标

### 1.1 原始实现的瓶颈

| 瓶颈 | 问题 |
|-----|------|
| RoPE 反转 | 每轮打分需要对所有 key 反转 RoPE，计算量大 |
| 多次显存读取 | 对 N 个未来位置打分需要从显存读取 K 共 N 次 |
| 冗余三角函数 | 每个 token × 每个 offset 都计算 $\cos$、$\sin$ |

### 1.2 优化后

| 优化 | 效果 |
|-----|------|
| 避免 RoPE 反转 | 直接用 $\mathbf{K}_{\text{rot}}$ 计算，通过相位校正等价 |
| 单次读取 | 每轮打分只从显存读取 K 一次 |
| 共享三角表 | 预计算 $\cos(t\omega)$、$\sin(t\omega)$，所有 token 共享 |

---

## 2. 优化 1：避免 RoPE 反转

### 2.1 问题

显存中存储的是旋转后的 key $\mathbf{K}_{\text{rot}}$，但打分公式需要原始 key $\mathbf{K}$。

**原来的做法**：
```
1. 读取 K_rot
2. 反转 RoPE：K = K_rot × e^{-ipω}  ← 复数乘法
3. 计算 φ = arg(Q̄ × K*)
4. 代入 cos((t-p)ω + φ)
```

### 2.2 解决方案

利用额外存储的位置 $p$，直接用 $\mathbf{K}_{\text{rot}}$ 计算：

**优化后**：
```
1. 读取 K_rot 和 p
2. 直接计算 φ_rot = arg(Q̄ × K_rot*)  ← 省略 RoPE 反转
3. 校正相位：φ = φ_rot + p·ω
4. 代入 cos((t-p)ω + φ) = cos(tω + φ_rot)
```

### 2.3 数学推导

由 $\mathbf{K}_{\text{rot}} = \mathbf{K} \cdot e^{ip\omega}$，可得：

$$
\bar{\mathbf{Q}} \cdot \mathbf{K}^* = \bar{\mathbf{Q}} \cdot \mathbf{K}_{\text{rot}}^* \cdot e^{ip\omega}
$$

取辐角：

$$
\boxed{\phi = \phi_{\text{rot}} + p \cdot \omega}
$$

代入原公式并简化：

$$
\cos\bigl((t-p)\omega + \phi\bigr) = \cos\bigl((t-p)\omega + \phi_{\text{rot}} + p\omega\bigr) = \cos(t\omega + \phi_{\text{rot}})
$$

**注意**：幅度不变，因为旋转不改变模长：$A = |\bar{\mathbf{Q}}| \cdot |\mathbf{K}| = |\bar{\mathbf{Q}}| \cdot |\mathbf{K}_{\text{rot}}|$

### 2.4 简化后的打分公式

$$
\boxed{\text{score} = \sum_{f} \left[ A_f \cdot s_f^2 \cdot \cos(t \cdot \omega_f + \phi_{\text{rot},f}) \right] + \text{extra\_term}}
$$

公式只依赖 $t$，不再显式依赖 $p$（$p$ 的影响隐含在 $\phi_{\text{rot}}$ 中）。

**重要修正**（2026-02-01）：经过实现验证，使用 $K_{\text{rot}}$ 时，正确的相位计算应该是 `phase = t * omega`，不需要显式加上 $\phi_{\text{rot}}$。详见 `RKV_EQUIVALENCE_FIX.md` 和 `FP32_EQUIVALENCE_FIX.md`。

---

## 3. 优化 2：单次读取多位置打分

### 3.1 问题

当前实现循环遍历 offset，每次迭代都从显存重新读取 K：

```python
for offset in offsets:  # 16 次
    k = load_from_memory()  # 每次都读取！
    score = compute_score(k, round_start + offset)
```

### 3.2 解决方案

只加载 K 一次，在寄存器中迭代所有 offset：

```python
k = load_from_memory()  # 只读取一次
amp, phi_rot = compute_amp_and_phase(k)  # 只算一次

for offset in offsets:  # 在寄存器中迭代
    t = round_start + offset
    score = amp * cos(t * omega + phi_rot) + extra_term
```

### 3.3 显存带宽对比

| 配置 | 优化前 | 优化后 | 减少 |
|-----|-------|-------|-----|
| 16 offset, 8192 token, 128 dim, BF16 | 32 MB | 2 MB | **16×** |

---

## 4. 优化 3：共享三角函数表

### 4.1 核心思想

使用三角恒等式将打分公式拆分：

$$
\cos(t \omega + \phi_{\text{rot}}) = \cos(t \omega) \cdot \cos(\phi_{\text{rot}}) - \sin(t \omega) \cdot \sin(\phi_{\text{rot}})
$$

### 4.2 位置无关系数

定义：

$$
\begin{aligned}
\mathcal{A}_f &= A_f \cdot s_f^2 \cdot \cos(\phi_{\text{rot},f}) \\
\mathcal{B}_f &= A_f \cdot s_f^2 \cdot \sin(\phi_{\text{rot},f})
\end{aligned}
$$

**关键**：$\mathcal{A}_f$ 和 $\mathcal{B}_f$ 只依赖于 $\mathbf{K}_{\text{rot}}$，与目标位置 $t$ 无关。

### 4.3 快速计算（无需三角函数）

设 $\bar{\mathbf{Q}}_f = q_r + iq_i$，$\mathbf{K}_{\text{rot},f} = k_r + ik_i$：

$$
\text{Re} = q_r k_r + q_i k_i, \quad \text{Im} = q_i k_r - q_r k_i
$$

由于 $A_f = |z|$ 且 $\cos\phi = \text{Re}/|z|$，约掉后：

$$
\boxed{
\mathcal{A}_f = s_f^2 \cdot \text{Re}, \quad \mathcal{B}_f = s_f^2 \cdot \text{Im}
}
$$

**优化效果**：只需乘法和加法，无需 atan2、cos、sin、sqrt。

**实现说明**（2026-02-01）：当前实现中，打分公式已简化为直接使用预计算的三角函数表，无需计算 $\phi_{\text{rot}}$。详见最终计算流程（第 5 节）。

### 4.4 打分变成点积

$$
\boxed{\text{score}(t) = \boldsymbol{\mathcal{A}} \cdot \mathbf{c}(t) - \boldsymbol{\mathcal{B}} \cdot \mathbf{s}(t) + E}
$$

其中：
- $\mathbf{c}(t) = [\cos(t\omega_1), \cos(t\omega_2), \ldots]$
- $\mathbf{s}(t) = [\sin(t\omega_1), \sin(t\omega_2), \ldots]$

### 4.5 预计算共享表（vLLM 初始化时缓存）

#### 核心思想

$\mathbf{c}(t)$ 和 $\mathbf{s}(t)$ 只依赖于 $t$ 和 $\omega$，不依赖于具体 token。

更关键的是：**压缩只在特定位置触发**。假设超参数 `compress_interval = 128`，则压缩只在 decode 位置能被 128 整除时激活：

```
可能的 round_start: 128, 256, 384, 512, 640, ...
```

因此，所有可能的查询位置 $t = \text{round\_start} + \text{offset}$ 是**有限且可预知的**。

#### 预计算策略

在 **vLLM 初始化时**（而非每轮打分时）一次性预计算所有可能的三角函数值：

```python
# ========== vLLM 初始化时（只执行一次）==========
def precompute_trig_tables(
    max_seq_len: int,      # 如 131072 (128K)
    compress_interval: int, # 如 128
    offsets: Tensor,        # 如 [1, 2, 4, 8, ..., 32768]，共 16 个
    omega: Tensor,          # [freq_count]，如 64 个频率
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    预计算所有可能压缩位置的 cos/sin 表。

    Returns:
        cos_table: [num_compress_pos, num_offsets, freq_count]
        sin_table: [num_compress_pos, num_offsets, freq_count]
    """
    # 所有可能的压缩触发位置
    round_starts = torch.arange(
        compress_interval,
        max_seq_len + 1,
        compress_interval,
        device=device
    )  # [128, 256, 384, ..., 131072]

    num_pos = round_starts.shape[0]
    num_offsets = offsets.shape[0]
    freq_count = omega.shape[0]

    # 计算所有 t = round_start + offset
    # t: [num_pos, num_offsets]
    t = round_starts[:, None] + offsets[None, :]

    # 计算 t * omega: [num_pos, num_offsets, freq_count]
    phase = t[:, :, None] * omega[None, None, :]

    cos_table = torch.cos(phase)
    sin_table = torch.sin(phase)

    return cos_table, sin_table

# 缓存到 GPU
self.cos_table, self.sin_table = precompute_trig_tables(...)
```

#### 打分时查表

```python
# ========== 每轮打分时（只需查表）==========
def get_trig_values(round_start: int, compress_interval: int):
    # 计算当前 round_start 对应的索引
    idx = (round_start // compress_interval) - 1

    # 直接查表，无需计算三角函数
    C = self.cos_table[idx]  # [num_offsets, freq_count]
    S = self.sin_table[idx]  # [num_offsets, freq_count]
    return C, S
```

#### 显存占用估算

| 参数 | 典型值 |
|------|--------|
| max_seq_len | 131072 (128K) |
| compress_interval | 128 |
| num_compress_pos | 131072 / 128 = **1024** |
| num_offsets | 16 |
| freq_count | 64 (head_dim=128) |
| dtype | float32 (4 bytes) |

**计算**：

$$
\text{显存} = 2 \times \text{num\_pos} \times \text{num\_offsets} \times \text{freq\_count} \times 4
$$

| 配置 | num_pos | 计算 | 显存占用 |
|------|---------|------|----------|
| 128K seq, interval=128, 64 freq | 1024 | 2 × 1024 × 16 × 64 × 4 | **8 MB** |
| 128K seq, interval=128, 128 freq | 1024 | 2 × 1024 × 16 × 128 × 4 | **16 MB** |
| 1M seq, interval=128, 64 freq | 8192 | 2 × 8192 × 16 × 64 × 4 | **64 MB** |
| 1M seq, interval=64, 128 freq | 16384 | 2 × 16384 × 16 × 128 × 4 | **256 MB** ⚠️ |

**结论**：在常见配置下（128K seq, interval≥128），显存占用 **< 100 MB**，完全可接受。

> ⚠️ **警告**：如果 `compress_interval < 64` 或 `max_seq_len > 1M`，显存可能超过 100MB，需要考虑动态计算或分块缓存。

---

## 5. 最终计算流程

```python
# ========== vLLM 初始化时（只执行一次）==========
# 预计算所有可能压缩位置的三角函数表
cos_table, sin_table = precompute_trig_tables(
    max_seq_len=131072,
    compress_interval=128,
    offsets=offsets,
    omega=omega,
    device=device,
)
# cos_table: [num_compress_pos, num_offsets, freq_count]
# sin_table: [num_compress_pos, num_offsets, freq_count]

# ========== 每轮打分开始时 ==========
# 查表获取当前 round_start 对应的三角函数值（无需计算）
idx = (round_start // compress_interval) - 1
C = cos_table[idx]  # [num_offsets, freq_count]
S = sin_table[idx]  # [num_offsets, freq_count]

# ========== 每个 token ==========
# Step 1: 加载 K_rot（只读取一次）
k_r, k_i = load_k_rot(token_idx)

# Step 2: 计算位置无关系数（只需乘法加法，无三角函数）
Re = q_r * k_r + q_i * k_i
Im = q_i * k_r - q_r * k_i
A_coef = freq_scale_sq * Re
B_coef = freq_scale_sq * Im

# Step 3: 对每个 offset 打分（两个点积，无三角函数）
for i in range(num_offsets):
    score[i] = dot(A_coef, C[i]) - dot(B_coef, S[i]) + extra_term

# Step 4: 聚合
final_score = aggregate(score)  # mean 或 max
```

> **注意**：打分阶段完全不需要计算三角函数，所有 cos/sin 值都来自预计算表。

---

## 6. 批量矩阵乘法形式

将所有 token 的系数组成矩阵：

$$
\mathbf{A}_{\text{all}} \in \mathbb{R}^{T \times F}, \quad \mathbf{B}_{\text{all}} \in \mathbb{R}^{T \times F}
$$

则所有 token 在所有 offset 的打分可以用两个矩阵乘法完成：

$$
\boxed{\mathbf{Scores} = \mathbf{A}_{\text{all}} \cdot \mathbf{C}^T - \mathbf{B}_{\text{all}} \cdot \mathbf{S}^T + \mathbf{E}}
$$

其中 $\mathbf{Scores} \in \mathbb{R}^{T \times N}$，GPU 上可用 cuBLAS 高度优化。

---

## 7. 性能提升预估

| 优化 | 显存带宽 | 计算量 |
|-----|---------|-------|
| 避免 RoPE 反转 | -20% | -30% |
| 单次读取多位置 | **-94%** | 0% |
| 位置分离 + 共享三角表 | -5% | **-60%** |
| **综合** | **~95%** | **~80%** |

---

## 8. 实现修正历史

### 2026-02-01: 相位计算公式修正

在实际实现和测试中发现，使用 $K_{\text{rot}}$ 时的相位计算需要修正：

**原文档描述**：`phase = t * omega + phi_rot`
**修正后实现**：`phase = t * omega`（无需显式加 phi_rot）

**原因**：复数乘积 `Q * conj(K_rot)` 已经包含了相位信息，通过系数 A 和 B 隐式编码。直接使用三角恒等式展开即可，无需计算 `phi_rot = atan2(prod_imag, prod_real)`。

**验证结果**：
- FP32 等价性测试通过（误差 < 1e-4）
- 核心打分 kernel 测试 33/33 通过
- 与 R-KV 参考实现等价性验证通过

**详细说明**：
- 数学推导：`../RKV_EQUIVALENCE_FIX.md`
- 实现修正：`../FP32_EQUIVALENCE_FIX.md`
- MLR 公式修正：`../MLR_FIX.md`

---

*文档版本：1.1*
*创建日期：2025-01-30*
*更新日期：2026-02-01（添加实现修正历史）*
