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

### 4.4 打分变成点积

$$
\boxed{\text{score}(t) = \boldsymbol{\mathcal{A}} \cdot \mathbf{c}(t) - \boldsymbol{\mathcal{B}} \cdot \mathbf{s}(t) + E}
$$

其中：
- $\mathbf{c}(t) = [\cos(t\omega_1), \cos(t\omega_2), \ldots]$
- $\mathbf{s}(t) = [\sin(t\omega_1), \sin(t\omega_2), \ldots]$

### 4.5 预计算共享表

$\mathbf{c}(t)$ 和 $\mathbf{s}(t)$ 只依赖于 $t$ 和 $\omega$，不依赖于具体 token，因此可以预计算：

```python
# 每轮打分开始时（所有 token 共享）
C = zeros(num_offsets, freq_count)
S = zeros(num_offsets, freq_count)
for i, offset in enumerate(offsets):
    t = round_start + offset
    C[i] = cos(t * omega)
    S[i] = sin(t * omega)
```

**显存占用**：

| 配置 | 计算 | 占用 |
|-----|------|------|
| 16 offset, 64 freq, bf16 | 2 × 16 × 64 × 2 | **4 KB** |

极小，可忽略。

---

## 5. 最终计算流程

```python
# ========== 每轮打分开始时 ==========
# 预计算共享三角函数表（一次性）
C = cos(offsets * omega)  # [num_offsets, freq_count]
S = sin(offsets * omega)  # [num_offsets, freq_count]

# ========== 每个 token ==========
# Step 1: 加载 K_rot（只读取一次）
k_r, k_i = load_k_rot(token_idx)

# Step 2: 计算位置无关系数（只需乘法加法）
Re = q_r * k_r + q_i * k_i
Im = q_i * k_r - q_r * k_i
A_coef = freq_scale_sq * Re
B_coef = freq_scale_sq * Im

# Step 3: 对每个 offset 打分（两个点积）
for i in range(num_offsets):
    score[i] = dot(A_coef, C[i]) - dot(B_coef, S[i]) + extra_term

# Step 4: 聚合
final_score = aggregate(score)  # mean 或 max
```

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

*文档版本：1.0*
*创建日期：2025-01-30*
