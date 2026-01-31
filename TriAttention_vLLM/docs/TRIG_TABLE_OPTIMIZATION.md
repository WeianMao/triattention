# 共享三角函数表优化

本文档详细推导 TriAttention 打分中的共享三角函数表优化。

---

## 1. 问题背景

打分需要对 $N$ 个未来位置（如 offset 0~15）都计算一遍，选最优。

**优化前**：每个 token × 每个 offset 都要算 $\cos(t \cdot \omega + \phi_{\text{rot}})$

**目标**：减少重复的三角函数计算

---

## 2. 原始打分公式

优化 RoPE 反转后的打分公式：

$$
\text{score}(t) = \sum_{f} \left[ A_f \cdot s_f^2 \cdot \cos(t \cdot \omega_f + \phi_{\text{rot},f}) \right] + E
$$

**符号定义**：

| 符号 | 含义 | 依赖 |
|-----|------|------|
| $t$ | 目标位置 $= \text{round\_start} + \text{offset}$ | 打分时指定 |
| $A_f$ | 第 $f$ 个频率分量的幅度 | 每个 token 不同 |
| $\phi_{\text{rot},f}$ | 第 $f$ 个频率分量的相位 | 每个 token 不同 |
| $\omega_f$ | 第 $f$ 个 RoPE 频率 | 固定（模型配置） |
| $s_f^2$ | 第 $f$ 个频率缩放因子 | 固定（stats 文件） |
| $E$ | extra term | 每个 token 不同 |

---

## 3. 优化推导

### 3.1 三角恒等式展开

$$
\cos(t \cdot \omega_f + \phi_{\text{rot},f}) = \cos(t \cdot \omega_f) \cdot \cos(\phi_{\text{rot},f}) - \sin(t \cdot \omega_f) \cdot \sin(\phi_{\text{rot},f})
$$

### 3.2 定义位置无关系数

对每个 token，定义：

$$
\begin{aligned}
\mathcal{A}_f &= A_f \cdot s_f^2 \cdot \cos(\phi_{\text{rot},f}) \\
\mathcal{B}_f &= A_f \cdot s_f^2 \cdot \sin(\phi_{\text{rot},f})
\end{aligned}
$$

**关键观察**：$\mathcal{A}_f$ 和 $\mathcal{B}_f$ 只依赖于 $\mathbf{K}_{\text{rot}}$，与目标位置 $t$ 无关。

### 3.3 加速 $\cos(\phi_{\text{rot}})$ 和 $\sin(\phi_{\text{rot}})$ 的计算

直接计算 $\cos(\phi_{\text{rot}})$ 需要 atan2 + cos，GPU 上很慢。可以利用复数性质完全避免。

**推导**：

设 $\bar{\mathbf{Q}}_f = q_r + i \cdot q_i$，$\mathbf{K}_{\text{rot},f} = k_r + i \cdot k_i$

复数乘积：
$$
z = \bar{\mathbf{Q}}_f \cdot \mathbf{K}_{\text{rot},f}^* = (q_r k_r + q_i k_i) + i(q_i k_r - q_r k_i)
$$

定义：
$$
\text{Re} = q_r k_r + q_i k_i, \quad \text{Im} = q_i k_r - q_r k_i
$$

由复数性质：
$$
\cos(\phi_{\text{rot}}) = \frac{\text{Re}}{|z|}, \quad \sin(\phi_{\text{rot}}) = \frac{\text{Im}}{|z|}
$$

代入 $\mathcal{A}_f = A_f \cdot s_f^2 \cdot \cos(\phi_{\text{rot}})$：

$$
\mathcal{A}_f = A_f \cdot s_f^2 \cdot \frac{\text{Re}}{|z|}
$$

**关键**：$A_f = |\bar{\mathbf{Q}}_f| \cdot |\mathbf{K}_{\text{rot},f}|$ 而 $|z| = |\bar{\mathbf{Q}}_f| \cdot |\mathbf{K}_{\text{rot},f}|$，所以 $A_f = |z|$，约掉！

**最终公式**：

$$
\boxed{
\begin{aligned}
\mathcal{A}_f &= s_f^2 \cdot \text{Re} = s_f^2 \cdot (q_r k_r + q_i k_i) \\
\mathcal{B}_f &= s_f^2 \cdot \text{Im} = s_f^2 \cdot (q_i k_r - q_r k_i)
\end{aligned}
}
$$

**优化效果**：只需乘法和加法，**不需要 atan2、cos、sin、sqrt**。

### 3.4 代入原公式

$$
\begin{aligned}
\text{score}(t) &= \sum_{f} \left[ A_f \cdot s_f^2 \cdot \left( \cos(t\omega_f)\cos(\phi_{\text{rot},f}) - \sin(t\omega_f)\sin(\phi_{\text{rot},f}) \right) \right] + E \\
&= \sum_{f} \left[ \mathcal{A}_f \cdot \cos(t \cdot \omega_f) \right] - \sum_{f} \left[ \mathcal{B}_f \cdot \sin(t \cdot \omega_f) \right] + E
\end{aligned}
$$

### 3.5 向量点积形式

$$
\boxed{\text{score}(t) = \boldsymbol{\mathcal{A}} \cdot \mathbf{c}(t) - \boldsymbol{\mathcal{B}} \cdot \mathbf{s}(t) + E}
$$

其中：

| 向量 | 定义 | 维度 |
|-----|------|------|
| $\boldsymbol{\mathcal{A}}$ | $[\mathcal{A}_1, \mathcal{A}_2, \ldots, \mathcal{A}_F]$ | $[F]$ |
| $\boldsymbol{\mathcal{B}}$ | $[\mathcal{B}_1, \mathcal{B}_2, \ldots, \mathcal{B}_F]$ | $[F]$ |
| $\mathbf{c}(t)$ | $[\cos(t\omega_1), \cos(t\omega_2), \ldots, \cos(t\omega_F)]$ | $[F]$ |
| $\mathbf{s}(t)$ | $[\sin(t\omega_1), \sin(t\omega_2), \ldots, \sin(t\omega_F)]$ | $[F]$ |

---

## 4. 预计算共享表

### 4.1 观察

$\mathbf{c}(t)$ 和 $\mathbf{s}(t)$ 只依赖于：
- 目标位置 $t$（有限个：$N$ 个 offset）
- RoPE 频率 $\omega_f$（固定）

**不依赖于具体 token**，因此可以预计算并共享。

### 4.2 表结构

对于 $N$ 个目标位置 $t_i = \text{round\_start} + \text{offset}_i$：

$$
\mathbf{C} = \begin{bmatrix}
\cos(t_0 \omega_1) & \cos(t_0 \omega_2) & \cdots & \cos(t_0 \omega_F) \\
\cos(t_1 \omega_1) & \cos(t_1 \omega_2) & \cdots & \cos(t_1 \omega_F) \\
\vdots & \vdots & \ddots & \vdots \\
\cos(t_{N-1} \omega_1) & \cos(t_{N-1} \omega_2) & \cdots & \cos(t_{N-1} \omega_F)
\end{bmatrix} \in \mathbb{R}^{N \times F}
$$

$$
\mathbf{S} = \begin{bmatrix}
\sin(t_0 \omega_1) & \sin(t_0 \omega_2) & \cdots & \sin(t_0 \omega_F) \\
\sin(t_1 \omega_1) & \sin(t_1 \omega_2) & \cdots & \sin(t_1 \omega_F) \\
\vdots & \vdots & \ddots & \vdots \\
\sin(t_{N-1} \omega_1) & \sin(t_{N-1} \omega_2) & \cdots & \sin(t_{N-1} \omega_F)
\end{bmatrix} \in \mathbb{R}^{N \times F}
$$

### 4.3 显存占用

$$
\text{显存} = 2 \times N \times F \times \text{sizeof}(\text{dtype})
$$

| 配置 | 计算 | 占用 |
|-----|------|------|
| 16 offset, 64 freq, bf16 | $2 \times 16 \times 64 \times 2$ | **4 KB** |
| 32 offset, 128 freq, bf16 | $2 \times 32 \times 128 \times 2$ | **16 KB** |

占用极小，可忽略。

---

## 5. 最终计算流程

### 5.1 伪代码

```python
# ========== 每轮打分开始时（所有 token 共享）==========
# 预计算三角函数表
C = zeros(num_offsets, freq_count)  # cos 表
S = zeros(num_offsets, freq_count)  # sin 表
for i, offset in enumerate(offsets):
    t = round_start + offset
    C[i] = cos(t * omega)  # omega: [freq_count]
    S[i] = sin(t * omega)

# ========== 每个 token ==========
for token in tokens:
    # Step 1: 计算位置无关系数（只需乘法加法，无三角函数）
    # Q_mean = q_r + i*q_i, K_rot = k_r + i*k_i
    Re = q_r * k_r + q_i * k_i  # 复数乘积实部
    Im = q_i * k_r - q_r * k_i  # 复数乘积虚部

    A_coef = freq_scale_sq * Re  # 𝒜: [freq_count]
    B_coef = freq_scale_sq * Im  # ℬ: [freq_count]

    # Step 2: 对每个 offset 打分（两个点积）
    for i in range(num_offsets):
        score[i] = dot(A_coef, C[i]) - dot(B_coef, S[i]) + extra_term

    # Step 3: 聚合（mean 或 max）
    final_score = aggregate(score)
```

### 5.2 计算量对比

| | 优化前 | 优化后 |
|--|-------|-------|
| 三角函数调用 | $N_{\text{token}} \times N_{\text{offset}} \times F$ | $N_{\text{offset}} \times F$（共享表，一次性） |
| 每个 token | atan2 + cos + sin | 仅乘法加法 |
| 每个 token × offset | cos/sin 计算 | 2 个点积 |

**优化要点**：
1. 共享三角函数表：$\cos(t\omega)$, $\sin(t\omega)$ 所有 token 共享
2. 位置无关系数：$\mathcal{A}$, $\mathcal{B}$ 只需乘法加法，无三角函数
3. 打分变成点积：GPU 上高度优化

---

## 6. 进一步优化：批量矩阵乘法

如果将所有 token 的 $\boldsymbol{\mathcal{A}}$, $\boldsymbol{\mathcal{B}}$ 组成矩阵：

$$
\begin{aligned}
\mathbf{A}_{\text{all}} &\in \mathbb{R}^{T \times F} & \text{（T 个 token）} \\
\mathbf{B}_{\text{all}} &\in \mathbb{R}^{T \times F}
\end{aligned}
$$

则所有 token 在所有 offset 的打分可以用两个矩阵乘法完成：

$$
\boxed{\mathbf{Scores} = \mathbf{A}_{\text{all}} \cdot \mathbf{C}^T - \mathbf{B}_{\text{all}} \cdot \mathbf{S}^T + \mathbf{E}}
$$

其中 $\mathbf{Scores} \in \mathbb{R}^{T \times N}$，第 $(i, j)$ 元素是第 $i$ 个 token 在第 $j$ 个 offset 的分数。

**这是最高效的实现方式**：两个 GEMM 操作，GPU 上有高度优化的实现（cuBLAS）。

---

*文档版本：1.0*
*创建日期：2025-01-30*
