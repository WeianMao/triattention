# TriAttention 计算优化

本文档详细说明 TriAttention 打分计算优化的数学推导和实现策略。

---

## 1. 问题描述

### 当前瓶颈

1. **RoPE 反转**：每轮打分需要对所有缓存的 key 反转 RoPE
2. **多位置显存读取**：对 N 个未来位置打分需要从显存读取 K 共 N 次
3. **冗余计算**：位置无关项对每个未来位置都重新计算

### 优化目标

- 完全消除 RoPE 反转
- 每轮打分只从显存读取 K 一次
- 分离位置相关和位置无关的计算

---

## 2. 符号定义与数学背景

### 2.1 符号表

| 符号 | 含义 | 来源 |
|-----|------|------|
| $\mathbf{K}$ | 原始 key 向量（未旋转） | 模型输出 |
| $\mathbf{K}_{\text{rot}}$ | RoPE 旋转后的 key 向量 | 显存中存储的 |
| $p$ | key 的原始位置索引 | 需要额外存储 |
| $t$ | 打分的目标未来位置 | 打分时指定 |
| $\omega_f$ | 第 $f$ 个频率分量的 RoPE 频率 | 模型配置 |
| $\bar{\mathbf{Q}}_{\text{mean}}$ | 统计得到的平均 query（复数形式） | stats 文件 |
| $\phi$ | Q 和原始 K 的相位差 | 需要计算 |
| $\phi_{\text{rot}}$ | Q 和旋转后 K 的相位差 | 直接从 $\mathbf{K}_{\text{rot}}$ 计算 |
| $A_f$ | 第 $f$ 个频率分量的幅度 | 计算得到 |
| $s_f^2$ | 频率缩放因子 | stats 文件 |
| $E_f$ | 位置无关幅度项 | stats 文件 |

### 2.2 RoPE 编码

对于位置 $p$ 处的 key 向量，RoPE 旋转可表示为复数乘法：

$$
\mathbf{K}_{\text{rot}} = \mathbf{K} \cdot e^{i \cdot p \cdot \omega}
$$

对于 "half" 风格 RoPE（Qwen, Llama）：向量前半部分为实部，后半部分为虚部。

### 2.3 原始打分公式

来自 `round_pruning_utils.py:score_keys_for_round`：

$$
\text{score}(\mathbf{k}, t) = \underbrace{\sum_{f} \left[ A_f \cdot s_f^2 \cdot \cos\bigl((t - p) \cdot \omega_f + \phi_f\bigr) \right]}_{\text{位置相关项}} + \underbrace{\sum_{f} \left[ E_f \cdot s_f^2 \right]}_{\text{位置无关项}}
$$

**幅度和相位的计算**（需要原始 K）：

$$
\begin{aligned}
\phi_f &= \arg\bigl(\bar{\mathbf{Q}}_{\text{mean},f} \cdot \mathbf{K}_f^*\bigr) & \text{（Q 和 K 的相位差）} \\
A_f &= |\bar{\mathbf{Q}}_{\text{mean},f}| \cdot |\mathbf{K}_f| & \text{（幅度乘积）}
\end{aligned}
$$

其中 $\mathbf{K}_f^*$ 表示 $\mathbf{K}$ 第 $f$ 个频率分量的复共轭。

---

## 3. 优化 1：避免 RoPE 反转

### 3.1 问题

原始打分需要原始 K（未旋转），但显存中存的是 $\mathbf{K}_{\text{rot}}$（旋转后）。

**原来的做法**（计算量大）：
1. 从显存读取 $\mathbf{K}_{\text{rot}}$
2. 反转 RoPE：$\mathbf{K} = \mathbf{K}_{\text{rot}} \cdot e^{-ip\omega}$ ← **需要对每个 token 做复数乘法**
3. 计算 $\phi = \arg(\bar{\mathbf{Q}} \cdot \mathbf{K}^*)$
4. 代入公式 $\cos((t-p)\omega + \phi)$

### 3.2 优化思路

既然额外存储了位置 $p$，可以直接用 $\mathbf{K}_{\text{rot}}$ 计算，然后用 $p$ 校正：

**优化后的做法**：
1. 从显存读取 $\mathbf{K}_{\text{rot}}$ 和 $p$
2. 直接计算 $\phi_{\text{rot}} = \arg(\bar{\mathbf{Q}} \cdot \mathbf{K}_{\text{rot}}^*)$ ← **省略 RoPE 反转**
3. 校正相位：$\phi = \phi_{\text{rot}} + p \cdot \omega$
4. 代入原公式 $\cos((t-p)\omega + \phi)$

### 3.3 数学推导

**相位校正公式的来源**：

由 $\mathbf{K}_{\text{rot}} = \mathbf{K} \cdot e^{ip\omega}$，可得 $\mathbf{K} = \mathbf{K}_{\text{rot}} \cdot e^{-ip\omega}$

$$
\begin{aligned}
\bar{\mathbf{Q}} \cdot \mathbf{K}^* &= \bar{\mathbf{Q}} \cdot (\mathbf{K}_{\text{rot}} \cdot e^{-ip\omega})^* \\
&= \bar{\mathbf{Q}} \cdot \mathbf{K}_{\text{rot}}^* \cdot e^{ip\omega}
\end{aligned}
$$

取辐角：

$$
\boxed{\phi = \phi_{\text{rot}} + p \cdot \omega}
$$

其中：
- $\phi_{\text{rot}} = \arg(\bar{\mathbf{Q}} \cdot \mathbf{K}_{\text{rot}}^*)$：直接用显存中的 $\mathbf{K}_{\text{rot}}$ 计算
- $p$：额外存储的位置索引

**幅度不变**：$A = |\bar{\mathbf{Q}}| \cdot |\mathbf{K}| = |\bar{\mathbf{Q}}| \cdot |\mathbf{K}_{\text{rot}}|$（旋转不改变模长）

### 3.4 最终打分公式

代入原公式：

$$
\cos\bigl((t-p)\omega + \phi\bigr) = \cos\bigl((t-p)\omega + \phi_{\text{rot}} + p\omega\bigr) = \cos(t\omega + \phi_{\text{rot}})
$$

**简化后的公式**：

$$
\boxed{\text{score} = \sum_{f} \left[ A_f \cdot s_f^2 \cdot \cos(t \cdot \omega_f + \phi_{\text{rot},f}) \right] + \text{extra\_term}}
$$

注意：公式简化后只依赖 $t$，不再显式依赖 $p$。但 $\phi_{\text{rot}}$ 的计算隐含了 $p$ 的影响。

### 3.5 位置索引存储

额外存储位置 $p$ 的开销：

| 类型 | 支持范围 | 内存开销（2048 token × 8 head） |
|-----|---------|------------------------------|
| bf16 | 整数精确到 256 | 32 KB |
| int32 | 0 ~ 2B | 64 KB |

**设备自适应策略**：
```python
# init 时检测
if torch.cuda.is_bf16_supported():
    position_dtype = torch.bfloat16  # 优先：与 KV cache 对齐，便于 kernel 实现
else:
    position_dtype = torch.int32     # 回退：兼容老 GPU（如 V100）
```

bf16 可精确表示 0~256 的整数，超过会有舍入误差，但对打分影响可忽略。

---

## 4. 优化 2：单次读取多位置打分

### 4.1 问题

当前实现循环遍历 offset：

```python
for offset in offsets:
    delta = round_start + offset - key_indices
    phase = delta * omega + phi
    scores_at_offset = (amp * cos(phase)).sum()
```

每次迭代都从显存重新读取 K。

### 4.2 融合 Kernel 设计

**核心思路**：

1. 每个 warp 只加载 $\mathbf{K}$ 一次
2. $A$、$\phi_{\text{rot}}$ 只计算一次
3. 仅使用寄存器迭代 offset

```python
@triton.jit
def triattention_scoring_kernel(
    # 输入指针
    K_rot_ptr,           # [num_tokens, head_dim]
    position_indices_ptr, # [num_tokens]
    Q_mean_real_ptr,     # [freq_count]
    Q_mean_imag_ptr,     # [freq_count]
    Q_abs_mean_ptr,      # [freq_count]
    omega_ptr,           # [freq_count]
    freq_scale_sq_ptr,   # [freq_count]
    offsets_ptr,         # [num_offsets]
    # 输出
    scores_ptr,          # [num_tokens]
    # 参数
    num_tokens, freq_count, num_offsets, round_start,
    aggregation,         # 'mean' 或 'max'
    BLOCK_SIZE,
):
    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 加载 K_rot（只加载一次！）
    k_rot_real = tl.load(K_rot_ptr + ...)
    k_rot_imag = tl.load(K_rot_ptr + ...)

    # 计算 relative_rot = Q_mean × conj(K_rot)
    rel_real = q_mean_real * k_rot_real + q_mean_imag * k_rot_imag
    rel_imag = q_mean_imag * k_rot_real - q_mean_real * k_rot_imag

    # amp = |relative_rot|, phi_rot = atan2(rel_imag, rel_real)
    amp = tl.sqrt(rel_real² + rel_imag²)
    phi_rot = tl.math.atan2(rel_imag, rel_real)

    # 迭代 offset（K 已在寄存器中）
    for i in range(num_offsets):
        t = round_start + tl.load(offsets_ptr + i)
        phase = t * omega + phi_rot
        score = tl.sum(amp * freq_scale_sq * tl.cos(phase)) + extra_term
        # 聚合...
```

### 4.3 显存带宽对比

| 配置 | 优化前 | 优化后 | 减少 |
|-----|-------|-------|-----|
| 16 offset, 8192 token, 128 dim, BF16 | 32 MB | 2 MB | **16×** |

$$
\text{优化前}: N_{\text{offset}} \times N_{\text{token}} \times D \times \text{sizeof}(\text{dtype})
$$

$$
\text{优化后}: N_{\text{token}} \times D \times \text{sizeof}(\text{dtype})
$$

---

## 5. 优化 3：位置相关/无关计算分离

> 详细推导见 [TRIG_TABLE_OPTIMIZATION.md](./TRIG_TABLE_OPTIMIZATION.md)

### 5.1 核心思路

使用三角恒等式将打分公式拆分为位置相关和位置无关部分：

$$
\text{score}(t) = \boldsymbol{\mathcal{A}} \cdot \mathbf{c}(t) - \boldsymbol{\mathcal{B}} \cdot \mathbf{s}(t) + E
$$

其中：
- $\boldsymbol{\mathcal{A}}, \boldsymbol{\mathcal{B}}$：位置无关系数（每个 token 算一次）
- $\mathbf{c}(t), \mathbf{s}(t)$：$\cos(t\omega)$, $\sin(t\omega)$ 向量（所有 token 共享）

### 5.2 位置无关系数的快速计算

设 $\bar{\mathbf{Q}}_f = q_r + iq_i$，$\mathbf{K}_{\text{rot},f} = k_r + ik_i$，定义：

$$
\text{Re} = q_r k_r + q_i k_i, \quad \text{Im} = q_i k_r - q_r k_i
$$

则位置无关系数可以直接计算（**无需 atan2、cos、sin、sqrt**）：

$$
\boxed{\mathcal{A}_f = s_f^2 \cdot \text{Re}, \quad \mathcal{B}_f = s_f^2 \cdot \text{Im}}
$$

### 5.3 共享三角函数表

$\cos(t\omega_f)$ 和 $\sin(t\omega_f)$ 只依赖于 $t$ 和 $\omega_f$，**不依赖于具体 token**。

预计算表大小：$2 \times N_{\text{offset}} \times F \times \text{sizeof(dtype)}$（约 4~16 KB，可忽略）

### 5.4 最终实现

打分变成两个点积（或批量矩阵乘法），GPU 上高度优化：

```python
# 预计算共享表（一次性）
C[i] = cos((round_start + offset_i) * omega)
S[i] = sin((round_start + offset_i) * omega)

# 每个 token（只需乘法加法）
A_coef = freq_scale_sq * (q_r * k_r + q_i * k_i)
B_coef = freq_scale_sq * (q_i * k_r - q_r * k_i)

# 打分（两个点积）
score[i] = dot(A_coef, C[i]) - dot(B_coef, S[i]) + extra_term
```

---

## 6. 实现路线图

| 阶段 | 任务 |
|-----|------|
| 1. 参考实现 | Python CPU 实现，验证数学正确性 |
| 2. Triton Kernel | 单次读取模式，正确性测试 |
| 3. 高级优化 | 位置分离，共享三角函数表 |
| 4. 集成 | vLLM attention 集成，端到端测试 |

---

## 7. 预期性能提升

| 优化 | 显存带宽 | 计算量 | 复杂度 |
|-----|---------|-------|-------|
| 避免 RoPE 反转 | -20% | -30% | 中 |
| 单次读取多位置 | **-94%** | 0% | 高 |
| 位置分离 | 0% | -40% | 中 |
| 共享三角函数表 | -5% | **-60%** | 低 |
| **综合** | **~95%** | **~80%** | - |

---

## 8. 验证计划

### 8.1 数学正确性

- [ ] 验证：$\cos(t\omega + \phi) = \mathcal{A}\cos(t\omega) - \mathcal{B}\sin(t\omega)$
- [ ] 验证：$\text{score}(\mathbf{K}_{\text{rot}}, p) = \text{score}(\mathbf{K})$
- [ ] 集成测试：完整打分轮次匹配原始实现

### 8.2 数值稳定性

| 精度 | 最大相对误差 |
|-----|------------|
| FP16 | < 1e-3 |
| BF16 | < 1e-2 |

---

## 9. 待确认问题

1. 对于 "half" 风格 RoPE，将前后两半视为复数的实部虚部，复数乘法是否正确？
2. 频率缩放因子 $s_f^2$ 在优化后的公式中是否需要调整？
3. 数值精度：$\phi_{\text{rot}} + p \cdot \omega$ 在 FP16/BF16 下是否稳定？

---

*文档版本：1.1*
*创建日期：2025-01-30*
*最后更新：2025-01-30*
