# Neural Network Architecture

## 概述

Module 1 (Key Pruning) 和 Module 2 (Bin-based Sparse Attention) 共享相同的输入编码方式：**Kernel-based Encoding**。

本文档详细描述：
1. 输入预处理（旋转参考系）
2. Kernel Encoding Layer
3. 网络参数量分析

---

## 1. 输入预处理：旋转参考系

### 1.1 核心思想

神经网络的输入是 **post-RoPE** 的 K/Q 向量，但在一个**旋转的参考系**中表示。

### 1.2 参考系定义

| 组件 | 定义 |
|------|------|
| **参考向量** | 零角度向量 (1, 0)（每个频段的单位向量） |
| **参考位置** | 当前 round 的中间位置（第 ~64 个 token） |
| **参考角度** | 由 RoPE 旋转决定：`reference_angle_j = ref_position × ω_j` |

> **简化说明**：原设计使用 Q 平均向量作为参考，但实际上只需要使用零角度向量 (1, 0)。因为我们只关心**相对角度**，零角度向量经过 RoPE 旋转后的角度就是纯粹的位置编码角度，足够作为参考系。

### 1.3 参考角度计算

**核心思想**：
1. 在每个频段 j 上，定义一个**零角度向量** `(1, 0)`（即角度为 0 的单位向量）
2. 将该零角度向量用 RoPE 旋转到参考位置 `ref_position`
3. 旋转后的角度就是参考角度：`reference_angle_j = ref_position × ω_j`

```python
def get_reference_angles(round_start, num_freqs=64, round_window=128):
    """
    获取当前 round 的参考角度

    原理：
    - 零角度向量 (1, 0) 的初始角度 = 0
    - RoPE 旋转公式：angle_after = angle_before + pos × ω_j
    - 因此：reference_angle_j = 0 + ref_position × ω_j = ref_position × ω_j
    """
    # 1. 计算参考位置（round 中间）
    ref_position = round_start + round_window // 2  # e.g., round_start + 64

    # 2. 计算每个频段的 RoPE 角频率
    # 对于 LLaMA/Qwen 等模型: ω_j = 1 / (10000^(2j/head_dim))
    dim_indices = torch.arange(0, num_freqs)
    omega = 1.0 / (10000 ** (2 * dim_indices / (num_freqs * 2)))  # shape: (num_freqs,)

    # 3. 参考角度 = 零角度向量 (1,0) 经过 RoPE 旋转后的角度
    #    = 0 + ref_position × ω_j
    #    = ref_position × ω_j
    reference_angles = ref_position * omega  # shape: (num_freqs,)

    return reference_angles
```

> **简化优势**：由于零角度向量 (1, 0) 的初始角度为 0，经过 RoPE 旋转后角度就是 `pos × ω_j`，无需额外的 atan2 计算。

### 1.4 RoPE 频段结构

对于 `head_dim = 128`：
- 共有 **64 个频段**
- 每个频段是一个 **2D 平面**
- 每个频段的旋转速度（角频率 ω）不同

```
频段 0: [dim_0, dim_1]   → ω_0 (最慢)
频段 1: [dim_2, dim_3]   → ω_1
...
频段 63: [dim_126, dim_127] → ω_63 (最快)
```

### 1.5 在旋转参考系下的表示

对于每个频段 j，输入向量（K 或 Q）可以表示为：
- **模长**: `magnitude_j = sqrt(x_{2j}^2 + x_{2j+1}^2)`
- **角度**: `angle_j = atan2(x_{2j+1}, x_{2j})`（直接使用原始角度）

### 1.6 参考系处理的优化

> **重要优化**：参考角度不在输入端处理，而是在 Kernel 参数端处理。

**原始方法（低效）**：
```python
# 每个 K 都要减去 reference_angle
angle_j = atan2(...) - reference_angle_j
kernel_output = von_mises(angle_j, mu_ij, kappa_ij)
```

**优化方法（推荐）**：
```python
# K 直接使用原始角度
angle_j = atan2(...)
# 在 round 开头，将 reference_angle 加到 mu 中（只计算一次）
mu_effective_ij = mu_ij + reference_angle_j
kernel_output = von_mises(angle_j, mu_effective_ij, kappa_ij)
```

**等价性证明**：
```
cos(angle_j - reference_angle_j - mu_ij)
= cos(angle_j - (mu_ij + reference_angle_j))
= cos(angle_j - mu_effective_ij)
```

**优势**：
- 原方法：每个 K 都要计算 `angle - reference_angle`（N 次减法）
- 优化方法：只在 round 开头调整一次 mu（1 次加法 per bin per freq）

---

## 2. Kernel Encoding Layer

### 2.1 角度编码：归一化 von Mises Kernel

```python
def normalized_von_mises_kernel(angle, mu, kappa):
    """
    归一化 von Mises kernel
    - 最大值为 1（规避配分函数计算）
    - 只需 cos 和 exp，GPU 友好
    """
    return exp(kappa * (cos(angle - mu) - 1))
```

**优势**：
1. 天然处理角度周期性，无边界问题
2. 通过 `-1` 保证最大值为 1，无需计算贝塞尔函数
3. 只需 `cos` 和 `exp`，计算高效

**参数**：
- `mu`: 可学习，kernel 中心位置
- `kappa`: 可学习，集中度（类似 1/σ²）

### 2.2 多 Kernel 结构

> **设计决策**：每个频段使用 **3 个 von Mises Kernel**，增强表达能力。

#### 结构说明

对于输出 bin i（共 128 个 bin）和频段 j（共 64 个频段）：
- 有 **3 个独立的 von Mises Kernel**（索引 m = 0, 1, 2）
- 每个 kernel 有 3 个参数：`mu_ijm`, `kappa_ijm`, `weight_ijm`
- 同一频段的 3 个 kernel 输出**累加**后，再乘以该频段的模长

```
bin_i_freq_j_contribution = magnitude_j × Σ_m [weight_ijm × vM(angle_j, mu_ijm, kappa_ijm)]
```

#### 参数初始化

> **重要**：kappa 初始化要**足够小**（方差足够大），确保 kernel 覆盖接近一整周 (2π)，避免远离 mu 的区域概率接近零导致梯度消失。

```python
def initialize_kernel_params(num_bins=128, num_freqs=64, num_kernels=3):
    """
    初始化 Kernel Encoding Layer 参数

    关键：kappa 初始化较小（~1.0），确保 kernel 较宽，覆盖接近 2π
    """
    # mu: 均匀分布在 [-π, π]
    # 3 个 kernel 的 mu 初始位置间隔 2π/3
    mu = torch.zeros(num_bins, num_freqs, num_kernels)
    for m in range(num_kernels):
        mu[:, :, m] = torch.randn(num_bins, num_freqs) * 0.1 + (2 * math.pi * m / num_kernels - math.pi)

    # kappa: 初始化较小，确保覆盖范围大
    # kappa ≈ 1.0 时，von Mises 的"标准差"约为 1 弧度，覆盖大部分圆周
    kappa = torch.ones(num_bins, num_freqs, num_kernels) * 1.0

    # weight: 初始化为较小值，防止输出过大
    weight = torch.randn(num_bins, num_freqs, num_kernels) * 0.1

    # bias: 初始化为零
    bias = torch.zeros(num_bins)

    return mu, kappa, weight, bias
```

### 2.3 单个输出的计算

```python
def compute_output_i_freq_j(magnitude_j, angle_j, params_i_j, reference_angle_j, num_kernels=3):
    """
    计算第 i 个 bin 在第 j 个频段的贡献

    Args:
        params_i_j: 包含 3 个 kernel 的参数
            - mu: (num_kernels,)
            - kappa: (num_kernels,)
            - weight: (num_kernels,)
    """
    kernel_sum = 0

    for m in range(num_kernels):
        mu_ijm = params_i_j['mu'][m]
        kappa_ijm = params_i_j['kappa'][m]
        weight_ijm = params_i_j['weight'][m]

        # 将 reference_angle 加到 mu 中
        mu_effective = mu_ijm + reference_angle_j

        # 计算单个 kernel 输出
        kernel_output = normalized_von_mises_kernel(angle_j, mu_effective, kappa_ijm)

        # 加权累加
        kernel_sum += weight_ijm * kernel_output

    # 乘以该频段的模长
    return kernel_sum * magnitude_j
```

> **注意**：`reference_angle_j` 在每个 round 开头计算一次，然后加到所有 kernel 的 mu 中。

### 2.4 完整前向传播

```python
def kernel_encoding_layer(K, params, reference_angles, num_bins=128, num_freqs=64, num_kernels=3):
    """
    Kernel Encoding Layer 前向传播

    Input:
        K: shape (head_dim,) - 原始 Key（无需预处理）
        params: 所有可学习参数
        reference_angles: shape (num_freqs,) - 当前 round 的参考角度
        num_bins: 输出维度 (128)
        num_freqs: 频段数量 (64)
        num_kernels: 每个频段的 kernel 数量 (3)

    Output:
        logits: shape (num_bins,)

    ⚠️ 实现注意：以下使用循环仅为表达逻辑，实际应使用 tensor 操作。
    """
    outputs = zeros(num_bins)

    for i in range(num_bins):
        for j in range(num_freqs):
            magnitude_j = get_magnitude(K, freq_j)
            angle_j = get_angle(K, freq_j)

            outputs[i] += compute_output_i_freq_j(
                magnitude_j, angle_j, params[i][j], reference_angles[j], num_kernels
            )

        # 加 bias
        outputs[i] += params[i]['bias']

    return outputs
```

---

## 3. 参数量分析

### 3.1 Kernel Encoding Layer 参数

> **设计**：每个频段 3 个 von Mises Kernel

| 参数类型 | 数量计算 | 值 |
|----------|----------|-----|
| mu | num_bins × num_freqs × num_kernels | 128 × 64 × 3 = 24,576 |
| kappa | num_bins × num_freqs × num_kernels | 128 × 64 × 3 = 24,576 |
| weight | num_bins × num_freqs × num_kernels | 128 × 64 × 3 = 24,576 |
| bias | num_bins | 128 |

**Kernel Layer 总参数**: 24,576 × 3 + 128 = **73,856 per head**

### 3.2 各模块总参数

| 模块 | 结构 | 参数量 (per head) |
|------|------|-------------------|
| Module 1 (Key Pruning) | Kernel + MLP(128→h→1) | 73,856 + MLP |
| Module 2 Key Binning | Kernel only | 73,856 |
| Module 2 Query Routing | Kernel only | 73,856 |

**每个 Query Head 总参数**: ~221,000 + MLP (约 225K-250K)

### 3.3 Module 2 参数共享选项

对于 Module 2 (Bin-based Sparse Attention)，Key Binning 和 Query Routing 有两个独立的神经网络。考虑以下两种参数共享策略：

#### 选项 A: 完全独立（默认）

Key 网络和 Query 网络完全独立，不共享任何参数。

| 参数 | Key 网络 | Query 网络 |
|------|----------|------------|
| mu | 独立 | 独立 |
| kappa | 独立 | 独立 |
| weight | 独立 | 独立 |
| bias | 独立 | 独立 |

**总参数**: 73,856 × 2 = **147,712 per Query head**

#### 选项 B: 共享 Kernel 参数（mu, kappa）

Key 网络和 Query 网络共享 kernel 的中心位置（mu）和集中度（kappa），但 weight 和 bias 独立。

| 参数 | Key 网络 | Query 网络 | 共享 |
|------|----------|------------|------|
| mu | ✓ | ✓ | **共享** |
| kappa | ✓ | ✓ | **共享** |
| weight | 独立 | 独立 | 否 |
| bias | 独立 | 独立 | 否 |

**共享参数**: 128 × 64 × 3 × 2 = 49,152
**独立参数**: (128 × 64 × 3 + 128) × 2 = 49,408
**总参数**: 49,152 + 49,408 = **98,560 per Query head**

**直觉**：mu 和 kappa 定义了"bin 的位置和形状"，这对 K 和 Q 应该是一致的。weight 和 bias 决定"如何组合"，可以不同。

#### 待实验对比

- [ ] 选项 A vs 选项 B 的性能对比
- [ ] 共享参数是否影响 attention recall

---

## 4. 向量化实现

实际实现时应完全向量化：

```python
def kernel_encoding_layer_vectorized(K, mu, kappa, weight, bias, reference_angles):
    """
    向量化实现（3 kernels per frequency band）

    Args:
        K: (head_dim,)
        mu: (num_bins, num_freqs, num_kernels) - 基础 mu 值
        kappa: (num_bins, num_freqs, num_kernels)
        weight: (num_bins, num_freqs, num_kernels)
        bias: (num_bins,)
        reference_angles: (num_freqs,) - 当前 round 的参考角度

    Returns:
        logits: (num_bins,)
    """
    num_bins, num_freqs, num_kernels = mu.shape  # 128, 64, 3

    # 提取模长和角度（原始值，无需预处理）
    K_complex = K.view(-1, 2)  # (num_freqs, 2)
    magnitude = K_complex.norm(dim=1)  # (num_freqs,)
    angle = atan2(K_complex[:, 1], K_complex[:, 0])  # (num_freqs,)

    # 关键优化：将 reference_angles 加到 mu 中（广播）
    # reference_angles: (num_freqs,) -> (1, num_freqs, 1)
    mu_effective = mu + reference_angles.view(1, -1, 1)  # (num_bins, num_freqs, num_kernels)

    # 计算 kernel（广播）
    # angle: (num_freqs,) -> (1, num_freqs, 1)
    angle_expanded = angle.view(1, -1, 1)  # (1, num_freqs, 1)
    kernel = exp(kappa * (cos(angle_expanded - mu_effective) - 1))  # (num_bins, num_freqs, num_kernels)

    # 加权求和：先对 kernels 求和，再乘以 magnitude，最后对 freqs 求和
    weighted_kernels = (kernel * weight).sum(dim=2)  # (num_bins, num_freqs)
    weighted = weighted_kernels * magnitude.unsqueeze(0)  # (num_bins, num_freqs)
    logits = weighted.sum(dim=1) + bias  # (num_bins,)

    return logits
```

### 4.1 Round 级别的优化

在每个 round 开头，可以预计算 `mu_effective`：

```python
def prepare_round(mu, reference_angles):
    """
    在 round 开头调用一次，预计算 mu_effective

    Args:
        mu: (num_bins, num_freqs, num_kernels)
        reference_angles: (num_freqs,)

    Returns:
        mu_effective: (num_bins, num_freqs, num_kernels)
    """
    # 只计算一次，用于该 round 内所有 K 和 Q
    mu_effective = mu + reference_angles.view(1, -1, 1)
    return mu_effective

def kernel_encoding_fast(K, mu_effective, kappa, weight, bias):
    """
    使用预计算的 mu_effective，更高效

    Args:
        mu_effective: (num_bins, num_freqs, num_kernels)
        kappa: (num_bins, num_freqs, num_kernels)
        weight: (num_bins, num_freqs, num_kernels)
    """
    K_complex = K.view(-1, 2)
    magnitude = K_complex.norm(dim=1)  # (num_freqs,)
    angle = atan2(K_complex[:, 1], K_complex[:, 0])  # (num_freqs,)

    angle_expanded = angle.view(1, -1, 1)  # (1, num_freqs, 1)
    kernel = exp(kappa * (cos(angle_expanded - mu_effective) - 1))  # (num_bins, num_freqs, num_kernels)

    weighted_kernels = (kernel * weight).sum(dim=2)  # (num_bins, num_freqs)
    weighted = weighted_kernels * magnitude.unsqueeze(0)  # (num_bins, num_freqs)
    logits = weighted.sum(dim=1) + bias  # (num_bins,)

    return logits
```

---

## 5. 旋转参考系的实现细节

### 5.1 RoPE 角频率预计算

```python
def get_rope_omega(num_freqs=64, base=10000):
    """
    获取 RoPE 的角频率（可预计算，与位置无关）
    """
    dim_indices = torch.arange(0, num_freqs)
    omega = 1.0 / (base ** (2 * dim_indices / (num_freqs * 2)))  # shape: (num_freqs,)
    return omega

# 预计算角频率（常量，整个推理过程不变）
ROPE_OMEGA = get_rope_omega()  # shape: (64,)
```

### 5.2 完整的 Round 初始化流程

```python
def initialize_round(round_start, mu, round_window=128):
    """
    在每个 round 开头调用，准备该 round 的参数

    Args:
        round_start: 当前 round 起始位置
        mu: (num_bins, num_freqs) - 基础 mu 参数

    Returns:
        mu_effective: (num_bins, num_freqs) - 调整后的 mu

    注意：不再需要 head_id 参数，因为参考角度只由位置决定
    """
    # Step 1: 计算参考位置
    ref_position = round_start + round_window // 2

    # Step 2: 计算参考角度（零角度向量旋转后的角度）
    reference_angles = ref_position * ROPE_OMEGA  # shape: (num_freqs,)

    # Step 3: 将参考角度加到 mu 中（关键优化）
    mu_effective = mu + reference_angles.unsqueeze(0)  # (num_bins, num_freqs)

    return mu_effective
```

> **简化优势**：
> - 无需预统计 Q 平均向量（PRETRAINED_Q_MEAN）
> - 无需调用 `apply_rope()` 函数
> - 无需 `atan2()` 计算
> - 只需一次乘法：`ref_position * ROPE_OMEGA`
> - 所有 head 共享相同的 `ROPE_OMEGA`（假设 head_dim 相同）

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-14 | 初始化文档；记录 von Mises vs 截断 Gaussian 讨论 |
| 2025-12-14 | 优化参考系处理：reference_angle 加到 mu 而非从 angle 减去；删除截断 Gaussian 选项；添加 K/Q 网络参数共享选项 |
| 2025-12-15 | 简化参考向量：从 Q 平均向量改为零角度向量 (1,0)；参考角度直接由 RoPE 位置决定（`ref_position × ω_j`）；移除 PRETRAINED_Q_MEAN 依赖 |
| 2025-12-15 | 增加表达能力：每个频段从 1 个改为 3 个 von Mises Kernel；更新参数量计算；添加初始化建议（kappa 初始化较小避免梯度消失） |
