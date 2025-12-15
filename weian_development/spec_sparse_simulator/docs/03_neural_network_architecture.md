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

> **待实验**：Kappa 正数约束
>
> Von Mises kernel 要求 `kappa > 0`。两种处理方式待对比：
> - **无约束**：直接学习 kappa，可能变负（行为未定义）
> - **Softplus 约束**：`kappa_positive = F.softplus(kappa_raw)` 确保正数
>
> 需要实验对比哪种方式训练更稳定、效果更好。

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

## 2.5 Position Scaling Layer（Module 1 专用）

> **重要**：此层**仅用于 Module 1 (Key Pruning)**，Module 2 (Bin-based Sparse Attention) **不使用**位置编码。
>
> Module 2 不需要位置编码的原因：Binning 只关心 Q-K 的语义相似性，位置信息已经通过 RoPE 编码在向量中。

### 2.5.1 设计概述

在 MLP 输出的 logit 上乘一个位置相关的可学习权重，调整不同位置 Key 的 drop 倾向。

```
MLP Output (logit)
    │
    ▼
┌─────────────────────────────────┐
│   Position Scaling             │
│   scaled_logit = logit × w(x)  │
│   w(x) = log-scale interpolated│
└─────────────────────────────────┘
    │
    ▼
Sigmoid
```

### 2.5.2 锚点权重结构

使用 **log 尺度**上的锚点进行线性插值：

| 锚点索引 | 位置 | log₁₀(位置) | 可学习权重 |
|----------|------|-------------|------------|
| 0 | 1,000 | 3 | w₀ |
| 1 | 10,000 | 4 | w₁ |
| 2 | 100,000 | 5 | w₂ |

**参数量**：3 个标量（可扩展为更多锚点）

### 2.5.3 向量化实现

```python
class PositionScalingLayer(nn.Module):
    """
    Position Scaling Layer (Module 1 专用)

    在 log 尺度上的锚点之间线性插值，得到位置权重
    使用 softplus 保证权重非负
    """
    def __init__(self, anchor_positions=[1000, 10000, 100000]):
        super().__init__()
        self.num_anchors = len(anchor_positions)

        # 预计算 log 尺度的锚点位置（常量）
        self.register_buffer(
            'log_anchors',
            torch.log10(torch.tensor(anchor_positions, dtype=torch.float32))
        )

        # 可学习的锚点权重（原始参数）
        # 初始化为 softplus 的逆值，使得 softplus(raw) ≈ 1.0
        # softplus(x) = log(1 + exp(x))，要使结果为 1.0，x = log(e - 1) ≈ 0.5413
        init_value = math.log(math.exp(1.0) - 1)  # ≈ 0.5413
        self.anchor_weights_raw = nn.Parameter(
            torch.full((self.num_anchors,), init_value)
        )

    @property
    def anchor_weights(self):
        """通过 softplus 保证权重非负"""
        return F.softplus(self.anchor_weights_raw)

    def forward(self, logits, positions):
        """
        Args:
            logits: (num_keys,) - MLP 输出的 logits
            positions: (num_keys,) - 每个 Key 的位置（绝对或相对）

        Returns:
            scaled_logits: (num_keys,) - 位置缩放后的 logits
        """
        # 计算每个位置的权重
        pos_weights = self._interpolate_weights(positions)

        # 乘到 logits 上
        return logits * pos_weights

    def _interpolate_weights(self, positions):
        """
        在 log 尺度上线性插值

        Args:
            positions: (num_keys,) - 位置值（>= 1）

        Returns:
            weights: (num_keys,) - 插值得到的权重（非负）
        """
        # 获取经过 softplus 的非负权重
        anchor_weights = self.anchor_weights  # (num_anchors,)

        # 避免 log(0)
        positions = positions.clamp(min=1)
        log_pos = torch.log10(positions.float())  # (num_keys,)

        # 初始化为第一个锚点权重
        weights = torch.full_like(log_pos, anchor_weights[0])

        # 遍历每个区间进行插值
        for i in range(self.num_anchors - 1):
            log_left = self.log_anchors[i]
            log_right = self.log_anchors[i + 1]
            w_left = anchor_weights[i]
            w_right = anchor_weights[i + 1]

            # 找到在此区间内的位置
            in_interval = (log_pos >= log_left) & (log_pos < log_right)

            # 计算插值系数 t ∈ [0, 1]
            t = (log_pos - log_left) / (log_right - log_left)
            t = t.clamp(0, 1)

            # 线性插值
            interpolated = w_left * (1 - t) + w_right * t
            weights = torch.where(in_interval, interpolated, weights)

        # 大于最大锚点的位置使用最后一个权重
        above_max = log_pos >= self.log_anchors[-1]
        weights = torch.where(above_max, anchor_weights[-1], weights)

        return weights
```

> **非负约束**：
> - 原始参数 `anchor_weights_raw` 初始化为 `log(e-1) ≈ 0.5413`
> - 通过 `softplus` 变换后，初始权重 ≈ 1.0
> - 训练过程中权重始终非负

### 2.5.4 与 Module 1 网络的集成

```python
class Module1KeyPruningNetwork(nn.Module):
    """
    Module 1: Key Pruning 完整网络
    """
    def __init__(self, num_bins=128, num_freqs=64, num_kernels=3,
                 mlp_hidden=64, anchor_positions=[1000, 10000, 100000]):
        super().__init__()

        # Kernel Encoding Layer（与 Module 2 共享设计）
        self.kernel_layer = KernelEncodingLayer(num_bins, num_freqs, num_kernels)

        # MLP Layer（Module 1 专用）
        self.mlp = nn.Sequential(
            nn.Linear(num_bins, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

        # Position Scaling Layer（Module 1 专用）
        self.position_scaling = PositionScalingLayer(anchor_positions)

    def forward(self, K, key_positions, reference_angles):
        """
        Args:
            K: (num_keys, head_dim) - Key 向量
            key_positions: (num_keys,) - Key 的位置
            reference_angles: (num_freqs,) - 当前 round 的参考角度

        Returns:
            drop_probs: (num_keys,) - drop 概率
        """
        # 1. Kernel Encoding
        encoded = self.kernel_layer(K, reference_angles)  # (num_keys, num_bins)

        # 2. MLP
        logits = self.mlp(encoded).squeeze(-1)  # (num_keys,)

        # 3. Position Scaling（Module 1 专用）
        scaled_logits = self.position_scaling(logits, key_positions)

        # 4. Sigmoid
        drop_probs = torch.sigmoid(scaled_logits)

        return drop_probs
```

### 2.5.5 Module 2 网络结构（无 Position Scaling）

> **对比**：Module 2 的网络更简洁，没有 MLP 和 Position Scaling。

```python
class Module2BinningNetwork(nn.Module):
    """
    Module 2: Key Binning / Query Routing 网络
    注意：没有 MLP，没有 Position Scaling
    """
    def __init__(self, num_bins=128, num_freqs=64, num_kernels=3):
        super().__init__()
        self.kernel_layer = KernelEncodingLayer(num_bins, num_freqs, num_kernels)

    def forward(self, x, reference_angles):
        """
        Args:
            x: (num_tokens, head_dim) - K 或 Q 向量
            reference_angles: (num_freqs,)

        Returns:
            bin_probs: (num_tokens, num_bins) - softmax 后的 bin 概率
        """
        # 1. Kernel Encoding
        logits = self.kernel_layer(x, reference_angles)  # (num_tokens, num_bins)

        # 2. 直接 Softmax（无 MLP，无 Position Scaling）
        bin_probs = F.softmax(logits, dim=-1)

        return bin_probs
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
| Module 1 (Key Pruning) | Kernel + MLP(128→h→1) + **Position Scaling** | 73,856 + MLP + **3** |
| Module 2 Key Binning | Kernel only | 73,856 |
| Module 2 Query Routing | Kernel only | 73,856 |

> **注意**：Position Scaling 仅用于 Module 1，只增加 3 个参数（锚点权重）。

**Module 1 详细参数**（假设 mlp_hidden=64）：
- Kernel Encoding: 73,856
- MLP: 128×64 + 64 + 64×1 + 1 = 8,256 + 65 = **8,321**
- Position Scaling: **3**
- **Module 1 总计**: ~82,180 per head

**每个 Query Head 总参数**: ~230,000 (Module 1 + Module 2 Key + Module 2 Query)

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

> **⚠️ RoPE 基数假设**：
>
> 上述代码固定使用 `base=10000`（标准 RoPE）。若目标模型使用不同配置，参考角度计算会与真实位置编码错位：
>
> | 模型变体 | 可能差异 |
> |----------|----------|
> | **RoPE Scaling** | 如 LLaMA 的 NTK-aware scaling、YaRN 等 |
> | **不同 base** | 如某些 Qwen 变体使用 base=1000000 |
> | **不同 head_dim** | 影响 num_freqs |
>
> **后续实验应**：
> - 从模型配置文件读取 `rope_base` / `rope_scaling` 参数
> - 或将 `base` 作为实验可调参数
>
> **当前 POC 阶段**：假设标准 RoPE (base=10000)，与参考实现一致。

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
| 2025-12-15 | 添加 Kappa 正数约束待实验说明（无约束 vs Softplus） |
| 2025-12-15 | 添加 Position Scaling Layer（Module 1 专用）：log 尺度锚点插值，乘在 logit 上；明确 Module 2 不使用位置编码 |
| 2025-12-15 | 添加 RoPE 基数假设警告：固定 base=10000，后续需支持 RoPE scaling 和不同 base |
