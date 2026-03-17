# Experiment 011: Learnable Probe Vectors

## Overview

基于 exp_006_module2_reverse_cross_trace_validation 的消融实验。仅修改网络架构设计部分，其他所有组件保持不变（数据加载、训练流程、softmax方向、损失函数、评估指标等）。

## Motivation

原有的 KernelEncodingLayer 使用 von Mises kernel 编码，设计较为复杂：
- 将向量分解为 64 个 frequency bands 的 (magnitude, angle) 对
- 每个 frequency band 使用 3 个 von Mises kernels
- 每个 kernel 有 3 个可学习参数：mu (中心), kappa (集中度), weight (权重)
- 参数量较大（约 73,856 per network）

本实验旨在简化探针计算方式，使用更直接的线性映射，**但保留位置信息的注入方式**。

## Method: Learnable Probe Vectors

### Core Idea

每个探针（bin）对应一个**可学习的向量**，该向量的形状与 key/query 向量完全一致（head_dim = 128）。

**关键设计**：可学习向量需要像原架构中的 kernel 均值 (mu) 一样，**根据 reference position 应用 RoPE 旋转**。

### 计算流程

给定：
- 一个 post-RoPE 的 key 向量 $k \in \mathbb{R}^{d}$
- 第 $i$ 个探针的可学习向量 $p_i \in \mathbb{R}^{d}$（基础向量，未旋转）
- reference position（round 的中点位置）

计算步骤：

1. **对可学习向量应用 RoPE 旋转**：$\tilde{p}_i = \text{RoPE}(p_i, \text{ref\_pos})$
2. **点积 + bias**：$\text{score}_i = k \cdot \tilde{p}_i + b_i$

---

## 核心机制：RoPE 旋转（重点）

### 背景：为什么需要旋转

输入的 key/query 向量是 **post-RoPE** 的，即已经根据其在序列中的位置进行了旋转。为了让探针能够感知位置信息，可学习向量也需要旋转到对应的 reference position。

**类比原架构**：
- 原架构中，kernel 的均值 mu 通过 `mu_effective = mu + reference_angles` 注入位置信息
- 新架构中，可学习向量通过 RoPE 旋转注入位置信息

### RoPE 旋转公式

RoPE 将 head_dim=128 的向量视为 64 个 2D 复数对。对于第 $j$ 个 frequency dimension：

**Angular frequency**：
$$
\omega_j = \frac{1}{\text{base}^{2j/d}}, \quad \text{base}=10000, \quad j \in [0, 63]
$$

**旋转角度**：
$$
\theta_j = \text{pos} \times \omega_j
$$

**2D 旋转**（对向量的第 $2j$ 和 $2j+1$ 个元素）：
$$
\begin{bmatrix} v'_{2j} \\ v'_{2j+1} \end{bmatrix} = \begin{bmatrix} \cos\theta_j & -\sin\theta_j \\ \sin\theta_j & \cos\theta_j \end{bmatrix} \begin{bmatrix} v_{2j} \\ v_{2j+1} \end{bmatrix}
$$

### Reference Position 计算

与原架构一致：
$$
\text{ref\_pos} = \text{round\_start} + \frac{\text{round\_window}}{2}
$$

### 旋转的物理意义

- **Key** 在位置 $\text{pos}_k$ 被旋转了 $\text{pos}_k \times \omega$
- **Probe** 被旋转到 reference position $\text{ref\_pos} \times \omega$
- 两者点积时，RoPE 的相对位置编码特性使得结果反映 key 与 reference position 的相对关系

---

## Implementation

### RoPE 旋转函数

```python
def apply_rope_rotation(vectors, position, base=10000):
    """
    对向量应用 RoPE 旋转。

    Args:
        vectors: shape (num_vectors, head_dim) or (head_dim,)
        position: 旋转到的目标位置（标量）
        base: RoPE base (default: 10000)

    Returns:
        rotated_vectors: 与输入相同 shape
    """
    is_single = vectors.dim() == 1
    if is_single:
        vectors = vectors.unsqueeze(0)

    num_vectors, head_dim = vectors.shape
    num_freqs = head_dim // 2  # 64

    # 计算 angular frequencies: omega_j = 1 / (base^(2j/d))
    dim_indices = torch.arange(num_freqs, device=vectors.device)
    omega = 1.0 / (base ** (2 * dim_indices / head_dim))

    # 计算旋转角度: theta_j = pos * omega_j
    theta = position * omega  # shape: (num_freqs,)

    # 计算 cos 和 sin
    cos_theta = torch.cos(theta)  # shape: (num_freqs,)
    sin_theta = torch.sin(theta)  # shape: (num_freqs,)

    # 重组向量为复数对: (num_vectors, num_freqs, 2)
    vectors_complex = vectors.view(num_vectors, num_freqs, 2)

    # 应用 2D 旋转
    # v'_0 = v_0 * cos - v_1 * sin
    # v'_1 = v_0 * sin + v_1 * cos
    rotated = torch.stack([
        vectors_complex[..., 0] * cos_theta - vectors_complex[..., 1] * sin_theta,
        vectors_complex[..., 0] * sin_theta + vectors_complex[..., 1] * cos_theta
    ], dim=-1)

    # 恢复原始 shape
    rotated = rotated.view(num_vectors, head_dim)

    if is_single:
        rotated = rotated.squeeze(0)

    return rotated
```

### LearnableProbeLayer 完整实现

```python
class LearnableProbeLayer(nn.Module):
    """
    Learnable Probe Layer with RoPE rotation.

    Each probe is a learnable vector that gets rotated by RoPE
    to the reference position before computing dot product with keys.
    """

    def __init__(self, num_bins=128, head_dim=128, base=10000):
        super().__init__()
        self.num_bins = num_bins
        self.head_dim = head_dim
        self.base = base

        # Probe vectors: each row is a probe vector (base vector, unrotated)
        # Shape: (num_bins, head_dim)
        # 初始化：使用与 Transformer QK 线性层相同的方式
        # std = 1 / sqrt(head_dim)，保证点积结果方差合理
        self.probes = nn.Parameter(
            torch.randn(num_bins, head_dim) / math.sqrt(head_dim)
        )

        # Bias for each probe
        # Shape: (num_bins,)
        self.bias = nn.Parameter(torch.zeros(num_bins))

    def _compute_reference_angles(self, round_start, round_window=128):
        """
        计算 reference position（与原架构一致）。
        """
        ref_position = round_start + round_window // 2
        return ref_position

    def forward(self, x, round_start, round_window=128):
        """
        Args:
            x: Input vectors (post-RoPE) of shape (num_vectors, head_dim) or (head_dim,)
            round_start: Starting position of current round
            round_window: Window size (default: 128)

        Returns:
            logits: Shape (num_vectors, num_bins) or (num_bins,)
        """
        is_single = x.dim() == 1
        if is_single:
            x = x.unsqueeze(0)

        # Step 1: 计算 reference position
        ref_pos = self._compute_reference_angles(round_start, round_window)

        # Step 2: 对所有 probe vectors 应用 RoPE 旋转
        # probes: (num_bins, head_dim) -> rotated_probes: (num_bins, head_dim)
        rotated_probes = apply_rope_rotation(self.probes, ref_pos, self.base)

        # Step 3: 点积 + bias
        # x: (num_vectors, head_dim)
        # rotated_probes.T: (head_dim, num_bins)
        # logits: (num_vectors, num_bins)
        logits = torch.matmul(x, rotated_probes.t()) + self.bias

        if is_single:
            logits = logits.squeeze(0)

        return logits
```

---

## 初始化策略

### 原理

Transformer 中 Q、K 的线性层初始化目标是让点积结果的方差合理（不会太大或太小）。

对于两个独立初始化的向量 $q, k \in \mathbb{R}^d$，如果每个元素 $q_i, k_i \sim \mathcal{N}(0, \sigma^2)$，则：
$$
\text{Var}(q \cdot k) = d \cdot \sigma^4
$$

为了让 $\text{Var}(q \cdot k) \approx 1$，设置 $\sigma = 1/\sqrt{d}$，即 $\sigma^4 = 1/d^2$，$d \cdot \sigma^4 = 1/d$。

考虑到 key 向量经过 RoPE 后幅度基本不变，使用 $\sigma = 1/\sqrt{\text{head\_dim}}$ 是合理的。

### 具体初始化

```python
# Probe vectors 初始化
self.probes = nn.Parameter(
    torch.randn(num_bins, head_dim) / math.sqrt(head_dim)
)

# Bias 初始化为 0
self.bias = nn.Parameter(torch.zeros(num_bins))
```

对于 head_dim = 128：
- $\sigma = 1/\sqrt{128} \approx 0.088$
- 这比之前的 0.01 大约 9 倍，向量长度更合理

---

## 与原架构的对比

| 方面 | 原架构 (KernelEncodingLayer) | 新架构 (LearnableProbeLayer) |
|------|------------------------------|------------------------------|
| 编码方式 | von Mises kernel + frequency decomposition | 直接点积 |
| 位置信息 | reference_angles 加到 mu 上 | **RoPE 旋转 probe vectors** |
| 参数 | mu, kappa, weight (per freq, per kernel) + bias | probe vectors + bias |
| 参数量 (per network) | 128 × 64 × 3 × 3 + 128 = 73,856 | 128 × 128 + 128 = 16,512 |
| 初始化 | mu: 均匀分布在 [-π, π], kappa: 1.0, weight: randn*0.1 | **randn / sqrt(head_dim)** |
| 计算复杂度 | 较高（exp, cos, atan2） | 较低（RoPE旋转 + 矩阵乘法） |

---

## Key Design Decisions

### 1. 保留 RoPE 位置信息（关键）

可学习向量必须根据 reference position 进行 RoPE 旋转，与原架构中 mu 的位置注入方式保持一致。

### 2. 保持 softmax 方向不变

- Key network: softmax over keys (dim=0)
- Query network: softmax over bins (dim=-1)

### 3. 独立的 Key/Query 网络

与原架构一致，Key network 和 Query network 使用**相同结构但独立参数**。

---

## Experiment Plan

### 1. 基础实验

使用与 exp_006 相同的：
- 训练数据、测试数据
- 训练超参数（learning rate, batch size, epochs）
- 评估指标（hit rate, bin utilization 等）

仅替换网络架构。

### 2. 消融变体（可选）

| 变体 | 描述 |
|------|------|
| A | 基础版本：RoPE旋转 + 点积 + bias |
| B | 无 RoPE：不旋转 probe vectors（验证 RoPE 的重要性） |
| C | 添加 temperature：logits / temperature |

---

## Expected Outcomes

1. **参数量减少**：从 ~148K (两个网络) 降至 ~33K
2. **计算效率提升**：简单矩阵乘法 vs 复杂 kernel 计算
3. **性能对比**：验证简化架构是否能保持或提升 hit rate

---

## Files to Create

```
exp_011_learnable_probe_vectors/
├── model.py              # LearnableProbeLayer + Module2Network
├── train.py              # 复用 exp_006 训练逻辑
├── evaluate.py           # 复用 exp_006 评估逻辑
├── run.py                # 实验入口
└── README.md             # 实验说明
```

---

## References

- 基础实验：`exp_006_module2_reverse_cross_trace_validation/`
- 原始网络设计：`exp_006_module2_reverse_cross_trace_validation/model.py`
