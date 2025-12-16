# 神经网络架构

## 概述

Module 1 和 Module 2 共享相同的输入编码方式：**Kernel-based Encoding**。

---

## 1. 输入预处理：旋转参考系

### 核心思想

神经网络输入是 **post-RoPE** 的 K/Q 向量，在**旋转的参考系**中表示。

### 参考角度计算

使用**零角度向量 (1, 0)** 作为参考，经过 RoPE 旋转后：

```python
def get_reference_angles(round_start, num_freqs=64, round_window=128):
    ref_position = round_start + round_window // 2  # round 中间位置
    omega = 1.0 / (10000 ** (2 * torch.arange(num_freqs) / (num_freqs * 2)))
    return ref_position * omega  # shape: (num_freqs,)
```

### RoPE 频段结构

对于 `head_dim = 128`：64 个频段，每个是 2D 平面

```
频段 j: [dim_{2j}, dim_{2j+1}] → 角频率 ω_j
```

### 关键优化

**参考角度不在输入端处理，而在 Kernel 参数端处理**：

```python
# 优化方法：round 开头将 reference_angle 加到 mu
mu_effective = mu + reference_angles  # 只计算一次
kernel_output = von_mises(angle, mu_effective, kappa)
```

---

## 2. Kernel Encoding Layer

### 变量定义

**输入变量**（从 K/Q 向量提取）：

| 变量 | 符号 | 定义 | Shape |
|------|------|------|-------|
| 输入向量 | `K` | post-RoPE 的 Key 或 Query 向量 | `(head_dim,)` |
| 复数表示 | `K_complex` | 将 K 按相邻维度配对，视为复数 | `(num_freqs, 2)` |
| 模长 | `magnitude_j` | 第 j 个频段的向量长度：`√(K[2j]² + K[2j+1]²)` | `(num_freqs,)` |
| 角度 | `angle_j` | 第 j 个频段的相位角：`atan2(K[2j+1], K[2j])` | `(num_freqs,)` |
| 参考角度 | `reference_angles` | 当前 round 的参考角度（见 Section 1） | `(num_freqs,)` |

**可学习参数**：

| 参数 | 符号 | 定义 | Shape | 初始化 |
|------|------|------|-------|--------|
| **mu** | `μ_ijm` | 第 i 个 bin、第 j 个频段、第 m 个 kernel 的中心角度 | `(num_bins, num_freqs, num_kernels)` | 3 个 kernel 间隔 2π/3 |
| **kappa** | `κ_ijm` | 集中度参数（类似 1/σ²，越大越集中） | `(num_bins, num_freqs, num_kernels)` | ~1.0（覆盖范围大） |
| **weight** | `w_ijm` | 各 kernel 的加权系数 | `(num_bins, num_freqs, num_kernels)` | ~0.1（小值） |
| **bias** | `b_i` | 每个 bin 的偏置项 | `(num_bins,)` | 0 |

### 归一化 von Mises Kernel

```python
def normalized_von_mises(angle, mu, kappa):
    return exp(kappa * (cos(angle - mu) - 1))  # 最大值为 1
```

**优势**：
- 天然处理角度周期性
- 无需计算贝塞尔函数
- 只需 `cos` 和 `exp`

### 计算公式

每个频段使用 **3 个 von Mises Kernel**，输出公式：

```
output_i = Σ_j [ magnitude_j × Σ_m (w_ijm × vM(angle_j, μ_ijm + ref_j, κ_ijm)) ] + b_i
```

其中 `vM(θ, μ, κ) = exp(κ × (cos(θ - μ) - 1))`

### 向量化实现

```python
def kernel_encoding(K, mu, kappa, weight, bias, reference_angles):
    """
    Args:
        K: (head_dim,) - 输入向量
        mu: (num_bins, num_freqs, num_kernels) - 【可学习】kernel 中心
        kappa: (num_bins, num_freqs, num_kernels) - 【可学习】集中度
        weight: (num_bins, num_freqs, num_kernels) - 【可学习】加权系数
        bias: (num_bins,) - 【可学习】偏置
        reference_angles: (num_freqs,) - 参考角度（非学习，每 round 计算一次）
    """
    # 提取模长和角度
    K_complex = K.view(-1, 2)                        # (num_freqs, 2)
    magnitude = K_complex.norm(dim=1)                # (num_freqs,)
    angle = atan2(K_complex[:, 1], K_complex[:, 0])  # (num_freqs,)

    # 参考角度加到 mu 上（优化：只算一次）
    mu_effective = mu + reference_angles.view(1, -1, 1)  # (num_bins, num_freqs, num_kernels)

    # von Mises kernel
    kernel = exp(kappa * (cos(angle.view(1, -1, 1) - mu_effective) - 1))  # (num_bins, num_freqs, num_kernels)

    # 加权求和
    weighted = (kernel * weight).sum(dim=2) * magnitude  # (num_bins, num_freqs)
    return weighted.sum(dim=1) + bias  # (num_bins,)
```

---

## 3. Position Scaling Layer（Module 1 专用）

### 设计动机

解码到后期时，KV Cache 中积累了大量 Key，**冗余的可能性更大**，因此需要更激进地 drop。Position Scaling 让模型学习到：随着序列变长，drop 概率应适当提高。

### 实现

在 **log 尺度**上设置锚点（1k, 10k, 100k），线性插值：

```python
class PositionScalingLayer(nn.Module):
    def __init__(self, anchor_positions=[1000, 10000, 100000]):
        self.log_anchors = torch.log10(torch.tensor(anchor_positions))
        # 初始化使 softplus(raw) ≈ 1.0
        init_val = math.log(math.exp(1.0) - 1)  # ≈ 0.5413
        self.anchor_weights_raw = nn.Parameter(torch.full((3,), init_val))

    def forward(self, logits, positions):
        weights = self._interpolate(positions)
        return logits * weights

    def _interpolate(self, positions):
        anchor_weights = F.softplus(self.anchor_weights_raw)  # 保证非负
        log_pos = torch.log10(positions.clamp(min=1).float())
        # 在 log 尺度上线性插值...
        return interpolated_weights
```

**应用位置**：在 MLP 输出的 logit 上，Sigmoid 之前

---

## 4. 完整网络结构

### Module 1: Key Pruning

```python
class Module1Network(nn.Module):
    def forward(self, K, positions, reference_angles):
        encoded = kernel_encoding(K, ...)         # (num_keys, 128)
        logits = mlp(encoded).squeeze(-1)         # (num_keys,)
        scaled = position_scaling(logits, positions)
        return sigmoid(scaled)                    # drop 概率
```

### Module 2: Binning

```python
class Module2Network(nn.Module):
    def forward(self, x, reference_angles):
        logits = kernel_encoding(x, ...)  # (num_tokens, 128)
        return softmax(logits, dim=-1)    # bin 概率
```

---

## 5. 参数量分析

| 组件 | 参数计算 | 数量 |
|------|----------|------|
| Kernel Layer | 128 × 64 × 3 × 3 + 128 | ~73,856 |
| MLP (128→64→1) | 128×64 + 64 + 64×1 + 1 | ~8,321 |
| Position Scaling | 3 | 3 |

| 模块 | 总参数 (per head) |
|------|-------------------|
| Module 1 | ~82,180 |
| Module 2 Key | ~73,856 |
| Module 2 Query | ~73,856 |

---

## 6. 待实验

- [ ] Kappa 正数约束：无约束 vs Softplus
- [ ] K/Q 网络参数共享（共享 mu/kappa，独立 weight/bias）
- [ ] RoPE 基数适配（当前假设 base=10000）
