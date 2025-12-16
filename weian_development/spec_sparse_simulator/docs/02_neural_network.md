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

### 归一化 von Mises Kernel

```python
def normalized_von_mises(angle, mu, kappa):
    return exp(kappa * (cos(angle - mu) - 1))  # 最大值为 1
```

**优势**：
- 天然处理角度周期性
- 无需计算贝塞尔函数
- 只需 `cos` 和 `exp`

### 多 Kernel 结构

每个频段使用 **3 个 von Mises Kernel**：

```
bin_i_freq_j = magnitude_j × Σ_m [weight_ijm × vM(angle_j, mu_ijm, kappa_ijm)]
```

### 参数初始化

```python
# mu: 3 个 kernel 间隔 2π/3
# kappa: 初始化 ~1.0（确保覆盖范围大，避免梯度消失）
# weight: 小值初始化 ~0.1
```

### 向量化实现

```python
def kernel_encoding(K, mu, kappa, weight, bias, reference_angles):
    """
    K: (head_dim,)
    mu, kappa, weight: (num_bins, num_freqs, num_kernels)
    """
    K_complex = K.view(-1, 2)  # (num_freqs, 2)
    magnitude = K_complex.norm(dim=1)
    angle = atan2(K_complex[:, 1], K_complex[:, 0])

    mu_effective = mu + reference_angles.view(1, -1, 1)
    kernel = exp(kappa * (cos(angle.view(1, -1, 1) - mu_effective) - 1))

    weighted = (kernel * weight).sum(dim=2) * magnitude  # (num_bins, num_freqs)
    return weighted.sum(dim=1) + bias  # (num_bins,)
```

---

## 3. Position Scaling Layer（Module 1 专用）

### 设计动机

不同位置的 Key 有不同的 drop 倾向基线：早期 Key 更易被 drop。

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
