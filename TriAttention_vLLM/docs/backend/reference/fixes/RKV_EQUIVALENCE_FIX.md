# R-KV 与 TriAttention 公式等价性问题分析与修复方案

## 1. 问题描述

当前 TriAttention 的 Triton kernel 与 R-KV 参考实现的打分结果**不一致**：

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| 分数相关性 | 0.24 | > 0.99 |
| TopK 重叠率 | ~20% | > 95% |

## 2. 根本原因

### 2.1 两个实现使用了不同的 K 表示

| 实现 | K 输入 | 文件位置 |
|------|--------|----------|
| R-KV | `K_unrot`（未旋转） | `R-KV/weian_development/speckv/round_pruning_utils.py:256-318` |
| TriAttention | `K_rot`（已旋转） | `TriAttention_vLLM/triattention/kernels/triton_scoring.py:116-163` |

### 2.2 K_rot 与 K_unrot 的关系

RoPE 旋转的数学定义（复数形式）：

$$
K_{rot}[f] = K_{unrot}[f] \cdot e^{i \cdot p \cdot \omega[f]}
$$

其中：
- $f$ = 频率索引 (0 到 freq_count-1)
- $p$ = token 的原始位置 (position)
- $\omega[f]$ = 第 $f$ 个频率的角频率

展开为实部/虚部形式：

$$
\begin{aligned}
K_{rot,r}[f] &= K_{unrot,r}[f] \cdot \cos(p \cdot \omega[f]) - K_{unrot,i}[f] \cdot \sin(p \cdot \omega[f]) \\
K_{rot,i}[f] &= K_{unrot,r}[f] \cdot \sin(p \cdot \omega[f]) + K_{unrot,i}[f] \cdot \cos(p \cdot \omega[f])
\end{aligned}
$$

## 3. R-KV 打分公式推导

### 3.1 R-KV 的计算流程

```python
# R-KV: round_pruning_utils.py
k_complex = to_complex_pairs(k_unrot)           # K_unrot -> 复数
relative = q_mean_complex * conj(k_complex)     # Q × conj(K_unrot)
phi = atan2(relative.imag, relative.real)       # 相位
amp = |Q| × |K_unrot|                           # 幅度

# 打分
phase = Δ × ω + φ
score = Σ_f(amp[f] × scale²[f] × cos(phase[f])) + extra
```

### 3.2 数学表达

定义：
- $Q = Q_r + i \cdot Q_i$ （Q 统计量，复数形式）
- $K = K_r + i \cdot K_i$ （K_unrot，复数形式）
- $\Delta = t - p$ （位置差，$t$ = round_start + offset, $p$ = key position）

**R-KV 打分公式**：

$$
\text{score}_{RKV} = \sum_f \left[ |Q[f]| \cdot |K[f]| \cdot s^2[f] \cdot \cos(\Delta \cdot \omega[f] + \phi[f]) \right] + \text{extra}
$$

其中：
$$
\phi[f] = \arg(Q[f] \cdot \overline{K[f]}) = \arctan2(Q_i K_r - Q_r K_i, \; Q_r K_r + Q_i K_i)
$$

### 3.3 展开三角函数

利用 $\cos(A + B) = \cos A \cos B - \sin A \sin B$：

$$
\begin{aligned}
|Q| |K| \cos(\Delta\omega + \phi)
&= |Q| |K| \left[ \cos(\Delta\omega) \cos\phi - \sin(\Delta\omega) \sin\phi \right] \\
&= |Q| |K| \cos\phi \cdot \cos(\Delta\omega) - |Q| |K| \sin\phi \cdot \sin(\Delta\omega)
\end{aligned}
$$

注意到：
$$
\begin{aligned}
|Q| |K| \cos\phi &= \text{Re}(Q \cdot \overline{K}) = Q_r K_r + Q_i K_i \\
|Q| |K| \sin\phi &= \text{Im}(Q \cdot \overline{K}) = Q_i K_r - Q_r K_i
\end{aligned}
$$

因此 **R-KV 公式等价于**：

$$
\boxed{
\text{score}_{RKV} = \sum_f s^2[f] \left[ \text{Re}(Q \cdot \overline{K_{unrot}}) \cos(\Delta\omega) - \text{Im}(Q \cdot \overline{K_{unrot}}) \sin(\Delta\omega) \right] + \text{extra}
}
$$

## 4. TriAttention 当前实现

### 4.1 当前代码逻辑

```python
# TriAttention: triton_scoring.py (当前实现)
# 输入: K_rot（已旋转的 K）

# 计算 Q × conj(K_rot)
prod_real = Q_r × K_rot_r + Q_i × K_rot_i   # Re(Q × conj(K_rot))
prod_imag = Q_i × K_rot_r - Q_r × K_rot_i   # Im(Q × conj(K_rot))

A = scale × prod_real
B = scale × prod_imag

# 相位只用位置差
phase = Δ × ω

# 打分
score = Σ_f(A[f] × cos(phase[f]) - B[f] × sin(phase[f])) + extra
```

### 4.2 当前公式

$$
\text{score}_{current} = \sum_f s^2[f] \left[ \text{Re}(Q \cdot \overline{K_{rot}}) \cos(\Delta\omega) - \text{Im}(Q \cdot \overline{K_{rot}}) \sin(\Delta\omega) \right] + \text{extra}
$$

### 4.3 问题所在

**当前实现使用的是 $K_{rot}$，而不是 $K_{unrot}$！**

由于：
$$
K_{rot} = K_{unrot} \cdot e^{i \cdot p \cdot \omega}
$$

所以：
$$
Q \cdot \overline{K_{rot}} = Q \cdot \overline{K_{unrot}} \cdot e^{-i \cdot p \cdot \omega}
$$

**两者相差一个相位因子 $e^{-i \cdot p \cdot \omega}$**，这就是公式不等价的根本原因。

## 5. 方案 B：修正相位补偿

### 5.1 目标

在使用 $K_{rot}$ 的情况下，通过修正相位计算，使得最终打分与 R-KV 一致。

### 5.2 数学推导

我们需要从 $Q \cdot \overline{K_{rot}}$ 恢复出 $Q \cdot \overline{K_{unrot}}$ 的效果。

**Step 1**: 建立关系

$$
Q \cdot \overline{K_{unrot}} = Q \cdot \overline{K_{rot}} \cdot e^{i \cdot p \cdot \omega}
$$

**Step 2**: 代入 R-KV 公式

$$
\begin{aligned}
\text{score}_{RKV} &= \sum_f s^2 \left[ \text{Re}(Q \cdot \overline{K_{unrot}}) \cos(\Delta\omega) - \text{Im}(Q \cdot \overline{K_{unrot}}) \sin(\Delta\omega) \right] \\
&= \sum_f s^2 \left[ \text{Re}(Q \cdot \overline{K_{rot}} \cdot e^{ip\omega}) \cos(\Delta\omega) - \text{Im}(Q \cdot \overline{K_{rot}} \cdot e^{ip\omega}) \sin(\Delta\omega) \right]
\end{aligned}
$$

**Step 3**: 展开 $e^{ip\omega}$ 乘法

设 $Z = Q \cdot \overline{K_{rot}} = A' + i B'$（这是当前 TriAttention 计算的值）

则：
$$
Z \cdot e^{ip\omega} = (A' + iB')(\cos(p\omega) + i\sin(p\omega))
$$

展开：
$$
\begin{aligned}
\text{Re}(Z \cdot e^{ip\omega}) &= A' \cos(p\omega) - B' \sin(p\omega) \\
\text{Im}(Z \cdot e^{ip\omega}) &= A' \sin(p\omega) + B' \cos(p\omega)
\end{aligned}
$$

**Step 4**: 代入打分公式

$$
\begin{aligned}
\text{score} &= \sum_f s^2 \Big[ (A' \cos(p\omega) - B' \sin(p\omega)) \cos(\Delta\omega) \\
&\quad\quad\quad - (A' \sin(p\omega) + B' \cos(p\omega)) \sin(\Delta\omega) \Big]
\end{aligned}
$$

**Step 5**: 展开并合并同类项

$$
\begin{aligned}
&= \sum_f s^2 \Big[ A' \cos(p\omega)\cos(\Delta\omega) - B' \sin(p\omega)\cos(\Delta\omega) \\
&\quad\quad\quad - A' \sin(p\omega)\sin(\Delta\omega) - B' \cos(p\omega)\sin(\Delta\omega) \Big] \\
&= \sum_f s^2 \Big[ A' (\cos(p\omega)\cos(\Delta\omega) - \sin(p\omega)\sin(\Delta\omega)) \\
&\quad\quad\quad - B' (\sin(p\omega)\cos(\Delta\omega) + \cos(p\omega)\sin(\Delta\omega)) \Big] \\
&= \sum_f s^2 \Big[ A' \cos((p + \Delta)\omega) - B' \sin((p + \Delta)\omega) \Big]
\end{aligned}
$$

**Step 6**: 代入 $\Delta = t - p$

$$
p + \Delta = p + (t - p) = t
$$

因此：

$$
\boxed{
\text{score}_{corrected} = \sum_f s^2[f] \left[ A'[f] \cos(t \cdot \omega[f]) - B'[f] \sin(t \cdot \omega[f]) \right] + \text{extra}
}
$$

其中：
- $A'[f] = \text{Re}(Q[f] \cdot \overline{K_{rot}[f]}) = Q_r K_{rot,r} + Q_i K_{rot,i}$
- $B'[f] = \text{Im}(Q[f] \cdot \overline{K_{rot}[f]}) = Q_i K_{rot,r} - Q_r K_{rot,i}$
- $t = \text{round\_start} + \text{offset}$

### 5.3 关键发现

**修正后的相位只依赖 $t$（查询位置），与 $p$（key 位置）无关！**

这意味着：
- 当前实现：`phase = (t - p) × ω`（依赖 key 位置 p）
- 修正实现：`phase = t × ω`（只依赖查询位置 t）

## 6. 修复方案

### 6.1 需要修改的代码

**文件**: `TriAttention_vLLM/triattention/kernels/triton_scoring.py`

**当前代码** (第 151-155 行):
```python
# 当前：phase = (round_start + offset - position) * omega
delta_t = round_start + offset - positions  # [BLOCK_N]
phase = delta_t[:, None] * omega[None, :]   # [BLOCK_N, BLOCK_F]
```

**修正后**:
```python
# 修正：phase = (round_start + offset) * omega（只依赖查询位置）
t = round_start + offset  # scalar
phase = t * omega[None, :]  # [1, BLOCK_F] broadcast to [BLOCK_N, BLOCK_F]
```

### 6.2 修改影响

1. **计算简化**：不再需要加载和使用 `position_indices`（在打分计算中）
2. **内存访问减少**：无需读取每个 token 的位置
3. **数学正确性**：与 R-KV 公式完全等价

### 6.3 position_indices 的作用

修正后，`position_indices` 在打分阶段**不再需要**。但它仍然可能在以下场景有用：
- 调试和验证
- 其他需要知道 token 原始位置的操作
- 保护特定位置范围的 token（如 prefill）

## 7. 验证方法

修复后，应该满足以下条件：

```python
# 对于相同的输入
# R-KV: 使用 K_unrot
# TriAttention: 使用 K_rot

score_rkv = rkv_score_keys_for_round(K_unrot, ...)
score_tri = triattention_scoring(K_rot, ...)

# 期望
assert torch.allclose(score_rkv, score_tri, atol=1e-5)
# 或
correlation = torch.corrcoef([score_rkv, score_tri])[0,1]
assert correlation > 0.99
```

## 8. 总结

| 项目 | 当前实现 | 修正后 |
|------|----------|--------|
| 相位计算 | `(t - p) × ω` | `t × ω` |
| 是否依赖 key 位置 | 是 | 否 |
| 与 R-KV 等价 | ❌ | ✅ |

**核心洞察**：当使用已旋转的 $K_{rot}$ 时，RoPE 的位置信息已经"烘焙"在 K 里面了。正确的做法是用查询位置 $t$ 直接计算相位，而不是用位置差 $\Delta = t - p$。
