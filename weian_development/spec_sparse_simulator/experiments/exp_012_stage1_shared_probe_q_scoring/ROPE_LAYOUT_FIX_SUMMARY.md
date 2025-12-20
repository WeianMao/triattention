# RoPE 向量布局修复总结

## 问题背景

Module 2（学习探针打分）的性能一直低于 Hybrid Frequency 基线：
- Module 2: K=50 命中率 86.61%
- Hybrid Frequency: K=50 命中率 99.04%

本文档记录了定位和修复这个问题的过程。

## 调试过程

### 1. L2 归一化实验

最初发现 L2 归一化对性能有负面影响：

| 设置 | K=50 | K=500 | K=1000 |
|------|------|-------|--------|
| 有 L2 归一化 | 50.51% | 96.14% | 98.77% |
| 无 L2 归一化 | 86.61% | 98.69% | 98.95% |

**结论**：去掉 L2 归一化后性能从 50% 提升到 86%，但仍低于 Hybrid Frequency 的 99%。

### 2. invert_to_origin 实验

尝试让每个 Q 旋转到位置 0（像 Hybrid Frequency 那样），而不是旋转到 round 的参考位置：

| 设置 | K=50 |
|------|------|
| 旋转到参考位置 | 86.61% |
| 旋转到原点 (invert_to_origin=True) | 82.96% |

**结论**：这个改动反而让性能变差，说明问题不在这里。

### 3. 向量布局分析

通过对比 Module 2 和 Hybrid Frequency 的代码，发现关键差异：

**Module 2 原始实现（交错布局）**：
```python
# 假设向量是 [x0, y0, x1, y1, ...] 交错排列
vectors_complex = vectors.view(num_vectors, num_freqs, 2)
rotated = torch.stack([
    vectors_complex[..., 0] * cos - vectors_complex[..., 1] * sin,
    vectors_complex[..., 0] * sin + vectors_complex[..., 1] * cos
], dim=-1)
```

**Hybrid Frequency / transformers 实现（前后分离布局）**：
```python
# 向量是 [real_0...real_63, imag_0...imag_63] 前后分离
def rotate_half(x):
    d = x.shape[-1] // 2
    x1 = x[..., :d]   # real part
    x2 = x[..., d:]   # imag part
    return torch.cat((-x2, x1), dim=-1)

rotated = vectors * cos + rotate_half(vectors) * sin
```

### 4. 旋转对比测试

编写测试脚本验证两种旋转方式的差异：

```
=== Rotation Comparison ===
Module2 (interleaved) result[:8]: [-0.179, -0.192, 0.391, ...]
FrontBack result[:8]: [0.023, -0.705, -0.199, ...]

Max difference between two methods: 8.78

=== Compare with Hybrid Frequency invert_rope ===
Max diff Module2 vs HF: 8.94
Max diff FrontBack vs HF: 6.22
Max diff FrontBack(Qwen3 inv_freq) vs HF: 4.77e-07  ← 完美匹配！
```

**结论**：
1. 两种布局产生完全不同的结果
2. 使用前后分离布局 + Qwen3 的 inv_freq 才能与 Hybrid Frequency 完美匹配

### 5. inv_freq 差异

Qwen3 使用 YaRN 缩放，其 inv_freq 与标准 RoPE 公式不同：

```
Qwen3 inv_freq[:8]: [1.0, 0.806, 0.649, 0.523, 0.422, 0.340, 0.274, 0.221]
标准 omega[:8]:     [1.0, 0.866, 0.750, 0.649, 0.562, 0.487, 0.422, 0.365]
```

标准 RoPE 公式 `omega = 1/(base^(2j/d))` 本身是正确的，但 Qwen3/DeepSeek 模型使用了 YaRN 缩放后的不同参数值。

## 修复内容

### 修改的文件

1. **compute_kmeans_init.py**
   - 添加 `rotate_half()` 函数
   - 修改 `apply_rope_rotation()` 使用前后分离布局
   - 修改 `compute_magnitude_features()` 使用前后分离布局
   - 修改 `compute_magnitude_init()` 使用前后分离布局
   - 添加 `load_model_inv_freq()` 加载模型的 inv_freq
   - 添加 `inv_freq` 和 `invert_to_origin` 参数

2. **model.py**
   - 添加 `rotate_half()` 函数
   - 修改 `apply_rope_rotation()` 使用前后分离布局并支持 inv_freq
   - 修改 `compute_magnitude_features()` 使用前后分离布局
   - 修改 `DistanceBasedQueryScorer.forward()` 使用前后分离布局计算距离
   - 修改 `SharedProbeLayer` 存储并使用 inv_freq
   - 修改 `Module2Network` 和 `create_model()` 传递 inv_freq

3. **test_init_only.py**
   - 添加 `invert_to_origin` 参数支持

## 修复结果

| 设置 | K=50 | K=500 | K=1000 |
|------|------|-------|--------|
| 修复前 (交错布局 + base=10000) | 86.61% | 98.69% | 98.95% |
| **修复后 (前后布局 + 模型inv_freq)** | **99.13%** | **99.81%** | **99.94%** |
| Hybrid Frequency 基线 | 99.04% | 99.60% | 99.76% |

## 关键教训

1. **向量布局必须匹配数据格式**：transformers/Qwen3 使用前后分离布局 `[real, imag]`，不是交错布局 `[r0,i0,r1,i1,...]`

2. **RoPE 频率必须使用模型的实际参数**：不同模型可能使用不同的 RoPE 变体（如 YaRN），需要加载模型的 inv_freq 而不是从 base=10000 自己计算

3. **初始化和推理必须一致**：如果初始化用一种旋转方式，推理用另一种，会导致严重的性能下降

## 相关文件

- `test_hybrid_freq_baseline.py`: Hybrid Frequency 基线测试
- `EXPERIMENT_LOG.md`: L2 归一化实验记录
