# Yarn 缩放误差排查说明

## 目的
`freq_magnitude_plots.py` 希望用频段统计量（`|Q|`, `|K|`, `|Q||K|`, `φ_f`）来重建注意力核，其中 `Σ_f |Q||K| cos(ω_f Δ + φ_f)` 应该贴近真实 attention map。

## 观察到的问题
当前带相位重构与原始注意力差异很大，但忽略相位的 `Σ_f |Q||K| cos(ω_f Δ)` 反而更接近真实图像。这说明我们计算的平均夹角 `φ_f` 被系统性扭曲。

## 根因
- DeepSeek-R1-Qwen3-8B 的 `config.json` 声明 `rope_scaling = {"rope_type": "yarn", ...}`。
- 无论是在 `scripts/yaml_runs_serialized/run_offline_deepseek_msgpack_64.sh` 生成 trace 还是在 `capture_qk_distributed.py` 捕获 Q/K，都会加载同一个模型目录，并通过 Qwen3 官方实现的 `Qwen3RotaryEmbedding` 生成 RoPE 表。
- Yarn 版本的 RoPE 会在返回 cos/sin 时乘上一个常数 `attention_scaling`（本模型约 1.1386，参考 `transformers/models/qwen3/modeling_qwen3.py:300-327` 及 `transformers/modeling_rope_utils.py:197-283`），而且从第一个 token 起就生效。
- `freq_magnitude_plots.py` 的 `invert_rope` 直接把捕获到的 Q/K 乘以 cos/sin 做反旋，忽略了这层缩放，导致“还原”的向量被放大。`angle_statistics` 因此得到错误的平均夹角 `φ_f`，带相位的重构被带偏。

## 修正思路
反旋时先恢复 Yarn 缩放，再做逆旋转。可以把 `invert_rope` 更新为：

```python
# weian_development/attention_qk_analysis/freq_magnitude_plots.py

def invert_rope(rotated: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, scale: float) -> torch.Tensor:
    base = rotated / scale
    cos_unit = cos / scale
    sin_unit = sin / scale
    return base * cos_unit - rotate_half(base) * sin_unit
```

调用处传入 `rotary.attention_scaling`：

```python
scale = float(rotary.attention_scaling)
q_orig = invert_rope(q_block, cos_table, sin_table, scale)
k_orig = invert_rope(k_block, cos_table, sin_table, scale)
```

也可以选择先把 `cos_table`、`sin_table` 除以 `scale` 后再调用旧的 `invert_rope`，效果等价。

## 验证建议
1. 用随机张量走一遍“应用 RoPE → 反 RoPE”闭环，若补偿正确，最大误差应 ~1e-6！否则会在 1e-1 的量级。
2. 重新生成频段诊断图；修正后 `Σ_f |Q||K| cos(ω_f Δ + φ_f)` 应明显更贴近真实 attention map。
