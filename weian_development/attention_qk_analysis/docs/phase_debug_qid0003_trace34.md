# Phase Reconstruction Debug — qid0003_trace34 / layer 00 head 00

## 背景
`freq_magnitude_plots.py` 利用频段统计量（以频段索引 `f` 为单位）重建注意力核。记：

- `|Q|`, `|K|`：RoPE 还原后单频段的 query/key 幅值。
- `E[·]`：对自回归合法配对 `(i ≥ j)` 的平均。
- `Δ = i - j`：token 距离。
- `ω_f`：第 `f` 个 RoPE 频率。
- `θ_{q,i}`, `θ_{k,j}`：RoPE 还原后在频段 `f` 的相位。
- `δ_{ij} = θ_{q,i} - θ_{k,j}`：相位差。
- `φ_f`：`angle_statistics` 求得的平均相位差。

脚本当前统计 `E[|K|]`, `E[|Q|]`, `E[|Q||K|]` 与 `φ_f`，并用
`Σ_f E[|Q||K|] · cos(ω_f Δ + φ_f)` 重构注意力曲线。

从傅立叶展开的角度，若把频段 `f` 上的加权贡献写成

```
E[|Q||K| cos δ_{ij}] = A_f,
E[|Q||K| sin δ_{ij}] = B_f,
```

则对应的信号可表示成：

```
A_f cos(ω_f Δ) − B_f sin(ω_f Δ) = R_f cos(ω_f Δ + φ_f),
```

其中

```
R_f = sqrt(A_f^2 + B_f^2),
φ_f = atan2(B_f, A_f).
```

当前实现没有显式追踪 `A_f`, `B_f`，而是假设
`A_f ≈ E[|Q||K|] · E[cos δ_{ij}]`、`B_f ≈ E[|Q||K|] · E[sin δ_{ij}]`，即认为相位差与幅值独立。

希望带相位曲线能贴近真实 attention map。但在 `qid0003_trace34` 的 `layer_00_head_00` 上发现：

- `Σ_f E[|Q||K|] cos(ω_f Δ)`（不加 φ）拟合良好
- `Σ_f E[|Q||K|] cos(ω_f Δ + φ_f)` 与真实注意力偏差大，近邻得分反而被压低

## 调试步骤
1. 编写 `debug_phase_components.py`，针对 `outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34/qk.pt`：
   - 还原 RoPE 前的 Q/K
   - 计算：
     - `A_actual = E[|Q||K| cos δ_{ij}]`（幅值加权的余弦分量）
     - `B_actual = E[|Q||K| sin δ_{ij}]`（幅值加权的正弦分量）
     - 当前脚本的近似：`A_approx = E[|Q||K|] · E[cos δ_{ij}]`、`B_approx = E[|Q||K|] · E[sin δ_{ij}]`
   - 与直接 `q·k` 曲线对比三种重构：
     - `plain`（φ=0），`existing φ`（现有实现），`weighted comps`（使用 `A_actual`/`B_actual`）

   运行结果：
   ```
   Curve MSE vs direct dot-product:
   plain (φ=0)    ≈ 7639
   existing φ     ≈ 7940  (更差)
   weighted comps ≈ 3243  (显著提升)
   ```
   
   并且逐频段检查发现：
   - 频段 15：`A_actual ≈ 0.78`，`A_approx ≈ 1.23`，`φ_unweighted ≈ 0.45 rad`、`φ_weighted ≈ 1.17 rad`（其中 `φ_unweighted = atan2(E[sin δ], E[cos δ])`，`φ_weighted = atan2(B_actual, A_actual)`）。现有近似把真实相位推远，导致近邻贡献被误抵消。
   - 高能量频段（如 6、10、11、14）普遍存在 `A_actual` 与 `A_approx`、`φ_unw` 与 `φ_w` 不一致，说明幅值与角度强烈相关。

2. 编写 `debug_angle_by_distance.py`，查看不同 token 距离 Δ 下的夹角：
   - Δ=1 时，主频段 `cos` 接近 0.7，说明近邻具有明显相位偏好。
   - Δ 增大后，`cos` 趋于 0，方差增大（相位接近随机）。
   - 全局地对 `cos δ` 做简单平均，会把远距噪声与近邻强信号混在一起，削弱 `φ_f` 的指示能力。

## 问题定位
- `freq_magnitude_plots.py:116-142` 的 `angle_statistics` 先单位化再平均，未对 `|Q||K|` 加权。
- `freq_magnitude_plots.py:280-284` 的重构公式把失真的 `φ_f` 套在 `E[|Q||K|]` 上，等价于假设 `δ` 与幅值独立。
- 实际上，强幅值往往对应近邻，对相位有明显偏好；弱幅值（远距噪声）会把平均角拉向 0 或 ±π。结果是加入 `φ_f` 反而削弱了近邻的权重，引起可视化“翻转”。

## 修正建议（仅建议，不修改）
1. 在统计阶段记录 `A_f = E[|Q||K| cos δ_{ij}]` 与 `B_f = E[|Q||K| sin δ_{ij}]`，或等价的幅度/角度：
   - `R_f = √(A_f² + B_f²)`
   - `φ_f = atan2(B_f, A_f)`
   - 可额外记录 `E[|Q||K|]` 以便比较。
2. 重构时用 `A_f`, `B_f` 直接组合：
   - `Σ_f (A_f cos(ω_f Δ) − B_f sin(ω_f Δ))`
   - 或 `Σ_f R_f cos(ω_f Δ + φ_f)`，其中 `R_f = √(A_f² + B_f²)`。
3. 若希望比较不同 Δ 的行为，可进一步分桶（例如按 Δ 或 log Δ 分段统计），但核心问题在于“应按幅值加权”。

## 验证思路
- 更新统计/重构逻辑后，再跑 `debug_phase_components.py`，`weighted comps` 与新实现应一致，且 MSE 接近 3243。
- 在 `freq_magnitude_plots.py` 生成的图像中，`Σ_f |Q||K| cos(ω_f Δ + φ_f)` 曲线应回到与真实注意力相近的趋势（近邻高、远距低）。

## 附件
- 调试脚本：
  - `weian_development/attention_qk_analysis/debug_phase_components.py`
  - `weian_development/attention_qk_analysis/debug_angle_by_distance.py`

这些脚本不修改原流程，可直接复现上述诊断。
