# 频段幅值图说明

本工具基于 `qk.pt` 中的 Q/K 张量，对 RoPE 频段的幅值进行聚合和可视化，帮助理解在不同旋转频率下的能量分布。与注意力热图不同，这里我们重点关注各频段的绝对幅值（未取对数），并将横轴映射为真实的 RoPE 周期。

## 数据来源
- **输入**：`qk.pt`（`capture_qk_distributed.py` 生成），内含 `q`、`k` 两个张量，形状均为 `[num_layers, num_heads, seq_len, head_dim]`。
- **RoPE 周期**：脚本通过 `AutoConfig` + `Qwen3RotaryEmbedding` 恢复 `inv_freq`，将其转换为实际周期 `period = 2π / inv_freq`，单位为 token，可复现与 Qwen3 forward 完全一致的频段划分。

## 聚合过程
1. **幅值计算**：每个 head 的维度按 `(x, y)` 配对，利用 `sqrt(x^2 + y^2)` 得到各频段的幅值。
2. **图 1（Avg |K|）**：对所有 key token 的幅值沿序列方向取平均，反映每个频段在 key 表征上的“能量”。
3. **图 2（Avg |Q|）**：同理，对 query 取平均。
4. **图 3（Avg |Q||K| over causal pairs）**：在自回归约束下，对 `i ≥ j` 的所有 (query_i, key_j) 对累积 `|Q|_i · |K|_j` 并取平均，显示频段间的交互强度。
5. **图 4（Mean angle φ_f）**：恢复 RoPE 旋转前的 `(x, y)` 坐标后，计算每个频段的签名夹角平均值。夹角用 `atan2` 表示，范围在 [-π, π]。
6. **图 5（Angle variance σ²_f）**：同样基于签名夹角，评估每个频段的方差。
7. **图 6（Reconstructed Σ_f |Q||K| cos(ω_f Δ)）**：假设旋转角仅受距离影响，用图 3 的幅值作为权重，对 token 距离 Δ 的理论内积进行重构。
8. **图 7（Reconstructed Σ_f |Q||K| cos(ω_f Δ + φ_f)）**：在前一图的基础上，加入每个频段的平均夹角偏移 φ_f，观察角度漂移的影响。
9. **归一化**：幅值保持在线性域；两条重构曲线直接绘制加权余弦和，无额外对数或归一化。

## 输出说明
- 输出目录结构与注意力热图一致：`outputs/.../freq_magnitude_plots/qidXXXX_traceYY/layer_##_head_##_freq.png`。
- 每张图片包含七个子图，前五个的横轴为 RoPE 周期（对数刻度），后两个的横轴为 token 距离 Δ（同样为对数刻度）；纵轴分别为对应的平均幅值、角度统计或重构值。
- 同目录会附带 `README.md`，总结数值口径和图示含义。

## 使用命令
```bash
scripts/run_freq_magnitude_plots.sh --max-distance 10000 --pool-size 32
```
可通过追加参数定制，如 `--max-layers 4 --max-heads 8 --device cuda:1 --max-distance 10000 --pool-size 16`。默认使用 float32 在 GPU 上计算，若显存紧张可改为 `--device cpu`。

## 注意事项
- 由于周期跨度巨大（从约 6 token 到数千万 token），横轴采用对数刻度便于观察低频段细节。
- 图 3 的平均是在自回归掩码下对全部 (i, j) 对求和后除以对数，因此不会因序列长度而失真；图 4/5 的角度统计在池化后的 token 上计算（池化窗口可通过 `--pool-size` 调整）。
- 若希望对幅值再做归一化或仅关注特定频段，可在脚本中调整聚合方式或添加筛选。
