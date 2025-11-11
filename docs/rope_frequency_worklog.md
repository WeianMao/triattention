# RoPE Frequency Worklog

## 背景与目标
- 需要在 `outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34` 上做快速可视化，避免重新跑 `scripts/run_freq_magnitude_plots.sh`。
- 目标是直接基于捕获的 Q/K 张量分析 RoPE 频段的行为，同时给出易于迭代的脚本和调试工具。

## 当前脚本与用途
| 脚本 | 作用 | 主要参数 |
| --- | --- | --- |
| `weian_development/attention_qk_analysis/freq_magnitude_single_plot.py` | 单头调试版，重建 Σ |Q||K| cos(ωΔ) 及 ground truth，用于快速 sanity check | `--layer` `--head` `--output-path`
| `weian_development/attention_qk_analysis/freq_magnitude_single_series.py` | 批量聚合版。对所有 query 采样历史打分、在 RoPE 频段做离散变换，绘制 (1) RoPE-frequency reconstruction、(2) 平均 raw Q·K、(3) FFT ground truth（log 轴对齐） | `--fft-window` (默认全长) `--energy-eps`
| `weian_development/attention_qk_analysis/freq_single_token_debug.py` | baseline：直接对单个 query 的历史打分做 DFT 重建，用来确认流程与模型前向一致 | `--token-index` `--fft-window`
| `weian_development/attention_qk_analysis/freq_single_token_debug_freqwise.py` | 副本，当前和 baseline 一致；计划在其上做逐频段拟合实验 | 同上 |

生成的关键图片：
- `outputs/deepseek_r1_qwen3_8b/layer_00_head_00_freq_series.png`
- `outputs/deepseek_r1_qwen3_8b/layer_00_head_00_token_debug.png`
- `outputs/deepseek_r1_qwen3_8b/layer_00_head_00_token_debug_freqwise.png`

## 数据来源与注意事项
- KV cache 由 `weian_development/attention_qk_analysis/capture_qk_distributed.py` 捕获，位置：每层 self-attention 的 `apply_rotary_pos_emb` 之后。
- 捕获到的 Q/K **已经包含** Qwen3/DeepSeek 的 RoPE attention scaling（YaRN 会在 cos/sin 上乘以 `attention_scaling`）。如果下游分析需要未缩放向量，要自行除回缩放因子。
- 读取模型配置时要保留 `rope_scaling` 字段：`rope_type: yarn`, `factor: 4.0`, `attn_factor: 0.878248856…`。`Qwen3RotaryEmbedding` 会自动兼容，把 `attn_factor` 转成 `attention_factor`。

## 目前方案概述
1. **单 token 验证**：
   - 直接对 `score(Δ)` 做 FFT/逆 FFT，确认与模型前向输出一致；误差级别 1e-5。
   - 证明捕获数据和频率表正确无误。
2. **聚合版频谱分析**：
   - 收集所有 query 的历史打分序列（默认使用全部 Δ，窗口可调）。
   - 逐 query 减去均值 → 用 RoPE 频率做离散变换 → 对每个 query 的频谱按能量归一 → 求平均并乘回平均能量 → 用 logspace 对 Δ 重新采样并可视化。
   - Ground truth 仍使用 FFT cross correlation（与 `freq_magnitude_plots.py` 相同），只截断到窗口长度。

## 未完事项 / 注意风险
- **逐频段拟合**：尝试分别处理 `r_f(Δ) = Re(q_f[t] · conj(k_f[t-Δ]))` 时，发现幅度随 Δ 的变化远大于纯粹 cos 波形，导致重建几乎是平线。后续若要继续，需要先对 `|Q_f||K_f|` 做建模或归一化；仅靠 `exp(-iω_fΔ)` 投影无法恢复细节。
- **YaRN scaling**：cos 表会 >1，不是 bug。若后续要比较“未放大”的 RoPE，需要在捕获阶段或分析阶段处理 scaling。
- **显存/内存占用**：全长窗口时会把历史打分一次搬到 CPU；当前默认的 `window=None` 会自动使用全长。若内存压力过大可以改 `--fft-window` 或在脚本里实现 chunked 处理。
- **多层/多头对齐**：当前示例集中在 layer0 head0；其他层/头只需调整参数即可运行。

## 下一步建议
1. 在 `freq_single_token_debug_freqwise.py` 上实现逐频段幅度归一后再聚合，查看能否贴近原始 `r_f(Δ)`。
2. 视需要把同样的思想迁移到 `freq_magnitude_single_series.py`，看看聚合后的逐频段谱是否能解释 Ground truth。
3. 如要比较“不含 attention scaling”的 RoPE，可以在 `capture_qk_distributed.py` 里额外保存 scaling 值或在分析脚本里除回缩放。

---
若要继续深挖，请先通读 `freq_magnitude_single_series.py` 和 `freq_single_token_debug.py`，理解 RoPE 频率变换的细节，再在 `freq_single_token_debug_freqwise.py` 上实验逐频段方案。
