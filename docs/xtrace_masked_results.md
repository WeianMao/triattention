XTrace 高频屏蔽变体实验记录（低命中头基准）

- 数据与采样：主 trace `outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34`，统计 trace `.../qid0008_trace46`，采样头 `weian_development/online_k_pruning_viz/hybrid_sample_heads_lowret_top10.json`。
- 共同参数：`--max-keys 2048 --round-window 64 --score-aggregation mean --offset-max-length 65536 --device cuda:0`。
- 脚本：`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace_masked_tuned.py`
  - 新增超参：`--period-threshold-scale`（周期阈值系数，period >= delta*scale 保留），`--mask-extra-term`（将频率掩码也应用到位置无关 extra 项）。
- 基线：原始 xtrace `attention_pruning_case_study_hybrid_rounds_xtrace.py` → overall 0.9647（`outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_xtrace_lowret/qid0003_trace34/agg_mean_max2048_w64/retention_metrics.json`）。

运行命令示例
```bash
# 屏蔽位置相关项+extra 项，周期阈值 scale=2.0（当前最佳）
conda run -n dc python weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace_masked_tuned.py \
  outputs/deepseek_r1_qwen3_8b/qk_bf16_traces \
  --trace qid0003_trace34 \
  --stats-trace outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0008_trace46 \
  --head-sample-file weian_development/online_k_pruning_viz/hybrid_sample_heads_lowret_top10.json \
  --device cuda:0 \
  --output-root outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_xtrace_masked_tuned_both_lowret_p20 \
  --period-threshold-scale 2.0 \
  --mask-extra-term
```

结果汇总（overall retention，10 heads）
- 屏蔽位置相关项+extra 项：
  - scale=0.1 → 0.9664（`..._both_lowret_p01/.../retention_metrics.json`）
  - 0.25 → 0.9670（`..._both_lowret_p025/.../retention_metrics.json`）
  - 0.50 → 0.9672（`..._both_lowret_p05/.../retention_metrics.json`）
  - 1.0 → 0.9673（`..._both_lowret_p10/.../retention_metrics.json`）
  - 2.0 → **0.9680（最佳）**（`..._both_lowret_p20/.../retention_metrics.json`）
  - 3.0 → 0.9676（`..._both_lowret_p30/.../retention_metrics.json`）
  - 4.0 → 0.9670（`..._both_lowret_p40/.../retention_metrics.json`）
- 仅屏蔽位置相关项（extra 不屏蔽，前序实验）：最佳 scale≈0.1，overall 0.9643（`..._xtrace_masked_tuned_lowret_p01/.../retention_metrics.json`），未超过基线。

结论
- 同时屏蔽位置相关+extra 项，阈值 scale≈2.0 时可小幅超越原始 xtrace（提升 ~0.0033）。继续增大或减小 scale 收益不明显，部分回落。
