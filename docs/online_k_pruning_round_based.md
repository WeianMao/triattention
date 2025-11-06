# Round-Based Online K Pruning Plan

This note distills the requested adjustments to the hybrid frequency-based key pruning workflow so the upcoming experiment matches the intended online cache maintenance semantics.

## Terminology & Symbols

- `M`: hard cap on the number of keys (`K`) we are willing to keep resident in GPU memory at any moment. Default `M = 2048`.
- `W`: window size (tokens per maintenance round). After decoding `W` tokens we trigger a maintenance step. Default `W = 64`.
- `Δ` (`delta`): relative position between the query to be decoded and a candidate key. Sign convention will match the original implementation (positive `Δ` means key is older than the query).
- `offset`: a geometric series of positional offsets `[1, 2, 4, 8, …, max_length]` used to sweep over future query positions. Default `max_length = 65536` (`2**16`).
- `S_phase(Δ; τ)`, `S_amp(τ)`: components of the existing hybrid score reused here. `τ` denotes the frequency index.
- `S_hybrid(Δ; τ) = S_phase(Δ; τ) + S_amp(τ)`: base per-frequency score.
- Aggregator: either `mean` (default) or `max` to collapse the per-frequency/per-offset scores into a single scalar per key before Top-K selection.

## Round-Based Maintenance Logic

1. **Initial state**: Start a decoding round with up to `M` existing keys.
2. **Maintenance trigger**: Before decoding the next batch of `W` tokens, recompute scores for every resident key and select the top `M - W` entries. All other pre-existing keys are considered evicted for the duration of the upcoming round.
3. **Decode window**: As the next `W` tokens are generated, their associated (query,key) entries are appended to the cache. During this round, the newly produced `W` keys are never evicted.
4. **Next round**: Repeat from step 2. The newly added keys participate in the next scoring cycle, where once again only `M - W` of the pre-existing keys survive to make room for the next `W` arrivals.

## Score Construction With Future Offsets

To emulate the online behaviour without managing a true rolling KV cache, we evaluate each resident key against a sweep of prospective future queries.

1. For a candidate key with relative position `Δ`, form a vector of synthetic positions `Δ + offset`. Each element corresponds to a hypothetical future query index recovered via the geometric offset list.
2. For every such position and frequency index `τ`, compute `S_hybrid(Δ + offset; τ)` using the existing phase/amplitude terms. Because the actual query index within the next round is unknown, this grid captures both near-term and far-future attention requirements.
3. Aggregate the per-offset hybrid scores into a single scalar per key via either:
   - `mean`: arithmetic mean of the scores across offsets (default experiment setting).
   - `max`: maximum score across offsets (secondary run for comparison).
4. Use these aggregated scores to rank keys and retain the top `M - W` entries during each maintenance step.

## Implementation Notes

- The prior code scored keys independently per query. We now mask out keys that were "evicted" in earlier rounds so they never reappear in later scoring passes, which mirrors the behaviour of a shrinking KV cache without physically reconstructing it.
- Only K tensors are available on disk. The experiment continues to ignore V vectors and focuses exclusively on the key-side pruning heuristics.
- The statistical terms `|E[q]|` and `E[|q|]` should still be sourced from the reference trace used previously (cross-trace statistics remain unchanged).
- Experiments must keep the original scripts untouched. Work from a duplicated script and direct outputs to a fresh directory to avoid overwriting prior artifacts.
- Run both aggregation variants (`mean`, `max`) across the 100 sampled heads from the pre-existing head list. Record hit-rate metrics and visualisations under the new output root.

## 轮次脚本使用方法

- 新脚本位于 `weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`，在原版基础上新增以下参数：
  - `--max-keys` (`--keep-keys` 别名)：每轮维护结束后允许保留的 Key 上限 `M`。
  - `--round-window`: 每轮解码 token 数 `W`。
  - `--offset-max-length`: 几何偏移序列的最大值（默认 65536，对应 `1,2,4,...`）。
  - `--score-aggregation`: `mean` 或 `max`，用于聚合偏移得分。
- 输出目录会追加 `agg_{aggregation}_max{M}_w{W}` 子目录，避免覆盖旧实验。
- 运行示例：

```bash
conda run -n dc python weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py \
  outputs/deepseek_r1_qwen3_8b/qk_bf16_traces \
  --trace qid0003_trace34 \
  --stats-trace outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0008_trace46 \
  --head-sample-file weian_development/online_k_pruning_viz/hybrid_sample_heads.json \
  --round-window 64 \
  --max-keys 2048 \
  --score-aggregation mean \
  --offset-max-length 65536 \
  --device cuda:0 --dtype float32 --verbose
```

## 当前实验结果

- 采样头数：100（`hybrid_sample_heads.json`）。
- 主 Trace：`qid0003_trace34`；统计 Trace：`qid0008_trace46`。
- `M = 2048`、`W = 64`、`offset_max_length = 65536`，其余保持默认。

| 聚合方式 | 总体命中率 | 指标路径 |
| --- | --- | --- |
| `mean` | 0.9902 | `outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_hybrid_rounds_runs/qid0003_trace34/agg_mean_max2048_w64/retention_metrics.json` |
| `max` | 0.9899 | `outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_hybrid_rounds_runs/qid0003_trace34/agg_max_max2048_w64/retention_metrics.json` |

- 两轮均显示 Layer 2 Head 31 为最弱点，其余大多数采样 head 命中率 > 0.95。
- 可视化、命中 logs 与逐头指标保存在对应输出目录下；图表文件名带有 `rounds_{mean|max}` 后缀以区分聚合策略。
