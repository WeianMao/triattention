# SpeckV KV 打分聚合风险记录（潜在单头主导问题）

## 问题描述
- 当前裁剪逻辑（`R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py` → `SparseRoundPruner._select_keep_indices`）流程：
  1) 对每个采样头生成 `per_head_scores`，形状 `[num_heads, candidate_count]`。
  2) 对每个头各自取 top_k（k=per_head_quota）形成并集 `union_indices`。
  3) 当 `union_indices` 数量 ≥ `keep_count` 时，在 union 内用 `combined = per_head_scores.max(dim=0)` 再做一次 top_k，得到最终保留索引。
- 漏洞：如果 `union_indices` 远大于 `keep_count`，最终决策几乎完全依赖 `combined` 的 max。若各头分数量纲差异大（某些头整体分值显著高），这些“高幅值头”会主导最终 KV 选择，即便其他头也有高分位置，仍可能被挤掉。
- 现状：代码未对 per-head 分数做归一化/均衡，也没有“每头至少保留 N 个”之类的硬约束，只有“先 union 各头 top_k 再 max”这一层弱保护。

## 影响
- 在多头分值分布不均的情况下，剪枝可能偏向少数高幅值头，导致其他头的信息被过度裁剪，可能损害性能。

## 解决思路（暂未实现，需要可配置）
- 在聚合前对每个头的分数做归一化（例如对该 head 的所有候选 KV 的分数做归一化），再参与后续的 max；归一化开关需可配置，默认不启用以兼容现有行为。
- 其他可选防护（待评估）：按头配额/均分配额、对 `combined` 使用 mean/加权而非 max、或对 union 后每头贡献设上限。
