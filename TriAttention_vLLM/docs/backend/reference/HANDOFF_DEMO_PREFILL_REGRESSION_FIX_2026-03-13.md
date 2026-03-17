# Handoff: Demo Prefill Regression Fix (2026-03-13)

## 1) 结论（给接手同事先看）

- **主问题已定位并修复**：长 prefill demo 场景下，`per_head + hf_aligned_global_per_head` 路径的选择器聚合逻辑有误，导致压缩后输出退化（重复/胡言乱语）。
- **修复后 demo 已复验通过**：在此前会坏的同配置下，输出恢复正常（`max_same_word_run=1`）。
- **不需要额外手工操作**：不需要改配置、不需要加额外 env，直接使用当前代码即可。

## 2) 根因是什么

问题在 vLLM runtime 的 HF 对齐 per-head 分组选择子链路：

- 文件：`TriAttention_vLLM/triattention_runtime/selector_hf.py`
- 模块：`_select_keep_indices_for_group_per_head(...)`
- 现象：跨层聚合使用了 `mean`，会导致该场景下 keep 集选择偏差，压缩后生成质量严重退化。

对照依据：

- R-KV 参考实现在对应 per-head 分组处使用的是 `max` 聚合：  
  `R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py:455`

## 3) 修复做了什么

- 在 `selector_hf.py` 的 group-selector 路径中，将跨层聚合默认行为改为 `max`（此前是 `mean`）。
- 保留了 debug 覆写开关用于后续排查，但默认行为已经是修复后的正确行为。

## 4) 验证结果（关键）

### 4.1 之前会坏的场景（现已恢复）

- 配置：  
  `pruning_mode=per_head`  
  `per_head_selection_semantics=hf_aligned_global_per_head`  
  `kv_budget=12000`  
  `protect_prefill=false`  
  OpenClaw-like long prefill dataset（`/tmp/tri_diag/openclaw_like_dataset.jsonl`）

- 修复前：`max_same_word_run=41`（明显重复退化）
- 修复后（无 debug 特殊开关）：`max_same_word_run=1`（输出可读，压缩正常触发）
  - 输出：`/tmp/tri_diag/20260313_diag_v58_afterfix_nodebug_out/shard00/run000.jsonl`

### 4.2 额外对照（定位闭环）

- “仅关闭 group selector”可恢复正常；
- “保留 group selector，但强制 group 聚合为 max”可恢复正常（norm on/off 都恢复）；
- 证明根因确实在该 group-selector 聚合子模块，而非 async/chunk/reclaim 主干。

## 5) 关于 legacy_layer_local

- `legacy_layer_local` 是另一条旧语义路径（每层本地 per-head 选择）。
- 在当前这个长 prefill demo case 下它仍有退化，**但这是独立问题**，不影响本次主修复闭环。
- 如果当前 demo 走的是 `hf_aligned_global_per_head`（主路径），这次问题已解决。

## 6) 对同事的使用说明（最简）

- 直接用当前分支代码即可，不需要改额外参数来绕过问题。
- demo 若沿用当前常规配置（HF 对齐 per-head 路径），应可直接运行，不再出现此前那类重复退化。
