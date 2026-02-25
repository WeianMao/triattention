# HF 对齐状态（当前默认实现）

- 更新时间：2026-02-25
- 当前主对齐参照脚本：
  - `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`

## 1. 当前结论（面向使用）

1. `per_head` 是当前主目标模式，已完成全量 anchor 验证（AIME24 sampled8，8 shards）。
2. 当前结果已处于 HF 参考线附近（同量级），可用于下一阶段验证。
3. `per_layer_per_head` 的代码路径已完成关键风险修复（GQA 聚合顺序），但端到端实验仍待后续执行。

## 2. 已确认的关键实现点（代码引用）

1. `per_head` 默认模式
   - `TriAttention_vLLM/triattention_v2/config.py:48`
2. `per_layer` 默认禁用（避免误用）
   - `TriAttention_vLLM/triattention_v2/config.py:204`
   - `TriAttention_vLLM/triattention_v2/selector_hf.py:30`
3. `per_layer_per_head` 在 GQA 下使用“先打分再按 KV-head 聚合”
   - 分组开关决策：`TriAttention_vLLM/triattention_v2/selector_hf.py:237`
   - 分组聚合实现：`TriAttention_vLLM/triattention_v2/selector_hf.py:263`
   - `per_layer_per_head` 聚合模式入口：`TriAttention_vLLM/triattention_v2/selector_hf.py:280`
   - 打分后聚合调用：`TriAttention_vLLM/triattention_v2/selector_hf.py:345`

## 3. `per_head` 全量 anchor 结果（记录）

1. 配置：
   - `TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor.yaml`
2. 结果（AIME24 sampled8）：
   - `acc = 42.9`
   - `num_scores = 240`
   - `timeout_samples = 2`

说明：
- 采样生成场景下，HF 与 vLLM 的逐 token 文本不一致不等于算法不对齐；
- 当前更看重同口径评测结果是否在合理区间、以及关键语义路径是否按代码审计对齐。

## 4. 当前不作为阻塞项（针对上面参照脚本）

以下项不在该 `per_head` 参照脚本的当前对齐范围内：
1. `disable_mlr`
2. `disable_trig`
3. `disable_top_n_high_freq`

如后续要做这些 ablation，再单独补对齐验证。
