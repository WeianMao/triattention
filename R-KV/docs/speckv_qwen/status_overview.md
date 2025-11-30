# SpecKV-Qwen 项目概览（初稿）

> 面向不了解上下文的读者：说明目标、参考实现、当前状态与下一步。单 Agent 开发，协作条款省略。

## 1. 目标与交付
- 在 R-KV 框架/风格下适配 SpecKV 到 Qwen，核心逻辑需与早期 Qwen 版本等价。
- 产物：两份脚本（`run_rkv_aime25_official_sampled8_qwen.sh` 的 SpecKV 版；`run_speckv_aime24_official_sampled8.sh` 改模型为 Qwen 的版本）+ 差异对比 MD（SpecKV-Qwen vs `run_sparse_prefill_keep_sharded_eval.sh`）。
- 关键要求：模型/数据/kv_budget/position & pruner 逻辑/Prompt 模板保持与早期 Qwen 版本一致；发现关键偏差立即停下并告警。

## 2. 目录与角色
- `R-KV/docs/speckv_qwen/`：本项目规划与留档（本目录）。
- `R-KV/weian_script/aime24_official_sampled8/`：目标脚本所在目录，沿用 R-KV 风格。
- 参考实现（必须比对逻辑一致性）：
  - 早期模拟：`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`（Qwen，表现好）。
  - LazyEviction 评测：`LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`（Qwen，表现好）。
  - R-KV 重构：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`（Llama，表现差，风格参考但算法需核对）。

## 3. 当前状态（2025-XX-XX）
- 文档已迁入 `R-KV/docs/speckv_qwen/`，尚未编写新的脚本或对比文档。
- 现有 R-KV Llama 版可能含配置/逻辑偏差，需边对齐边验证。

## 4. 已知风险与停机条件
- 三版本若出现关键算法差异（pruner 状态、裁剪触发、kv_budget、position/RoPE、prompt 模板、模型/数据），必须先告警再开发。
- 脚本对比需做到“苹果对苹果”：同模型（Qwen）、同基准/数据；仅允许采样等次要差异。
- 现有 Llama 版效果差，可能来源：重构 bug/配置漂移或模型差异，不能盲信。

## 5. 下一步（简要）
- 梳理三份参考实现的算法路径与超参，列出关键点对照表。
- 起草两份目标脚本的修改草案（基于 R-KV 目录），并规划与 LazyEviction 版本的差异核查。
- 搭建差异对比 MD 框架，明确“允许的次要差异”和“必须一致的关键点”。***
