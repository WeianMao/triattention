# 开发准则（SpecKV-Qwen 版，单 Agent）

## 1. 范围与目录
- 所有实现放在 `R-KV/` 内，目标脚本位于 `R-KV/weian_script/aime24_official_sampled8/`。
- 参考但不修改：`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`、`LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`。
- 现有 R-KV Llama 版 (`run_speckv_aime24_official_sampled8.sh`) 仅用作风格/接口参考，算法需核对。

## 2. 算法一致性要求
- 关键要素必须与早期 Qwen/LazyEviction 版本等价：pruner 状态/触发、kv_budget、位置/RoPE 处理、round/窗口逻辑、prompt 模板、模型/数据源。
- 仅允许的差异：采样参数等非核心超参（需在对比文档中显式列出）。
- 发现关键不一致或无法对齐时，立即停止开发并在 `message.md` 记录告警。

## 3. 开发与验证
- 优先沿用 R-KV 脚本结构与配置方式，避免散落脚本；新增文件放在当前目录或既有子目录。
- 轻量验证优先：`python -m compileall`，必要时单样本/最小数据 smoke；不要贸然跑长任务。
- 变更说明：每次提交前更新 `status_overview.md` 进展要点，`project_todo.md` 勾选对应项。

## 4. 记录与留档
- 所有收到的关键信息与用户消息放入 `message.md` 归档（保留原文或简要转写）。
- 差异对比、假设与风险写入专门的差异 MD；重要发现需加“警告”标注。***

## 5. 计划管理与收尾
- 每次开始执行 `project_todo.md` 中的任一事项前，先在对应条目下细化当次执行计划（可拆子弹点，写清具体动作与预期输出）。
- 开发过程中若需重新规划：先通知用户；随后立即在 `project_todo.md` 更新新计划/调整；并在 `status_overview.md` 或 `message.md` 记录当前状态、问题、重规划原因与新的路径。
- 每个 Agent 完成当次开发后，需第一时间同步 `status_overview.md`/`project_todo.md` 状态并提交一次 commit，避免遗漏进度。***
