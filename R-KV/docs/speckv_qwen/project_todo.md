# SpecKV-Qwen TODO（单 Agent 用）

> 说明：使用 `- [ ]` 未完成，`- [x]` 已完成。无需填写 Agent 信息，按顺序推进即可；遇到关键算法偏差先停下并在 message 中记录。每次开始执行其中一项前，请在该条下补充可执行的分解计划；如需重规划，先通知用户，再在本文档更新新计划和状态。***

## 一、基线对齐与差异梳理
- [x] 通读三份参考实现（`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`、`LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`、`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`）  
  - 目标：锁定 SpecKV 关键路径和接口差异源。  
  - 动作：逐段标记 pruner 状态机、kv_budget 计算、position/RoPE 处理、round/窗口更新、prompt 模板、采样接口/参数、模型加载/分片、数据入口，做成对照表草稿。  
  - 验证：不跑代码，输出对照表初稿（可在差异 MD 草稿中形成表格）。  
  - 执行计划：1) 先通读 `weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`，按“pruner/kv_budget/position+RoPE/round窗口/prompt+数据/采样+前向patch/模型加载+分片”摘要；2) 同维度阅读 `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`；3) 再读 `R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`，标记与前两者的异同；4) 将要点写入 `R-KV/docs/speckv_qwen/diff_draft.md` 对照表草稿，标出潜在不一致/待确认项，并标记“允许差异/可能问题”。
- [x] 定义“必须一致 vs 允许差异”清单  
  - 目标：约束后续实现不偏离关键点。  
  - 动作：基于对照表标注硬性一致项（pruner/RoPE/kv_budget/prompt/数据/模型）和可放宽项（采样温度、logprobs 开关、日志路径等）。  
  - 验证：在差异 MD 草稿中列清单，用于后续自检。  
  - 执行计划：基于 `R-KV/docs/speckv_qwen/diff_draft.md` 的对照表，抽取必须一致项与允许差异项，分区列在同一文件；对未决/需确认项加显式“待确认”标记，预留后续同步/告警位。

## 二、脚本设计草稿（Qwen SpecKV 两套）
- [ ] SpecKV on `run_rkv_aime25_official_sampled8_qwen.sh` 骨架  
  - 目标：设计改动点列表（算法换 SpecKV，其余保持 R-KV 风格）。  
  - 动作：标出需要改的函数/参数/flag（SpecKV 前向 patch、pruner 配置、kv_budget、prompt/数据源）。  
  - 验证：设计审阅，自检是否覆盖所有关键一致项。  
  - 执行计划：阅读 `run_rkv_aime25_official_sampled8_qwen.sh` 当前结构，结合差异清单在 `diff_draft.md` 中列出需替换的模块/函数/参数（前向 patch、pruner/kv_budget、prompt+数据源、模型配置、采样接口）；逐项注明参考的 Qwen 基线（模拟/LazyEviction）以供后续落地。
- [ ] SpecKV on `run_speckv_aime24_official_sampled8.sh` 改模型为 Qwen  
  - 目标：明确从 Llama 切到 Qwen 的所有配置/依赖改动。  
  - 动作：列出模型路径、tokenizer、RoPE/position 适配、分布式/分片配置、采样接口调整。  
  - 验证：设计审阅，自检是否覆盖所有关键一致项。  
  - 执行计划：通读 Llama 版脚本，收集与模型/分片/采样/position+RoPE 相关的配置；在 `diff_draft.md` 中写出切换到 Qwen 所需的替换项与风险（路径、tokenizer、rope/position 处理、采样/并行配置），并关联到关键一致项以便后续执行。

## 三、核心逻辑对齐与实现
- [ ] 抽取/复核 SpecKV 核心逻辑（pruner 状态、kv_budget、position/RoPE、round/窗口、prefix 保留），必要时从早期 Qwen 版移植。
- [ ] 校准 prompt/数据入口与早期 Qwen 基线一致（模板、特殊 token、round 滚动）。
- [ ] 实施 SpecKV 版 `run_rkv_aime25_official_sampled8_qwen.sh`（套用设计清单，完成脚本落地）。
- [ ] 实施 Qwen 版 `run_speckv_aime24_official_sampled8.sh`（模型切换完、算法一致性复核）。
- [ ] 若发现与早期 Qwen/LazyEviction 关键逻辑不一致，立即停下并在 message.md 记录告警。

## 四、对比与文档
- [ ] 编写差异对比 MD：SpecKV-Qwen vs `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`，列出允许差异与实际差异，标警告项。
- [ ] 在 status_overview.md 更新阶段进展、风险与待办。

## 五、验证（轻量）
- [ ] 静态检查：`python -m compileall R-KV/weian_script`（或最小子集）。
- [ ] 规划 smoke 命令（单样本/小 qid，含 logprobs on/off），记录待跑方案与前置。
- [ ] 如运行 smoke：记录命令与结果；若无法运行，记录原因与缺口。

## 六、收尾
- [ ] 同步文档：更新 status_overview.md、project_todo.md 勾选完成项。
- [ ] 汇总已跑/未跑验证与结论，准备提交记录。
- [ ] 复核高风险点并记录结论/待办：Prompt 模板差异、pruner 绑定方式与位置处理等价性、RoPE/Stats 模型匹配、kv_budget/round_window 取值合理性、采样温度是否需统一。***
