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
- [x] SpecKV on `run_rkv_aime25_official_sampled8_qwen.sh` 骨架  
  - 目标：设计改动点列表（算法换 SpecKV，其余保持 R-KV 风格）。  
  - 动作：标出需要改的函数/参数/flag（SpecKV 前向 patch、pruner 配置、kv_budget、prompt/数据源）。  
  - 验证：设计审阅，自检是否覆盖所有关键一致项。  
  - 执行计划（当前）：1) 阅读 `run_rkv_aime25_official_sampled8_qwen.sh` 与对应 YAML，梳理现有数据/模型/kv_budget/采样配置。2) 对照 LazyEviction Qwen 版与 SpeckV 核心要求，列出 speckv 必需的新增字段（stats/round_window/offset 等）与需要切换的参数，并标记潜在不兼容点。3) 将设计结果写入 `diff_draft.md`，注明参考来源与风险。
- [x] SpecKV on `run_speckv_aime24_official_sampled8.sh` 改模型为 Qwen  
  - 目标：明确从 Llama 切到 Qwen 的所有配置/依赖改动。  
  - 动作：列出模型路径、tokenizer、RoPE/position 适配、分布式/分片配置、采样接口调整。  
  - 验证：设计审阅，自检是否覆盖所有关键一致项。  
  - 执行计划（当前）：1) 通读 Llama 版脚本+YAML，收集模型/rope_style/attn_impl/kv_budget/prompt/输出目录等配置。2) 对照 Qwen 基线，拟定需要替换/保留的项（模型+tokenizer+stats+round_window+kv_budget+dtype/attn_impl+prompt路径），标注高风险依赖。3) 将设计清单和风险写入 `diff_draft.md`。

## 三、核心逻辑对齐与实现
- [x] 抽取/复核 SpecKV 核心逻辑（pruner 状态、kv_budget、position/RoPE、round/窗口、prefix 保留），必要时从早期 Qwen 版移植。  
  - 执行计划（当前）：1) 检查 `weian_development/speckv/` 相关实现（generate patch、pruner、stats 校验）中与模型/position/RoPE 相关的逻辑，确认对 Qwen 兼容性与 LazyEviction 的一致点。2) 记录必要调整或确认无差异，输出到 `diff_draft.md`/备注，为脚本配置提供依据。
- [x] 校准 prompt/数据入口与早期 Qwen 基线一致（模板、特殊 token、round 滚动）。  
  - 执行计划（当前）：1) 复核 R-KV Qwen 脚本的 prompt 构造路径/标志（plain vs chat），确认 SpeckV 默认值。2) 对照早期 Qwen/LazyEviction 模板，标记当前选择及可能需重算 stats 的风险，写入 `diff_draft.md`。
- [x] 实施 SpecKV 版 `run_rkv_aime25_official_sampled8_qwen.sh`（套用设计清单，完成脚本落地）。
- [x] 实施 Qwen 版 `run_speckv_aime24_official_sampled8.sh`（模型切换完、算法一致性复核）。
- [x] 若发现与早期 Qwen/LazyEviction 关键逻辑不一致，立即停下并在 message.md 记录告警。  
  - 已记录：plain prompt vs LazyEviction chat 模板；kv_budget/window_size=2048/128 vs 1492/363；环境缺失 flash_attn2，SpeckV Qwen 改为 sdpa+fp16（与 LazyEviction 一致）并按此生成 stats。

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
