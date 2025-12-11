# RKV 重构概览（LazyEviction 内部版）

> 面向不熟悉上下文的读者，阐明目标、范围、安全约束与当前状态。规范沿用 `R-KV/docs/fullkv_ablation` 的结构，但所有实现与脚本均落在 `LazyEviction/` 下，避免影响现有实验。

## 1. 目标与交付
- 把 R-KV 的 AIME 流程（参考 `run_rkv_aime25_official_sampled8_qwen.sh`）以 LazyEviction 的代码规范/运行方式重构，形成独立实现与可复现实验入口。
- 公平对比目标 = `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`（当前为单样本贪心、前缀固定且不计入 budget），RKV 重构默认也应与之同口径，不要求 sample-8/聚合，除非额外复现官方脚本。
- 交付物：文档体系（本目录）、隔离的调度/runner/配置、以及一键运行脚本（放在 `LazyEviction/weian_script/`）。

## 2. 范围与安全
- **不改动现有 LazyEviction 实验默认值**；所有新行为须隔离或通过 cfg 开关，默认 = 旧逻辑。
- 代码与配置仅新增到 `LazyEviction/`（含 `weian_development/` 下的复用模块），不得依赖仓库外部文件。
- Prefill 阶段固定不压缩且不计入 KV budget；KV budget 计算方法必须与 LazyEviction 其他方法一致。
- 长跑实验使用 `PD-L1_binder` 进程前缀；遵循 AGENTS 约束与原有 .gitignore。

## 3. 基准与对照
- 主要对照脚本：`LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`（调度样式、参数对齐）。
- 参考源实现：`R-KV/weian_script/aime24_official_sampled8/run_rkv_aime24_official_sampled8_qwen.sh` 及其 YAML（例如 `sample8_rkv_aime25_official_qwen.yaml`）。

## 4. 当前状态
- 已完成：需求梳理与约束收敛（prefill 不计 budget、默认贪心单样本、隔离路径）、现有 R-KV 配置/runner 盘点（采样固定、plain prompt、max_length 计总长、kv_budget 未显式排除 prefill/padding），以及 LazyEviction 版 cfg/调度/runner 设计草案与 smoke/对比计划。
- 实现：`run_rkv_sharded_eval.sh` + `rkv_lazy_dispatch.py` + `rkv_lazy_runner.py` 已落地，提示词/数据改为与 SparsePrefillKeep 一致（`LazyEviction/datasets/aime/test.jsonl` + Qwen chat `<think>`），默认 `count_prefill_in_budget=false`、`kv_budget=1492`、单样本贪心。
- 运行与评测：已用 8 卡全量跑完 AIME（30 题），输出 `outputs/rkv_lazy_aime/merged/merged.jsonl`，评测使用 LazyEviction 同款提取/判分，得到 `accuracy=0.5333 (16/30)`，指标写入 `outputs/rkv_lazy_aime/merged/metrics.json`。
- 风险/疑点罗列见 `rebuild_plan.md`；任务分解见 `project_todo.md`。

## 5. 风险与关注
- KV budget 定义/计数方式若与 LazyEviction 不同，可能造成不公平；需要显式校准（prefill、padding、chunk、head 维度）。当前 R-KV monkeypatch 内部逻辑需确认。
- 数据/提示词/解码策略差异（采样 vs 贪心、多样本平均等）可能放大差距；R-KV 现实现忽略 chat 模板、强制采样，需要在重构时提供可配置对齐。
- 不允许影响现有流水线：新增 cfg 默认值需回放到旧行为；调度/日志路径需隔离，避免覆盖历史输出。
- 现状缺陷：RKV 路径依赖外部 `R-KV/` 代码、prefix 未被固定/免预算，预算口径与 SparsePrefillKeep 不一致，eval 步骤未实现；如需复现 sample-8 仅作为可选补充，默认与基线对齐应为单样本贪心。
