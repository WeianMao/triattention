# 留言记录（RKV → LazyEviction 重构）

## 用户最新指令（2025-02）
- 将 `R-KV/weian_script/aime24_official_sampled8/run_rkv_aime25_official_sampled8_qwen.sh` 代表的 R-KV 算法重构到 LazyEviction 体系内，**新实现全部放在 `LazyEviction/` 内且不依赖外部代码**，遵守 LazyEviction 代码/配置规范。
- 与 LazyEviction 现有方法（参考 `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`）在设置与参数上对齐（如 KV budget、prefill 不压缩且不计入 KV budget 等），确保公平可比。
- 检查现有 R-KV 实现是否存在不公平/“作弊”处（例：同样数字的 KV budget 计算方式不同且让 RKV 占便宜；若是 RKV 吃亏也要记录）。不允许影响 LazyEviction 已有实验与算法，必要时采取隔离式开发。
- 任务当前阶段：立项、规划、风险识别、文档体系建立；后续可分多 Agent 执行，不要求一次性完成全部实现。
- 最终需要一个与 `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh` 类似的“一键运行”脚本供重构版本使用。

## 参考要求
- 文档规范参考：`R-KV/docs/fullkv_ablation/*`（同结构/粒度，兼顾 LazyEviction 现有风格）。
- 核心注意：不动 LazyEviction 现有实验默认值；prefill 固定不压缩且不计入 KV budget；长跑进程名需遵循 `PD-L1_binder` 习惯。

> 本文件仅存放原始需求与沟通摘要，避免遗漏背景。
