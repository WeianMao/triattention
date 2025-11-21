# R-KV 复现修复 TODO（执行 checklist）

- 开发约束：全部改动/新增文件仅放在 `R-KV/` 目录内，遵守规范以支持后续论文复现，避免低级错误导致结果不可靠。

## A. 生成与采样对齐
- [x] `run_math.py`：新增/接线 `num_samples`、`temperature`、`top_p` 参数（默认 64/0.6/0.95，AIME 强制 max_length=32768）；循环采样写 `draw_idx` 字段，保持单次 `num_return_sequences=1`。
- [x] `weian_development/rkv_sharded_eval.py`：同样支持 `num_samples`/温度/top_p，写 `draw_idx`，确保 `reset_cache_each_batch` 逻辑与单卡一致。
- [x] 采样设置：默认 64（论文对齐），保留 `num_samples=8` 快捷参数/配置用于快速实验。
- [ ] 测试：小规模采样烟测（num_samples=2/3）确认字段和输出格式正确。（尚未跑真实模型，仅做 dry-run/合成数据验证）

## B. 安全约束
- [x] 强制 `eval_batch_size=1`：在单卡和 sharded 路径上做前置检查/报错提示（不尝试修复 batch>1）。
- [x] 保留已有 `self.length` 重置行为，不再改动。
- [x] 测试：batch>1 触发预期报错（命令行前置报错）。

## C. 多样本评测
- [x] 新增 `evaluation/eval_math_multi.py`（或等效）：按 `sample_idx` 聚合多次输出，计算 pass@1（均值），预留 pass@k；兼容仅 1 次输出的 jsonl。
- [x] 评测输出：在新目录（例如 `.../eval_multi/`）写 metrics JSON；若无多样本字段，行为与原 eval 保持。
- [x] 测试：对合成多样本 jsonl 跑 eval_multi，指标输出正常（conda env rkv）；单样本未额外跑。

## D. 并行与负载均衡
- [x] Sharded 策略改为“每卡跑全量题目，采样次数均分”。示例：30 题、64 次采样、8 卡 → 每卡跑 8 次全集题。
- [x] 每卡随机种子区分：基于全局 seed + shard_id 偏移，确保不同卡的采样不同。
- [x] 仍保留自动 merge（`merge_rkv_shards.py`）和自动评测流程。
- [ ] 测试：多卡小配置烟测（如 2 卡，每卡 1-2 次采样）验证 seeds 差异、merge 顺序、评测链路。（当前仅 dry-run + 合成数据 merge 验证）

## E. 一键脚本与配置
- [x] 为 rkv/fullkv/snapkv/streamingllm/h2o 各自添加 sharded 脚本 + YAML（sdpa+fp16+fp32_topk+reset），新目录/文件名简洁，不复用旧输出路径。
- [x] 提供两套配置：`num_samples=64`（论文对齐）和 `num_samples=8`（快速实验），各自独立输出/日志目录（保留旧结果不覆写）。
- [x] 环境假设：conda env `rkv`，8 卡，`HF_HOME`/`PIP_CACHE_DIR` 继承现有约定。
- [ ] 测试：至少对一个方法跑 8 样本快速链路（含自动评测），确认输出/日志路径和命名。（尚未跑实机，仅 dry-run）
- [x] 支持在 YAML 中配置基础 seed（runner_args.seed），便于重复试验。

## F. 文档与记录（完成后检查路径/参数在文档中可查）
- [x] 更新 `docs/CHANGELOG_weian.md`（或补充说明文件）：记录强制 batch=1、采样参数、种子策略、负载拆分方式、新脚本路径与输出目录。
- [ ] 如执行中遇到特殊问题/限制，写入此文件末尾或新增 “issues” 小节。

## G. 阻塞/问题记录
- 若执行中遇到阻塞，请在此补充：
  - [ ] ____________________________________________
