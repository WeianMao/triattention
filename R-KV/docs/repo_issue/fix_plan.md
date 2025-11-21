# R-KV 复现修复方案草稿（请确认后再实施）

## 已知问题（导致与论文结果不一致）
- **解码仍是贪心 + 单样本**：`run_math.py` 与分片版 `weian_development/rkv_sharded_eval.py` 固定 `do_sample=False`，只生成 1 条输出；评估脚本 `evaluation/eval_math.py` 也假设单样本，只取最后一条 `pred`。这与论文要求的温度 0.6、top_p 0.95、每题 64 次采样取 pass@1 的设置不符，精度偏低。
- **batchify Bug**：`eval_batch_size>1` 时压缩调度跨样本共享 `self.length`，且 padding token 参与压缩评分，导致性能显著下降（作者在 GitHub issue 中也提示仅支持 batch=1）。目前脚本未强制或提示；本轮仅强制/断言 batch=1，不处理批量支持。
- **单卡 vs 分片曾错位**：历史上 `self.length` 未按样本重置导致压缩点偏移；已在 `rkv/modeling.py` 修复，后续改动需保留该行为。
- **端到端链条缺口**：现有分片脚本需要手动跑分片→merge→eval，多步且只评单样本，缺少“一键多采样+评估”的入口。

## 拟议修改方案
1) **采样与多次生成对齐论文**  
   - 为 `run_math.py`、`weian_development/rkv_sharded_eval.py` 增加 `num_samples`、`temperature`、`top_p`（默认 64/0.6/0.95，自动重写 max_length 为 AIME 的 32768），按题循环生成多次，记录 `draw_idx` 写入同一 jsonl。保留另一个快捷 setting：`num_samples=8` 便于快速实验。
2) **batch 行为**  
   - 强制/断言 `eval_batch_size=1`，并在日志/README 提示；本轮不支持批量路径。
3) **pass@k 评估**  
   - 新增多样本评估器（例如 `evaluation/eval_math_multi.py`），按 `sample_idx` 汇总每题的多次输出，计算 pass@1（均值）/可扩展 pass@k；保持对单样本文件的兼容。
4) **并行与负载均衡方案（新的 sharded 策略）**  
   - 改为“每卡跑全量题目，多次采样切分到卡”模式：有 N 卡、每题 M 次采样，则每卡跑 `M/N` 次全集 30 题，避免按题目分片造成长题偏载。确保每卡使用不同随机种子（可基于全局 seed + shard_id）。
5) **一键多采样脚本/配置（新命名）**  
   - 为 rkv/fullkv/snapkv/streamingllm/h2o 提供 sharded 一键脚本 + YAML（新目录名、简洁命名，不复用旧输出路径），自动：多卡并行 → merge → 多样本评测。提供两套常用配置：`num_samples=64`（论文对齐）与 `num_samples=8`（快速实验），输出目录各自独立。
6) **文档/记录**  
   - 在 `docs/CHANGELOG_weian.md` 或新增笔记中记录以上变更、默认 batch=1 约束、采样设定、种子策略、负载拆分说明。

## 待确认点
- **已确认**：强制 batch=1；不修 kv_budget 校验；保持 `self.length` 重置修复；保持自动评测链路。  
- **设定**：温度 0.6、top_p 0.95，对齐官方；AIME max_length=32768。  
- **采样数**：提供 64（论文对齐）与 8（快速实验）两套。  
- **并行**：多卡跑全集题目，采样次数均分到卡，种子随 shard_id 区分。  
- **方法范围**：rkv/fullkv/snapkv/streamingllm/h2o 均需“一键多采样+评估”版本。  
- **环境**：8 卡、conda env `rkv`，不变。  
- **输出目录**：使用全新、简洁的目录名；保留历史结果。
