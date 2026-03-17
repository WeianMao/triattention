# SpecKV 快速实验 resume 修复说明（含 run 定义）

## 背景
- 原始问题：`--skip-existing` 只按 shard 文件行数判定（`lines >= local_count`），旧目录残留不完整输出时被误判为完成，导致 merge/eval 数据残缺、acc=0。
- 术语澄清：run = 全量题目的一次抽样；每题抽样数 = run 总数 = `num_samples`。例如 30 题、32 抽样、8 卡 → `num_shards=8`，每个 shard 负责 4 个 run（run_id 0-3, 4-7, …, 28-31）。

## 本次修复（已完成）
- `rkv_sharded_dispatch.py`
  - skip 逻辑按 run 粒度：每个 shard 拿到自己的 run_id 区间（与原 local_samples/start_draw 相同），只有缺 run 时才启动进程。
  - 期望行数按题目数检查：`expected_records = dataset_rows (受 max_examples 限制)`，并要求 meta 标记完成。
- `rkv_sharded_eval.py`
  - 每个 run 单独写文件：`output_dir/shardXX/runYYY.jsonl.tmp` → 写完 rename 为 `runYYY.jsonl`，同时写 `runYYY.meta.json`（status=complete, records, run_id, shard_id）。缺 meta 或行数不足会被视为未完成并重跑。
  - run_id 分配：每 shard 处理自己的 run 段（start_draw/local_samples），与 GPU 数一致的切分；run_id = draw_idx。
  - 随机种子：`seed = base_seed + run_id * RUN_SEED_STRIDE (1_000_000) + sample_idx`，与 GPU/shard 数无关，保证改卡数/重跑也能复现。
- `merge_rkv_shards.py`
  - 支持新目录结构 `shard*/run*.jsonl`，仅 merge 带 meta 且 status=complete 的文件；保留可选 pattern 覆盖。若找不到 run 结构则回退到旧的 `*.jsonl`（无 meta）。
- 其他：`python -m compileall R-KV/weian_development` 已通过。

## 运行与目录结构
- 输出目录默认：`<method_output_dir>/shards/`。
- 每个 shard 独立子目录：`shard00/`, `shard01/`, …；每个 run：`run000.jsonl` + `run000.meta.json`（写盘时有 `.tmp`）。
- `--skip-existing` 会逐 run 检查：文件存在、行数达到题目数、meta status=complete 才算完成；否则重跑该 run。
- seed 不再依赖 `shard_id`，调整 GPU 数或重启不会改变同一 run + sample_idx 的随机性。

### 示例命令
```bash
PYTHONPATH=/data/rbg/users/weian/project/rl/dc \
conda run -n rkv python R-KV/weian_development/rkv_sharded_dispatch.py \
  --config R-KV/weian_script/configs/sample8_speckv_aime24_quick.yaml \
  --gpus 3,4,5,6 --num-shards 4 --skip-existing \
  --method-output-dir R-KV/outputs/sample8_speckv_aime24_quick_clean \
  --log-dir R-KV/logs/sample8_speckv_aime24_quick_clean \
  --output-dir R-KV/outputs/sample8_speckv_aime24_quick_clean/shards
```

## resume 伪代码速记
1) 调度器：读取 config+数据集行数 → 计算每 shard run_id 范围 → 若 run 缺失则启动该 shard，传参 `--shard_id`/`--num_shards` 等。
2) worker：加载数据集（可截断 `max_examples`）→ 对分配的 run_id 循环：
   - 如 `runYYY.jsonl`+meta 完成则跳过。
   - 写 `runYYY.jsonl.tmp`，对每题设置 `seed = base_seed + run_id * STRIDE + sample_idx` 生成一条记录，含 `draw_idx=run_id`。
   - 写 `runYYY.meta.json`（status=complete, records=题目数）后 rename tmp 为正式文件。
3) merge：读取 `shard*/run*.jsonl` 且 meta=complete，按 `(sample_idx, draw_idx)` 排序输出到 `merged/merged.jsonl`。

## 潜在问题/注意
- 修改 `num_shards` / `num_samples` / `max_examples` 后 resume 时期望行数会变化，可能触发全量重跑；保持配置稳定。
- 如果输出目录里有老的（无 meta）run 文件会被判为未完成并被覆盖；如需保留旧数据请备份到别处。
- meta 写在数据写完之后，理论上掉电可能留下 `.jsonl.tmp` 或 meta 与数据不匹配；重跑会清理并覆盖。
- merge 回退到旧模式只在找不到 run 结构时启用；混合结构可能导致旧文件被忽略，建议清理目录或使用 `--pattern` 指定。

## Handoff（给后续接手人）
- 修改文件：
  - `R-KV/weian_development/rkv_sharded_dispatch.py`：run 级 skip 判定、期望行数=题目数、路径改为 `shardXX/runYYY`。
  - `R-KV/weian_development/rkv_sharded_eval.py`：run 级写盘 + meta、run_id 种子公式、按 run 文件 resume。
  - `R-KV/weian_development/merge_rkv_shards.py`：支持新 run 目录、只 merge 完整 run。
  - 文档：`R-KV/docs/resume_fix_plan.md`（本文件）。
- 运行新实验时优先使用全新输出目录；若要复用旧目录，请先手工清理残缺文件或备份。
