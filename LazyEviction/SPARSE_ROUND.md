# SparseRound 集成说明

本节描述如何在 LazyEviction benchmark 框架下运行 `SparseRound`（基于 `hf_offline_runner_sparse` 的稀疏 KV 剪枝算法），并保持与原 Window_LAZY 设置一致的评测流程。

## 目录结构

- `weian_development/lazy_eviction_sparse_evaluation_sharded.py`  
  拷贝自原 `lazy_eviction_evaluation_sharded.py`，在检测到 `--method sparse_round` 时切换到 `SparseRoundPruner` 并沿用 HF offline 生成循环。
- `weian_development/lazy_eviction_sparse_runner.py`  
  与原 runner 相同，仅指向新的评测脚本，继续使用 `PD-L1_binder` 的进程掩码。
- `LazyEviction/eval_qwen_aime_sparse_sharded.sh`  
  Window_LAZY 入口脚本的镜像版，只把 `method` 固定为 `sparse_round` 并透传稀疏相关参数。
- `LazyEviction/weian_script/run_sparse_sharded_eval.sh`  
  启动多个 shard 的 wrapper，内部自动挑选空闲 GPU。
- `LazyEviction/weian_script/eval_sparse.sh`  
  复用 `merge_lazy_eviction_shards.py` 将 `sparse_round` 的 shard 输出合并。

## 运行步骤

1. **准备环境**  
   ```bash
   conda activate lazy_evict
   cd /data/rbg/users/weian/project/rl/dc
   ```

2. **启动稀疏评测**  
   ```bash
   OUTPUT_ROOT=/path/to/LazyEviction/outputs/sparse_round_runX \
   LazyEviction/weian_script/run_sparse_sharded_eval.sh
   ```

   - 默认会读取 `nvidia-smi`，挑出显存占用 ≤200MiB 的 GPU，并自动设置 `NUM_SHARDS` 与 `GPUS`（可通过 `GPU_MEMORY_THRESHOLD` 或手动导出 `GPUS/NUM_SHARDS` 覆盖）。
   - 运行日志写入 `logs/lazy_eviction_sparse_round/sparse_shard*.log`，输出位于 `LazyEviction/outputs/.../sparse_round/shard_*`。

3. **合并分片**  
   待所有 shard 完成后执行：
   ```bash
   LazyEviction/weian_script/eval_sparse.sh
   ```
   若输出目录不在默认路径，可设置 `METHOD_OUTPUT_DIR=/custom/path`。

## 关键参数

- `max_kv_capacity` / `decoding_recent_size`：与 Window_LAZY 保持一致，确保比较公平。
- `SPARSE_STATS_PATH`：默认指向 `weian_development/hf_offline_runner_sparse/stats/distill_qwen7b_qid9001_trace00_stats.pt`（与当前 `DeepSeek-R1-Distill-Qwen-7B` 对齐），如需在其他模型上复用请重新采集并通过环境变量覆盖。
- `OUTPUT_ROOT`：用于隔离不同实验的结果目录，务必设置在 `LazyEviction/outputs/` 下以保持仓库整洁。

## 注意事项

- 脚本依赖 `hf_offline_runner_sparse` 中的统计文件，若与当前模型层数不匹配会自动过滤越界 heads。
- 部分外部依赖（`dynasor`、`deepconf`、`msgpack` 等）提供了回退实现，若环境中已有正式依赖则会优先使用原版。
- 若需要手动终止任务，可通过 `pkill -f lazy_eviction_sparse_runner.py` 或 `pkill -f PD-L1_binder` 快速清理所有 shard。

以上步骤即可在 LazyEviction 基准上复用 SparseRound 算法，并保持除稀疏策略外的所有评测设置不变。
