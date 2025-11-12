## 背景 / 目标
- 把我们在 `weian_development/hf_offline_runner_sparse` 中实现的 SparseRound 注意力方法，迁移到 LazyEviction 的 Benchmark 管线下，做到“只替换稀疏策略，其他设置全部保持一致”，以便与现有 Window_LAZY/ROOP 等实现公平对比。
- 运行入口需要通过 `LazyEviction/weian_script/run_sparse_sharded_eval.sh` 与 `LazyEviction/weian_script/eval_sparse.sh`，并确保长跑任务遵循 `PD-L1_binder` 命名前缀。

## 目录 / 资源路径
- 仓库根目录：`/data/rbg/users/weian/project/rl/dc`
- LazyEviction 工程：`/data/rbg/users/weian/project/rl/dc/LazyEviction`
- SparseRound HF Runner：`/data/rbg/users/weian/project/rl/dc/weian_development/hf_offline_runner_sparse`
- 默认统计文件（DeepSeek-R1-Distill-Qwen-7B 专用）：`/data/rbg/users/weian/project/rl/dc/weian_development/hf_offline_runner_sparse/stats/distill_qwen7b_qid9001_trace00_stats.pt`
- 新增评测入口脚本：`/data/rbg/users/weian/project/rl/dc/LazyEviction/weian_script/run_sparse_sharded_eval.sh`、`/data/rbg/users/weian/project/rl/dc/LazyEviction/weian_script/eval_sparse.sh`
- 合并脚本：`/data/rbg/users/weian/project/rl/dc/weian_development/merge_lazy_eviction_shards.py`
- 数据集（示例）：`/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B`
- 输出根目录（示例）：`/data/rbg/users/weian/project/rl/dc/outputs/DeepSeek-R1-Distill-Qwen-7B`

## 现有成果
- 新增了 `run_sparse_sharded_eval.sh`、`eval_sparse.sh`、`lazy_eviction_sparse_runner.py`、`lazy_eviction_sparse_evaluation_sharded.py` 等文件，实现了：
  - 多 GPU 分片评测入口，与 LazyEviction 原有脚本对齐。
  - 通过 `weian_development/hf_offline_runner_sparse` 下的 `SparseRoundPruner` 进行 HuggingFace 推理时的 KV 剪枝。
  - `eval_sparse.sh` 可合并各分片输出。
- `weian_development/hf_offline_runner_sparse` 里已有完整的 SparseRound 推理/剪枝逻辑（`example_offline_hf_serialized.py`、`sparse_round_pruner.py`、`round_pruning_utils.py` 等），并提供了针对 DeepSeek-R1-0528-Qwen3-8B 的历史统计文件 `stats/qid0008_trace46_stats.pt`，以及如今为 LazyEviction/Distill-Qwen-7B 重新采集的 `stats/distill_qwen7b_qid9001_trace00_stats.pt`。

## 主要问题
1. **统计文件与模型不匹配**
   - 早期 `run_sparse_sharded_eval.sh` 默认的 `SPARSE_STATS_PATH` 指向 `/weian_development/hf_offline_runner_sparse/stats/qid0008_trace46_stats.pt`，该文件的 `model_path`、`head_dim`、layer/head 采样均针对 `DeepSeek-R1-0528-Qwen3-8B`。
   - 当评测模型切换到 `DeepSeek-R1-Distill-Qwen-7B` 时若继续沿用旧统计，`SparseRoundPruner` 得分会严重错位且没有元数据校验。
   - 已使用 `outputs/DeepSeek-R1-Distill-Qwen-7B/qk_fp16_traces/qid9001_trace00` 重新采集 q/k 并导出 `stats/distill_qwen7b_qid9001_trace00_stats.pt`，同时脚本默认值改为指向该文件；仍需在更换模型时重复采集流程。

2. **初始化路径与 LazyEviction 不一致**
   - LazyEviction 基线在 `run_lazy_method` 内会执行 `model.monkeypatch.replace_qwen/replace_llama`、设置 `TempCache.alpha`，并把 `max_kv_capacity` / `decoding_recent_size` / `obs_size` 写入每一层 `self_attn.config`。
   - 新的 `run_sparse_method` 直接加载 HF 原生模型，没有上述 monkeypatch 和 per-layer 配置，实际修改了注意力实现与 cache 行为，违反“非侵入式”要求。

3. **缺少防御性检查**
   - `SparseRoundPruner` 仅按层数裁剪 `sampled_heads`，不会验证 stats 中的 `model_path`、`head_dim`、rope_scaling 是否和当前模型一致。
   - `round_pruning_utils` 需在 `build_rotary`/`invert_rope` 等处考虑 transformers 版本差异，目前刚好能 fallback，但没有明确记录。

## 修复计划
1. **重新生成统计文件**
   - 参考 `weian_development/attention_qk_analysis/` 的工具或旧 pipeline，针对 `DeepSeek-R1-Distill-Qwen-7B` 采集 q/k traces，生成新的 `stats/*.pt`。
   - 将 `SPARSE_STATS_PATH` 默认值切换到新文件，并在文档中说明每换模型都必须重新统计。

2. **在 Sparse 路径复用 LazyEviction 初始化**
   - `run_sparse_method` 中引入 `replace_qwen/replace_llama`、`TempCache.alpha` 以及逐层 `self_attn.config` 写入逻辑，确保模型除剪枝策略外与 `run_lazy_method` 完全一致。
   - 确保 `attn_implementation`、`obs_size` 等参数由脚本传入，避免硬编码。

3. **增加一致性校验**
   - 在 `SparseRoundPruner.__init__` 内校验 stats 元数据：模型名、`head_dim`、`sequence_length`（或 rope_scaling）必须匹配，否则报错。
   - 可在 `round_pruning_utils.build_rotary` 中记录最终使用的 rope 配置，并在 pruner 中与 stats 的 `metadata` 比对。

4. **补充文档与使用说明**
   - 在 `docs/` 下撰写操作手册：如何采集 stats、如何配置脚本、如何合并输出。
   - 标注公平对比时需要统一的 `max_kv_capacity`、`decoding_recent_size`、`dataset split` 等关键超参数。

## 后续可选事项
- 为 `SparseRoundPruner` 添加单元测试，使用小型模型+伪造 stats 验证采样与剪枝流程。
- 结合已有 `merge_lazy_eviction_shards.py` 输出结果，制作对比表格（Window_LAZY vs SparseRound）以验证修复效果。
