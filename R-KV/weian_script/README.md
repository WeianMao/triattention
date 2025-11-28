# weian_script helpers

Convenience launchers for HuggingFace-based math experiments.

## 目录分组
- `aime24_official_sampled8/`：官方风格（flash_attn2 + bfloat16，无 reset/fp32_topk）8 次采样，包含 FullKV / H2O / R-KV / SnapKV / StreamingLLM。
- `aime24_official_sampled64/`：官方风格 FullKV 64 次采样。
- `aime24_sampled/`：sdpa + fp16 + fp32_topk + reset 的多次采样版本（默认 8 次），覆盖 FullKV / H2O / R-KV / SnapKV / StreamingLLM。
- `aime24_sharded/`：sdpa + fp16 的分片版本（同上多种方法）。
- `aime24_baseline/`：R-KV 单卡基线（sdpa + fp16 + fp32_topk + reset）及便捷封装。
- `configs/`：对应 YAML 配置；`archive/`：历史/废弃脚本。

从 `R-KV/weian_script` 目录调用，直接 `bash aime24_official_sampled8/run_*.sh` 即可。

## 默认推荐（official）
官方风格指：flash_attention_2 + bfloat16，默认不启用 reset_cache_each_batch / fp32_topk。
- `aime24_official_sampled8/run_fullkv_aime24_official_sampled8.sh` → `configs/sample8_fullkv_aime24_official.yaml`
- `aime24_official_sampled64/run_fullkv_aime24_official_sampled64.sh` → `configs/sample64_fullkv_aime24_official.yaml`
- `aime24_official_sampled8/run_snapkv_aime24_official_sampled8.sh` → `configs/sample8_snapkv_aime24_official.yaml`
- `aime24_official_sampled8/run_streamingllm_aime24_official_sampled8.sh` → `configs/sample8_streamingllm_aime24_official.yaml`
- `aime24_official_sampled8/run_h2o_aime24_official_sampled8.sh` → `configs/sample8_h2o_aime24_official.yaml`
- `aime24_official_sampled8/run_rkv_aime24_official_sampled8.sh` → `configs/sample8_rkv_aime24_official.yaml`
- `aime24_official_sampled8/run_sparseprefillkeep_aime24_official_sampled8.sh` → `configs/sample8_sparseprefillkeep_aime24_official.yaml`
- `wei/run_sparseprefillkeep_aime24_official64.sh` → `configs/sample64_sparseprefillkeep_aime24_official.yaml`

日志文件名沿用 `rkv_aime24_shardXX.log` 前缀是历史命名，与实际 method 无关。`--skip-existing`（默认开启）可用于断点续跑。

## SparseRound（prefill-keep / SpecKV）使用方法
1. 先生成稀疏统计（保持与对照一致的 chat template + prompt），示例：
   ```bash
   python R-KV/weian_development/rkv_sparse_round_calibrate.py \
     --trace-root R-KV/outputs/sample8_fullkv_aime24_official \
     --output-path R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt \
     --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B
   ```
   - 默认取 3 条 FullKV trace 求均值；需要更多样本可调 `--num-traces`。
   - 头采样文件默认 `R-KV/weian_development/speckv/stats/deepseek_r1_llama8b_heads.json`，不存在会自动生成。
   - 统计文件会记录 prompt 模板 / system prompt / attn 实现 / dtype / kv_budget，并在 SpeckV 运行时强校验；切换 chat 模板、system prompt、kv_budget、dtype 或 attn 实现后请重新运行校准生成新的 `outputs/.../stats/*.pt`。
2. 运行 8 次或 64 次抽样的稀疏版本（与 R-KV official 设置对齐：flash_attn2 + bf16，kv_budget=2048，prompt 不压缩）：
   ```bash
   # 8 抽样
   bash R-KV/weian_script/aime24_official_sampled8/run_sparseprefillkeep_aime24_official_sampled8.sh
   # 64 抽样
   bash R-KV/weian_script/wei/run_sparseprefillkeep_aime24_official64.sh
   ```
   - 可通过 `--gpus`、`--num-shards`、`--skip-merge` 等参数覆盖默认。
   - 若更换模型/数据集，请同步更新 YAML 中的 `model_path`、`dataset_path`、`sparse_stats_path`。

提示：稀疏脚本与 R-KV baselines 共用调度器/评测流程，保持 kv_budget 与前缀保留策略一致即可公平对齐（prompt 全保留，解码阶段再裁剪）。

## 旧版/非 official 路径（保留兼容）
- R-KV baseline（sdpa + fp16 + fp32_topk + reset）：`aime24_baseline/run_rkv_aime24_single.sh`、`aime24_baseline/run_rkv_aime24.sh`（单卡）、`aime24_sharded/run_rkv_aime24_sharded.sh`（多卡）。
- 多次采样 sdpa + fp16 + fp32_topk + reset：`aime24_sampled/run_rkv_aime24_sampled.sh`、`aime24_sampled/run_fullkv_aime24_sampled_fp16_sdpa_reset.sh`、`aime24_sampled/run_snapkv_aime24_sampled.sh`、`aime24_sampled/run_streamingllm_aime24_sampled.sh`、`aime24_sampled/run_h2o_aime24_sampled.sh`（`SAMPLES=8` 可切 8 次）。
- 其他 sdpa + fp16 变体（分片）：`aime24_sharded/run_fullkv_aime24_sharded.sh`、`aime24_sharded/run_snapkv_aime24_sharded.sh`、`aime24_sharded/run_streamingllm_aime24_sharded.sh`、`aime24_sharded/run_h2o_aime24_sharded.sh`。

Legacy/test 脚本与历史 ablation 仍放在 `archive/`，默认优先使用含 `official` 的脚本作为默认设置。
