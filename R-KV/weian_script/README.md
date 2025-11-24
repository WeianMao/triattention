# weian_script helpers

Convenience launchers for HuggingFace-based math experiments.

## 默认推荐（official）
官方风格指：flash_attention_2 + bfloat16，默认不启用 reset_cache_each_batch / fp32_topk。以脚本名含 `official` 为准，目前常用：
- `run_fullkv_aime24_official_sampled8.sh` → `configs/sample8_fullkv_aime24_official.yaml`
- `run_fullkv_aime24_official_sampled64.sh` → `configs/sample64_fullkv_aime24_official.yaml`
- `run_snapkv_aime24_official_sampled8.sh` → `configs/sample8_snapkv_aime24_official.yaml`
- `run_streamingllm_aime24_official_sampled8.sh` → `configs/sample8_streamingllm_aime24_official.yaml`
- `run_h2o_aime24_official_sampled8.sh` → `configs/sample8_h2o_aime24_official.yaml`
- `run_rkv_aime24_official_sampled8.sh` → `configs/sample8_rkv_aime24_official.yaml`

日志文件名沿用 `rkv_aime24_shardXX.log` 前缀是历史命名，与实际 method 无关。

## 旧版/非 official 路径（保留兼容）
- R-KV baseline（sdpa + fp16 + fp32_topk + reset）：`run_rkv_aime24_single.sh` / `run_rkv_aime24.sh`（单卡），`run_rkv_aime24_sharded.sh`（多卡）。
- 多次采样 sdpa+fp16+fp32_topk+reset：`run_rkv_aime24_sampled.sh`、`run_fullkv_aime24_sampled_fp16_sdpa_reset.sh`、`run_snapkv_aime24_sampled.sh`、`run_streamingllm_aime24_sampled.sh`、`run_h2o_aime24_sampled.sh`（`SAMPLES=8` 可切 8 次）。
- 其他 sdpa+fp16 变体：`run_fullkv_aime24_sharded.sh`、`run_snapkv_aime24_sharded.sh`、`run_streamingllm_aime24_sharded.sh`、`run_h2o_aime24_sharded.sh`。

Legacy/test 脚本与历史 ablation 位于 `archive/`。推荐优先使用含 `official` 的脚本作为默认设置。
