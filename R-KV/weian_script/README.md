# weian_script helpers

Convenience launchers for HuggingFace-based math experiments.

## R-KV AIME24 baseline (sdpa + fp16 + fp32_topk + reset)
- `run_rkv_aime24_single.sh` (or `run_rkv_aime24.sh` wrapper): single-GPU run, output to `R-KV/outputs/rkv_aime24_single_sdpa_fp16_reset/output.jsonl`, auto-eval to `R-KV/HuggingFace/outputs/output_sdpa_fp16_reset_eval/`.
- `run_rkv_aime24_sharded.sh`: multi-GPU dispatcher using `configs/rkv_aime24_sharded.yaml` (8 shards by default), output to `R-KV/outputs/rkv_aime24_sharded_sdpa_fp16_reset/`, eval to its `eval` subdir.

## Other methods (aligned settings)
- `run_fullkv_aime24_sharded.sh` → `configs/fullkv_aime24_sharded.yaml` (sdpa + fp16 + reset; no compression).
- `run_snapkv_aime24_sharded.sh` → `configs/snapkv_aime24_sharded.yaml` (sdpa + fp16 + fp32_topk + reset).
- `run_streamingllm_aime24_sharded.sh` → `configs/streamingllm_aime24_sharded.yaml` (sdpa + fp16 + fp32_topk + reset).
- `run_h2o_aime24_sharded.sh` → `configs/h2o_aime24_sharded.yaml` (sdpa + fp16 + fp32_topk + reset).

Other compression variants (fullkv/h2o/snapkv/streamingllm) keep their original scripts in this folder. Legacy/test R-KV launchers and ablations have been moved into `archive/` for reference.
