R-KV task log
==============

User requirements
- Recreate conda env `rkv` from scratch using official instructions under `R-KV/HuggingFace` (Python 3.10). Pin `transformers` to 4.48.1 (avoid newer) and install other deps.
- Hugging Face assets may already be cached (check backup `R-KV-backup-20250219` and existing HF cache under `/data/rbg/users/weian/.cache/huggingface`); reuse if possible.
- Run experiments via `R-KV/HuggingFace/scripts/run.sh` and `scripts/eval.sh`; README misses the `HuggingFace` prefix.
- Ensure `run_math.py` calls `mask_process_command("PD-L1_binder")` so visible process name is `PD-L1_binder` (requirement for long runs).
- Tests pass only if accuracy > 0; wrong deps (e.g., wrong transformers version) may silently give acc = 0.

Plan & checklist
- [x] Remove any prior `rkv` conda env and create a clean one.
- [x] Install `R-KV/HuggingFace` requirements (transformers==4.48.1, numpy/tqdm/matplotlib) and evaluation deps; install the local package in editable mode if needed.
- [x] Ensure `run_math.py` includes `mask_process_command("PD-L1_binder")`.
- [x] Run `scripts/run.sh` and `scripts/eval.sh` with process name set to `PD-L1_binder`; confirm accuracy > 0.
- [x] Log versions used, caches reused, and experiment outputs/paths.

Progress notes
- [x] Task log created.
- [x] Built fresh `rkv` env (Python 3.10) with torch==2.3.1+cu121, transformers==4.48.1, accelerate==0.33.0, flash-attn==2.5.8 (CUDA_HOME=/usr/local/cuda-12.4, PIP_CACHE_DIR=/data/rbg/users/weian/.cache/pip), numpy==2.2.2, tqdm==4.67.1, matplotlib==3.10.0; installed `evaluation/latex2sympy2`, `evaluation/requirements.txt`, and `rkv` editable.
- [x] Added `mask_process_command("PD-L1_binder")` to `run_math.py` and guarded optional qwen3 imports in `rkv/monkeypatch.py` and `rkv/modeling.py` for transformers 4.48.1 compatibility.
- [x] Ran `scripts/run.sh` (DeepSeek-R1-Distill-Llama-8B, method=rkv, kv_budget=128) -> `outputs/output.jsonl` (30 samples) with PD-L1_binder process name; eval via `scripts/eval.sh` produced `evaluation/aime24/default-aime24_math_eval_cot_metrics.json` (acc=3.3%, num_samples=30, empty_samples=1).
- [x] Additional run with kv_budget=2048: `run_math.py --kv_budget 2048 --save_path ./outputs/output_kv2048.jsonl` (PD-L1_binder). Eval with `evaluation/eval_math.py --exp_name evaluation_kv2048 --base_dir ./outputs_kv2048 --dataset aime24` yielded `evaluation/evaluation_kv2048/aime24/default-aime24_math_eval_cot_metrics.json` (acc=43.3%, num_samples=30, empty_samples=0).

Recent maintenance (2025-03-xx)
- Updated R1KV modeling: reset `self.length` per sample; fixes compression phase mismatch (single vs sharded outputs now对齐).
- Refreshed AIME24 baselines: `run_rkv_aime24_single.sh` (wrapper `run_rkv_aime24.sh`) 和 `run_rkv_aime24_sharded.sh` 统一为 sdpa + fp16 + fp32_topk + reset，输出归一到 `R-KV/outputs/rkv_aime24_single_sdpa_fp16_reset` / `R-KV/outputs/rkv_aime24_sharded_sdpa_fp16_reset`，单卡自动评测写入 `R-KV/HuggingFace/outputs/output_sdpa_fp16_reset_eval`，分片评测在对应 `eval/` 子目录；`configs/rkv_aime24_sharded.yaml` 已更新。
- Tidied `weian_script`: 历史/测试型 R-KV 启动脚本与 ablations 归档到 `weian_script/archive/`（配置到 `weian_script/configs/archive/`），README 标注现行入口。
- 其他 sharded 方法（fullkv/snapkv/streamingllm/h2o）对齐 sdpa + fp16 + reset（适用时附 fp32_topk），输出/日志目录带 `_sdpa_fp16_reset` 后缀；新增 fullkv flash-attn 变体 `run_fullkv_aime24_sharded_flash.sh`。
