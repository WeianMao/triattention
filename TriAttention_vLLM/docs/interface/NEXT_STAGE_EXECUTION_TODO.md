# Next Stage Execution TODO

Updated: 2026-03-01
Status: Active
Owner: Codex + Weian

## 1) Goal (What must be achieved)

1. Add support for Qwen3 and Qwen3-Coder in `TriAttention_vLLM` using the same main code path.
2. Keep Qwen2.5 behavior stable (no regression in old baseline behavior).
3. Add prefill auto-chunk + per-chunk compression (pre-chunk path only) for demo stability (avoid OOM in long prefill scenarios).

## 2) User Intent / Design Decisions to Remember

1. Discussion style: concise, chunk-by-chunk; avoid overloading with details.
2. For prefill optimization:
   - `pre-chunk` is selected.
   - Do **not** pursue `post-chunk` fallback path.
   - Main objective is demo success (avoid OOM), not perfection.
3. Prefill optimization should be auto-triggered by threshold/chunk size, both configurable.
4. Suggested default for first implementation: threshold = 2048, chunk size = 2048.
5. Algorithm internals can be pragmatic as long as behavior objective is met.
6. Only escalate to user for important decisions / blocked paths / no-good-option cases.

## 3) Non-Goals / Low-Priority

1. Not required to make strict-equality peak-memory guarantees.
2. Not required to implement both pre-chunk and post-chunk.
3. Not required to optimize for paper-level perfect quality in this stage.

## 4) Environment Notes

1. Hugging Face cache root requested by user: `/data/rbg/users/weian/env/huggingface`.
2. Current env check:
   - default shell env may vary; execution commands in this stage explicitly used
     `HF_HOME=/data/rbg/users/weian/env/huggingface`.
3. Model download actions must follow Hugging Face default cache structure under the user-specified root.

## 5) Execution Roadmap (Open Road)

1. Block A: Qwen3 / Qwen3-Coder compatibility + Qwen2.5 no-regression protection.
2. Block B: Prefill auto-chunk + per-chunk compression (pre-chunk only), demo-oriented.
3. Block C: End-to-end verification for:
   - Old baseline compatibility
   - New model support
   - Long prefill no-OOM demonstration path

## 6) Active TODO Checklist

1. [x] A1 - Audit current runtime/model assumptions that may be Qwen2.5-specific.
2. [x] A2 - Implement Qwen3/Qwen3-Coder compatibility changes in runtime path.
3. [x] A3 - Add/adjust tests for compatibility and no-regression (Qwen2.5).
4. [x] A4 - Build/align a Qwen3 anchor config against target HF script behavior.
5. [x] B1 - Locate integration point for prefill auto-chunk execution.
6. [x] B2 - Implement configurable threshold/chunk-size pre-chunk policy.
7. [x] B3 - Ensure per-chunk compression is executed and observable.
8. [x] B4 - Add tests for long prefill chunk behavior and safety checks.
9. [x] C1 - Run smoke/regression checks and summarize outcomes.
10. [x] C2 - Prepare concise demo run recipe (fullkv vs triattention contrast).
11. [x] D1 - Run full-scale TriAttention experiment (all available GPUs) for HF-script alignment check.
12. [ ] D2 - Produce side-by-side metric comparison against
    `R-KV/weian_script/aime_sampled8_qwen3/speckv/aime24/run_speckv_aime24_qwen_norm_aligned.sh`.

## 8) Progress Log

1. 2026-02-27:
   - Added this execution TODO file with user-intent memory + roadmap.
   - Verified HF cache env is under user-requested root:
     `HF_HOME=/data/rbg/users/weian/env/huggingface`.
   - Completed A1 audit snapshot:
     primary compatibility risk is rotary frequency source for Qwen3-family.
   - Implemented compatibility hardening:
     `triattention.utils` now preserves/derives `metadata.inv_freq` for R-KV stats;
     `triattention.compressor` now prefers metadata `inv_freq` over `rope_theta`
     fallback.
   - Added tests:
     `tests_runtime/test_compressor_rope_init.py`,
     and updated `tests_runtime/test_utils_rkv_stats.py`.
   - Validation:
     `12 passed` on targeted tests.
2. 2026-02-27:
   - Added Qwen3 anchor dispatch config:
     `evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor_qwen3.yaml`.
   - Implemented pre-chunk controls in runtime runner:
     - `--prefill-auto-chunk`
     - `--prefill-chunk-threshold`
     - `--prefill-chunk-size`
     - current integration enforces threshold == chunk_size (single vLLM knob).
   - Added pre-chunk demo-oriented Qwen3 config:
     `evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor_qwen3_prefill_chunk.yaml`.
   - Tests / checks:
     - `tests_runtime/test_runtime_eval_runner.py` -> `11 passed`
     - `tests_runtime/test_utils_rkv_stats.py` + `tests_runtime/test_compressor_rope_init.py` -> `4 passed`
     - `tests_runtime/test_config.py` -> `8 passed`
     - Runner CLI help confirms pre-chunk args are exposed.
     - compile checks passed for modified Python files.
3. 2026-02-27:
   - Started downloading `Qwen/Qwen3-Coder-30B-A3B-Instruct` with:
     `HF_HOME=/data/rbg/users/weian/env/huggingface` (Hugging Face default cache layout).
   - Current cache path:
     `/data/rbg/users/weian/env/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct`
   - Download completed (no `.incomplete` blobs remain).
4. 2026-02-27:
   - Added chunk observability for compression events:
     - `scheduled_tokens`
     - `estimated_cache_len`
     - `prefill_len`
     in `runner_compression_actions` emitted events.
   - Added/updated tests:
     - `tests_runtime/test_runner_compression_actions.py`
     - `tests_runtime/test_planner.py`
   - Stabilized `test_selection_planner` by explicitly registering KV layout
     axis hints for test tensors (no runtime behavior change).
   - Full runtime test suite passed:
     - `pytest -q TriAttention_vLLM/tests_runtime`
     - Result: `179 passed`.
5. 2026-02-27:
   - Added fullkv-vs-triattention demo configs:
     - `evaluation/dispatch/configs/triattention_aime24_fullkv_qwen3_demo.yaml`
     - `evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor_qwen3_prefill_chunk.yaml`
   - Added concise runbook:
     - `docs/interface/PREFILL_CHUNK_DEMO_RUNBOOK.md`
6. 2026-02-27:
   - Verified local cache usability for downloaded Qwen3-Coder:
     - `AutoConfig.from_pretrained(..., local_files_only=True)` OK
     - `AutoTokenizer.from_pretrained(..., local_files_only=True)` OK
7. 2026-02-27:
   - Ran Qwen3 TriAttention smoke end-to-end:
     - output: `evaluation/outputs/qwen3_smoke_run/shard00/run000.jsonl`
     - result: completed without OOM (`prefill_tokens=62`, `total_tokens=256`).
   - Verified runtime auto-chunk wiring in live logs:
     - `enable_chunked_prefill=True`
     - `max_num_batched_tokens=16` (for smoke).
8. 2026-02-27:
   - Ran long-prefill validation (Qwen3, TriAttention, pre-chunk):
     - dataset: `evaluation/outputs/qwen3_longprefill_dataset.jsonl`
     - output: `evaluation/outputs/qwen3_longprefill_run2/shard00/run000.jsonl`
     - result: completed without OOM with long prompt
       (`prefill_tokens=3058`, `total_tokens=4096`).
   - Re-ran full runtime tests:
     - `pytest -q TriAttention_vLLM/tests_runtime`
     - Result: `179 passed in 13.02s`.
9. 2026-02-27:
   - Ran ultra-long prefill pair on the same sample (`prefill_tokens=12058`, `total_tokens=13000`):
     - fullkv-like run:
       `evaluation/outputs/qwen3_verylong_fullkv_run/shard00/run000.jsonl`
       (`disable_compression=true`, `prefill_auto_chunk=false`)
     - TriAttention pre-chunk run:
       `evaluation/outputs/qwen3_verylong_tri_run/shard00/run000.jsonl`
       (`disable_compression=false`, `prefill_auto_chunk=true`, `chunk=2048`)
   - Both completed without OOM on current hardware; this means demo OOM contrast
     still needs stronger memory pressure conditions (e.g., larger model / longer
     context / lower memory budget setup).
10. 2026-02-27:
   - Started full 8-GPU alignment run against HF SpeckV target setting
     (`triattention_aime24_hf_perhead_anchor` parameters, AIME24 sampled8).
   - Initial dispatch via `conda run` showed shard launch lock contention (no log growth).
   - Added backward-compatible dispatch override:
     `TRIATTN_DISPATCH_PYTHON_BIN` to use direct interpreter for shard launches.
   - Re-launched successfully with all 8 GPUs active:
     - config: `evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor.yaml`
     - logs: `evaluation/logs/triattention_aligncheck_hf_qwen7b_20260227_125917`
     - outputs: `evaluation/outputs/triattention_aligncheck_hf_qwen7b_20260227_125917/shards`
   - Status: running (in progress), shard outputs currently in `.tmp` accumulation.
11. 2026-02-27:
   - Full alignment run live progress snapshot:
     - run root:
       `evaluation/outputs/triattention_aligncheck_hf_qwen7b_20260227_125917`
     - logs:
       `evaluation/logs/triattention_aligncheck_hf_qwen7b_20260227_125917`
   - Milestones observed:
     - shard completed: `shard01`, `shard05` (`[DONE]` logged)
     - remaining shards continue running with active `.tmp` growth.
12. 2026-02-28:
   - User要求“只使用空闲卡，不使用已占用显卡”后，重新启动 Qwen3 对齐全量实验。
   - 启动前 GPU 状态确认：
     - 空闲：`0,1,2,3`
     - 被占用：`4,5,6,7`（其他用户任务）
   - 本次运行显式绑定空闲卡：
     - `--gpus 0,1,2,3`
     - config: `evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor_qwen3.yaml`
     - logs: `evaluation/logs/triattention_aligncheck_hf_qwen3_20260228_155536`
     - output root (target): `evaluation/outputs/triattention_aligncheck_hf_qwen3_20260228_155536/shards`
   - 运行态确认：
     - 4 个 shard worker 正常启动并进入 Qwen3 权重加载；
     - `nvidia-smi` 观测到仅 `0-3` 有本任务显存占用，未触碰 `4-7`。
   - 当前状态：进行中（等待完整 shard 结果后进入 D2 指标对比）。
13. 2026-03-01:
   - Qwen3 对齐全量实验已完整结束（无中断残留）：
     - run root:
       `evaluation/outputs/triattention_aligncheck_hf_qwen3_20260228_155536`
     - logs:
       `evaluation/logs/triattention_aligncheck_hf_qwen3_20260228_155536`
   - 完整性检查：
     - 8/8 shard 均 `[DONE]`
     - 每个 shard 30 条记录（共 240 条）
     - merged 文件存在且 `wc -l = 240`
     - 无 `.tmp` 残留
   - 评测结果：
     - metrics:
       `evaluation/outputs/triattention_aligncheck_hf_qwen3_20260228_155536/eval/triattention_aime24_hf_perhead_anchor_qwen3/aime24/default-default_math_multi_eval_cot_metrics.json`
     - `acc = 47.5`, `timeout_samples = 4`, `empty_samples = 3`
   - 运行关键参数确认：
     - `kv_budget = 2048`
     - `top_k = 50`
     - 模型：`/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B`

## 7) Important Risks to Track

1. Qwen3 RoPE/metadata/stats compatibility drift may cause hidden alignment issues.
2. Prefill chunk trigger timing can affect runtime semantics if not isolated cleanly.
3. Any change must preserve Qwen2.5 behavior by default.
4. On current GPU + model setup, fullkv baseline may not OOM for 12k prefill;
   explicit high-pressure demo setup is required for visible OOM contrast.
