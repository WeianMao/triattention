# Next Stage Execution TODO

Updated: 2026-03-02
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
13. [x] E1 - Run Qwen3-Coder long-prefill stress check (fullkv vs TriAttention)
    on free GPUs only; fix runtime issues blocking execution.
14. [x] F1 - Locate canonical R-KV calibration script and confirm expected trace input form.
15. [x] F2 - Generate Qwen3-Coder calibration stats from
    `R-KV/outputs/aime_sampled8_qwen3/fullkv/aime24`.
16. [x] F3 - Re-run TriAttention FP8 long-prefill after stats are wired.
17. [x] D3 - Execute demo-oriented long-prefill stress comparison (baseline fail vs TriAttention pass).

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
14. 2026-03-01:
   - 执行 Qwen3-Coder 长 prefill 压测（仅使用空闲 GPU）并修复阻塞问题：
     - 模型：`Qwen/Qwen3-Coder-30B-A3B-Instruct`
     - 初始单卡 A100-40G 失败：模型初始化 OOM（权重装载阶段）。
     - 切到 TP=2 后再次失败：默认 cudagraph capture OOM。
     - 修复：设置 `--enforce-eager true`，规避 cudagraph 额外显存开销。
     - 发现原随机数据集超长（`decoder prompt length = 306427`），
       超过 `max_model_len=32768`，并非有效 prefill OOM 场景。
     - 修复：构造新数据集
       `evaluation/outputs/qwen3coder_longprefill_30k_dataset.jsonl`
       （token 长度约 30001）。
   - 在同一参数基线下完成 fullkv 与 TriAttention 对照（TP=2，空闲卡 1,2）：
     - fullkv:
       `evaluation/outputs/qwen3coder_longprefill30k_fullkv_tp2_eager90_20260301_191537/shards/shard00/run000.jsonl`
       - `prefill_tokens=30053`, `output_tokens=2715`, `total_tokens=32768`
     - TriAttention + prefill chunk:
       `evaluation/outputs/qwen3coder_longprefill30k_tri_tp2_eager90_20260301_191537/shards/shard00/run000.jsonl`
       - `prefill_tokens=30053`, `output_tokens=365`, `total_tokens=30418`
   - 结论：两条都可稳定完成，无运行时 OOM；压测链路已可复现。
15. 2026-03-01:
   - W8A16/FP8 单卡验证（不启用 TP）：
     - 模型：`Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`
     - 运行时日志确认：
       - `quantization=fp8`
       - `Using MARLIN Fp8 MoE backend`
       - A100 上为 `weight-only FP8 compression`（无原生 FP8 计算）。
   - 已成功运行：
     - fullkv smoke:
       `evaluation/outputs/qwen3coder_fp8_fullkv_smoke_tp1_20260301_2012/shards/shard00/run000.jsonl`
       (`prefill_tokens=62`, `output_tokens=99`, `total_tokens=161`)
     - TriAttention smoke:
       `evaluation/outputs/qwen3coder_fp8_tri_smoke_tp1_20260301_2012/shards/shard00/run000.jsonl`
       (`prefill_tokens=62`, `output_tokens=104`, `total_tokens=166`)
     - fullkv long prefill (30k):
       `evaluation/outputs/qwen3coder_fp8_longprefill30k_fullkv_tp1_20260301_2012/shards/shard00/run000.jsonl`
       (`prefill_tokens=30053`, `output_tokens=2715`, `total_tokens=32768`)
   - 发现未解决兼容问题（TriAttention + FP8 + long prefill）：
     - 失败类型 A：
       `TRIATTN_FATAL_TRITON_SCORING_REQUIRED:stats_path_not_set`
     - 失败类型 B（显式关闭 triton scoring 后）：
       `enable_experimental_kv_compaction requires require_triton_scoring=True`
   - 当前判断：
     - W8A16 在 fullkv 路径可运行；
     - TriAttention 压缩路径对 FP8 长 prefill 仍需专项适配（strict 约束链路冲突）。
16. 2026-03-02:
   - 已确认 calibration 使用脚本与输入形式：
     - script:
       `R-KV/weian_development/rkv_sparse_round_calibrate.py`
     - 输入支持：
       `--trace-root <dir>`（优先读取 `merged/merged.jsonl`，否则回退 `shards/*.jsonl`）。
   - 已按用户给定 trace 目录准备并启动 coder stats 生成（单卡 GPU 4）：
     - trace root:
       `R-KV/outputs/aime_sampled8_qwen3/fullkv/aime24`
     - target stats:
       `R-KV/outputs/repository/sample8_fullkv_aime24_official_qwen3coder/stats/qwen3_coder_30b_a3b_fp8_plain_stats.pt`
   - 当前阻塞：
     - 机器出现明显 NFS I/O wait（进程状态 `D`），包括 `torch/transformers` 导入阶段；
     - calibration 进程处于 I/O 阻塞，尚未进入有效计算阶段。
   - 结论：
     - “缺 stats 导致 strict 报错”已被确认；
     - 现阶段主阻塞是机器 I/O 状态，不是算法/配置逻辑。
17. 2026-03-02:
   - 切换新机器后，按“仅使用空闲 GPU”继续执行：
     - 空闲卡确认：`4,5,6,7`。
   - calibration 兼容修复（最小改动，默认行为不变）：
     - 文件：`R-KV/weian_development/rkv_sparse_round_calibrate.py`
     - 新增可选参数：`--device-map`（例如 `auto`）
     - 当设置 `--device-map` 时，将模型放置委托给 transformers/accelerate，
       并自动解析输入设备；未设置时沿用旧逻辑（单设备 `model.to(device)`）。
   - 已成功生成 Qwen3-Coder stats：
     - 输出：
       `R-KV/outputs/repository/sample8_fullkv_aime24_official_qwen3coder/stats/qwen3_coder_30b_a3b_fp8_plain_stats.pt`
     - 命令关键参数：
       - `model-path`：
         `.../models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646...`
       - `--device-map auto`
       - `--num-traces 1`
   - 已成功复跑 FP8 TriAttention 长 prefill（单卡，无 TP）：
     - 输出：
       `TriAttention_vLLM/evaluation/outputs/qwen3coder_fp8_longprefill30k_tri_tp1_statsfix_20260302_104934/shards/shard00/run000.jsonl`
     - 关键结果：
       - `prefill_tokens=30053`
       - `output_tokens=2715`
       - `total_tokens=32768`
       - `enable_experimental_kv_compaction=true`
       - `require_triton_scoring=true`
   - 结论：
     - 先前 FP8 TriAttention 长 prefill 失败根因已闭环为“缺 coder stats”；
     - 产出 coder stats 后，同配置路径可正常完成运行。
18. 2026-03-02:
   - 执行目标测试 D（demo 导向：长 prefill 内存压力对照）：
     - 数据集：
       `TriAttention_vLLM/evaluation/outputs/qwen3coder_longprefill_120k_dataset.jsonl`
       （单条样本，问题 prefill 约 120k token）。
   - Baseline（fullkv）失败样例：
     - 设定：`disable_compression=true`，并将 prefill chunk 设为近似“不分块”
       （`prefill_chunk_size=123000`）。
     - 结果：引擎初始化阶段失败（KV cache memory 不足）。
     - 关键报错：`ValueError ... max seq len (123000) ... needed 11.26 GiB ... available 7.14 GiB ...`
     - 可复现日志：
       `TriAttention_vLLM/evaluation/outputs/qwen3coder_fp8_longprefill120k_fullkv_tp1_demoD_20260302_chunk123k_fail/baseline_fail.log`
   - TriAttention 成功样例：
     - 设定：`kv_budget=2048`，`prefill_chunk_size=2048`，启用压缩与 reclaim。
     - 输出：
       `TriAttention_vLLM/evaluation/outputs/qwen3coder_fp8_longprefill120k_tri_tp1_demoD_20260302_chunk2048/shards/shard00/run000.jsonl`
     - 关键结果：
       - `prefill_tokens=120053`
       - `output_tokens=2947`
       - `total_tokens=123000`
       - `status=complete`
   - 结论：
     - 在超长 prefill 压力场景下，baseline 配置可触发显存容量失败；
     - TriAttention 的 chunk + 压缩路径可在同长度输入下完成运行，满足 demo 目标。
19. 2026-03-04:
   - 新增对外交付使用文档（仓库根目录）：
     - `TRIATTENTION_VLLM_USAGE.md`
   - 文档覆盖内容：
     - HF_HOME 规范（`/data/rbg/users/weian/env/huggingface`）
     - Qwen3 / Qwen3-Coder 下载方式
     - dispatch 启动命令
     - FP8 Coder calibration 前置
     - 压缩激活与运行状态的日志判据
   - 目的：
     - 让外部使用者按文档直接完成环境准备与运行，不依赖历史对话上下文。

## 7) Important Risks to Track

1. Qwen3 RoPE/metadata/stats compatibility drift may cause hidden alignment issues.
2. Prefill chunk trigger timing can affect runtime semantics if not isolated cleanly.
3. Any change must preserve Qwen2.5 behavior by default.
4. On current GPU + model setup, fullkv baseline may not OOM for 12k prefill;
   explicit high-pressure demo setup is required for visible OOM contrast.
