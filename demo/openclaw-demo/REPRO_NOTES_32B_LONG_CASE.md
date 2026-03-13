# 32B Long-Case Repro Notes (Benchmark + Demo)

This note documents the exact scripts used to reproduce the recent 32B long-context checks.

## 1) Known-good benchmark script (32B speaks normal language)

Script:

`weian_development/demo_debug/run_32b_openclaw_like_benchmark.sh`

Default setup:

- Model: `JunHowie/Qwen3-32B-GPTQ-Int4` local snapshot
- Stats: `demo/openclaw-demo/stats/qwen3_32b_int4_speckv_stats.pt`
- Dataset: `/tmp/tri_diag/openclaw_like_dataset.jsonl`
- `kv_budget=12000`
- `max_length=12600`

Run:

```bash
bash weian_development/demo_debug/run_32b_openclaw_like_benchmark.sh
```

Expected:

- Output text should be readable (not repeated gibberish).
- `total_tokens` should exceed `kv_budget` only mildly in this setup.
- Compression events should appear in runtime logs (depending on logger settings).

## 2) Demo quality probe script (for teammate's demo stack)

Script:

`weian_development/demo_debug/check_demo_completion_quality.sh`

Run against a running demo gateway:

```bash
DEMO_BASE_URL=http://127.0.0.1:8125 \
MAX_TOKENS=4000 \
bash weian_development/demo_debug/check_demo_completion_quality.sh
```

Expected output summary includes:

- `usage.prompt_tokens`
- `usage.completion_tokens`
- `max_same_ws_run`
- `max_same_char_run`

The script also prints output head/tail for quick manual inspection.

## 3) If teammate says "budget=14k still repeats"

Please verify these first:

1. The actual server process is from the current branch/commit (not an older running backend).
2. Runtime init log contains the expected budget, for example:
   - `TriAttention monkeypatched Scheduler initialized: budget=14000 ...`
3. Model/stats match:
   - model: `Qwen3-32B-GPTQ-Int4`
   - stats: `qwen3_32b_int4_speckv_stats.pt`
4. Prompt and output length are comparable to the long-case test:
   - long prompt (~7k+ tokens) and long generation (2k~4k tokens).
5. Use the same API path when comparing:
   - `demo`/OpenClaw typically goes via `/v1/completions`.

If any one of the above differs, quality conclusions are not directly comparable.
