# TriAttention vLLM V2 Audit Checklist (2026-02-17)

## 0. Audit Goal

Primary goal for this audit:

1. Verify obvious bug risk in current V2 implementation.
2. Verify whether experiments are valid and whether V2 is truly aligned with HF baseline intent.
3. Identify next required work for:
   - Short-term goal: HF reproduction/alignment.
   - Long-term goal: production-grade generalized serving (beyond reasoning-only).

This file is the single source of truth for this audit turn.

## 1. Scope and Non-Goals

In scope:

- `TriAttention_vLLM/triattention_v2/*` core runtime path.
- `TriAttention_vLLM/evaluation/runner/vllm_triattention_v2_runner.py`.
- `TriAttention_vLLM/evaluation/dispatch/*` task split and launch flow.
- `TriAttention_vLLM/benchmarks/reasoning/compare_results.py` and eval path.
- V2 docs in `TriAttention_vLLM/docs/interface/*` and key backend design refs.

Out of scope for this turn:

- Full production architecture refactor.
- New kernel development from scratch.

## 2. Check Criteria

### 2.1 HF Alignment Validity (highest priority)

- Config-level parity: key parameters and switches map correctly.
- Semantics-level parity: pruning mode behavior matches intended HF mode.
- Eval-level parity: metric script/method equivalence (same parser/grader logic).
- Run-level validity: shard split/merge/eval chain produces complete and non-corrupt outputs.

### 2.2 Obvious Bug Screening

- State transitions consistent (planner/scheduler/executor/state).
- Compression trigger conditions sane (no unintentional every-step compress loop).
- Reclaim/reuse logic safe (no silent corruption path).
- Triton path behavior explicit (no hidden slow fallback when strict is required).

### 2.3 Efficiency Readiness for HF Reproduction

- No avoidable large-scale Python loops in hot path.
- Scoring path uses Triton as required for current stage.
- No obvious redundant K/V data movement in scoring path.

## 3. Work Breakdown

1. Read current status docs and goal docs.
2. Inspect V2 modules one by one.
3. Inspect eval/dispatch flow and output integrity checks.
4. Cross-check with HF reference result path and known baseline numbers.
5. Produce findings and ranked action list (short-term vs long-term).

## 4. Findings Log (rolling)

### 4.1 Docs / Goal Alignment

1. Long-term goal and acceptance criteria are clearly documented in `docs/interface/PROJECT_GOAL.md`.
2. V2 SSOT organization is in place and usable (`V2_OVERVIEW.md`, `CURRENT_STATUS.md`, `OPEN_ISSUES.md`).
3. Current documentation still mixes two different “HF alignment” narratives:
   - “HF strict” (historical runs and mixed settings);
   - “HF per-head anchor” (current parity target).
   This is manageable but still confusing for new contributors.

### 4.2 Implementation Findings

1. [P0 Alignment Risk] R-KV stats conversion currently maps `q_abs_mean` to `freq_scale_sq` (`triattention/utils.py:198`), while HF SpeckV uses rotary-derived frequency scaling in scoring path.
2. [P0 Reliability Risk] Triton selector is not hard-required in all failure modes:
   - missing stats path or unsupported mode returns selector `None` and falls back (`triattention_v2/hook_impl.py:133`, `triattention_v2/hook_impl.py:138`, `triattention_v2/hook_impl.py:152`).
   - this can silently run non-Triton fallback under misconfiguration.
3. [P1 Alignment Risk] Trigger/execute timing is not strictly HF-identical:
   - scheduler estimates `effective_base + scheduled_tokens` (`triattention_v2/scheduler.py:108`);
   - hook clamps to current computed length and compresses before `execute_model` (`triattention_v2/hook_impl.py:385`, `triattention_v2/hook_impl.py:389`).
4. [P1 Alignment Risk] `max_new_tokens` is estimated using external tokenizer length (`evaluation/runner/vllm_triattention_v2_runner.py:526`), not vLLM internal prompt tokenization.
5. [P1 Perf Risk] eval runner forces eager mode (`evaluation/runner/vllm_triattention_v2_runner.py:405`), which caps throughput.
6. Positive confirmation:
   - scheduler reclaim path is now fail-fast on inconsistent block mapping (good safety guard).
   - V2 smoke and tests_v2 pass on current codebase.

### 4.3 Experiment Validity Findings

1. V2 per-head anchor full run output integrity is valid:
   - 8 shard files, each `30` lines;
   - merged output `240` lines;
   - all `(sample_idx, draw_idx)` pairs unique and complete (30 samples × 8 draws).
2. Eval artifact exists and is complete:
   - `TriAttention_vLLM/evaluation/outputs/triattention_v2_aime24_hf_perhead_anchor_fix_20260216_174000/eval/triattention_v2_aime24_hf_perhead_anchor/aime24/default-default_math_multi_eval_cot_metrics.json`
   - `acc = 43.8`.
3. Regression checks on current code:
   - `tests_v2/run_smoke.py` => `smoke passed: 54 tests`;
   - `pytest TriAttention_vLLM/tests_v2 -q` => `54 passed`.

### 4.4 HF Alignment Findings

1. Official eval metric comparison (same `eval_math_multi.py` family):
   - HF reference per-head: `42.5`;
   - V2 per-head anchor fix: `43.8`;
   - delta: `+1.3` points (close, but not yet “strictly proven equivalent”).
2. `compare_results.py` reports a different metric definition:
   - it uses “question correct if any draw correct” (`benchmarks/reasoning/compare_results.py:156`);
   - official eval uses per-question mean over draws then dataset mean (`R-KV/HuggingFace/evaluation/evaluate.py:87`).
3. Therefore `73.33/76.67` and `42.5/43.8` are both valid but not comparable to each other; they are different metrics.
4. Per-head aggregation semantics:
   - HF per-head uses “mean(layer-wise max per KV head)” (`R-KV/weian_development/speckv/speckv_rkv_style.py:409`);
   - V2 hf-aligned path currently averages per-layer per-head score tensors directly (`triattention_v2/hook_impl.py:309`).
   - this is semantically close but not yet mathematically identical.

## 5. Required Next Actions

### 5.1 Short-term (HF reproduction)

1. [P0] Fix `freq_scale_sq` source for R-KV stats path to match HF scoring semantics (do not derive from `q_abs_mean`).
2. [P0] Make Triton scoring strictly mandatory in HF-alignment mode:
   - missing stats / selector unavailability should fail-fast, not fallback.
3. [P0] Align compression trigger timing with HF intent (decide and implement one exact contract; validate with token-level parity on fixed subset).
4. [P1] Separate and standardize reporting:
   - “official acc” (eval script) and “any-correct accuracy” (compare script) must be printed together with explicit labels.
5. [P1] Add parity test matrix for pruning modes:
   - `per_layer`, `per_head`, `per_layer_per_head`;
   - include semantic assertions, not only smoke pass.
6. [P1] Add throughput sanity gate for HF reproduction runs:
   - optional non-eager toggle A/B to ensure no accidental severe slowdown path.

### 5.2 Long-term (production readiness)

1. Complete robust block reclaim + page/layout safety under multi-group and long-run stress.
2. Support batch>1 end-to-end with strict request isolation and trigger consistency.
3. Mature prefill policy controls:
   - protect/trim switch;
   - include-prefill budget semantics under real traffic.
4. Add KV-usage-driven trigger as first-class strategy (with hysteresis and observability).
5. Establish CI gates:
   - smoke + parity subset + output integrity checks + metric sanity checks.

## 6. Decision/Assumption Log

- 2026-02-17: This audit prioritizes correctness/alignment evidence over adding new features.
- 2026-02-17: Current status is “close-to-HF but not yet strict-equivalent”, mainly due scoring data-path and semantic edge differences listed above.
