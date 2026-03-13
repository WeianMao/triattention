# Selector Bug Investigation Plan (2026-03-13)

## Intent (Must Keep)
- Keep the SpeckKV algorithm unchanged.
- Find and fix the real bug in V1 runtime integration path.
- Debug instrumentation must be env-gated and must not impact non-debug default performance.

## Non-Goals (Explicit)
- Do **not** use fallback-keep as final product behavior to "hide" the bug.
- Do **not** switch to another compression algorithm as a permanent fix.

## Evidence (Confirmed Facts)
1. Same model/stats/settings, only path differs:
   - Model: `DeepSeek-R1-0528-Qwen3-8B`
   - Stats: `R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen3/stats/deepseek_r1_qwen3_8b_plain_stats.pt`
   - Dataset: `/tmp/tri_diag/openclaw_like_dataset.jsonl`
   - Context: `prefill=7817`, `max_length=12600`, `kv_budget=12000`, `protect_prefill=false`

2. `async` itself is not sufficient to explain failure:
   - `head snapshot + async0` (`v34`) is readable (`max_run=1`).
   - So "async on/off" is not the direct root cause.

3. The degradation reproduces when selector path is used in this long-prefill case:
   - `auto_fallback=0` (`v36`) => bad tail repetition (`2 2 2 ...`, `max_run=43`)
   - compression reason: `kv_compacted:per_head:hf_aligned_global_per_head`

4. Fallback path is stable in this case:
   - `auto_fallback=1` (`v35`) => readable (`max_run=1`)
   - compression reason: `kv_compacted:fallback`

5. Legacy per-head semantics is also not healthy enough:
   - `auto_fallback=0 + per_head_selection_semantics=legacy_layer_local` (`v37`)
   - avoids `2 2 2`, but still degrades to punctuation repetition (`. . .`, `max_run=37`)
   - compression reason: `kv_compacted:per_head`

## Rejected Hypotheses (Based On Evidence)
- "The root cause is async scheduling itself." -> Rejected.
- "Only hf-aligned branch is broken; legacy branch is fully good." -> Rejected.

## Current Hypotheses (Need Verification)
1. **High confidence**: issue is in V1 selector-driven keep-index pipeline for long-prefill (not compression kernel itself).
2. **Medium confidence**: paged/streaming scoring path and its chunk-merge behavior may produce unstable keep sets under this workload.
3. **Medium confidence**: selector output may be formally valid (shape/range) but semantically weak for this context (degrades generation).

## Next Debug Plan (No Non-Debug Performance Impact)
1. Add debug-only trace of selector outputs at each compression step:
   - keep-count per head / shared
   - min/max index
   - overlap across adjacent compression steps (Jaccard)
   - gated by env flag (off by default).

2. Add debug-only "selector quality sanity checks" (fail-fast mode):
   - range check, monotonic/sorted check, duplicate ratio check
   - abrupt drift check between adjacent compression steps
   - only active when `TRIATTN_RUNTIME_SELECTOR_DEBUG=1`.

3. Differential run matrix (same seed/prompt/model/stats):
   - A: `fallback` (reference stable output)
   - B: `hf_aligned_global_per_head`
   - C: `legacy_layer_local`
   - Compare first divergence step and selector traces.

4. If divergence is pinned to selector output instability:
   - locate exact substage (score compute / normalization / chunk merge / topk merge).
   - prepare minimal fix in selector path only (no algorithm replacement).

## Acceptance Criteria For Root-Cause Lock
- Can reproduce bad case with selector path and stable case with fallback path on same setup.
- Can point to one concrete selector substage where traces diverge abnormally.
- A debug-only targeted patch at that substage restores readable output without changing non-debug path behavior.

## Artifacts (Latest Key Outputs)
- Good (fallback on): `debug/openclaw_async0_autofallback_on_output.jsonl`
- Bad (fallback off): `debug/openclaw_async0_autofallback_off_output.jsonl`
- Trace:
  - `/tmp/tri_diag/tri_trace_v35_current_async0_dcenv.jsonl`
  - `/tmp/tri_diag/tri_trace_v36_current_async0_noautofallback.jsonl`
  - `/tmp/tri_diag/tri_trace_v37_current_async0_noautofallback_legacy.jsonl`

## Rollback Review (Changes Introduced During "Async-As-Root-Cause" Phase)
> Purpose: record what must be reviewed/possibly rolled back after root cause is fixed.

### A. Suspected Wrong-Direction Runtime Changes (Need Re-evaluation)
1. `triattention_runtime/runner_state_updates.py`
- Added local-estimate progression for compressed requests (`TRIATTN_RUNTIME_LEGACY_COMPRESSED_ESTIMATE` path).
- Status: likely tied to async-root-cause hypothesis; keep for now as experiment control; revisit rollback later.

2. `evaluation/runner/vllm_triattention_runtime_runner.py`
- Added env-forced async scheduling toggles:
  - `TRIATTN_RUNTIME_FORCE_DISABLE_ASYNC_OUTPUT_PROC`
  - `TRIATTN_RUNTIME_FORCE_ASYNC_SCHEDULING`
- Status: debug/control convenience only. Keep during investigation; can be removed/hidden later.

3. `triattention_runtime/request_key_compat.py`
4. `triattention_runtime/scheduler.py`
5. `triattention_runtime/effective_overrides.py`
6. `triattention_runtime/worker_reclaim_sync.py`
7. `triattention_runtime/selector_hf.py`
- req_id typing relaxed from `str` to generic (`str|int`/non-None).
- Status: not proven wrong; likely compatibility hardening. Keep but mark for post-fix audit.

### B. Temporary Mitigation (Not Final Product Fix)
1. `triattention_runtime/hook_impl.py`
- auto fallback keep for long-prefill (`TRIATTN_RUNTIME_AUTO_FALLBACK_KEEP_LONG_PREFILL`).
- Status: currently useful to keep demo stable, but not accepted as final bug fix.

### C. Debug/Tracing Additions (Planned For Cleanup)
1. `triattention_runtime/debug_trace.py`
2. Trace event calls in:
- `triattention_runtime/runner_state_updates.py`
- `triattention_runtime/runner_compression_actions.py`
- Status: debug-only and env-gated; keep during investigation; remove or narrow once root cause is fixed.

## Async Misdiagnosis Changes: Rollback Priority List
> This section is the handoff-critical view: what came from the async misdiagnosis and
> how we should treat it after root cause is proven.

1. **Priority P0 (re-evaluate first; likely rollback if root cause is selector-only)**
- `triattention_runtime/runner_state_updates.py`
  - Added compressed-request local estimate progression (`_LEGACY_COMPRESSED_ESTIMATE` branch and local recurrence).
  - Why risky: it changes core runtime state evolution in non-debug path.
  - Current status: keep temporarily for controlled reproduction; do not expand scope.

2. **Priority P1 (control knobs; keep during investigation, hide/remove later)**
- `evaluation/runner/vllm_triattention_runtime_runner.py`
  - Added env overrides for async scheduling and async output processor.
  - Why added: isolate async hypothesis quickly.
  - Why not final: not part of product behavior, mainly diagnostic control.

3. **Priority P1 (possibly valid compatibility hardening; keep unless disproven)**
- `triattention_runtime/request_key_compat.py`
- `triattention_runtime/scheduler.py`
- `triattention_runtime/effective_overrides.py`
- `triattention_runtime/worker_reclaim_sync.py`
- `triattention_runtime/selector_hf.py`
  - Change type: req_id typing from strict `str` to generic non-None key (`str|int` compatible).
  - Why likely safe: avoids key mismatch when scheduler/request key type differs.
  - Action: keep for now; re-verify after root cause fix.

4. **Priority P2 (temporary quality mitigation, not root-cause fix)**
- `triattention_runtime/hook_impl.py`
  - Auto fallback keep for long-prefill (`TRIATTN_RUNTIME_AUTO_FALLBACK_KEEP_LONG_PREFILL`).
  - Purpose: keep demo usable.
  - Action: keep as mitigation toggle until selector bug is fixed; not accepted as final fix.

5. **Priority P3 (debug-only instrumentation; remove/narrow after close)**
- `triattention_runtime/debug_trace.py`
- Trace callsites in:
  - `triattention_runtime/runner_state_updates.py`
  - `triattention_runtime/runner_compression_actions.py`
  - `tests_runtime/test_request_key_compat.py`
  - related debug test updates
  - Action: cleanup once root cause is pinned and fixed.

## Evidence vs Guess (Strict Boundary For Handoff)
- **Evidence**:
  - async0 can be good (v34), so async is not sole root cause.
  - selector path (`auto_fallback=0`) degrades in long-prefill case (v36/v37).
  - fallback path (`auto_fallback=1`) stays readable (v35).
- **Guess (to be proven)**:
  - bug is inside selector chain under this workload (score/merge/guard path).
  - current async-related runtime changes are likely not the true fix.

## Final Execution Plan (Current)
1. Keep non-debug runtime behavior frozen (no new non-debug fixes now).
2. Add only debug-gated observability for selector pipeline.
3. Reproduce with fixed setup and isolate first divergence step between:
- fallback stable path
- selector path (`hf_aligned_global_per_head`, `legacy_layer_local`)
4. Identify concrete failing substage inside selector chain:
- score compute / normalization / chunk merge / top-k merge / guard application.
5. After root cause is proven, prepare minimal non-debug fix proposal.
6. Then perform rollback audit using this section (A/B/C), and clean temporary debug code.

## Immediate Next Actions (This Session)
1. Keep current non-debug code unchanged.
2. Run debug-only differential experiments that isolate selector sub-stages.
3. Produce one clear mechanism statement:
- expected module contract
- actual broken behavior
- why it leads to degraded generation.
4. Only after mechanism lock, propose minimal non-debug fix and explicit rollback set.

## 2026-03-13 Late Findings (Major Progress)

### A. Differential Results (OpenClaw-like, 8B + matched stats + budget=12000)

1. Baseline bad case (current default HF per-head path):
- `pruning_mode=per_head`
- `per_head_selection_semantics=hf_aligned_global_per_head`
- `sparse_normalize_scores=false`
- Result: bad (`max_same_word_run=41`)

2. Debug-only bypass group selector:
- Same as above + `TRIATTN_RUNTIME_DEBUG_DISABLE_GROUP_SELECTOR=1`
- Result: good (`max_same_word_run=1`)

3. Debug-only keep group selector, but switch cross-layer aggregation in group selector:
- Same as baseline bad + `TRIATTN_RUNTIME_DEBUG_GROUP_PERHEAD_AGG_MODE=max`
- Result: good (`max_same_word_run=1`) for both:
  - `sparse_normalize_scores=false`
  - `sparse_normalize_scores=true`

### B. Locked Mechanism (HF Per-Head Path)

- Root-cause is now strongly localized to **group selector aggregation logic** in
  `selector_hf.py::_select_keep_indices_for_group_per_head`:
  - current behavior uses **mean** across layers for per-head chunk scores
  - debug replacement with **max** restores readable output while compression still triggers.
- This matches the suspicion that this substage diverges from expected HF-aligned behavior.
- Reference alignment clue:
  - `R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py:455`
    uses `group_scores.max(dim=0)` for per-head grouped aggregation.

### C. What Is Already Ruled Out

- Not async itself (already ruled out earlier).
- Not chunk-merge artifact alone (single-chunk still bad in affected modes).
- Not reclaim trigger absence (compression is applied 4 times in all compared runs).
- Not cross-head overlap collapse (debug per-head overlap metrics are similar in good/bad runs).

### D. Remaining Scope

- Legacy per-head path (`per_head + legacy_layer_local`) remains degraded in this
  long-prefill demo case and is likely a separate issue.
- For the immediate demo blocker (`hf_aligned_global_per_head`), the main failing
  stage is now identified.

## 2026-03-13 Fix Validation (HF Demo Blocker)

- Applied minimal runtime fix in `selector_hf.py`:
  - group selector cross-layer aggregation default switched to `max`
    (debug env can still override for comparison).
- Non-debug recheck with same OpenClaw-like setup:
  - `per_head + hf_aligned_global_per_head + normalize=false`: recovered (`max_run=1`)
  - `per_head + hf_aligned_global_per_head + normalize=true`: recovered (`max_run=1`)
  - compression still applied 4 times (`kv_compacted:per_head:hf_aligned_global_per_head`).
- Targeted runtime unit tests:
  - `46 passed` across hook/integration/runner state suites (with proper `PYTHONPATH`).
