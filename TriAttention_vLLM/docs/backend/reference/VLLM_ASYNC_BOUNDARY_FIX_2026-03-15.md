# vLLM Async Boundary Fix

Updated: 2026-03-15
Status: Fixed on isolated branch and validated
Owner: Codex
Branch: `codex/vllm-async-commit-fix-20260315`

## 1. Purpose

This document records the final localization and repair for the remaining
`formal path + async scheduling` corruption problem.

The goal is to preserve:

1. what the system was expected to do
2. what it actually did
3. why earlier fixes were not sufficient
4. what was changed
5. what evidence now supports the fix

## 2. Expected system behavior

For one request on the formal vLLM path, the intended runtime logic is:

1. scheduler decides whether the current batch should trigger compression
2. runner executes the compression and produces a new compressed KV state
3. scheduler consumes the compression result and updates its own view
4. future decode steps continue from that new compressed state

If async scheduling is enabled, that async behavior should improve throughput,
but it must not allow later batches to continue using the pre-compression view
once the compression batch has become the next semantic boundary.

## 3. Observed broken behavior

We already had strong evidence that:

1. `formal path + sync scheduling + compression` was good
2. `formal path + async scheduling + compression` was bad

That meant the remaining issue was not:

1. the compression algorithm itself
2. formal-path compatibility in general
3. simple KV copy corruption

The remaining issue had to be something specific to the async lifecycle.

## 4. Root-cause mechanism

The decisive mechanism is:

1. vLLM async scheduling uses a small batch queue to run ahead
2. in that loop, the engine may schedule a later batch before it has processed
   the output of the earlier batch
3. if the earlier batch is exactly the batch that triggers compression, then the
   later batch may still be scheduled from the old pre-compression scheduler
   state
4. that creates a narrow but real stale-state window
5. once that stale-state window exists, later decode can drift and degrade

In short:

1. compression itself was not the problem
2. the queue lookahead across a compression boundary was the problem

## 5. Why the previous async fix was not enough

An earlier repair attempt forced a synchronous sample-output commit after an
applied compression event.

That helped confirm the boundary mattered, but it was still too late.

Why:

1. the batch queue had already been allowed to run ahead
2. so synchronizing at sample-output time did not prevent the stale future batch
   from being scheduled on the old state

That is why the previous fix produced:

1. successful completion
2. some improvement
3. but still visible tail degradation

## 6. Final fix strategy

The final repair keeps ordinary async decode unchanged and only changes behavior
at compression boundaries.

Strategy:

1. detect whether the currently scheduled batch is predicted to hit a
   compression boundary
2. mark that batch as a boundary batch
3. once a boundary batch is in the async queue, do not let the queue schedule
   newer work ahead of it
4. drain that boundary batch first
5. let scheduler absorb the compression event
6. then resume ordinary async lookahead

This gives the desired property:

1. no per-token synchronization
2. no global shutdown of async scheduling
3. only a local queue barrier around batches that may compress

## 7. Implementation summary

The implementation lives in:

1. `TriAttention_vLLM/triattention_runtime/integration_monkeypatch.py`

Main change:

1. patch vLLM `EngineCore.step_with_batch_queue`
2. add a TriAttention-specific queue barrier for compression-boundary batches
3. keep ordinary async fast path unchanged

Supporting tests:

1. `TriAttention_vLLM/tests_runtime/test_integration_monkeypatch.py`
2. `TriAttention_vLLM/tests_runtime/test_runner.py`

## 8. Validation evidence

### 8.1 Unit tests

Command:

```bash
/data/rbg/users/weian/env/miniconda3/envs/trivllm/bin/python -m pytest \
  TriAttention_vLLM/tests_runtime/test_integration_monkeypatch.py \
  TriAttention_vLLM/tests_runtime/test_runner.py -q
```

Result:

1. `18 passed`

### 8.2 End-to-end async compression run

Configuration:

1. formal vLLM path
2. async scheduling enabled
3. compression enabled
4. `kv_budget = 7000`
5. long-prefill demo-like case

Output:

1. `debug/v2_formal_async_compress_budget7000_boundaryfix_good.jsonl`

Observed:

1. compression triggered multiple times
2. run completed cleanly
3. output stayed in the same readable quality band as the sync control
4. previous bad tail repetition no longer appeared as a runtime-corruption
   symptom

### 8.3 Sync control run

Configuration:

1. same setup
2. force sync scheduling

Output:

1. `debug/v2_formal_sync_compress_budget7000_boundaryfix_control.jsonl`

Observed:

1. output quality matched the repaired async result
2. this is important because it shows the repaired async path is no longer
   diverging from the known-good sync path

## 9. Performance reading

This fix does not revert the engine to full-time sync behavior.

Practical reading from the end-to-end runs:

1. repaired async run generation phase was about `35s`
2. sync control generation phase was about `42s`

This is not a rigorous benchmark, but it is enough to support the intended
claim:

1. the fix preserves a real async advantage
2. the overhead is concentrated near compression boundaries
3. it is not equivalent to syncing every decode token

## 10. Final conclusion

Current conclusion:

1. the remaining corruption problem on the formal path was caused by async queue
   lookahead across a compression boundary
2. a boundary-local queue barrier fixes that problem
3. the repaired async path now completes and matches sync-path quality on the
   target long-prefill probe
4. this fix is a reasonable candidate to merge back after branch review

## 11. Remaining caution

This closes the main async corruption issue for the validated probe case.

What it does not claim:

1. every possible workload has now been exhaustively benchmarked
2. all future async edge cases are impossible

Recommended next step after review:

1. merge this isolated-branch fix into the main development branch
2. run one broader smoke matrix on the main branch
3. then use that branch for downstream demo work
