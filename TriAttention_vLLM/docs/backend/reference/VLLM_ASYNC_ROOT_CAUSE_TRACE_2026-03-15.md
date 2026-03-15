# vLLM Async Root-Cause Trace

Updated: 2026-03-15
Status: Investigation milestone reached
Owner: Codex

Related docs:

1. `TriAttention_vLLM/docs/backend/reference/MILESTONE_HF_PREFILL_COMPRESSION_PROBE_2026-03-13.md`
2. `TriAttention_vLLM/docs/backend/reference/VLLM_PREFILL_COMPRESSION_FINDINGS_2026-03-14.md`
3. `TriAttention_vLLM/docs/backend/reference/VLLM_FORMAL_PATH_COMPAT_STATUS_2026-03-15.md`

## 1. Purpose

This document records the full reasoning chain that narrowed the long-prefill
quality problem from:

1. "maybe the compression algorithm is wrong"

to:

2. "the main remaining problem sits in the async runtime path"

The point is not only to record the final conclusion, but also to preserve:

1. what was tested
2. what each test was trying to prove
3. which hypotheses were falsified
4. which conclusions are now supported by direct evidence

## 2. Initial symptom

Observed symptom on the long demo-like case:

1. long prefill triggers compression
2. `TriAttention_vLLM` output can degrade badly after compression
3. the degradation is stronger than what we see in ordinary long-tail baseline
   drift

This immediately raised two possibilities:

1. the compression algorithm itself is not robust for this case
2. the `vLLM` integration/runtime path is applying the algorithm incorrectly

## 3. First separation: algorithm vs implementation

### 3.1 Why HF was used

The first major decision was to separate:

1. algorithm quality
2. `vLLM` runtime implementation quality

To do that, a separate HF-side probe was built instead of continuing to debug
only inside `vLLM`.

### 3.2 HF probe design

The HF-side probe was intentionally simple:

1. same model family and long-prompt case
2. run a normal baseline
3. run a "manual prefill compression once, then continue decode" variant

The probe was not intended to perfectly simulate the whole `vLLM` runtime.
Its purpose was narrower:

1. if the model collapses immediately even in HF, the algorithm itself becomes
   the main suspect
2. if the model stays normal in HF, suspicion shifts toward `vLLM`

### 3.3 HF probe result

Result:

1. HF baseline was normal
2. HF prefill-compression probe was also normal
3. no catastrophic corruption was reproduced there

What this supported:

1. the algorithm is not "obviously broken by construction"
2. the main direction should move toward `vLLM` runtime integration

What it did not prove:

1. it did not mathematically prove the algorithm is perfect
2. it did not prove all implementation paths are equivalent

But it was strong enough to justify moving the main effort into the `vLLM`
integration path.

## 4. Second separation: old/default path vs formal path

### 4.1 What was suspected next

Once suspicion shifted to `vLLM`, the next question became:

1. is the runtime using the compressed semantics correctly after compression?

This led to the discovery of a real bug on the old/default runtime path:

1. compression could happen
2. but the downstream attention-input preparation path was not actually using
   the compressed semantic overrides correctly

### 4.2 What was done

A debug-only compatibility probe was added on the old/default path to verify:

1. whether compressed effective-length / position semantics were really being
   propagated

### 4.3 What was observed

Observed:

1. before the probe, the override path was effectively empty on the real
   default path
2. after the probe, the override path became non-empty and active

This confirmed a real runtime bug.

### 4.4 Why that was not the end

Even after that bug was repaired in debug mode:

1. output still did not fully recover

This changed the reading:

1. the old/default path did contain real bugs
2. but continuing to patch that path was looking increasingly fragile
3. the better next direction was to move toward the formal runtime path

## 5. Third separation: did compaction itself corrupt KV?

### 5.1 Why this mattered

Before blaming runtime semantics, we also needed to rule out a simpler failure:

1. maybe compaction was physically copying the wrong KV contents

### 5.2 Validation

Debug-only content validation was added to the compaction code.

The validation checked:

1. after compaction, do the retained KV contents exactly match the intended kept
   contents?

### 5.3 Result

The compaction-content validation passed.

This did not prove every layout issue was impossible, but it strongly lowered
the probability that:

1. the main problem was a trivial "KV got copied incorrectly" bug

## 6. Formal path compatibility phase

### 6.1 Why move to the formal path

At this point, the most reasonable interpretation was:

1. algorithm is not the top suspect
2. old/default runtime path contains real bugs but still does not fully recover
3. the formal path is more likely to express the correct compressed semantics,
   if its compatibility blockers can be removed

### 6.2 What happened when formal path was first enabled

Initially, the formal path did not even reach the quality-comparison stage.
Instead, several compatibility blockers appeared one after another, including:

1. request identity mismatch between runtime stages
2. lifecycle mismatch between execute/sample stages
3. request-context mismatch in the compression hook preflight

This was important:

1. it explained why the formal path had not been giving useful quality results
2. it also showed the problem was not "the formal path computes bad output"
3. rather, the formal path first needed to be made compatible enough to run

### 6.3 What was repaired

The most important early repair was:

1. making the compression hook understand formal-path request state, instead of
   assuming the old path's request container shape

After this repair, the formal path finally entered the real compression chain.

## 7. The decisive experiment set

Once the formal path could actually run compression end to end, three runs were
executed on the same long-prefill probe case.

### 7.1 Run A: formal path + sync scheduling + compression

Goal:

1. check whether the formal path can produce normal output if we remove async
   scheduling from the equation

Observed:

1. compression triggered multiple times
2. the run completed
3. output stayed readable and usable
4. no catastrophic collapse appeared

Reference output:

1. `debug/v2_formal_sync_compress_budget7000_good.jsonl`

### 7.2 Run B: formal path + sync scheduling + no compression

Goal:

1. establish a same-path baseline for comparison

Observed:

1. baseline was normal
2. output quality was at the same rough level as Run A

Reference output:

1. `debug/v2_formal_sync_nocompress_budget12000_baseline.jsonl`

### 7.3 Run C: formal path + async scheduling + compression

Goal:

1. isolate whether the remaining problem is specifically tied to async behavior

Observed:

1. compression triggered normally
2. the run completed
3. output degraded again in the tail into obvious repetitive drift

Reference output:

1. `debug/v2_formal_async_compress_budget7000_bad.jsonl`

## 8. Why these three runs matter

Taken together, these three runs sharply narrow the remaining root cause.

They support all of the following:

1. the algorithm is not the main problem
2. the formal path itself is not generically broken
3. compression on the formal path can work correctly
4. the remaining failure is strongly tied to async scheduling / async state

## 9. Final repair result

The async corruption issue has now been repaired on an isolated fix branch.

Final mechanism statement:

1. async batch-queue lookahead was allowed to run across a compression boundary
2. that allowed a later batch to be scheduled from stale pre-compression state
3. the correct repair point was the queue boundary itself, not a later
   sample-output sync

Final fix:

1. detect a scheduled batch that may trigger compression
2. mark it as a compression-boundary batch
3. do not allow queue lookahead to pass that batch
4. let scheduler absorb the compression event first
5. then resume ordinary async scheduling

Validated outcome:

1. async compression run completes
2. output quality matches sync control on the same long-prefill probe
3. the repaired async path remains faster than the sync control

Detailed evidence:

1. `TriAttention_vLLM/docs/backend/reference/VLLM_ASYNC_BOUNDARY_FIX_2026-03-15.md`
   handoff

This is the strongest direct narrowing achieved in the investigation.

## 9. Async-specific hypothesis that was tested and rejected

### 9.1 Hypothesis

One natural idea was:

1. maybe the async issue exists because compression events are attached too early
2. so maybe delaying them until the later sample-output object would fix the
   phase mismatch

### 9.2 Test

An async-only experimental patch was tried:

1. when formal async path returned `None` from execute-model
2. compression events were kept pending
3. and only attached later to the sample-output stage

### 9.3 Result

This hypothesis was rejected.

Observed:

1. scheduler no longer learned early enough that compression had already
   happened
2. repeated over-triggering increased
3. behavior became worse, not better

### 9.4 Why that matters

This falsified an overly simple reading:

1. the bug is not just "events are attached too early"

It tells us something more precise:

1. scheduler still needs timely visibility of compression state
2. but the current async path still introduces enough phase skew to damage
   quality
3. the final fix must keep scheduler state timely while still making the
   compression boundary semantically consistent

## 10. Current best explanation

At the system level, the best current explanation is:

1. in sync mode, once compression happens, the rest of the system sees the new
   compressed state in a sufficiently aligned way, so output quality remains
   normal
2. in async mode, compression and the rest of the runtime do not switch state at
   a sufficiently tight boundary
3. this leaves a window where some modules have effectively moved to the new
   compressed state while others are still acting on the old state
4. that window does not necessarily crash the model, but it is enough to cause
   quality drift and eventual repetitive degeneration

## 11. What this implies for the repair strategy

The repair direction that follows from the evidence is:

1. do not change the algorithm
2. do not keep stacking random patches on the old/default path
3. do not switch the whole runtime into globally synchronous behavior

Instead:

1. keep the formal path
2. keep async behavior for the normal decode path
3. introduce a minimal request-level synchronization boundary only when
   compression is actually committed

That is the next repair target.

## 12. Open question that remains

The remaining open question is no longer "where is the bug generally?"

It is now much narrower:

1. what is the smallest async compression-commit mechanism that restores state
   consistency
2. without paying the cost of per-token synchronization in the hot path

That is the next phase of work.
