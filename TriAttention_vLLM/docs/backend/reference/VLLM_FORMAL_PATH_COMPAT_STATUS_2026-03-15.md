# vLLM Formal-Path Compatibility Status

Updated: 2026-03-15
Status: Active investigation
Owner: Codex

Related docs:

1. `TriAttention_vLLM/docs/backend/reference/MILESTONE_HF_PREFILL_COMPRESSION_PROBE_2026-03-13.md`
2. `TriAttention_vLLM/docs/backend/reference/VLLM_PREFILL_COMPRESSION_FINDINGS_2026-03-14.md`
3. `TriAttention_vLLM/docs/backend/reference/VLLM_PREFILL_COMPRESSION_SUSPECT_AUDIT_2026-03-13.md`

## 1. Why this document exists

This document records the current milestone after the HF-side probe and the next
repair direction on the vLLM side.

The goal is to make handoff easy and to keep "confirmed facts" separate from
"working hypotheses".

## 2. High-level conclusion

Current evidence supports the following reading:

1. The prefill-compression algorithm itself is not the leading suspect.
2. The main problem is more likely in the `TriAttention_vLLM` runtime integration.
3. Continuing to add patches on the old/default runtime path is no longer the
   best main strategy.
4. The better direction is to make the more formal vLLM runtime path compatible
   enough to run end-to-end and then judge output quality there.

This is a milestone, not the final repair.

## 3. What is already confirmed

### 3.1 HF-side evidence

The isolated HF probe already showed:

1. On the same 32B INT4 model and long prefill case, manual prefill compression
   can still produce normal, readable output.
2. This does not fully prove the algorithm is perfect.
3. But it is strong evidence against the idea that "prefill compression itself
   necessarily causes the model to collapse".

### 3.2 Old/default vLLM path is not enough

We already confirmed a real bug on the old/default runtime path:

1. After compression, the new effective context semantics were not correctly
   propagated into the downstream attention-input preparation chain.
2. A debug-only probe showed that reconnecting this missing semantic path does
   change runtime behavior in the expected direction.

However:

1. Fixing only this issue was not enough to fully recover normal output.
2. So this is a real bug, but not yet the full story.

### 3.3 KV compaction content itself is not the main suspect

Debug-only validation showed:

1. After compaction, the kept KV content matches the intended selected content.
2. So the current evidence does not support "the compaction copy itself is
   corrupting KV values" as the main explanation.

This does not eliminate every layout-related issue, but it greatly lowers the
probability of a simple "KV content copied wrong" bug.

## 4. What is still only a hypothesis

The following statements are not yet final conclusions:

1. The formal vLLM runtime path will automatically fix output quality once its
   compatibility issues are repaired.
2. The remaining output corruption is caused by exactly one missing compatibility
   layer.
3. Async behavior is irrelevant.

The current hypothesis is narrower:

1. The formal path is architecturally more promising.
2. But it is currently blocked by multiple compatibility bugs before we can even
   reach a fair output-quality test.

## 5. Why the focus shifted to the formal path

At this point, the old/default runtime path has become a weak place to continue
stacking fixes:

1. It can run, but it can silently express the wrong compressed semantics.
2. We already found and fixed one real semantic propagation bug there.
3. Even after that, output quality still does not recover.

That suggests the old path is increasingly expensive to patch and hard to trust.

By contrast, the formal path looks conceptually closer to the right place to
express the compressed runtime semantics, but it still fails earlier due to
compatibility bugs.

This is why the repair strategy is now:

1. Treat the old path as evidence-gathering territory.
2. Treat the formal path as the main repair target.

## 6. Confirmed blockers on the formal path

The formal path is currently blocked by a chain of runtime compatibility issues.

These are confirmed blockers, not guesses:

1. Request-identity mapping mismatch
   - The runtime stages disagree about which request IDs they are talking about.
   - Result: downstream bookkeeping breaks before we can evaluate quality.

2. Sample / execute lifecycle mismatch
   - Once the first request-ID blocker is bypassed in debug mode, the runtime
     exposes another incompatibility in how sampling/execution state is managed.
   - Result: the run still crashes before reaching a valid end-to-end test.

3. Hook request-context mismatch
   - Another stage expects request state to exist in an old structure shape.
   - On the formal path that assumption no longer holds.
   - Result: compression hook preflight fails before the formal path can proceed.

These three blockers explain why the formal path has not yet produced a clean
"compression-on and output-normal" verdict.

## 7. What this means for the next repair phase

The next phase is no longer "guess broadly and patch the old route".

The next phase is:

1. Repair the formal-path compatibility chain step by step.
2. After each blocker is removed, rerun the same long-prefill compressed case.
3. Stop only when we get an end-to-end run where the model output is again normal
   and readable.

## 8. Practical priorities

Recommended order:

1. Formal-path request-context compatibility
2. Formal-path request-ID bookkeeping compatibility
3. Formal-path sampling/execution lifecycle compatibility
4. End-to-end long-prefill compressed run on the same probe case
5. Only if output is still bad after the formal path runs cleanly:
   - revisit selector/layout/reclaim semantics as the next tier

## 9. What should not be over-read from current evidence

The current evidence does not mean:

1. "The algorithm has been fully proven correct."
2. "All old-path bugs are now irrelevant."
3. "Only one line of code remains to fix."

The safe reading is:

1. We have passed an important milestone.
2. The main direction has narrowed.
3. The next debugging work should focus on formal-path compatibility rather than
   restarting the algorithm debate.

## 10. Success criterion for this phase

This phase is considered successful only when:

1. The formal path runs the long-prefill compression case end-to-end.
2. Compression really triggers.
3. The model output remains normal and readable instead of drifting into obvious
   corruption or pathological repetition.

## 11. New validation result (2026-03-15, later same session)

This section records the first end-to-end quality result on the formal path.

### 11.1 Confirmed good path

Configuration:

1. formal vLLM path enabled
2. synchronous scheduling forced
3. long-prefill compression case
4. `kv_budget = 7000`

Result:

1. compression really triggered multiple times
2. output stayed readable
3. output quality returned to the same rough level as the no-compression
   baseline on the same formal path
4. the previous catastrophic corruption did not appear

Reference outputs:

1. `debug/v2_formal_sync_compress_budget7000_good.jsonl`
2. `debug/v2_formal_sync_nocompress_budget12000_baseline.jsonl`

This is the first strong end-to-end evidence that:

1. the formal path can carry the compressed semantics correctly enough
2. the remaining corruption is not a blanket "formal path still broken"

### 11.2 Confirmed bad path

Configuration:

1. same formal vLLM path
2. same long-prefill compression case
3. same `kv_budget = 7000`
4. async scheduling enabled

Result:

1. compression still triggered normally
2. the run completed
3. but the tail of the output degraded into obvious repetitive drift

Reference output:

1. `debug/v2_formal_async_compress_budget7000_bad.jsonl`

### 11.3 What this proves

This narrows the remaining root cause substantially:

1. it is no longer reasonable to say "the algorithm is the problem"
2. it is no longer reasonable to say "the whole formal path is broken"
3. the strongest remaining suspect is now the async scheduling / async state
   propagation chain on the formal path

### 11.4 Practical reading

At this milestone, the most accurate statement is:

1. formal path + sync scheduling: quality recovered
2. formal path + async scheduling: still degrades

So the next repair target is not generic compression logic anymore.
It is the async lifecycle / event handoff path.

### 11.5 Rejected async hypothesis

One additional async-only hypothesis was tested and rejected:

1. hypothesis:
   - "the fix is simply to delay compression events until the later
     sample-output object instead of attaching them earlier"
2. result:
   - this made async behavior worse
   - the scheduler no longer learned in time that compression had already
     happened
   - repeated over-triggering appeared

This matters because it narrows the remaining async issue further:

1. the problem is not just "event is attached too early"
2. the scheduler still needs timely visibility of compression state
3. the remaining bug is more subtle than a simple event-retiming change
