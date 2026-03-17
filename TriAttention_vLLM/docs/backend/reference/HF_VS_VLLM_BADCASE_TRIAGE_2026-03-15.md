# HF vs vLLM Bad-Case Triage

Updated: 2026-03-16
Status: Milestone reached
Owner: Codex
Branch: `codex/vllm-async-commit-fix-20260315`

## 1. Why this note exists

This note records the latest triage milestone for the long weekly-report bad case.

The key question was:

`Is the bad output caused by the compression algorithm itself, or by the vLLM implementation path?`

## 2. Current known-good and known-bad paths

### Known-good

1. The repaired `probe/eval` vLLM path is good on the long probe case.
2. The same bad weekly-report prompt is also good on the isolated HF compression probe.

### Still bad

1. The `demo serve/gateway` TriAttention path still degrades on the long weekly-report case.
2. A direct vLLM TriAttention run on that same prompt also degrades once compression is active.

## 3. What was tested in this round

We tested the same bad prompt content across two different implementation paths.

Prompt source:

1. `/tmp/openclaw_demo_prompt.txt`

This prompt is the long weekly-report payload used in the bad demo-style case.

### HF path

Implementation:

1. `weian_development/demo_debug/hf_prefill_manual_compression_probe.py`

Mode:

1. `baseline`
2. `compress_once`

Important setup:

1. model: `JunHowie/Qwen3-32B-GPTQ-Int4`
2. stats: `demo/openclaw-demo/stats/qwen3_32b_int4_speckv_stats.pt`
3. `kv_budget = 7000`
4. `top_k = 20`
5. same bad prompt content, directly injected
6. no extra math/AIME wrapper was used for this validation

### vLLM path

Direct TriAttention vLLM run on the same prompt content had already shown visible degradation once compression was active.

Reference artifact:

1. `/tmp/A_impl_on_demo_prompt_direct_generate.json`

## 4. Result summary

### HF baseline on the bad prompt

1. `prompt_tokens = 7794`
2. `compressed_prompt_tokens = 7794`
3. `output_tokens = 1000`
4. output remained readable
5. no obvious repetition collapse

Artifact:

1. `/tmp/hf_on_bad_prompt_probe/baseline.json`

### HF compress_once on the bad prompt

1. `prompt_tokens = 7794`
2. `compressed_prompt_tokens = 7000`
3. real compression happened: reclaimed `794` prompt tokens
4. `output_tokens = 1000`
5. output remained readable
6. no obvious repetition collapse

Artifact:

1. `/tmp/hf_on_bad_prompt_probe/compress_once.json`

### Direct vLLM TriAttention on the same prompt

1. compression path active
2. output degraded
3. visible repeated structure and bad tail behavior appeared

Artifact:

1. `/tmp/A_impl_on_demo_prompt_direct_generate.json`

### HF vs vLLM first-compression keep indices

We then aligned the important compression settings and directly compared the
first per-head keep indices.

Aligned settings:

1. same model: `JunHowie/Qwen3-32B-GPTQ-Int4`
2. same stats: `demo/openclaw-demo/stats/qwen3_32b_int4_speckv_stats.pt`
3. same prompt: `/tmp/openclaw_demo_prompt.txt`
4. same `kv_budget = 7000`
5. same per-head mode and aggregation:
   - `per_head`
   - `hf_aligned_global_per_head`
   - `score_aggregation = mean`
   - `layer_perhead_aggregation = max`
   - `per_layer_aggregation = max`
6. `normalize_scores` was also cross-checked and ruled out as the root cause

Artifacts:

1. HF keep dump: `/tmp/hf_first_keep_indices.json`
2. vLLM keep dump: `/tmp/vllm_first_keep_indices.jsonl`

Old observed state before the latest fixes:

1. HF first compression is at pure prefill boundary with `seq_len = 7794`
2. vLLM first successful compression is later, at `total_tokens = 7795`
3. removing vLLM's extra latest token does **not** make the keep sets align
4. per-head similarity is still only about `0.82` Jaccard on average
5. every head still differs by about `660-708` indices

Old implication:

1. the bad-case divergence is already present at or before the first keep-selection result
2. this is much stronger evidence than small config mismatches
3. the remaining bug range is now substantially narrowed to:
   - selector inputs before/at first compression
   - or selector logic itself on the vLLM path

### Selector fixes confirmed after the old mismatch

After the earlier mismatch was identified, we continued the selector-only debug
path and confirmed two real implementation problems:

1. `paged per-head` cross-layer aggregation in the vLLM selector path had been
   using `max` where the HF-aligned path should use cross-layer `mean`
2. the PyTorch fallback scoring path was not mathematically equivalent to the
   optimized K-rot scoring path
3. the runtime selector was loading / passing scoring stats and trig tables at
   low precision, which materially perturbed keep decisions on this long case

These have now been fixed in code:

1. `TriAttention_vLLM/triattention_runtime/selector_hf.py`
2. `TriAttention_vLLM/triattention/scoring.py`

Most importantly, after these fixes:

1. offline selector keep indices on the long bad prompt are now exactly aligned
   with HF:
   - `compare_hf.mean_jaccard = 1.0`
   - `compare_hf.mean_symdiff = 0`
2. paged-vs-dense selector parity unit tests pass:
   - `TriAttention_vLLM/tests_runtime/test_hook_impl.py -k paged_matches_dense_with_normalize`
   - result: `2 passed`

Artifacts:

1. old offline compare: `/tmp/offline_vllm_selector_on_hf_keys_round0.json`
2. new offline compare after fixes:
   `/tmp/offline_vllm_selector_on_hf_keys_round0_after_fp32all_local.json`

New implication:

1. selector-side math is no longer the leading suspect
2. paged-vs-dense mismatch is no longer the leading suspect
3. if keep indices still differ in a fresh runtime run, the remaining gap is now
   more likely in runtime integration / stale-path / request execution wiring
   rather than selector semantics themselves

### `req_state_not_found` is real, but likely secondary for this bad case

We also investigated the repeated `req_state_not_found` skip at vLLM step 1.

Artifacts:

1. `/tmp/vllm_req_state_missing_trace.jsonl`
2. `/tmp/vllm_req_state_missing_trace2.jsonl`

Observed:

1. step 1 signal fires at the pure prefill boundary
2. compression is skipped with `req_state_not_found`
3. at that exact moment:
   - `req_states` exists
   - `req_states.req_id_to_index` exists
   - but the dict is still empty
4. so the failure is not "all state missing"
5. it is specifically "request id not yet registered into the req-id map at first compression boundary"

This confirms `req_state_not_found` is a real bug.

However, a debug-only attempt to work around this did **not** materially improve
the long bad-case output, which still degraded badly.

Implication:

1. `req_state_not_found` is a true vLLM integration bug
2. but for this weekly-report bad case, it currently looks like a secondary bug
3. it does **not** explain the full output corruption by itself

## 5. Current interpretation after the new selector fixes

The stronger current conclusion is:

1. the weekly-report bad case is still not evidence that the compression
   algorithm itself is wrong
2. the same prompt is normal on the HF compression probe
3. the offline vLLM selector is now also exactly aligned with HF on the first
   keep result
4. therefore the remaining gap has been pushed further down into the runtime
   execution path

## 6. What has been ruled out

### Not ruled in as root cause

1. prompt domain mismatch between calibration traces and inference prompt type
2. prompt length ratio alone
3. "the bad prompt is inherently toxic even without compression"

### Strongly ruled out

1. obvious model/stats shape mismatch such as using 8B stats on 32B inference

Current stats sanity check:

1. stats metadata model path points to `Qwen3-32B-INT4`
2. stats shape is `64 layers x 64 heads`
3. this is structurally consistent with the current 32B model family

Important nuance:

1. stats metadata still shows a math-style trace/prompt provenance
2. this is a configuration smell
3. but based on prior ablations and current evidence, it is not yet the leading explanation for the current bad output

## 7. Most useful next step

The next step should not be broad trial-and-error debugging.

Instead, the most useful plan is:

1. use HF as the reference implementation
2. compare vLLM vs HF at a few carefully chosen semantic checkpoints
3. narrow the bug to the first point where the two paths stop agreeing

Refined priority after the new fixes:

1. rerun a fresh runtime first-keep dump with the patched selector
2. verify that runtime first keep now matches the HF keep dump
3. if runtime first keep also aligns, move directly to the bad-case output check
4. only if runtime first keep still diverges, continue inside runtime wiring
   around request execution / integration boundaries

## 8. Practical decision implied by this note

Current decision:

1. continue prioritizing vLLM implementation debugging
2. do not pivot back to "algorithm is probably broken" based on the current evidence

## 9. New narrowing results after long-prompt checkpoint alignment

### Correct long prompt vs earlier mistaken short prompt

We found that one intermediate comparison had accidentally used the wrong prompt file:

1. wrong short prompt: `/tmp/openclaw_demo_prompt.txt`
   - only about `226` tokens
2. correct long prompt: `/tmp/openclaw_like_prompt.txt`
   - the real long weekly-report bad case

After correcting this, all later HF/vLLM comparisons in this section use the same
long prompt.

### First compression boundary is now aligned

After fixing the earlier paged per-head cross-layer aggregation mismatch
(`max` vs HF-aligned `mean` across layers), we re-ran the first-compression dump
on the correct long prompt.

Observed:

1. HF first compression:
   - pure prefill boundary
   - `seq_len = 7765`
2. vLLM first compression:
   - also at pure prefill boundary
   - `prefill_len = 7765`
   - `total_tokens = 7765`

Implication:

1. the remaining mismatch is not explained by "HF compresses at prefill, vLLM compresses later"
2. the old timing explanation is no longer sufficient for this long-prompt checkpoint

### K inputs already match closely

We then dumped sampled K vectors from HF and vLLM at that same first compression
boundary.

Artifacts:

1. HF sampled K dump: `/tmp/hf_longprompt_k_sample.json`
2. vLLM sampled K dump: `/tmp/vllm_longprompt_k_sample.json`

Observed:

1. sample positions are identical
2. sampled K vectors are essentially identical up to expected floating-point noise
3. example comparison summary:
   - layer 0 / head 0: `max_abs_diff = 0.0`
   - layer 0 / head 1: `max_abs_diff = 0.0`
   - layer 1 / head 0: `max_abs_diff = 0.015625`
   - layer 1 / head 1: `max_abs_diff = 0.0078125`

Implication:

1. the remaining mismatch is not in KV transport / gather / layout before scoring
2. the bug range narrows to:
   - scoring semantics
   - or later score-to-keep logic

### Keep indices are still clearly different

Even after:

1. same long prompt
2. same pure prefill first-compression point
3. same K sample behavior
4. same fixed cross-layer aggregation mode

the first keep sets still differ substantially.

Observed:

1. average Jaccard across heads is still only about `0.83`
2. average symmetric difference is still about `1260`
3. this is far beyond tiny numeric tie noise

Implication:

1. the mismatch happens after K gathering and before-or-at keep selection

### The main score mismatch was implementation/input-chain, not proof that the optimized math is wrong

After re-checking against the design notes and rebuilding the comparison more
carefully, the earlier "scoring theory is wrong" interpretation was too strong.

What we actually confirmed:

1. the optimized `K_rot` scoring path is intended to be mathematically equivalent
   to the HF reference path
2. two concrete implementation/input-chain bugs were breaking the preconditions of
   that equivalence:
   - `rope_style` was not being forwarded into the Triton kernel, so Qwen-style
     `half` layout was being interpreted as `interleaved`
   - runtime was not passing the real `model_path`, so `TriAttentionCompressor`
     silently fell back to legacy/default RoPE frequencies instead of the model's
     actual `inv_freq`
3. after fixing both of those issues, the raw-score mismatch collapsed from
   very large errors to small residual differences on the sampled positions

Implication:

1. the optimized scoring path should not be replaced with the slower HF-style
   unrotation implementation
2. the right fix direction is to repair the optimized path's inputs and layout
   assumptions, not to downgrade the algorithm

### Effect of the rope-style + real-model-frequency fixes

Once the two bugs above were fixed:

1. vLLM first-keep Jaccard against the original HF long-prompt dump improved from
   about `0.83` to about `0.96`
2. this was already a large step and showed that those two bugs were major causes
3. however, there was still a visible residual mismatch

### The remaining mismatch is mostly first-compression timing, not another large scoring bug

The next high-value check was to align HF with vLLM's *actual* first compression
timing.

Observed:

1. HF original long-prompt first compression happened at `seq_len = 7765`
2. vLLM's first real compression for this case happened at `total_tokens = 7766`
   with `prefill_len = 7765`
3. when HF was rerun on the same prompt plus one extra token (so `seq_len = 7766`)
   and compared against vLLM at that same compression point:
   - average Jaccard rose to about `0.993`
   - average symmetric difference fell to about `48`
   - per-head mismatch ratio fell to about `0.5% - 0.9%`

Implication:

1. the dominant remaining difference is that vLLM's first compression is still
   happening one token later than the original HF probe
2. once that timing is aligned, the keep sets are already in the "very close /
   likely numeric-tie-level" regime
3. the current residual mismatch no longer looks like a large selector/scoring bug

### vLLM internal paged-vs-dense check does not currently point to a separate paged bug

We also ran a short vLLM self-check with:

1. same long prompt
2. same first compression boundary
3. `TRIATTN_DEBUG_COMPARE_PAGED_DENSE_KEEP=1`

The run completed without a selector-compare failure.

Implication:

1. after the earlier aggregation fix, there is no new evidence that the main
   remaining mismatch is caused by paged chunk-merging alone
2. the stronger explanation remains:
   - paged and dense are both consuming a common scoring stack
   - and that common scoring stack is still semantically misaligned with HF

## 10. Updated root-cause ranking

Highest-priority remaining causes:

1. vLLM first-compression timing is still one token later than the original HF probe
2. there may still be a small residual numeric / tie-breaking difference after
   timing alignment, but it is now much smaller
3. the previously found scoring-chain bugs (`rope_style`, real model frequency source)
   were real and major, and are now fixed

Still real but now lower priority for this specific bad case:

1. `req_state_not_found` on first compression boundary
2. earlier async-boundary bug

These are still genuine implementation bugs, but they no longer look like the
main explanation for the current long-prompt keep mismatch.

## 11. New narrowing on the real OpenClaw bad case

The latest round used the real normalized DocMind prompt:

1. `/tmp/openclaw_real_payload_rendered_prompt_docmind_normalized.txt`
2. long prompt length around `15165` tokens
3. real serve trace shows first compression on a chunked-prefill boundary at
   `round_start = total_tokens = prefill_len = 10240`

This immediately changed the debugging target:

1. the relevant first-compression scene is not the earlier `7765/7766` probe
2. it is the real serve chunked-prefill scene at `10240`

### HF chunked-prefill K drift does exist

I simulated HF prefill in `2048`-token chunks up to the same `10240` boundary.

Observed:

1. sampled deep-layer K values from `HF chunked-prefill` drift away from the
   earlier HF monolithic prefill
2. the size of that deep-layer drift is surprisingly close to the runtime
   sampled K drift

Implication:

1. chunked prefill does change deep-layer activations
2. so "runtime deep layers differ from monolithic HF" is not by itself enough
   to prove a runtime-only bug

### But chunked prefill itself is still not the main explanation

This is the most important new result.

I then took the `HF chunked-prefill` KV at the same `10240` point and ran the
vLLM selector offline on top of those HF-generated keys.

Observed:

1. `HF chunked-prefill + vLLM selector` stays almost identical to
   `HF monolithic`
2. `HF chunked-prefill + vLLM selector` is still very far from the real
   runtime keep set

Measured:

1. `chunked HF selector` vs `HF monolithic`: `avg_jaccard ≈ 0.9993`
2. `chunked HF selector` vs runtime first keep: `avg_jaccard ≈ 0.4946`

Implication:

1. chunked prefill semantics alone do **not** explain the bad runtime keep
2. the remaining mismatch is still runtime-specific
3. the problem is now much more likely to be in the real runtime input chain
   that feeds the selector during serve/chunked-prefill execution

### Offline selector math remains consistent on identical HF-generated KV

Another strong cross-check:

1. offline `paged selector` on HF-generated KV
2. offline `dense selector` on the same HF-generated KV
3. HF reference keep

All three agree exactly in the tested `10240` scene.

Implication:

1. selector math in isolation is no longer the lead suspect
2. the highest-priority remaining direction is:
   - what exact runtime K / state / score inputs the live serve path provides
   - not whether the offline selector formula itself is wrong

### Direct evidence that runtime online keep is not reproduced by offline selector on the same dumped runtime K

I also ran a direct-backend bad-path probe and dumped, from the same live
request:

1. the runtime first keep JSON
2. the full dense K tensors for all 64 layers

Then I replayed the selector offline on those dumped runtime dense K tensors.

Observed:

1. offline `dense selector` on dumped runtime K and offline `paged selector`
   on the exact same dumped runtime K agree perfectly with each other
2. but both still differ substantially from the live runtime keep dump

Measured on the direct-backend `8192` first-compression scene:

1. `offline dense` vs `offline paged`: `avg_jaccard = 1.0`
2. `offline dense/paged` vs live runtime keep: `avg_jaccard ≈ 0.7286`

Implication:

1. the remaining mismatch is **not** explained by:
   - dumped runtime K values alone
   - dense vs paged selector math
2. the bug is now strongly localized to an online-only runtime path:
   - either hidden selector inputs/state
   - or the exact online invocation path that produces the live keep result
