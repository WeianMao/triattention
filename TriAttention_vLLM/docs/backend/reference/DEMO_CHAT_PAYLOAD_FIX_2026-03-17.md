# Demo Chat Payload Fix 2026-03-17

## Summary

We closed the remaining `serve/chat/tool-calling` corruption bug for the real OpenClaw-style payload on the TriAttention vLLM runtime path.

The final fix was **not** to downgrade vLLM scoring to the HF implementation. The efficient vLLM-side scoring path remains in place.

Instead, the final root causes were:

1. `selector` recent-window protection in live runtime was under-protecting the trailing window.
2. `slot_mapping` position shifting after compression was wrong for this runtime path.

With those fixed:

1. `keep index` alignment returned to near-identical / exact agreement in the relevant debug comparisons.
2. The real chat payload stopped degrading into zeros / spaces / repeated special tokens.
3. The real payload now produces normal `reasoning + tool_call` output again under compression.

## Final Root Causes

### 1. Recent window protection diverged in live runtime

In live `serve` runs, the selector could see `recent_count=0`, while fresh/offline selection protected the full trailing `window_size`.

That caused live runtime to drop recent tokens that should always have been protected, which changed the `keep index` set and made live selection diverge from HF/fresh selection.

The fix was to restore stable trailing-window semantics:

- always protect `min(window_size, total_tokens)` recent tokens
- do not tie that protection to transient `recent_unabsorbed_tokens` bookkeeping

This matches the intended selector behavior and does not add extra hot-path work.

### 2. Slot/position shifting after compression was corrupting continuation semantics

After compression, we still need `seq_lens` to reflect the compressed effective history length so attention masking and cache-length semantics stay correct.

But shifting runtime `positions` / slot-mapping inputs was wrong in the real `serve/chat/tool-calling` continuation path.

Evidence:

1. `disable_seq_override` did **not** help: output still degraded into long zero strings.
2. `disable_slot_shift` **did** help immediately: output recovered to normal `reasoning + tool_call`.

So the final runtime rule is:

- keep the `seq_lens` override
- do **not** shift slot-mapping positions by default

The old shift path is still retained behind a debug-only opt-in for future bisection, but it is no longer the default behavior.

## Important Non-Root-Causes That Were Ruled Out

These were real investigations, but they did **not** explain the final failure:

1. HF algorithm mismatch
   - HF baseline and HF `compress_once` on the same bad payload both remained normal.

2. `normalize_scores` mismatch
   - not the cause.

3. Chunked-prefill simulation mismatch
   - not the main cause.

4. Dense-vs-paged selector math alone
   - a real selector bug existed earlier (`group aggregation` default), and we fixed it,
   - but that was not the last remaining issue for the real chat payload.

5. Compaction content corruption
   - conservative compaction + content validation did not report corruption,
   - output still degraded before the slot/position fix.

## Evidence Chain

### Keep alignment

Before the final fixes, live `keep index` diverged significantly from HF/fresh comparison.

After forcing the full recent window, the multi-round keep comparison aligned:

- `avg_jaccard = 1.0`
- `min_jaccard = 1.0`

This established that selector divergence had been removed.

### Real payload output recovery

With the final default fixes in place and **no debug-only runtime toggles**:

- backend: TriAttention vLLM runtime
- model: `JunHowie/Qwen3-32B-GPTQ-Int4`
- stats: `demo/openclaw-demo/stats/qwen3_32b_int4_speckv_stats.pt`
- budget: `7000`
- API path: `/v1/chat/completions`
- payload: `weian_development/demo_debug/fixtures/openclaw_real_payload_20260313_2252.json` (model field patched to current local snapshot path)

the replay result was:

- `status_code = 200`
- `finish_reasons = ["tool_calls"]`
- `tool_delta_count = 11`
- output content was normal planning/reasoning text, not zeros / spaces / repeated special tokens

Result file:

- `/tmp/openclaw_real_payload_512_finalfix.out.json`

## Code Changes That Matter

### Runtime fixes

1. `TriAttention_vLLM/triattention_runtime/selector_hf.py`
   - trailing recent protection now consistently uses `window_size`

2. `TriAttention_vLLM/triattention_runtime/input_patch_vllm_backend.py`
   - slot/position shifting is disabled by default
   - legacy shift behavior remains behind debug-only opt-in:
     - `TRIATTN_DEBUG_ENABLE_SLOT_SHIFT=1`

### Tests updated

1. `TriAttention_vLLM/tests_runtime/test_input_patch_vllm_backend.py`
   - updated for the new default no-shift semantics
   - old shift-specific failure paths now require explicit debug opt-in

## Verification

Targeted runtime tests:

```bash
conda run -n trivllm python -m pytest -q \
  TriAttention_vLLM/tests_runtime/test_input_patch_vllm_backend.py \
  TriAttention_vLLM/tests_runtime/test_hook_impl.py \
  TriAttention_vLLM/tests_runtime/test_effective_overrides.py \
  TriAttention_vLLM/tests_runtime/test_input_adapter.py \
  TriAttention_vLLM/tests_runtime/test_runner.py
```

Result:

- `66 passed`

Additional targeted subset after the final slot/selector changes:

- `39 passed`

## Practical Takeaway

The remaining demo corruption was **not** due to the algorithm needing to be replaced with the slower HF-style path.

The correct fix was:

1. restore correct recent-window protection in live runtime selector semantics
2. keep `seq_lens` override
3. stop shifting runtime slot-mapping positions by default

That combination preserves the efficient vLLM-side implementation while making the real chat/tool payload behave correctly again.
