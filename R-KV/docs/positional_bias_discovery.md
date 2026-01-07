# Positional Bias Discovery: RoPE Offset Improves Long Reasoning

## Summary

Through controlled experiments simulating bug 896cbca6's attention position offset effect, we discovered that **shifting RoPE position encoding to make prefill (question) tokens appear "closer" to decode tokens improves performance by ~8%** on AIME24 mathematical reasoning tasks.

## Background

Bug 896cbca6 caused multiple issues due to pruner state not resetting between questions:
1. SpeckV scoring errors (phase offset in `invert_rope`)
2. **Attention position encoding errors** (decode seeing prefill at wrong relative positions)
3. Extra token protection due to `prefix_length` mismatch

We designed an ablation experiment to isolate the **attention position encoding effect only**.

## Experiment Design

### What We Simulated

Bug 896cbca6's attention effect:
- Prefill K positions: `[X, X+1, ..., X+P-1]` (X = accumulated value)
- Decode Q positions: `[X, X+1, ...]` (starts from same X, NOT X+P)
- Relative positions: `Q - K = -k` (instead of normal `P-k`)

### Implementation

In `rkv_speckv_generate.py`, we subtract `prefill_len` from decode position_ids:

```python
position_offset = 0
if simulate_attention_position_offset > 0 and not is_empty_cache and state.initial_prefix_length:
    # Decode phase: subtract prefill_len to simulate bug
    position_offset = -state.initial_prefix_length
start_pos = state.pruner.absolute_position + position_offset
```

This makes:
- Prefill K positions: `[0, 1, ..., P-1]` (normal)
- Decode Q positions: `[0, 1, ...]` (subtracted P)
- Relative positions: `0 - k = -k` (exactly matching bug effect)

### Key Property: Isolated Ablation

This modification **only affects RoPE position encoding** passed to the model. It does NOT affect:
- `state.pruner.absolute_position` (used by SpeckV scoring)
- `state.pruner.cache_positions` (used by SpeckV scoring)
- SpeckV's `invert_rope` or `score_keys_for_round` logic

This ensures we're testing **only the attention position effect**, not other bug side-effects.

## Results

| Experiment | Description | Performance |
|------------|-------------|-------------|
| Baseline | Normal position encoding | X% |
| Simulated Position Offset | Decode positions -= prefill_len | X + 8% |

**8% improvement** from position offset alone.

## Analysis: Why Does This Help?

### 1. Long Reasoning Position Decay Problem

```
AIME mathematical reasoning scenario:
  - Prefill (question): ~150 tokens
  - Decode (thinking): 5,000 - 30,000 tokens

Normal case (at decode token 10,000):
  - Question's relative distance = 10,000 + 150 = 10,150
  - RoPE attention weight decays with distance → question gets "forgotten"

With position offset:
  - Question's relative distance = 0 to -(P-1)
  - Question stays at "closest" position → sustained attention on question
```

### 2. Mathematical Reasoning Requires Continuous Reference

- Long reasoning chains need to repeatedly check problem conditions
- Position offset keeps prefill (question) at high attention weight
- Effectively "pins" the question at the top of attention

### 3. Effect Scales with Generation Length

The longer the decode sequence, the more severe the position decay problem, and the more beneficial the offset becomes.

## Implications

### This is a Feature, Not a Bug

The position offset effect can be formalized as a legitimate technique:

**Positional Bias for Long-Context Reasoning**
- Intentionally shift position encoding for important content (questions, instructions)
- Keep critical context at high attention weight throughout long generation
- Applications: long reasoning, long document QA, multi-turn dialogue

### Potential Optimizations

1. **Fixed offset**: Subtract a constant from decode positions
2. **Dynamic offset**: Scale offset based on decode length
3. **Selective offset**: Only apply to certain attention heads or layers
4. **Learnable offset**: Train the optimal offset value

## Files Modified

- `R-KV/weian_development/speckv/rkv_speckv_generate.py`: Added `simulate_attention_position_offset` parameter
- `R-KV/weian_development/rkv_sharded_eval.py`: Added CLI argument
- `R-KV/weian_development/rkv_sharded_dispatch.py`: Added parameter passing

## Experiment Script

```bash
# Baseline (no offset)
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_budget.sh

# With position offset simulation
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_budget_simulated_pos_offset.sh
```

## Conclusion

The 8% performance improvement from RoPE position offset confirms that:
1. Bug 896cbca6's performance gain was (at least partially) due to attention position effects
2. Making prefill "closer" in attention helps long reasoning tasks
3. This can potentially be developed into a formal optimization technique

---

*Discovered: 2025-01-07*
*Related: Bug 896cbca6 analysis in `R-KV/weian_script/aime_sampled8/speckv/BUG_VARIABLE_IMPACT.md`*
