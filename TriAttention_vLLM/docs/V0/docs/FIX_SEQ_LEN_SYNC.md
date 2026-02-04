# Fix: seq_len Synchronization in vLLM Integration

**Date**: 2026-02-02
**Status**: ✓ FIXED
**Files Modified**:
- `triattention/state.py`
- `triattention/compressor.py`

## Problem Description

### Symptom
Compression was triggering on **every decode step** instead of every `divide_length` tokens:

```
[TriAttention] Compressing: seq_len=320 -> budget=256
[TriAttention] Compressing: seq_len=321 -> budget=256  # WRONG! Should wait until 320
[TriAttention] Compressing: seq_len=322 -> budget=256  # WRONG!
[TriAttention] Compressing: seq_len=323 -> budget=256  # WRONG!
```

### Root Cause
After compression, vLLM's internal `seq_len` tracking doesn't update to reflect the compressed cache size:

1. **Initial state**: Prefill with 320 tokens
2. **First compression**: Cache reduced from 320 → 256 tokens
3. **Next decode step**: vLLM reports `seq_len=321` (doesn't know cache is only 256 tokens)
4. **should_compress() check**: Used vLLM's `seq_len` directly → always triggers (321 >= 320)

**Why vLLM doesn't update**:
- vLLM's `seq_len` is tracked by the scheduler
- Our compression happens in the attention layer (after scheduler decisions)
- We cannot modify vLLM's scheduler state without invasive changes

## Solution: Internal State Tracking

### Design Principle
**Track the true cache length internally**, independent of vLLM's `seq_len`:

1. **Initialization** (prefill):
   - Use vLLM's `seq_len` to initialize state
   - Set `current_cache_len = seq_len`

2. **Incremental Updates** (decode):
   - Calculate new tokens: `new_tokens = vllm_seq_len - absolute_position`
   - Update internal state: `current_cache_len += new_tokens`
   - Use `current_cache_len` for compression decisions

3. **After Compression**:
   - Update internal state: `current_cache_len = budget`
   - vLLM's `seq_len` continues incrementing (we ignore it)

### Implementation Changes

#### 1. State Tracking (`triattention/state.py`)

**Before**:
```python
def should_compress(self, current_len: int) -> bool:
    # Used vLLM's seq_len directly
    effective_size = current_len
    trigger_threshold = self.config.kv_budget + self.config.divide_length
    return effective_size >= trigger_threshold
```

**After**:
```python
def should_compress(self, current_len: int) -> bool:
    # Auto-initialize on first call
    if self.absolute_position == 0:
        self.initialize(current_len)
        effective_cache_len = current_len
    else:
        # Track new tokens and update internal state
        new_tokens = current_len - self.absolute_position
        if new_tokens > 0:
            self.append_tokens(new_tokens)
        # Use internally tracked cache length
        effective_cache_len = self.current_cache_len

    # Calculate effective size (excluding protected prefill if enabled)
    if self.config.protect_prefill:
        effective_size = max(0, effective_cache_len - self.prefill_length)
    else:
        effective_size = effective_cache_len

    trigger_threshold = self.config.kv_budget + self.config.divide_length
    return effective_size >= trigger_threshold
```

**Key Changes**:
- Auto-initialize state on first call (prefill)
- Track incremental tokens using `vllm_seq_len - absolute_position`
- Use `current_cache_len` (internal tracking) instead of `current_len` (vLLM's seq_len)

#### 2. Compressor Update (`triattention/compressor.py`)

Added documentation clarifying that state tracking happens in `should_compress()`:

```python
# Check if compression is needed
# NOTE: should_compress() will auto-initialize state on first call and track incremental updates
if not self.state.should_compress(seq_len):
    # No compression needed, return as-is with identity indices
    keep_indices = torch.arange(seq_len, device=key_states.device)
    return key_states, value_states, keep_indices
```

## Verification

### Test 1: State Tracking Logic (`test_seq_len_sync.py`)
Simulates vLLM behavior without full initialization:

```bash
python test_seq_len_sync.py
```

**Results**:
```
1. Prefill: vLLM seq_len=320
   -> Compression triggered (320 >= 320)
   -> Cache reduced to 256

Step 10: vLLM seq_len=330, cache_len=266, compress=False
Step 20: vLLM seq_len=340, cache_len=276, compress=False
...
Step 64: vLLM seq_len=384, cache_len=320, compress=True  ✓
   -> Compression triggered (320 >= 320)
   -> Cache reduced to 256

✓ TEST PASSED: Compression triggers at correct intervals
```

### Test 2: Minimal Simulation (`test_compression_fix_minimal.py`)
More detailed step-by-step verification:

```bash
python test_compression_fix_minimal.py
```

**Results**:
```
Compression log:
  Compression 1: step=64, vLLM seq_len=384, actual cache_len=320

Verifying compression intervals:
  ✓ Interval = 64 steps (matches divide_length)

Key Observations:
  - vLLM seq_len keeps growing (doesn't update after compression)
  - Internal cache_len fluctuates between 256 and 320
  - Compression triggers every ~64 tokens
  - No compression on every decode step (bug is FIXED!)
```

## Expected Behavior

### Correct Compression Timeline

**Configuration**: `kv_budget=256`, `divide_length=64`

| Step | vLLM seq_len | Internal cache_len | Action |
|------|--------------|-------------------|--------|
| Prefill | 320 | 320 | ✓ Compress (320 >= 320) → 256 |
| Decode 1 | 321 | 257 | Skip |
| Decode 2 | 322 | 258 | Skip |
| ... | ... | ... | ... |
| Decode 64 | 384 | 320 | ✓ Compress (320 >= 320) → 256 |
| Decode 65 | 385 | 257 | Skip |
| ... | ... | ... | ... |
| Decode 128 | 448 | 320 | ✓ Compress (320 >= 320) → 256 |

### Log Output (After Fix)

**Before** (bug):
```
[TriAttention] Compressing: seq_len=320 -> budget=256
[TriAttention] Compressing: seq_len=321 -> budget=256  # Every step!
[TriAttention] Compressing: seq_len=322 -> budget=256
[TriAttention] Compressing: seq_len=323 -> budget=256
```

**After** (fixed):
```
[TriAttention] Compressing: seq_len=320 -> budget=256
[TriAttention] Compressing: seq_len=384 -> budget=256  # Every 64 tokens!
[TriAttention] Compressing: seq_len=448 -> budget=256
```

## Impact

### Performance
- **Before**: Compression overhead on every decode step
- **After**: Compression overhead only every `divide_length` tokens
- **Speedup**: ~64x reduction in compression calls (for `divide_length=64`)

### Correctness
- Proper R-KV slack mode behavior: cache fluctuates in `[budget, budget + divide_length]`
- Aligns with original TriAttention/SpeckV algorithm design
- Matches R-KV reference implementation behavior

## Technical Notes

### Why Not Modify vLLM's seq_len?

We considered several approaches:

1. **Update vLLM scheduler state** (rejected):
   - Requires invasive changes to vLLM core
   - Breaks encapsulation
   - Fragile across vLLM versions

2. **Hook into scheduler callbacks** (rejected):
   - Complex integration
   - Version-specific APIs
   - Still requires understanding vLLM internals

3. **✓ Internal state tracking** (chosen):
   - Non-invasive (all code in TriAttention module)
   - Version-agnostic (only uses public APIs)
   - Clean separation of concerns
   - Easy to maintain and test

### State Tracking Algorithm

**Invariants**:
- `absolute_position`: Tracks vLLM's reported seq_len (monotonically increasing)
- `current_cache_len`: Tracks actual cache length (fluctuates after compression)
- `new_tokens = vllm_seq_len - absolute_position`: Incremental tokens added

**Update Flow**:
```python
# On each should_compress() call:
new_tokens = current_len - self.absolute_position
if new_tokens > 0:
    self.current_cache_len += new_tokens
    self.absolute_position += new_tokens

# After compression:
self.current_cache_len = budget  # Reset to budget
# absolute_position stays the same (tracks vLLM's view)
```

## Files Modified

### `triattention/state.py`
- Modified `should_compress()` to auto-initialize state and track incremental updates
- Uses internal `current_cache_len` instead of external `current_len`

### `triattention/compressor.py`
- Added documentation comment about state auto-initialization

## Testing

### Automated Tests
1. `test_seq_len_sync.py`: State tracking logic verification
2. `test_compression_fix_minimal.py`: End-to-end simulation

### Manual Verification
Run with actual vLLM (when model path available):
```bash
python test_vllm_compression_intervals.py
```

## Related Issues

- **Original Issue**: Compression triggering on every decode step
- **Root Cause**: vLLM seq_len not synchronized with compressed cache
- **Fix Strategy**: Internal state tracking independent of vLLM

## References

- R-KV Paper: Slack mode with periodic compression
- TriAttention Design: `docs/design/optimization.md`
- vLLM Integration: `triattention/vllm_integration.py`
