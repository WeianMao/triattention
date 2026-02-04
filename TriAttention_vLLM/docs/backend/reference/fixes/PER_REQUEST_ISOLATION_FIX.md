# Per-Request State Isolation Fix

## Problem (P1 Issue)

The original implementation in `triattention/vllm_integration.py` had a critical concurrency bug:

```python
class TriAttentionWrapper:
    def __init__(self, config, enabled_layers=None):
        # PROBLEM: All requests share the same compressors!
        self.compressors: Dict[int, TriAttentionCompressor] = {}
```

**Impact**: When vLLM processes multiple requests in the same batch, they share compressor state:
- `absolute_position` - tracks sequence position (per-request)
- `position_indices` - tracks which tokens are kept (per-request)
- `prefill_length` - initial sequence length (per-request)

This causes **state interference**: Request A's compression affects Request B's state, leading to incorrect KV cache compression.

## Solution

### 1. Per-Request Storage Structure

Changed from per-layer to per-request-per-layer storage:

```python
# Before: Dict[layer_idx, compressor]
self.compressors: Dict[int, TriAttentionCompressor] = {}

# After: Dict[request_id, Dict[layer_idx, compressor]]
self.request_compressors: Dict[str, Dict[int, TriAttentionCompressor]] = {}
```

### 2. Request Lifecycle Management

Added explicit request management methods:

```python
# Register new request (initializes state)
wrapper.register_request(request_id)

# Process request (maintains isolated state)
wrapper.compress_kv_cache(..., request_id=request_id)

# Cleanup when request completes
wrapper.unregister_request(request_id)
```

### 3. State Isolation Guarantees

Each request now maintains completely independent state:

```python
# Request 1
comp1 = wrapper.get_compressor(layer_idx=0, request_id="req1")
comp1.state.absolute_position = 100  # Only affects req1

# Request 2
comp2 = wrapper.get_compressor(layer_idx=0, request_id="req2")
comp2.state.absolute_position == 0  # Unaffected by req1
```

### 4. Layer Index Validation

Added validation to catch invalid layer indices early:

```python
def get_compressor(self, layer_idx: int, request_id: Optional[str] = None):
    if self.config.num_layers is not None:
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise ValueError(f"Invalid layer_idx {layer_idx}. ...")
```

### 5. Backward Compatibility

API remains compatible with existing code:

```python
# Old code (no request_id) - still works
wrapper.compress_kv_cache(keys, values, positions, layer_idx)

# New code (with request_id) - isolated state
wrapper.compress_kv_cache(keys, values, positions, layer_idx, request_id="req1")
```

Uses default request ID (`__default__`) when `request_id=None`.

## Updated API

### TriAttentionWrapper

**New Methods:**
- `register_request(request_id: str)` - Initialize request state
- `unregister_request(request_id: str)` - Cleanup request state
- `get_active_requests() -> list` - List active request IDs
- `get_request_state_summary(request_id: str) -> dict` - Get state summary

**Modified Methods:**
- `get_compressor(layer_idx, request_id=None)` - Added request_id parameter
- `should_compress(layer_idx, seq_len, request_id=None)` - Added request_id parameter
- `compress_kv_cache(..., request_id=None)` - Added request_id parameter
- `reset_all(request_id=None)` - Added request_id parameter (reset specific request)

### PagedKVCacheCompressor

**New Methods:**
- `register_request(request_id: str)` - Initialize request state
- `unregister_request(request_id: str)` - Cleanup request state
- `_get_compressor(request_id=None)` - Get per-request compressor

**Modified Methods:**
- `compress_paged_cache(..., request_id=None)` - Added request_id parameter

## Usage Example

```python
from triattention.config import TriAttentionConfig
from triattention.vllm_integration import TriAttentionWrapper

# Initialize wrapper
config = TriAttentionConfig(...)
wrapper = TriAttentionWrapper(config)

# Process multiple requests concurrently
for request_id in ["req1", "req2", "req3"]:
    # Register request
    wrapper.register_request(request_id)

    # Process through layers
    for layer_idx in range(num_layers):
        if wrapper.should_compress(layer_idx, seq_len, request_id):
            keys, values, positions = wrapper.compress_kv_cache(
                key_cache, value_cache, cache_positions,
                layer_idx, request_id
            )

    # Cleanup when done
    wrapper.unregister_request(request_id)
```

## Integration with vLLM

**When to call lifecycle methods:**

1. **`register_request()`** - Call when:
   - New request starts (after scheduler assigns slot)
   - Slot is being reused for new request

2. **`compress_kv_cache()`** - Call during:
   - Forward pass in each layer
   - Pass request ID from vLLM's `SequenceGroupMetadata`

3. **`unregister_request()`** - Call when:
   - Request completes successfully
   - Request is cancelled or fails
   - Before slot reuse (before new `register_request()`)

**Request ID Source:**
- vLLM's `SequenceGroup` has unique `request_id` field
- Pass this to all TriAttention calls for proper state isolation

## Verification

Run verification script to confirm fix:

```bash
python TriAttention_vLLM/test/verify_per_request_fix.py
```

Expected output:
```
============================================================
Per-Request State Isolation Fix Verification
============================================================

Test 1: Basic per-request isolation
  ✓ Different requests have separate compressor instances
  ✓ State changes in one request don't affect others
  PASSED

[... 6 tests total ...]

Results: 6/6 tests passed
============================================================
```

## Memory Management

**Automatic cleanup benefits:**
- Prevents memory leaks from abandoned requests
- Ensures state freshness for slot reuse
- Explicit lifecycle makes debugging easier

**Memory footprint per request:**
- Per layer: 1 `CompressionState` + frequency stats
- Typical: ~100KB per request (4 layers × 25KB)
- Cleanup reclaims memory immediately on `unregister_request()`

## Testing

Comprehensive test suite in `test/test_per_request_isolation.py` covers:
- State isolation between requests
- State independence across layers
- Proper cleanup on unregister
- Layer index validation
- Backward compatibility
- Memory management
- Auto-registration behavior
- Re-registration state reset

Run with:
```bash
pytest test/test_per_request_isolation.py -v
```

Or standalone:
```bash
python test/verify_per_request_fix.py
```

## Migration Guide

**For existing code without request IDs:**
- No changes required - uses default request ID
- Behavior unchanged for single-request scenarios

**For multi-request scenarios:**
- Add `request_id` parameter to all API calls
- Add lifecycle calls (`register_request`, `unregister_request`)
- Extract request ID from vLLM's `SequenceGroup` object

**Example migration:**

```python
# Before (shared state - BUGGY with concurrent requests)
wrapper.compress_kv_cache(keys, values, positions, layer_idx)

# After (isolated state - CORRECT)
wrapper.compress_kv_cache(keys, values, positions, layer_idx, request_id)
```

## Implementation Details

**Key design decisions:**

1. **Auto-registration**: Compressors auto-register on first access if not already registered
   - Simplifies API for single-request cases
   - Explicit registration still recommended for clarity

2. **Default request ID**: Uses `__default__` for backward compatibility
   - Excluded from `get_active_requests()` output
   - Single request usage remains simple

3. **Layer validation**: Validates layer indices to catch integration errors early
   - Only validates if `num_layers` configured
   - Raises `ValueError` with clear message

4. **State reset on re-registration**: Re-registering existing request resets its state
   - Prevents stale state from previous use
   - Useful for slot reuse scenarios

## Files Modified

- `triattention/vllm_integration.py` - Core implementation
- `test/test_per_request_isolation.py` - Comprehensive test suite
- `test/verify_per_request_fix.py` - Standalone verification script
- `PER_REQUEST_ISOLATION_FIX.md` - This document

## Status

**✅ P1 Issue Resolved**

- Per-request state isolation implemented
- All verification tests passing
- Backward compatibility maintained
- Memory management verified
- Ready for integration with vLLM multi-request batching
