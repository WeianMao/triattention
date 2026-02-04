# API Changes: Per-Request State Isolation

## Quick Reference

### TriAttentionWrapper

#### New Methods

```python
# Register a new request
wrapper.register_request(request_id: str) -> None

# Cleanup completed request
wrapper.unregister_request(request_id: str) -> None

# Get list of active requests (excludes default)
wrapper.get_active_requests() -> list

# Get state summary for debugging
wrapper.get_request_state_summary(request_id: str) -> Optional[dict]
```

#### Modified Methods (Added `request_id` parameter)

```python
# Get compressor (now per-request)
wrapper.get_compressor(
    layer_idx: int,
    request_id: Optional[str] = None  # NEW: defaults to __default__
) -> TriAttentionCompressor

# Check if compression needed
wrapper.should_compress(
    layer_idx: int,
    seq_len: int,
    request_id: Optional[str] = None  # NEW
) -> bool

# Compress KV cache
wrapper.compress_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    layer_idx: int,
    request_id: Optional[str] = None  # NEW
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Reset state
wrapper.reset_all(
    request_id: Optional[str] = None  # NEW: reset specific request or all
) -> None
```

### PagedKVCacheCompressor

#### New Methods

```python
# Register a new request
compressor.register_request(request_id: str) -> None

# Cleanup completed request
compressor.unregister_request(request_id: str) -> None

# Get compressor for request (internal use)
compressor._get_compressor(request_id: Optional[str] = None) -> TriAttentionCompressor
```

#### Modified Methods

```python
# Compress paged cache
compressor.compress_paged_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    layer_idx: int,
    request_id: Optional[str] = None  # NEW
) -> Tuple[int, torch.Tensor]
```

## Usage Patterns

### Pattern 1: Single Request (Backward Compatible)

```python
# No changes needed - uses default request ID
wrapper = TriAttentionWrapper(config)

# All existing calls work as before
if wrapper.should_compress(layer_idx, seq_len):
    k, v, p = wrapper.compress_kv_cache(keys, values, positions, layer_idx)
```

### Pattern 2: Multiple Requests (Recommended)

```python
wrapper = TriAttentionWrapper(config)

# Register request when it starts
wrapper.register_request(request_id)

# Use request_id in all operations
for layer in layers:
    if wrapper.should_compress(layer, seq_len, request_id):
        k, v, p = wrapper.compress_kv_cache(
            keys, values, positions, layer, request_id
        )

# Cleanup when request completes
wrapper.unregister_request(request_id)
```

### Pattern 3: vLLM Integration

```python
class TriAttentionBackend:
    def __init__(self):
        self.wrapper = TriAttentionWrapper(config)

    def on_request_start(self, sequence_group):
        """Called when scheduler assigns slot to request."""
        request_id = sequence_group.request_id
        self.wrapper.register_request(request_id)

    def forward_layer(self, layer_idx, kv_cache, metadata):
        """Called during attention forward pass."""
        request_id = metadata.request_id
        seq_len = metadata.seq_len

        if self.wrapper.should_compress(layer_idx, seq_len, request_id):
            return self.wrapper.compress_kv_cache(
                kv_cache.key, kv_cache.value, metadata.positions,
                layer_idx, request_id
            )
        return kv_cache.key, kv_cache.value, metadata.positions

    def on_request_complete(self, sequence_group):
        """Called when request finishes or is cancelled."""
        request_id = sequence_group.request_id
        self.wrapper.unregister_request(request_id)

    def on_slot_reuse(self, old_request_id, new_request_id):
        """Called when KV cache slot is reused."""
        # Cleanup old request
        self.wrapper.unregister_request(old_request_id)
        # Register new request
        self.wrapper.register_request(new_request_id)
```

## State Isolation Guarantee

Each request maintains completely independent state:

```python
# Request 1
wrapper.register_request("req1")
comp1 = wrapper.get_compressor(0, "req1")
comp1.state.absolute_position = 100

# Request 2
wrapper.register_request("req2")
comp2 = wrapper.get_compressor(0, "req2")
comp2.state.absolute_position = 200

# States are isolated
assert comp1.state.absolute_position == 100  # unchanged
assert comp2.state.absolute_position == 200  # unchanged
```

## Migration Checklist

- [ ] Identify request ID source (e.g., `SequenceGroup.request_id`)
- [ ] Add `register_request()` call at request start
- [ ] Add `request_id` parameter to all API calls
- [ ] Add `unregister_request()` call at request completion
- [ ] Handle slot reuse (unregister old, register new)
- [ ] Test with concurrent requests
- [ ] Verify memory cleanup (no leaks)

## Error Handling

```python
# Invalid layer index
try:
    wrapper.get_compressor(999, request_id)
except ValueError as e:
    print(f"Invalid layer: {e}")
    # Output: Invalid layer_idx 999. Expected 0 <= layer_idx < 4

# Request not found (returns None for state summary)
summary = wrapper.get_request_state_summary("unknown_req")
assert summary is None
```

## Debugging

```python
# List active requests
active = wrapper.get_active_requests()
print(f"Active requests: {active}")

# Get state for specific request
summary = wrapper.get_request_state_summary("req1")
for layer_idx, state in summary.items():
    print(f"Layer {layer_idx}: {state}")
    # Output includes: absolute_position, compression_count, etc.

# Reset specific request for debugging
wrapper.reset_all("req1")
```
