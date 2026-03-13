# Runtime Compatibility Handoff (2026-03-13)

## Scope
This handoff covers **only low-risk runtime compatibility/state-consistency fixes** that are required for current TriAttention vLLM integration.

It does **not** include strategy-level selector changes or demo-specific fallback policy changes.

## Why These Fixes
- vLLM scheduler/request keys are not always `str` on all paths (can be `int` or keyed objects).
- Patched KV allocation wrapper must accept extra positional args in newer call shapes.
- New-request metadata fields differ across vLLM structures (`prefill_token_ids`, `prompt_token_ids`, `num_prompt_tokens`, `prompt_token_ids_len`).
- Compressed-request runner cache length should advance conservatively from local state to avoid stale scheduler estimate glitches after compression.

## Code Changes
- `TriAttention_vLLM/triattention_runtime/request_key_compat.py`
  - Accept `int` request ids and object-key request ids.
- `TriAttention_vLLM/triattention_runtime/effective_overrides.py`
  - Applied-event req-id extraction now accepts non-`None` ids (not only `str`).
- `TriAttention_vLLM/triattention_runtime/scheduler.py`
  - Compression-event req-id guard aligned to non-`None` ids.
- `TriAttention_vLLM/triattention_runtime/worker_reclaim_sync.py`
  - Worker reclaim event req-id guard aligned to non-`None` ids.
- `TriAttention_vLLM/triattention_runtime/integration_monkeypatch.py`
  - `_patched_kv_cache_allocate_slots(...)` now forwards `*args` to stay signature-compatible.
- `TriAttention_vLLM/triattention_runtime/runner_state_updates.py`
  - Robust new-request prefill-length extraction across field variants.
  - Resume handling supports both `resumed_req_ids` and `req_ids`.
  - For already-compressed requests, cache-length estimate advances from local state (`prev + scheduled_tokens`) by default, with env escape hatch:
    - `TRIATTN_RUNTIME_LEGACY_COMPRESSED_ESTIMATE=1` to restore legacy behavior.
- `TriAttention_vLLM/triattention_runtime/debug_trace.py`
  - Optional env-gated JSONL tracing utility used by runtime modules.
  - No output unless `TRIATTN_RUNTIME_TRACE_PATH` is set.

## Tests Added/Updated
- Added: `TriAttention_vLLM/tests_runtime/test_request_key_compat.py`
- Updated:
  - `TriAttention_vLLM/tests_runtime/test_integration_monkeypatch.py`
  - `TriAttention_vLLM/tests_runtime/test_runner_state_updates.py`

## Verification Run
Executed in `conda` env `dc`:

```bash
PYTHONPATH=. conda run -n dc pytest \
  tests_runtime/test_request_key_compat.py \
  tests_runtime/test_integration_monkeypatch.py \
  tests_runtime/test_runner_state_updates.py \
  tests_runtime/test_effective_overrides.py \
  -q
```

Result: `25 passed`.

## Notes For Teammates
- This patch is intentionally conservative and does not alter selector algorithm policy.
- If you need old compressed-estimate behavior for A/B checks, set:
  - `TRIATTN_RUNTIME_LEGACY_COMPRESSED_ESTIMATE=1`
