# Legacy V0/V1 Code Archive (2026-02-20)

This directory stores legacy TriAttention code paths that are not part of the
current V2 default implementation.

Archived items in this snapshot:

1. `evaluation/runner/vllm_triattention_runner.py`
2. `evaluation/dispatch/configs/triattention_aime24.yaml`
3. `evaluation/dispatch/configs/triattention_aime24_seed42.yaml`
4. `examples/`
5. `test/`
6. `root_legacy_tests/` (top-level legacy test scripts)

Notes:

- V2 active path remains under:
  - `triattention_v2/`
  - `evaluation/runner/vllm_triattention_v2_runner.py`
  - `evaluation/dispatch/configs/triattention_v2_*.yaml`
- This is a `git mv` archive so history remains traceable.
