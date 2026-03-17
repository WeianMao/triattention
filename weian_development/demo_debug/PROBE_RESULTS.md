# Demo Serve Path Probe Results (2026-03-07)

## Purpose

Validate whether current `master` can run TriAttention in `vllm serve` mode via:

- `VLLM_PLUGINS=triattention`
- `--attention-backend CUSTOM`

without changing production code.

## Diagnostic Scripts

- `weian_development/demo_debug/verify_custom_backend_registration.py`
- `weian_development/demo_debug/stream_stutter_probe.py`

## Reproduction Outcome

`verify_custom_backend_registration.py` reports:

- `Legacy V1 backend plugin registration is retired` => `true`
- `Backend CUSTOM must be registered before use` => `true`
- `Engine core initialization failed` => `true`

Conclusion:

- `custom_backend_not_registered_via_retired_plugin`

## Evidence Files (local run artifacts)

- `weian_development/demo_debug/artifacts/custom_backend_probe_report.json`
- `weian_development/demo_debug/artifacts/custom_backend_probe.log`

## Impact on Current Demo Task

Before fixing this startup blocker, we cannot run a valid
`baseline vs triattention` streaming stutter comparison in current `master`
serve-mode path, because the TriAttention CUSTOM backend never becomes active.
