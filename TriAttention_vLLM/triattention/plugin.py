"""vLLM plugin entrypoint for TriAttention runtime (V2) integration.

Default behavior:
- Install runtime scheduler/worker monkeypatches for TriAttention V2 path.
- Bridge legacy `TRIATTENTION_*` env vars into `TRIATTN_RUNTIME_*` when needed.

Legacy V1 custom backend registration is retired.
"""

from __future__ import annotations

import os


def _truthy(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _set_if_absent(target: str, source: str) -> None:
    if os.environ.get(target):
        return
    value = os.environ.get(source)
    if value is not None and value != "":
        os.environ[target] = value


def _bridge_legacy_env_to_runtime() -> None:
    # Core runtime controls.
    _set_if_absent("TRIATTN_RUNTIME_KV_BUDGET", "TRIATTENTION_KV_BUDGET")
    _set_if_absent("TRIATTN_RUNTIME_DIVIDE_LENGTH", "TRIATTENTION_DIVIDE_LENGTH")
    _set_if_absent("TRIATTN_RUNTIME_WINDOW_SIZE", "TRIATTENTION_WINDOW_SIZE")
    _set_if_absent("TRIATTN_RUNTIME_LOG_DECISIONS", "TRIATTENTION_LOG_DECISIONS")
    _set_if_absent("TRIATTN_RUNTIME_SPARSE_STATS_PATH", "TRIATTENTION_STATS_PATH")

    # Keep default runtime behavior strict enough for real compression runs.
    os.environ.setdefault("TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION", "true")
    os.environ.setdefault("TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM", "true")
    os.environ.setdefault("TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING", "true")
    os.environ.setdefault("TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM", "true")

    pruning_mode = os.environ.get("TRIATTN_RUNTIME_PRUNING_MODE")
    if not pruning_mode:
        pruning_mode = os.environ.get("TRIATTENTION_PRUNING_MODE")
        if pruning_mode:
            mode = pruning_mode.strip().lower()
            if mode == "per_layer_head":
                mode = "per_layer_per_head"
            os.environ["TRIATTN_RUNTIME_PRUNING_MODE"] = mode


def register_triattention_backend():
    """Install TriAttention runtime integration when plugin is loaded by vLLM."""
    quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
    interface_mode = os.environ.get("TRIATTENTION_INTERFACE", "runtime").strip().lower()

    if interface_mode in {"legacy", "legacy_custom", "v1", "custom"}:
        if not quiet:
            print(
                "[TriAttention] Legacy V1 backend plugin registration is retired; "
                "use runtime interface (TRIATTENTION_INTERFACE=runtime)."
            )
        return

    _bridge_legacy_env_to_runtime()

    patch_scheduler = _truthy(
        os.environ.get("TRIATTN_RUNTIME_PATCH_SCHEDULER"),
        default=True,
    )
    patch_worker = _truthy(
        os.environ.get("TRIATTN_RUNTIME_PATCH_WORKER"),
        default=True,
    )

    try:
        from triattention_runtime.integration_monkeypatch import (
            install_vllm_integration_monkeypatches,
        )

        install_vllm_integration_monkeypatches(
            patch_scheduler=patch_scheduler,
            patch_worker=patch_worker,
        )
        if not quiet:
            print(
                "[TriAttention] Runtime (V2) plugin activated: "
                f"patch_scheduler={patch_scheduler} patch_worker={patch_worker}"
            )
    except Exception as exc:  # pragma: no cover - safety guard
        if not quiet:
            print(f"[TriAttention] Runtime plugin activation failed: {type(exc).__name__}: {exc}")
        raise
