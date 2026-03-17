"""Runner output bridge helpers for TriAttention runtime.

Keeps `TriAttentionModelRunner` focused on orchestration while this module owns:
- base runner execute_model invocation under effective-input overrides
- side-channel compression event attachment to execute_model/sample_tokens outputs
"""

from __future__ import annotations

import time
from typing import Any

from .input_adapter import active_effective_input_overrides, prepare_effective_input_overrides
from .input_patch_backend import assert_effective_overrides_consumed
from .runner_struct_compat import debug_v1_override_path_enabled


def execute_base_model_with_effective_overrides(
    *,
    base_runner: Any,
    state_store: Any,
    scheduler_output: Any,
    intermediate_tensors: Any = None,
    use_effective_overrides: bool = True,
    perf_out: dict[str, float] | None = None,
) -> Any:
    """Execute base runner with current effective-length overrides applied."""
    perf_enabled = isinstance(perf_out, dict)
    if not use_effective_overrides:
        if perf_enabled:
            t0 = time.perf_counter()
        output = base_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )
        if perf_enabled:
            t1 = time.perf_counter()
            perf_out["override_prep_ms"] = 0.0
            perf_out["base_exec_ms"] = (t1 - t0) * 1000.0
        return output

    if perf_enabled:
        t0 = time.perf_counter()
    overrides = prepare_effective_input_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
    )
    if perf_enabled:
        t1 = time.perf_counter()
    if (
        overrides.seq_base_map is None
        and overrides.pos_delta_map is None
        and overrides.single_seq_base is None
        and overrides.single_pos_delta == 0
    ):
        if perf_enabled:
            t2 = time.perf_counter()
        output = base_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )
        if perf_enabled:
            t3 = time.perf_counter()
            perf_out["override_prep_ms"] = (t1 - t0) * 1000.0
            perf_out["base_exec_ms"] = (t3 - t2) * 1000.0
        return output
    # Use sparse overrides in hot path to avoid per-step dense tensor copies.
    with active_effective_input_overrides(overrides):
        if perf_enabled:
            t2 = time.perf_counter()
        output = base_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )
        if perf_enabled:
            t3 = time.perf_counter()
        if getattr(base_runner, "req_states", None) is not None or debug_v1_override_path_enabled():
            assert_effective_overrides_consumed()
        if perf_enabled:
            perf_out["override_prep_ms"] = (t1 - t0) * 1000.0
            perf_out["base_exec_ms"] = (t3 - t2) * 1000.0
        return output


def attach_execute_model_compression_events(
    *,
    output: Any,
    pending_events: list[dict[str, Any]],
    scheduler_output: Any = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Attach compression events to ModelRunnerOutput when possible.

    In vLLM V1's async path, ``execute_model`` returns ``None`` (the actual
    ``ModelRunnerOutput`` is produced later).  When that happens, attach
    events to ``scheduler_output`` instead — the same Python object is
    passed through to ``scheduler.update_from_output()``, so the events
    will arrive without serialization.

    Returns ``(output, remaining_pending_events)``.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)
    applied_count = sum(1 for e in pending_events if e.get("status") == "applied")
    if output is None:
        if scheduler_output is not None and pending_events:
            setattr(
                scheduler_output,
                "triattention_compression_events",
                pending_events,
            )
            _log.info(
                "attach_events: output=None, attached %d events (%d applied) to scheduler_output (id=%d)",
                len(pending_events), applied_count, id(scheduler_output),
            )
            return output, []
        if pending_events:
            _log.warning(
                "attach_events: output=None scheduler_output=None, DROPPING %d events (%d applied)",
                len(pending_events), applied_count,
            )
        return output, pending_events
    try:
        setattr(output, "triattention_compression_events", pending_events)
        if applied_count > 0:
            _log.info(
                "attach_events: attached %d events (%d applied) to output type=%s",
                len(pending_events), applied_count, type(output).__name__,
            )
    except Exception:
        # Keep pending events for sample_tokens fallback path.
        return output, pending_events
    return output, []


def attach_sample_tokens_compression_events(
    *,
    output: Any,
    pending_events: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    """Attach compression events to sample_tokens output (fallback path)."""
    if output is None:
        return None, []
    setattr(output, "triattention_compression_events", pending_events)
    return output, []
