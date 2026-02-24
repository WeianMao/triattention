"""Runner output bridge helpers for TriAttention V2.

Keeps `TriAttentionModelRunner` focused on orchestration while this module owns:
- base runner execute_model invocation under effective-input overrides
- side-channel compression event attachment to execute_model/sample_tokens outputs
"""

from __future__ import annotations

from typing import Any

from .input_adapter import active_effective_input_overrides, prepare_effective_input_overrides
from .input_patch_backend import assert_effective_overrides_consumed


def execute_base_model_with_effective_overrides(
    *,
    base_runner: Any,
    state_store: Any,
    scheduler_output: Any,
    intermediate_tensors: Any = None,
    use_effective_overrides: bool = True,
) -> Any:
    """Execute base runner with current effective-length overrides applied."""
    if not use_effective_overrides:
        return base_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )
    overrides = prepare_effective_input_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
    )
    if (
        overrides.seq_base_map is None
        and overrides.pos_delta_map is None
        and overrides.single_seq_base is None
        and overrides.single_pos_delta == 0
    ):
        return base_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )
    # Use sparse overrides in hot path to avoid per-step dense tensor copies.
    with active_effective_input_overrides(overrides):
        output = base_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )
        if getattr(base_runner, "req_states", None) is not None:
            assert_effective_overrides_consumed()
        return output


def attach_execute_model_compression_events(
    *,
    output: Any,
    pending_events: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    """Attach compression events to ModelRunnerOutput when possible.

    Returns `(output, remaining_pending_events)`.
    """
    if output is None:
        return output, pending_events
    try:
        setattr(output, "triattention_compression_events", pending_events)
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
