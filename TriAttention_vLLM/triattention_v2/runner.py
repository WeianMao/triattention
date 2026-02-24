"""TriAttention v2 model runner proxy."""

from __future__ import annotations

import logging
from typing import Any

from .config import TriAttentionV2Config
from .executor import CompressionExecutor, RunnerHookCompressionExecutor
from .input_patch_backend import install_runtime_input_patch
from .request_key_compat import get_scheduled_token_items
from .runner_compression_actions import execute_runner_compression_actions
from .runner_output_bridge import (
    attach_execute_model_compression_events,
    attach_sample_tokens_compression_events,
    execute_base_model_with_effective_overrides,
)
from .runner_state_updates import (
    cleanup_finished_requests,
    consume_runner_signals,
    mark_preemptions,
    mark_resumed,
    register_new_requests,
)
from .signals import CompressionSignal
from .state import RequestStateStore
from .worker_reclaim_sync import apply_worker_block_reclaim_events

class TriAttentionModelRunner:
    """Proxy wrapper around vLLM model runner.

    Phase 1 behavior:
    - consume scheduler-side compression signals;
    - maintain request lifecycle-safe state;
    - keep vLLM forward path untouched.
    """

    def __init__(self, base_runner: Any, config: TriAttentionV2Config | None = None):
        self._base_runner = base_runner
        self.config = config or TriAttentionV2Config.from_env()
        self.state_store = RequestStateStore()
        # Expose request-level compression state to the installed hook so it can
        # apply recent-window semantics without relying on logical token order.
        setattr(base_runner, "_triattention_state_store", self.state_store)
        self.executor: CompressionExecutor = RunnerHookCompressionExecutor(base_runner)
        self._last_step = 0
        self._logger = logging.getLogger(__name__)
        self._pending_compression_events: list[dict[str, Any]] = []
        self._strict_no_downgrade = bool(self.config.enable_experimental_kv_compaction)
        self._runtime_input_patch_installed = False
        self._allowed_strict_skip_reasons = {
            "under_budget",
            "prefill_exceeds_budget",
            "req_state_not_found",
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_runner, name)

    def _register_new_requests(self, scheduler_output: Any) -> None:
        register_new_requests(
            state_store=self.state_store,
            scheduler_output=scheduler_output,
            protect_prefill=bool(self.config.protect_prefill),
        )

    def _cleanup_finished_requests(self, scheduler_output: Any) -> None:
        cleanup_finished_requests(state_store=self.state_store, scheduler_output=scheduler_output)

    def _mark_preemptions(self, scheduler_output: Any) -> None:
        mark_preemptions(state_store=self.state_store, scheduler_output=scheduler_output)

    def _mark_resumed(self, scheduler_output: Any) -> None:
        mark_resumed(state_store=self.state_store, scheduler_output=scheduler_output)

    def _consume_signals(self, scheduler_output: Any) -> dict[str, CompressionSignal]:
        step, signals = consume_runner_signals(
            state_store=self.state_store,
            scheduler_output=scheduler_output,
            last_step=self._last_step,
            logger=self._logger,
            log_decisions=bool(self.config.log_decisions),
        )
        self._last_step = step
        return signals

    def _execute_compression_actions(
        self,
        scheduler_output: Any,
        signals: dict[str, CompressionSignal],
    ) -> None:
        self._pending_compression_events = execute_runner_compression_actions(
            executor=self.executor,
            state_store=self.state_store,
            scheduler_output=scheduler_output,
            signals=signals,
            strict_no_downgrade=self._strict_no_downgrade,
            allowed_strict_skip_reasons=self._allowed_strict_skip_reasons,
            logger=self._logger,
            log_decisions=bool(self.config.log_decisions),
        )

    def _apply_worker_block_reclaim_events(self) -> None:
        """Apply reclaim shrink to worker-side block tables before prepare_inputs()."""
        apply_worker_block_reclaim_events(
            base_runner=self._base_runner,
            events=self._pending_compression_events,
        )

    def _needs_effective_input_overrides(self, scheduler_output: Any) -> bool:
        # Tighten scope to "current scheduled batch includes a compressed
        # request". Compression application updates request-local state before
        # this check, so we do not need to keep a separate step-local event path
        # here.
        scheduled_items = get_scheduled_token_items(scheduler_output)
        scheduled_req_ids: list[str] = [req_id for _raw_key, req_id, _scheduled_tokens in scheduled_items]
        if not scheduled_req_ids:
            return False
        checker = getattr(self.state_store, "has_compressed_request_in", None)
        if callable(checker):
            try:
                return bool(checker(scheduled_req_ids))
            except Exception:
                return False
        # Backward-compatible fallback if state_store is substituted in tests.
        checker_any = getattr(self.state_store, "has_active_compressed_requests", None)
        if callable(checker_any):
            try:
                return bool(checker_any())
            except Exception:
                return False
        return False

    def _ensure_runtime_input_patch_if_needed(self, need_effective_overrides: bool) -> None:
        if not need_effective_overrides:
            return
        # Unit tests may instantiate TriAttentionModelRunner with a lightweight fake
        # base runner that does not expose vLLM GPU input-prep internals.
        if getattr(self._base_runner, "req_states", None) is None:
            return
        if self._runtime_input_patch_installed:
            return
        patch_ok = install_runtime_input_patch()
        if not patch_ok:
            raise RuntimeError(
                "TriAttention V2 requires gpu seq_len/slot_mapping patch when "
                "effective-length overrides are active, but patch installation failed"
            )
        self._runtime_input_patch_installed = True

    def execute_model(
        self,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
    ) -> Any:
        self._register_new_requests(scheduler_output)
        self._cleanup_finished_requests(scheduler_output)
        self._mark_preemptions(scheduler_output)
        self._mark_resumed(scheduler_output)
        signals = self._consume_signals(scheduler_output)
        self._execute_compression_actions(scheduler_output, signals)
        self._apply_worker_block_reclaim_events()
        need_effective_overrides = self._needs_effective_input_overrides(scheduler_output)
        self._ensure_runtime_input_patch_if_needed(need_effective_overrides)
        output = execute_base_model_with_effective_overrides(
            base_runner=self._base_runner,
            state_store=self.state_store,
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
            use_effective_overrides=need_effective_overrides,
        )
        output, self._pending_compression_events = attach_execute_model_compression_events(
            output=output,
            pending_events=self._pending_compression_events,
        )
        return output

    def sample_tokens(self, grammar_output: Any) -> Any:
        # Kept for compatibility in case a runner path still calls this method.
        output = self._base_runner.sample_tokens(grammar_output)
        output, self._pending_compression_events = attach_sample_tokens_compression_events(
            output=output,
            pending_events=self._pending_compression_events,
        )
        return output

    def snapshot_states(self) -> dict[str, Any]:
        """Return debug snapshot for observability tests."""
        return self.state_store.snapshot()
