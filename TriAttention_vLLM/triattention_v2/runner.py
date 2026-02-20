"""TriAttention v2 model runner proxy."""

from __future__ import annotations

import logging
from typing import Any

from .config import TriAttentionV2Config
from .executor import CompressionExecutor, RunnerHookCompressionExecutor
from .signals import CompressionSignal
from .state import RequestStateStore

TRITON_SCORING_REQUIRED_MARKER = "TRIATTN_FATAL_TRITON_SCORING_REQUIRED"


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
        self.executor: CompressionExecutor = RunnerHookCompressionExecutor(base_runner)
        self._last_step = 0
        self._logger = logging.getLogger(__name__)
        self._pending_compression_events: list[dict[str, Any]] = []
        self._strict_no_downgrade = bool(self.config.enable_experimental_kv_compaction)
        self._allowed_strict_skip_reasons = {
            "under_budget",
            "prefill_exceeds_budget",
            "req_state_not_found",
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_runner, name)

    def _register_new_requests(self, scheduler_output: Any) -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.prefill_token_ids is not None:
                prefill_len = len(new_req.prefill_token_ids)
            elif new_req.prompt_token_ids is not None:
                prefill_len = len(new_req.prompt_token_ids)
            else:
                prefill_len = 0
            self.state_store.ensure(
                req_id=new_req.req_id,
                prefill_len=prefill_len,
                protect_prefill=self.config.protect_prefill,
            )

    def _cleanup_finished_requests(self, scheduler_output: Any) -> None:
        for req_id in scheduler_output.finished_req_ids:
            self.state_store.remove(req_id)

    def _mark_preemptions(self, scheduler_output: Any) -> None:
        preempted_req_ids = getattr(scheduler_output, "preempted_req_ids", None)
        if not preempted_req_ids:
            return
        for req_id in preempted_req_ids:
            self.state_store.mark_preempted(req_id)

    def _mark_resumed(self, scheduler_output: Any) -> None:
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        for req_id in resumed_req_ids:
            self.state_store.mark_resumed(req_id)

    def _consume_signals(self, scheduler_output: Any) -> dict[str, CompressionSignal]:
        step = getattr(scheduler_output, "triattention_step", self._last_step + 1)
        self._last_step = step
        signals: dict[str, CompressionSignal] = getattr(
            scheduler_output, "triattention_signals", {}
        )

        for req_id, signal in signals.items():
            state = self.state_store.ensure(
                req_id=req_id,
                prefill_len=signal.prefill_len,
                protect_prefill=signal.protect_prefill,
            )
            self.state_store.update_cache_len(req_id, signal.estimated_cache_len)
            if signal.should_compress:
                self.state_store.mark_trigger(req_id, signal.reason, signal.step)
                if self.config.log_decisions:
                    self._logger.debug(
                        "TriAttention trigger req=%s step=%d reason=%s len=%d mode=%s",
                        req_id,
                        signal.step,
                        signal.reason,
                        signal.estimated_cache_len,
                        state.mode,
                    )
        return signals

    def _execute_compression_actions(
        self,
        scheduler_output: Any,
        signals: dict[str, CompressionSignal],
    ) -> None:
        events: list[dict[str, Any]] = []
        for req_id, signal in signals.items():
            if not signal.should_compress:
                continue
            try:
                result = self.executor.execute(
                    req_id=req_id,
                    signal=signal,
                    scheduler_output=scheduler_output,
                )
            except Exception as exc:  # pragma: no cover - safety fallback
                if self._strict_no_downgrade:
                    self._logger.exception(
                        "TriAttention strict mode fatal: compression executor exception "
                        "req=%s step=%d",
                        req_id,
                        signal.step,
                    )
                    raise RuntimeError(
                        f"{TRITON_SCORING_REQUIRED_MARKER}:executor_exception:"
                        f"req={req_id}:step={signal.step}:type={type(exc).__name__}"
                    ) from exc
                if TRITON_SCORING_REQUIRED_MARKER in str(exc):
                    self._logger.exception(
                        "TriAttention fatal: Triton scoring is required. "
                        "req=%s step=%d",
                        req_id,
                        signal.step,
                    )
                    raise
                self.state_store.mark_compression_skipped(
                    req_id=req_id,
                    reason=f"executor_exception:{type(exc).__name__}",
                    step=signal.step,
                )
                self._logger.exception(
                    "TriAttention compression executor failed req=%s step=%d",
                    req_id,
                    signal.step,
                )
                events.append(
                    {
                        "req_id": req_id,
                        "step": signal.step,
                        "status": "error",
                        "reason": f"executor_exception:{type(exc).__name__}",
                        "cache_len_after": None,
                    }
                )
                continue

            if (
                self._strict_no_downgrade
                and not result.applied
                and result.reason not in self._allowed_strict_skip_reasons
            ):
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:unexpected_skip:"
                    f"req={req_id}:step={signal.step}:reason={result.reason}"
                )

            if result.applied:
                cache_len_after = (
                    signal.estimated_cache_len
                    if result.cache_len_after is None
                    else result.cache_len_after
                )
                details = result.details if isinstance(result.details, dict) else {}
                before_len = details.get("effective_tokens_before")
                budget_total = details.get("budget_total")
                reclaimed_block_count = details.get("reclaimed_block_count")
                self.state_store.mark_compressed(
                    req_id=req_id,
                    step=signal.step,
                    cache_len=cache_len_after,
                )
                if self.config.log_decisions:
                    self._logger.debug(
                        "TriAttention compression applied req=%s step=%d reason=%s",
                        req_id,
                        signal.step,
                        result.reason,
                    )
                if isinstance(before_len, int):
                    self._logger.info(
                        "TriAttention compression summary req=%s step=%d before=%d after=%d "
                        "budget=%s reclaimed_blocks=%s reason=%s",
                        req_id,
                        signal.step,
                        before_len,
                        cache_len_after,
                        budget_total,
                        reclaimed_block_count,
                        result.reason,
                    )
                events.append(
                    {
                        "req_id": req_id,
                        "step": signal.step,
                        "status": "applied",
                        "reason": result.reason,
                        "cache_len_after": cache_len_after,
                        "details": result.details,
                        "block_reclaim": (
                            result.details.get("block_reclaim")
                            if isinstance(result.details, dict)
                            else None
                        ),
                    }
                )
                continue

            self.state_store.mark_compression_skipped(
                req_id=req_id,
                reason=result.reason,
                step=signal.step,
            )
            if self.config.log_decisions:
                self._logger.debug(
                    "TriAttention compression skipped req=%s step=%d reason=%s",
                    req_id,
                    signal.step,
                    result.reason,
                )
            events.append(
                {
                    "req_id": req_id,
                    "step": signal.step,
                    "status": "skipped",
                    "reason": result.reason,
                    "cache_len_after": result.cache_len_after,
                    "details": result.details,
                    "block_reclaim": (
                        result.details.get("block_reclaim")
                        if isinstance(result.details, dict)
                        else None
                    ),
                }
            )
        self._pending_compression_events = events

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
        output = self._base_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )
        # vLLM scheduler consumes ModelRunnerOutput from execute_model();
        # attach side-channel events here so scheduler can update effective
        # cache length and avoid repeated over-triggering.
        if output is not None:
            try:
                setattr(
                    output,
                    "triattention_compression_events",
                    self._pending_compression_events,
                )
            except Exception:
                # Keep pending events for sample_tokens fallback path.
                pass
            else:
                self._pending_compression_events = []
        return output

    def sample_tokens(self, grammar_output: Any) -> Any:
        # Kept for compatibility in case a runner path still calls this method.
        output = self._base_runner.sample_tokens(grammar_output)
        if output is None:
            self._pending_compression_events = []
            return None
        setattr(output, "triattention_compression_events", self._pending_compression_events)
        self._pending_compression_events = []
        return output

    def snapshot_states(self) -> dict[str, Any]:
        """Return debug snapshot for observability tests."""
        return self.state_store.snapshot()
