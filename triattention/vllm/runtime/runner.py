"""TriAttention v2 model runner proxy."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from .config import TriAttentionRuntimeConfig
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
from .perf_profile import TriAttentionPerfProfile
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

    def __init__(self, base_runner: Any, config: TriAttentionRuntimeConfig | None = None):
        self._base_runner = base_runner
        self.config = config or TriAttentionRuntimeConfig.from_env()
        self.state_store = RequestStateStore()
        # Expose request-level compression state to the installed hook so it can
        # apply recent-window semantics without relying on logical token order.
        setattr(base_runner, "_triattention_state_store", self.state_store)
        self.executor: CompressionExecutor = RunnerHookCompressionExecutor(base_runner)
        self._last_step = 0
        self._logger = logging.getLogger(__name__)
        self._perf = TriAttentionPerfProfile.from_env(self._logger)
        self._pending_compression_events: list[dict[str, Any]] = []
        self._strict_no_downgrade = bool(self.config.enable_experimental_kv_compaction)
        self._runtime_input_patch_installed = False
        self._allowed_strict_skip_reasons = {
            "under_budget",
            "prefill_exceeds_budget",
            "req_state_not_found",
            "batch_queue_dedup",
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

    def _get_actual_kv_from_block_table(self, req_id: str) -> int | None:
        """Return actual KV token count from Worker's block table.

        Uses ``num_blocks_per_row * block_size`` which reflects the true
        physical state, free of async Scheduler lag.  Returns *None* when
        the information is unavailable (e.g. request not yet in input_batch).
        """
        input_batch = getattr(self._base_runner, "input_batch", None)
        if input_batch is None:
            return None
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if not isinstance(req_id_to_index, dict):
            return None
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            return None
        block_table_obj = getattr(input_batch, "block_table", None)
        if block_table_obj is None:
            return None
        inner_tables = getattr(block_table_obj, "block_tables", None)
        first_table = (
            inner_tables[0]
            if isinstance(inner_tables, list) and inner_tables
            else block_table_obj
        )
        num_blocks_per_row = getattr(first_table, "num_blocks_per_row", None)
        if num_blocks_per_row is None:
            return None
        cache_config = getattr(self._base_runner, "cache_config", None)
        blk_size = int(getattr(cache_config, "block_size", 0))
        if blk_size <= 0:
            return None
        return int(num_blocks_per_row[req_index]) * blk_size

    def _supplement_worker_self_triggers(
        self,
        scheduler_output: Any,
        signals: dict[str, CompressionSignal],
    ) -> dict[str, CompressionSignal]:
        """Generate self-trigger signals for requests the Scheduler missed.

        The Scheduler's estimate can lag behind the Worker's actual KV length
        by several chunks due to async scheduling.  This check uses the real
        Worker-side state so compression triggers at the right time.
        """
        if not self.config.enable_experimental_kv_compaction:
            return signals
        scheduled_items = get_scheduled_token_items(scheduler_output)
        for _raw_key, req_id, scheduled_tokens in scheduled_items:
            # If Scheduler already sent a trigger, check if we should
            # override it with a more accurate block-table-based estimate.
            existing = signals.get(req_id)
            if existing is not None and existing.should_compress:
                # During decode, Scheduler's signal is fine — skip.
                if int(getattr(existing, "scheduled_tokens", 1)) <= 1:
                    continue
                # During prefill, Scheduler's estimated_cache_len may lag.
                # Fall through to compute block-table-based actual_kv and
                # replace the signal with a corrected estimate.
            state = self.state_store.get(req_id) if hasattr(self.state_store, "get") else None
            if state is None:
                continue
            prefill_len = state.prefill_len
            # Compute actual KV length on the Worker side.
            # kv_from_blocks: block table already includes current step's
            # allocated blocks (update_states runs before execute_model),
            # so scheduled_tokens must NOT be added again.
            kv_from_blocks = False
            if state.compression_count > 0 and state.current_cache_len > 0:
                actual_kv = state.current_cache_len
            else:
                # First-time compression: use block table for ground truth.
                # num_computed_tokens from base_runner.requests has async lag
                # (Scheduler runs 2-3 chunks ahead), so we derive actual KV
                # from the physical block count on the Worker side.
                actual_kv_bt = self._get_actual_kv_from_block_table(req_id)
                if actual_kv_bt is not None:
                    actual_kv = actual_kv_bt
                    kv_from_blocks = True
                else:
                    # Fallback: base_runner.requests (may be stale).
                    req_state = None
                    requests = getattr(self._base_runner, "requests", None)
                    if isinstance(requests, dict):
                        req_state = requests.get(req_id)
                    if req_state is not None:
                        actual_kv = int(getattr(req_state, "num_computed_tokens", 0))
                    else:
                        continue
            # Threshold: budget + divide_length (same formula as Scheduler).
            threshold = self.config.kv_budget + self.config.divide_length
            if self.config.protect_prefill and not self.config.include_prefill_in_budget:
                threshold += max(prefill_len, 0)
            if kv_from_blocks:
                # Block table capacity already covers scheduled tokens.
                effective_kv = actual_kv
            else:
                effective_kv = actual_kv + max(1, int(scheduled_tokens))
            if effective_kv < threshold:
                continue
            self._logger.info(
                "TriAttention worker self-trigger: req=%s actual_kv=%d "
                "effective_kv=%d scheduled=%d threshold=%d "
                "from_blocks=%s (scheduler had no signal)",
                req_id, actual_kv, effective_kv, scheduled_tokens,
                threshold, kv_from_blocks,
            )
            signals[req_id] = CompressionSignal(
                req_id=req_id,
                should_compress=True,
                reason="length_threshold",
                estimated_cache_len=effective_kv,
                step=self._last_step,
                kv_usage=None,
                protect_prefill=self.config.protect_prefill,
                prefill_len=prefill_len,
                scheduled_tokens=max(1, int(scheduled_tokens)),
            )
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

    def _patch_scheduler_output_for_compressed_reqs(self, scheduler_output: Any) -> None:
        """Trim stale new_block_ids for compressed requests (V1 batch-queue).

        After compression, the worker's block table has been shrunk via
        worker_reclaim_sync.  But the scheduler may still send excess
        new_block_ids based on its stale view.  This trims those to fit
        within the worker's actual block capacity.
        """
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is None:
            return
        req_ids = getattr(cached_reqs, "req_ids", None)
        new_block_ids_list = getattr(cached_reqs, "new_block_ids", None)
        if not isinstance(req_ids, list) or not isinstance(new_block_ids_list, list):
            return
        if len(req_ids) != len(new_block_ids_list):
            return

        # Get block table info from worker.
        input_batch = getattr(self._base_runner, "input_batch", None)
        block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None
        if block_table_obj is None:
            return
        max_blocks = getattr(block_table_obj, "max_num_blocks_per_req", None)
        if not isinstance(max_blocks, int) or max_blocks <= 0:
            return

        # Get num_blocks_per_row from the (possibly single) inner table.
        inner_tables = getattr(block_table_obj, "block_tables", None)
        first_table = inner_tables[0] if isinstance(inner_tables, list) and inner_tables else block_table_obj
        num_blocks_per_row = getattr(first_table, "num_blocks_per_row", None)

        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if not isinstance(req_id_to_index, dict):
            return

        for i, req_id in enumerate(req_ids):
            rt_state = self.state_store.get(req_id)
            if rt_state is None or rt_state.compression_count <= 0:
                continue

            new_block_ids = new_block_ids_list[i]
            if new_block_ids is None:
                continue

            req_index = req_id_to_index.get(req_id)
            if not isinstance(req_index, int):
                continue

            # Check if appending would overflow.
            if num_blocks_per_row is not None:
                current = int(num_blocks_per_row[req_index])
            else:
                continue

            if isinstance(new_block_ids, (list, tuple)):
                max_new = max(len(g) if isinstance(g, (list, tuple)) else 0 for g in new_block_ids)
            else:
                continue

            if current + max_new > max_blocks:
                # Trim to fit: only keep blocks that fit within max_blocks.
                available = max(0, max_blocks - current)
                trimmed = tuple(
                    list(g)[:available] if isinstance(g, (list, tuple)) else g
                    for g in new_block_ids
                )
                new_block_ids_list[i] = trimmed
                self._logger.info(
                    "TriAttention patched new_block_ids: req=%s "
                    "current_blocks=%d max=%d new_trimmed=%d->%d",
                    req_id, current, max_blocks, max_new, available,
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
        #
        # vLLM 0.15 runtime paths may populate `input_batch` while leaving
        # `req_states` unset, so treat either surface as sufficient evidence
        # that the native GPU input-prep hooks are available.
        if (
            getattr(self._base_runner, "req_states", None) is None
            and getattr(self._base_runner, "input_batch", None) is None
            and os.environ.get("TRIATTN_DEBUG_ENABLE_V1_OVERRIDE_PATH", "0") != "1"
        ):
            return
        if self._runtime_input_patch_installed:
            return
        patch_ok = install_runtime_input_patch()
        if not patch_ok:
            raise RuntimeError(
                "TriAttention runtime requires gpu seq_len/slot_mapping patch when "
                "effective-length overrides are active, but patch installation failed"
            )
        self._runtime_input_patch_installed = True

    def execute_model(
        self,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
    ) -> Any:
        perf_enabled = bool(getattr(self._perf, "enabled", False))
        t_total = time.perf_counter() if perf_enabled else 0.0
        t0 = time.perf_counter() if perf_enabled else 0.0
        self._register_new_requests(scheduler_output)
        self._cleanup_finished_requests(scheduler_output)
        self._mark_preemptions(scheduler_output)
        self._mark_resumed(scheduler_output)
        signals = self._consume_signals(scheduler_output)
        signals = self._supplement_worker_self_triggers(scheduler_output, signals)
        t_state_ms = (time.perf_counter() - t0) * 1000.0 if perf_enabled else 0.0
        t0 = time.perf_counter() if perf_enabled else 0.0
        self._execute_compression_actions(scheduler_output, signals)
        t_compress_ms = (time.perf_counter() - t0) * 1000.0 if perf_enabled else 0.0
        self._perf.record_compression_events(self._pending_compression_events)
        t0 = time.perf_counter() if perf_enabled else 0.0
        self._apply_worker_block_reclaim_events()
        self._patch_scheduler_output_for_compressed_reqs(scheduler_output)
        t_reclaim_ms = (time.perf_counter() - t0) * 1000.0 if perf_enabled else 0.0
        need_effective_overrides = self._needs_effective_input_overrides(scheduler_output)
        self._ensure_runtime_input_patch_if_needed(need_effective_overrides)
        bridge_perf: dict[str, float] | None = {} if perf_enabled else None
        output = execute_base_model_with_effective_overrides(
            base_runner=self._base_runner,
            state_store=self.state_store,
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
            use_effective_overrides=need_effective_overrides,
            perf_out=bridge_perf,
        )
        self._perf.record_model_output(output)
        t_total_exec_ms = (time.perf_counter() - t_total) * 1000.0 if perf_enabled else 0.0
        has_trigger = any(bool(sig.should_compress) for sig in signals.values()) if signals else False
        self._perf.record_step(
            has_trigger=has_trigger,
            uses_overrides=bool(need_effective_overrides),
            t_state_ms=t_state_ms,
            t_compress_ms=t_compress_ms,
            t_reclaim_ms=t_reclaim_ms,
            t_override_prep_ms=float((bridge_perf or {}).get("override_prep_ms", 0.0)),
            t_base_exec_ms=float((bridge_perf or {}).get("base_exec_ms", 0.0)),
            t_total_exec_ms=t_total_exec_ms,
        )
        output, self._pending_compression_events = attach_execute_model_compression_events(
            output=output,
            pending_events=self._pending_compression_events,
            scheduler_output=scheduler_output,
        )
        return output

    def sample_tokens(self, grammar_output: Any) -> Any:
        # In vLLM V1 async path, execute_model returns None and the actual
        # ModelRunnerOutput (with sampled_token_ids) is produced here.
        output = self._base_runner.sample_tokens(grammar_output)
        output, self._pending_compression_events = attach_sample_tokens_compression_events(
            output=output,
            pending_events=self._pending_compression_events,
        )
        return output

    def snapshot_states(self) -> dict[str, Any]:
        """Return debug snapshot for observability tests."""
        return self.state_store.snapshot()
