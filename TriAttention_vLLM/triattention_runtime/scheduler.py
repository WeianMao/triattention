"""TriAttention v2 scheduler integration."""

from __future__ import annotations

from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.structured_output import StructuredOutputManager

from .config import TriAttentionV2Config
from .effective_len_tracker import EffectiveCacheLenTracker
from .planner import CompressionPlanner
from .request_key_compat import iter_scheduled_token_items
from .signals import CompressionSignal

logger = init_logger(__name__)


class TriAttentionScheduler(Scheduler):
    """Scheduler subclass that emits per-request compression signals."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        self.triattention_config = TriAttentionV2Config.from_env()
        self._planner = CompressionPlanner(self.triattention_config)
        self._effective_len_tracker = EffectiveCacheLenTracker()
        self._prefill_lens: dict[str, int] = {}
        self._length_threshold_cache: dict[str, int] = {}
        self._triattention_step = 0

        logger.info(
            "TriAttentionScheduler initialized: budget=%d divide_length=%d "
            "protect_prefill=%s kv_usage_trigger_enabled=%s block_reclaim_enabled=%s",
            self.triattention_config.kv_budget,
            self.triattention_config.divide_length,
            self.triattention_config.protect_prefill,
            self.triattention_config.enable_kv_usage_trigger,
            self.triattention_config.enable_experimental_block_reclaim,
        )

    def _resolve_prefill_len(self, req_id: str) -> int:
        if req_id in self._prefill_lens:
            return self._prefill_lens[req_id]
        request = self.requests.get(req_id)
        if request is None:
            return 0
        return request.num_prompt_tokens

    def _compute_length_threshold(self, prefill_len: int) -> int:
        threshold = self.triattention_config.kv_budget + self.triattention_config.divide_length
        if self.triattention_config.protect_prefill and not self.triattention_config.include_prefill_in_budget:
            threshold += max(0, int(prefill_len))
        return threshold

    def _sync_prefill_lens(self, scheduler_output: SchedulerOutput) -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            # Treat newly scheduled request as lifecycle reset for tracker state.
            self._effective_len_tracker.reset_request(
                new_req.req_id,
                new_req.num_computed_tokens,
            )
            if new_req.prefill_token_ids is not None:
                prefill_len = len(new_req.prefill_token_ids)
            elif new_req.prompt_token_ids is not None:
                prefill_len = len(new_req.prompt_token_ids)
            else:
                prefill_len = 0
            self._prefill_lens[new_req.req_id] = prefill_len
            self._length_threshold_cache[new_req.req_id] = self._compute_length_threshold(prefill_len)

        for req_id in scheduler_output.finished_req_ids:
            self._prefill_lens.pop(req_id, None)
            self._length_threshold_cache.pop(req_id, None)
            self._effective_len_tracker.remove_request(req_id)

        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        for req_id in resumed_req_ids:
            if req_id not in self._prefill_lens:
                prefill_len = self._resolve_prefill_len(req_id)
                self._prefill_lens[req_id] = prefill_len
                self._length_threshold_cache[req_id] = self._compute_length_threshold(prefill_len)

    def _has_active_effective_len_overrides(self) -> bool:
        checker = getattr(self._effective_len_tracker, "has_any_effective_len_overrides", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False

    def _build_signals(self, scheduler_output: SchedulerOutput) -> dict[str, CompressionSignal]:
        kv_usage_enabled = bool(self.triattention_config.enable_kv_usage_trigger)
        kv_usage = self.kv_cache_manager.usage if kv_usage_enabled else None
        compression_disabled = bool(self.triattention_config.disable_compression)
        signals: dict[str, CompressionSignal] = {}
        for _raw_key, req_id, scheduled_tokens in iter_scheduled_token_items(scheduler_output):
            request = self.requests.get(req_id)
            if request is None:
                continue
            has_override = self._effective_len_tracker.has_effective_len_override(req_id)
            if has_override:
                effective_base_len = self._effective_len_tracker.observe_num_computed(
                    req_id=req_id,
                    num_computed_tokens=request.num_computed_tokens,
                )
            else:
                # Common pre-compression path: effective cache length is exactly
                # num_computed_tokens, so avoid tracker writes in the decode hot path.
                effective_base_len = request.num_computed_tokens
            estimated_cache_len = effective_base_len + scheduled_tokens

            if not has_override:
                if compression_disabled and not kv_usage_enabled:
                    continue
                if not kv_usage_enabled and not compression_disabled:
                    threshold = self._length_threshold_cache.get(req_id)
                    if threshold is None:
                        prefill_len = self._resolve_prefill_len(req_id)
                        self._prefill_lens[req_id] = prefill_len
                        threshold = self._compute_length_threshold(prefill_len)
                        self._length_threshold_cache[req_id] = threshold
                    if estimated_cache_len < threshold:
                        continue

            prefill_len = self._prefill_lens.get(req_id)
            if prefill_len is None:
                prefill_len = self._resolve_prefill_len(req_id)
                self._prefill_lens[req_id] = prefill_len
                self._length_threshold_cache[req_id] = self._compute_length_threshold(prefill_len)
            signal = self._planner.build_signal(
                req_id=req_id,
                estimated_cache_len=estimated_cache_len,
                prefill_len=prefill_len,
                step=self._triattention_step,
                kv_usage=kv_usage,
            )
            # Keep scheduler->runner side-channel sparse to reduce per-step IPC
            # metadata overhead in the common no-compression decode path.
            #
            # Runner only needs full signal payload for:
            # 1) compression trigger execution in this step; or
            # 2) requests that have already been compressed and still need
            #    effective-length updates for runtime input overrides.
            if signal.should_compress or has_override:
                signals[req_id] = signal
        return signals

    def schedule(self) -> SchedulerOutput:
        scheduler_output = super().schedule()
        self._triattention_step += 1
        self._sync_prefill_lens(scheduler_output)
        if (
            self.triattention_config.disable_compression
            and not self.triattention_config.enable_kv_usage_trigger
            and not self._has_active_effective_len_overrides()
        ):
            # FullKV / no-compression path: avoid per-step planner work entirely.
            triattention_signals = {}
        else:
            triattention_signals = self._build_signals(scheduler_output)

        # Attach v2 side-channel metadata to scheduler output.
        setattr(scheduler_output, "triattention_step", self._triattention_step)
        setattr(scheduler_output, "triattention_signals", triattention_signals)

        if self.triattention_config.log_decisions and triattention_signals:
            hits = [
                req_id
                for req_id, signal in triattention_signals.items()
                if signal.should_compress
            ]
            if hits:
                logger.debug(
                    "TriAttention schedule step=%d trigger_reqs=%s",
                    self._triattention_step,
                    hits,
                )

        return scheduler_output

    def _apply_compression_events(self, compression_events: list[dict[str, Any]]) -> None:
        coordinator = getattr(self.kv_cache_manager, "coordinator", None)
        managers = getattr(coordinator, "single_type_managers", None)
        block_size = int(getattr(self, "block_size", 1))
        if block_size <= 0:
            block_size = 1

        def _num_required_blocks(token_len: int) -> int:
            if token_len <= 0:
                return 0
            return (token_len + block_size - 1) // block_size

        for event in compression_events:
            if event.get("status") != "applied":
                continue
            req_id = event.get("req_id")
            if not isinstance(req_id, str):
                continue
            cache_len_after = event.get("cache_len_after")
            if not isinstance(cache_len_after, int):
                continue
            req = self.requests.get(req_id)
            if req is None:
                continue
            self._effective_len_tracker.apply_compression(
                req_id=req_id,
                cache_len_after=cache_len_after,
                num_computed_tokens=req.num_computed_tokens,
            )

            if not self.triattention_config.enable_experimental_block_reclaim:
                continue
            required_blocks = _num_required_blocks(cache_len_after)
            expected_shrink_gids: set[int] = set()
            if isinstance(managers, (list, tuple)):
                for gid, manager in enumerate(managers):
                    req_blocks = manager.req_to_blocks.get(req_id)
                    if req_blocks and required_blocks < len(req_blocks):
                        expected_shrink_gids.add(gid)

            block_reclaim = event.get("block_reclaim")
            if not isinstance(block_reclaim, dict):
                if (
                    getattr(self.triattention_config, "require_physical_reclaim", False)
                    and expected_shrink_gids
                ):
                    raise RuntimeError(
                        "TriAttention block reclaim missing while shrink expected: "
                        f"req={req_id} expected_shrink_gids={sorted(expected_shrink_gids)} "
                        f"required_blocks={required_blocks}"
                    )
                continue
            groups = block_reclaim.get("groups")
            if not isinstance(groups, list):
                if (
                    getattr(self.triattention_config, "require_physical_reclaim", False)
                    and expected_shrink_gids
                ):
                    raise RuntimeError(
                        "TriAttention block reclaim groups invalid while shrink expected: "
                        f"req={req_id} expected_shrink_gids={sorted(expected_shrink_gids)} "
                        f"required_blocks={required_blocks}"
                    )
                continue
            if not isinstance(managers, (list, tuple)):
                continue

            seen_gids: set[int] = set()
            for group in groups:
                if not isinstance(group, dict):
                    continue
                gid = group.get("gid")
                block_ids_after = group.get("block_ids_after")
                if not isinstance(gid, int) or gid < 0 or gid >= len(managers):
                    continue
                if not isinstance(block_ids_after, list):
                    continue
                if not all(isinstance(block_id, int) for block_id in block_ids_after):
                    continue

                manager = managers[gid]
                req_blocks = manager.req_to_blocks.get(req_id)
                if not req_blocks:
                    continue

                seen_gids.add(gid)
                curr_ids = [block.block_id for block in req_blocks]
                kept_len = len(block_ids_after)
                if kept_len > len(curr_ids):
                    raise RuntimeError(
                        "TriAttention block reclaim invalid length: "
                        f"req={req_id} gid={gid} kept_len={kept_len} "
                        f"curr_len={len(curr_ids)}"
                    )
                if len(set(block_ids_after)) != kept_len:
                    raise RuntimeError(
                        "TriAttention block reclaim contains duplicate block ids: "
                        f"req={req_id} gid={gid} block_ids_after={block_ids_after}"
                    )
                expected_prefix = curr_ids[:kept_len]
                if expected_prefix != block_ids_after:
                    raise RuntimeError(
                        "TriAttention block reclaim prefix mismatch: "
                        f"req={req_id} gid={gid} expected_prefix={expected_prefix} "
                        f"actual_after={block_ids_after}"
                    )
                if (
                    getattr(self.triattention_config, "require_physical_reclaim", False)
                    and gid in expected_shrink_gids
                    and kept_len != required_blocks
                ):
                    raise RuntimeError(
                        "TriAttention block reclaim insufficient shrink: "
                        f"req={req_id} gid={gid} kept_len={kept_len} "
                        f"required_blocks={required_blocks}"
                    )

                kept_blocks = req_blocks[:kept_len]
                removed_blocks = req_blocks[kept_len:]

                manager.req_to_blocks[req_id] = kept_blocks
                if req_id in manager.num_cached_block:
                    manager.num_cached_block[req_id] = min(
                        manager.num_cached_block[req_id], len(kept_blocks)
                    )
                if removed_blocks:
                    manager.block_pool.free_blocks(reversed(removed_blocks))

            if (
                getattr(self.triattention_config, "require_physical_reclaim", False)
                and expected_shrink_gids
            ):
                missing_gids = sorted(expected_shrink_gids - seen_gids)
                if missing_gids:
                    raise RuntimeError(
                        "TriAttention block reclaim missing groups: "
                        f"req={req_id} missing_gids={missing_gids} "
                        f"required_blocks={required_blocks}"
                    )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, Any]:
        outputs = super().update_from_output(scheduler_output, model_runner_output)

        compression_events = getattr(
            model_runner_output,
            "triattention_compression_events",
            None,
        )
        if compression_events:
            if self.triattention_config.log_decisions:
                logger.debug(
                    "TriAttention compression events step=%d events=%s",
                    self._triattention_step,
                    compression_events,
                )
            self._apply_compression_events(compression_events)

        for req_id in scheduler_output.finished_req_ids:
            self._prefill_lens.pop(req_id, None)
            self._effective_len_tracker.remove_request(req_id)
        return outputs
