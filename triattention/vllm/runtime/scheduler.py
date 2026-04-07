"""TriAttention v2 scheduler integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.structured_output import StructuredOutputManager

from .config import TriAttentionRuntimeConfig
from .effective_len_tracker import EffectiveCacheLenTracker
from .kv_allocation_sync import (
    clear_request_allocation_sync_state,
    prepare_request_effective_num_computed,
    update_request_effective_kv_offset,
)
from .planner import CompressionPlanner
from .request_key_compat import iter_scheduled_token_items
from .signals import CompressionSignal

logger = init_logger(__name__)

_DUMP_FIRST_PROMPT_TOKEN_IDS = (
    os.environ.get("TRIATTN_DEBUG_DUMP_FIRST_PROMPT_TOKEN_IDS", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
_DUMP_FIRST_PROMPT_TOKEN_IDS_PATH = os.environ.get(
    "TRIATTN_DEBUG_DUMP_FIRST_PROMPT_TOKEN_IDS_PATH",
    "",
).strip()
_FIRST_PROMPT_TOKEN_IDS_DUMPED = False


def _evict_reclaimed_block_metadata(block_pool: Any, block: Any) -> None:
    """Best-effort clear of prefix-cache metadata before reusing a block."""
    if block_pool is None or block is None:
        return
    block_hash = getattr(block, "block_hash", None)
    if block_hash is None:
        return

    maybe_evict = getattr(block_pool, "_maybe_evict_cached_block", None)
    if callable(maybe_evict):
        maybe_evict(block)


def _free_reclaimed_blocks(manager: Any, removed_blocks: list[Any]) -> bool:
    """Free reclaimed tail blocks after clearing any stale prefix-cache identity."""
    if not removed_blocks:
        return False
    block_pool = getattr(manager, "block_pool", None)
    for block in removed_blocks:
        _evict_reclaimed_block_metadata(block_pool, block)
    if block_pool is None:
        return False
    block_pool.free_blocks(reversed(removed_blocks))
    return True


def _resolve_full_prefill_len_from_request_like(request_like: Any) -> int:
    candidates: list[int] = []

    prompt_token_ids = getattr(request_like, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        try:
            candidates.append(len(prompt_token_ids))
        except Exception:
            pass

    for attr_name in ("prompt_token_ids_len", "num_prompt_tokens"):
        raw_value = getattr(request_like, attr_name, None)
        if raw_value is None:
            continue
        try:
            candidates.append(int(raw_value))
        except (TypeError, ValueError):
            continue

    prefill_token_ids = getattr(request_like, "prefill_token_ids", None)
    if prefill_token_ids is not None:
        try:
            candidates.append(len(prefill_token_ids))
        except Exception:
            pass

    return max(candidates, default=0)


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
        self.triattention_config = TriAttentionRuntimeConfig.from_env()
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
        global _FIRST_PROMPT_TOKEN_IDS_DUMPED
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            is_first_seen = req_id not in self._prefill_lens
            if is_first_seen:
                # Chunked-prefill may surface the same request multiple times in
                # scheduled_new_reqs. Only the first appearance should reset the
                # effective-length tracker; later repeats are continuation of the
                # same lifecycle, not a new request.
                self._effective_len_tracker.reset_request(
                    req_id,
                    new_req.num_computed_tokens,
                )
            prompt_token_ids = getattr(new_req, "prompt_token_ids", None)
            prefill_len = _resolve_full_prefill_len_from_request_like(new_req)
            if (
                is_first_seen
                and _DUMP_FIRST_PROMPT_TOKEN_IDS
                and not _FIRST_PROMPT_TOKEN_IDS_DUMPED
                and _DUMP_FIRST_PROMPT_TOKEN_IDS_PATH
                and isinstance(prompt_token_ids, list)
            ):
                try:
                    dump_path = Path(_DUMP_FIRST_PROMPT_TOKEN_IDS_PATH)
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    dump_path.write_text(
                        json.dumps(
                            {
                                "req_id": req_id,
                                "prefill_len": int(prefill_len),
                                "prompt_token_ids_len": len(prompt_token_ids),
                                "prompt_token_ids": [int(x) for x in prompt_token_ids],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    _FIRST_PROMPT_TOKEN_IDS_DUMPED = True
                except Exception:
                    pass
            self._prefill_lens[req_id] = prefill_len
            self._length_threshold_cache[req_id] = self._compute_length_threshold(prefill_len)

        for req_id in scheduler_output.finished_req_ids:
            req = self.requests.get(req_id)
            if req is not None:
                clear_request_allocation_sync_state(req)
            self._prefill_lens.pop(req_id, None)
            self._length_threshold_cache.pop(req_id, None)
            self._effective_len_tracker.remove_request(req_id)

        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is None:
            resumed_req_ids: list[str] = []
        else:
            resumed_req_ids = getattr(cached_reqs, "resumed_req_ids", None)
            if resumed_req_ids is None:
                resumed_req_ids = getattr(cached_reqs, "req_ids", []) or []
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
        scheduled_items = list(iter_scheduled_token_items(scheduler_output))
        if not scheduled_items and self._triattention_step % 500 == 0:
            raw = getattr(scheduler_output, "num_scheduled_tokens", "MISSING")
            logger.info(
                "TriAttention _build_signals: no scheduled items step=%d "
                "num_scheduled_tokens_type=%s len=%s",
                self._triattention_step,
                type(raw).__name__,
                len(raw) if isinstance(raw, dict) else "N/A",
            )
        for _raw_key, req_id, scheduled_tokens in scheduled_items:
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
                        logger.info(
                            "TriAttention threshold computed req=%s threshold=%d "
                            "prefill_len=%d budget=%d divide_length=%d",
                            req_id, threshold, prefill_len,
                            self.triattention_config.kv_budget,
                            self.triattention_config.divide_length,
                        )
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
                scheduled_tokens=scheduled_tokens,
            )
            # Keep scheduler->runner side-channel sparse to reduce per-step IPC
            # metadata overhead in the common no-compression decode path.
            #
            # Runner only needs full signal payload for:
            # 1) compression trigger execution in this step; or
            # 2) requests that have already been compressed and still need
            #    effective-length updates for runtime input overrides.
            if signal.should_compress or has_override:
                if signal.should_compress:
                    logger.info(
                        "TriAttention signal triggered req=%s step=%d "
                        "estimated_cache_len=%d reason=%s",
                        req_id, self._triattention_step,
                        estimated_cache_len, signal.reason,
                    )
                signals[req_id] = signal
        return signals

    def _sync_effective_kv_offsets_before_schedule(self) -> None:
        running = getattr(self, "running", None)
        if not isinstance(running, list):
            return
        for request in running:
            prepare_request_effective_num_computed(request)

    def _compute_max_chunk_for_compression(self) -> int | None:
        """Max tokens per step to allow compression cycling within physical KV."""
        block_pool = getattr(getattr(self, "kv_cache_manager", None), "block_pool", None)
        if block_pool is None:
            return None
        total_blocks = getattr(block_pool, "num_gpu_blocks", 0)
        if total_blocks <= 0:
            return None
        block_size = int(getattr(self, "block_size", 16) or 16)
        physical_kv = total_blocks * block_size
        headroom = physical_kv - self.triattention_config.kv_budget
        if headroom <= 0:
            return None
        # Leave a small margin (one block) for allocation bookkeeping.
        headroom = max(1, headroom - block_size)
        return headroom

    def schedule(self) -> SchedulerOutput:
        self._sync_effective_kv_offsets_before_schedule()

        orig_max_scheduled = None
        if not self.triattention_config.disable_compression:
            max_chunk = self._compute_max_chunk_for_compression()
            if max_chunk is not None:
                current_max = getattr(self, "max_num_scheduled_tokens", None)
                if current_max is not None and max_chunk < current_max:
                    orig_max_scheduled = current_max
                    self.max_num_scheduled_tokens = max_chunk

        scheduler_output = super().schedule()

        if orig_max_scheduled is not None:
            self.max_num_scheduled_tokens = orig_max_scheduled

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
        logger.info(
            "TriAttention _apply_compression_events: kv_cache_manager=%s "
            "coordinator=%s managers=%s block_size=%d reclaim_enabled=%s",
            type(self.kv_cache_manager).__name__,
            type(coordinator).__name__ if coordinator else None,
            type(managers).__name__ if managers else None,
            block_size,
            getattr(self, "triattention_config", None)
            and self.triattention_config.enable_experimental_block_reclaim,
        )

        def _num_required_blocks(token_len: int) -> int:
            if token_len <= 0:
                return 0
            return (token_len + block_size - 1) // block_size

        for event in compression_events:
            if event.get("status") != "applied":
                continue
            req_id = event.get("req_id")
            if req_id is None:
                continue
            event_step = int(event.get("step", -1))
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
            _evt_scheduled = int(event.get("scheduled_tokens", 1))
            expected_shrink_gids: set[int] = set()
            reclaim_applied_any = False
            req_groups_seen = 0
            if isinstance(managers, (list, tuple)):
                for gid, manager in enumerate(managers):
                    req_blocks = manager.req_to_blocks.get(req_id)
                    if req_blocks and required_blocks < len(req_blocks):
                        expected_shrink_gids.add(gid)
                    if req_blocks:
                        req_groups_seen += 1

            block_reclaim = event.get("block_reclaim")
            groups = (
                block_reclaim.get("groups")
                if isinstance(block_reclaim, dict)
                else None
            )
            logger.info(
                "TriAttention block reclaim: req=%s required_blocks=%d "
                "expected_shrink_gids=%s block_reclaim=%s groups=%s",
                req_id, required_blocks, expected_shrink_gids,
                type(block_reclaim).__name__ if block_reclaim else None,
                bool(groups),
            )
            if not isinstance(groups, list):
                # In V1 batch-queue mode, consecutive compression steps can
                # race: the worker already truncated blocks in an earlier step
                # whose events the scheduler hasn't consumed yet.  When that
                # happens the later event legitimately has block_reclaim=None.
                # Synthesize the reclaim by truncating to required_blocks.
                #
                # Safety: during chunked prefill (scheduled_tokens > 1),
                # _update_states may have appended new blocks after the hook
                # ran.  Without block_ids_before we cannot distinguish old
                # blocks from new ones — skip synthesis to avoid freeing
                # blocks the worker is still using.
                if _evt_scheduled > 1:
                    logger.info(
                        "TriAttention block reclaim: skipping synthesized "
                        "reclaim during prefill (no groups, "
                        "scheduled_tokens=%d) req=%s",
                        _evt_scheduled, req_id,
                    )
                elif expected_shrink_gids and isinstance(managers, (list, tuple)):
                    for gid in sorted(expected_shrink_gids):
                        manager = managers[gid]
                        req_blocks = manager.req_to_blocks.get(req_id)
                        if not req_blocks or required_blocks >= len(req_blocks):
                            continue
                        kept_blocks = req_blocks[:required_blocks]
                        removed_blocks = req_blocks[required_blocks:]
                        manager.req_to_blocks[req_id] = kept_blocks
                        if req_id in manager.num_cached_block:
                            manager.num_cached_block[req_id] = min(
                                manager.num_cached_block[req_id],
                                len(kept_blocks),
                            )
                        if _free_reclaimed_blocks(manager, removed_blocks):
                            reclaim_applied_any = True
                if reclaim_applied_any:
                    update_request_effective_kv_offset(
                        request=req,
                        cache_len_after=cache_len_after,
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

                # Use block_ids_before to distinguish old blocks (present
                # when the hook ran) from new blocks appended by
                # _update_states after the hook.  Only free old tail blocks
                # that the hook removed; preserve new blocks the worker needs.
                block_ids_before = group.get("block_ids_before")
                if isinstance(block_ids_before, list):
                    original_count = len(block_ids_before)
                else:
                    original_count = len(req_blocks)
                new_blocks_this_step = list(req_blocks[original_count:])
                kept_old_blocks = list(req_blocks[:kept_len])
                removed_old_blocks = list(req_blocks[kept_len:original_count])
                # Reassemble: kept old prefix + new blocks from this step
                reassembled = kept_old_blocks + new_blocks_this_step
                manager.req_to_blocks[req_id] = reassembled
                if req_id in manager.num_cached_block:
                    manager.num_cached_block[req_id] = min(
                        manager.num_cached_block[req_id], len(reassembled)
                    )
                if removed_old_blocks:
                    logger.info(
                        "TriAttention scheduler FREE_BLOCKS: req=%s gid=%d "
                        "freed=%d kept=%d new=%d",
                        req_id, gid, len(removed_old_blocks),
                        len(kept_old_blocks), len(new_blocks_this_step),
                    )
                    if _free_reclaimed_blocks(manager, removed_old_blocks):
                        reclaim_applied_any = True

            # Synthesize reclaim for groups that were expected but not
            # covered by the explicit block_reclaim payload (V1 batch-queue
            # race — worker already truncated in an earlier step).
            # Same safety as above: skip during chunked prefill without
            # block_ids_before to avoid freeing new blocks.
            missing_gids = expected_shrink_gids - seen_gids
            if missing_gids and _evt_scheduled <= 1:
                for gid in sorted(missing_gids):
                    manager = managers[gid]
                    req_blocks = manager.req_to_blocks.get(req_id)
                    if not req_blocks or required_blocks >= len(req_blocks):
                        continue
                    kept_blocks = req_blocks[:required_blocks]
                    removed_blocks = req_blocks[required_blocks:]
                    manager.req_to_blocks[req_id] = kept_blocks
                    if req_id in manager.num_cached_block:
                        manager.num_cached_block[req_id] = min(
                            manager.num_cached_block[req_id],
                            len(kept_blocks),
                        )
                    if _free_reclaimed_blocks(manager, removed_blocks):
                        reclaim_applied_any = True
            elif missing_gids and _evt_scheduled > 1:
                logger.info(
                    "TriAttention block reclaim: skipping synthesized "
                    "reclaim for missing gids %s during prefill "
                    "(scheduled_tokens=%d) req=%s",
                    sorted(missing_gids), _evt_scheduled, req_id,
                )

            if reclaim_applied_any:
                update_request_effective_kv_offset(
                    request=req,
                    cache_len_after=cache_len_after,
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
            usage = float(self.kv_cache_manager.usage)
            for engine_output in outputs.values():
                scheduler_stats = getattr(engine_output, "scheduler_stats", None)
                if scheduler_stats is not None:
                    scheduler_stats.kv_cache_usage = usage

        for req_id in scheduler_output.finished_req_ids:
            self._prefill_lens.pop(req_id, None)
            self._effective_len_tracker.remove_request(req_id)
        return outputs
