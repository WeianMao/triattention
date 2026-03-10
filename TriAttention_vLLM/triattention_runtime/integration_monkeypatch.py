"""Monkey patch vLLM V1 scheduler/worker for TriAttention runtime integration.

This keeps vLLM class identities unchanged (native Scheduler/Worker) while
injecting the minimum TriAttention hooks needed for current runtime behavior.
"""

from __future__ import annotations

from typing import Any, Callable

from vllm.logger import init_logger

from .config import TriAttentionRuntimeConfig
from .effective_len_tracker import EffectiveCacheLenTracker
from .kv_allocation_sync import resolve_request_effective_num_computed
from .planner import CompressionPlanner
from .request_key_compat import iter_scheduled_token_items
from .scheduler import TriAttentionScheduler
from .signals import CompressionSignal
from .worker import TriAttentionWorker

logger = init_logger(__name__)

_PATCHED = False
_PATCHED_SCHEDULER_ACTIVE = False
_PATCHED_WORKER_ACTIVE = False
_ORIG_SCHED_INIT: Callable[..., Any] | None = None
_ORIG_SCHED_SCHEDULE: Callable[..., Any] | None = None
_ORIG_SCHED_UPDATE_FROM_OUTPUT: Callable[..., Any] | None = None
_ORIG_WORKER_INIT_DEVICE: Callable[..., Any] | None = None
_ORIG_WORKER_EXECUTE_MODEL: Callable[..., Any] | None = None
_ORIG_KVCACHE_ALLOCATE_SLOTS: Callable[..., Any] | None = None


def _refresh_scheduler_stats_kv_usage(outputs: Any, kv_usage: float) -> None:
    """Best-effort refresh for scheduler_stats.kv_cache_usage in returned outputs.

    In V1, TriAttention reclaim is applied after the base scheduler emits stats.
    Refreshing this field keeps the per-step exported usage aligned with the
    post-reclaim block-pool state without changing core scheduling behavior.
    """
    if not isinstance(outputs, dict):
        return
    usage = float(kv_usage)
    for engine_output in outputs.values():
        scheduler_stats = getattr(engine_output, "scheduler_stats", None)
        if scheduler_stats is not None:
            scheduler_stats.kv_cache_usage = usage


def _patched_scheduler_init(self, *args, **kwargs):
    assert _ORIG_SCHED_INIT is not None
    _ORIG_SCHED_INIT(self, *args, **kwargs)
    cfg = TriAttentionRuntimeConfig.from_env()
    # Always attach config/state once patched to keep behavior deterministic.
    self.triattention_config = cfg
    self._planner = CompressionPlanner(cfg)
    self._effective_len_tracker = EffectiveCacheLenTracker()
    self._prefill_lens = {}
    self._length_threshold_cache = {}
    self._triattention_step = 0
    logger.info(
        "TriAttention monkeypatched Scheduler initialized: budget=%d divide_length=%d "
        "protect_prefill=%s disable_compression=%s kv_usage_trigger_enabled=%s",
        cfg.kv_budget,
        cfg.divide_length,
        cfg.protect_prefill,
        cfg.disable_compression,
        cfg.enable_kv_usage_trigger,
    )


def _patched_scheduler_schedule(self):
    assert _ORIG_SCHED_SCHEDULE is not None
    TriAttentionScheduler._sync_effective_kv_offsets_before_schedule(self)
    scheduler_output = _ORIG_SCHED_SCHEDULE(self)

    cfg = getattr(self, "triattention_config", None)
    if cfg is None:
        return scheduler_output

    self._triattention_step += 1
    TriAttentionScheduler._sync_prefill_lens(self, scheduler_output)

    if (
        cfg.disable_compression
        and not cfg.enable_kv_usage_trigger
        and not TriAttentionScheduler._has_active_effective_len_overrides(self)
    ):
        triattention_signals = {}
    else:
        triattention_signals = TriAttentionScheduler._build_signals(self, scheduler_output)

    setattr(scheduler_output, "triattention_step", self._triattention_step)
    setattr(scheduler_output, "triattention_signals", triattention_signals)
    return scheduler_output


def _patched_scheduler_update_from_output(self, scheduler_output, model_runner_output):
    assert _ORIG_SCHED_UPDATE_FROM_OUTPUT is not None
    outputs = _ORIG_SCHED_UPDATE_FROM_OUTPUT(self, scheduler_output, model_runner_output)

    cfg = getattr(self, "triattention_config", None)
    if cfg is None:
        return outputs

    # Prefer events from model_runner_output (V0 / sync path), fall back to
    # scheduler_output (V1 async path where execute_model returns None).
    compression_events = getattr(
        model_runner_output,
        "triattention_compression_events",
        None,
    )
    source = "model_runner_output" if compression_events else None
    if not compression_events:
        compression_events = getattr(
            scheduler_output,
            "triattention_compression_events",
            None,
        )
        if compression_events:
            source = "scheduler_output"
    if compression_events:
        applied = [e for e in compression_events if e.get("status") == "applied"]
        logger.info(
            "TriAttention update_from_output: received %d events (%d applied) via %s",
            len(compression_events), len(applied), source,
        )
        TriAttentionScheduler._apply_compression_events(self, compression_events)
        _refresh_scheduler_stats_kv_usage(outputs, self.kv_cache_manager.usage)

    for req_id in scheduler_output.finished_req_ids:
        self._prefill_lens.pop(req_id, None)
        self._length_threshold_cache.pop(req_id, None)
        self._effective_len_tracker.remove_request(req_id)
    return outputs


def _patched_worker_init_device(self):
    assert _ORIG_WORKER_INIT_DEVICE is not None
    _ORIG_WORKER_INIT_DEVICE(self)
    if not _PATCHED_WORKER_ACTIVE:
        return
    if getattr(self, "_triattention_runner_proxy_installed", False):
        return
    # Reuse TriAttentionWorker lazy-injection fields on native Worker instance.
    self._triattention_runtime_config = TriAttentionRuntimeConfig.from_env()
    self._triattention_runner_proxy_installed = False


def _patched_worker_execute_model(self, scheduler_output):
    assert _ORIG_WORKER_EXECUTE_MODEL is not None
    if _PATCHED_WORKER_ACTIVE:
        signals = getattr(scheduler_output, "triattention_signals", None)
        if signals:
            TriAttentionWorker._ensure_triattention_runner_proxy(self)
    return _ORIG_WORKER_EXECUTE_MODEL(self, scheduler_output)


def _patched_kv_cache_allocate_slots(
    self,
    request,
    num_new_tokens,
    num_new_computed_tokens=0,
    new_computed_blocks=None,
    num_lookahead_tokens=0,
    delay_cache_blocks=False,
    num_encoder_tokens=0,
):
    """Keep vLLM allocation math aligned with TriAttention effective KV length."""
    assert _ORIG_KVCACHE_ALLOCATE_SLOTS is not None
    effective_num_computed = resolve_request_effective_num_computed(request)
    if effective_num_computed is None:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
        )
    logical_num_computed = getattr(request, "num_computed_tokens", None)
    if not isinstance(logical_num_computed, int):
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
        )
    if effective_num_computed >= logical_num_computed:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
        )
    setattr(request, "num_computed_tokens", int(effective_num_computed))
    try:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
        )
    finally:
        setattr(request, "num_computed_tokens", logical_num_computed)


def install_vllm_integration_monkeypatches(
    *,
    patch_scheduler: bool = True,
    patch_worker: bool = True,
) -> None:
    global _PATCHED, _ORIG_SCHED_INIT, _ORIG_SCHED_SCHEDULE, _ORIG_SCHED_UPDATE_FROM_OUTPUT
    global _ORIG_WORKER_INIT_DEVICE, _ORIG_WORKER_EXECUTE_MODEL
    global _ORIG_KVCACHE_ALLOCATE_SLOTS
    global _PATCHED_SCHEDULER_ACTIVE, _PATCHED_WORKER_ACTIVE
    if _PATCHED:
        _PATCHED_SCHEDULER_ACTIVE = _PATCHED_SCHEDULER_ACTIVE or bool(patch_scheduler)
        _PATCHED_WORKER_ACTIVE = _PATCHED_WORKER_ACTIVE or bool(patch_worker)
        return

    import vllm.v1.core.sched.scheduler as sched_mod
    import vllm.v1.core.kv_cache_manager as kv_cache_manager_mod
    import vllm.v1.worker.gpu_worker as worker_mod

    Scheduler = sched_mod.Scheduler
    KVCacheManager = kv_cache_manager_mod.KVCacheManager
    Worker = worker_mod.Worker

    if patch_scheduler:
        _ORIG_SCHED_INIT = Scheduler.__init__
        _ORIG_SCHED_SCHEDULE = Scheduler.schedule
        _ORIG_SCHED_UPDATE_FROM_OUTPUT = Scheduler.update_from_output
        Scheduler.__init__ = _patched_scheduler_init
        Scheduler.schedule = _patched_scheduler_schedule
        Scheduler.update_from_output = _patched_scheduler_update_from_output
        # Attach helper methods used by the patched wrappers.
        Scheduler._resolve_prefill_len = TriAttentionScheduler._resolve_prefill_len
        Scheduler._compute_length_threshold = TriAttentionScheduler._compute_length_threshold
        Scheduler._sync_prefill_lens = TriAttentionScheduler._sync_prefill_lens
        Scheduler._has_active_effective_len_overrides = (
            TriAttentionScheduler._has_active_effective_len_overrides
        )
        Scheduler._build_signals = TriAttentionScheduler._build_signals
        Scheduler._sync_effective_kv_offsets_before_schedule = (
            TriAttentionScheduler._sync_effective_kv_offsets_before_schedule
        )
        Scheduler._apply_compression_events = TriAttentionScheduler._apply_compression_events
        _ORIG_KVCACHE_ALLOCATE_SLOTS = KVCacheManager.allocate_slots
        KVCacheManager.allocate_slots = _patched_kv_cache_allocate_slots

    if patch_worker:
        _ORIG_WORKER_INIT_DEVICE = Worker.init_device
        _ORIG_WORKER_EXECUTE_MODEL = Worker.execute_model
        Worker.init_device = _patched_worker_init_device
        Worker.execute_model = _patched_worker_execute_model
        Worker._ensure_triattention_runner_proxy = TriAttentionWorker._ensure_triattention_runner_proxy

    _PATCHED_SCHEDULER_ACTIVE = bool(patch_scheduler)
    _PATCHED_WORKER_ACTIVE = bool(patch_worker)
    _PATCHED = True
    logger.info(
        "Installed TriAttention runtime monkeypatch integration: patch_scheduler=%s patch_worker=%s",
        patch_scheduler,
        patch_worker,
    )
