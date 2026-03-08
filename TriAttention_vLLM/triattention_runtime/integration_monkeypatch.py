"""Monkey patch vLLM V1 scheduler/worker for TriAttention runtime integration.

This keeps vLLM class identities unchanged (native Scheduler/Worker) while
injecting the minimum TriAttention hooks needed for current runtime behavior.
"""

from __future__ import annotations

from typing import Any, Callable

from vllm.logger import init_logger

from .config import TriAttentionRuntimeConfig
from .effective_len_tracker import EffectiveCacheLenTracker
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
    if not compression_events:
        compression_events = getattr(
            scheduler_output,
            "triattention_compression_events",
            None,
        )
    if compression_events:
        TriAttentionScheduler._apply_compression_events(self, compression_events)

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


def install_vllm_integration_monkeypatches(
    *,
    patch_scheduler: bool = True,
    patch_worker: bool = True,
) -> None:
    global _PATCHED, _ORIG_SCHED_INIT, _ORIG_SCHED_SCHEDULE, _ORIG_SCHED_UPDATE_FROM_OUTPUT
    global _ORIG_WORKER_INIT_DEVICE, _ORIG_WORKER_EXECUTE_MODEL
    global _PATCHED_SCHEDULER_ACTIVE, _PATCHED_WORKER_ACTIVE
    if _PATCHED:
        _PATCHED_SCHEDULER_ACTIVE = _PATCHED_SCHEDULER_ACTIVE or bool(patch_scheduler)
        _PATCHED_WORKER_ACTIVE = _PATCHED_WORKER_ACTIVE or bool(patch_worker)
        return

    import vllm.v1.core.sched.scheduler as sched_mod
    import vllm.v1.worker.gpu_worker as worker_mod

    Scheduler = sched_mod.Scheduler
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
        Scheduler._apply_compression_events = TriAttentionScheduler._apply_compression_events

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
