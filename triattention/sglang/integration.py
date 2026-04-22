"""Monkey-patch registration and installation for sglang.

All hooks are collected here and applied in a single
``install_all_hooks()`` call so that the installation order is explicit
and auditable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install_all_hooks() -> None:
    """Register every TriAttention monkey-patch on the live sglang modules.

    Intended to be called exactly once per process, from
    :func:`triattention.sglang.install_sglang_integration`.

    Patches applied (in order):
      1. Scheduler.__init__         — attach TriAttention runtime state
      2. Scheduler.get_next_batch_to_run — trigger check + compression
      3. Scheduler.process_batch_result_decode — post-decode bookkeeping
      4. ScheduleBatch.prepare_for_decode — seq_lens override (Phase G)
      5. ForwardBatch.init_new      — positions override (Phase G)
    """
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    from triattention.sglang.scheduler_hooks import (
        _patched_get_next_batch_to_run,
        _patched_process_batch_result_decode,
        _patched_scheduler_init,
    )

    # 1. Patch Scheduler.__init__
    original_init = Scheduler.__init__
    Scheduler.__init__ = _patched_scheduler_init(original_init)
    logger.info("TriAttention: patched Scheduler.__init__")

    # 2. Patch Scheduler.get_next_batch_to_run
    original_get_next = Scheduler.get_next_batch_to_run
    Scheduler.get_next_batch_to_run = _patched_get_next_batch_to_run(
        original_get_next
    )
    logger.info("TriAttention: patched Scheduler.get_next_batch_to_run")

    # 3. Patch process_batch_result_decode (defined in the mixin)
    original_process_decode = SchedulerOutputProcessorMixin.process_batch_result_decode
    SchedulerOutputProcessorMixin.process_batch_result_decode = (
        _patched_process_batch_result_decode(original_process_decode)
    )
    logger.info(
        "TriAttention: patched "
        "SchedulerOutputProcessorMixin.process_batch_result_decode"
    )

    # Phase F: No worker patch needed — compression runs entirely in the
    # scheduler (same-process architecture).  See worker_hooks.py docstring.

    # 4. Patch ScheduleBatch.prepare_for_decode (Phase G — seq_lens override)
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    from triattention.sglang.input_patches import _patched_prepare_for_decode

    original_prepare = ScheduleBatch.prepare_for_decode
    ScheduleBatch.prepare_for_decode = _patched_prepare_for_decode(original_prepare)
    logger.info("TriAttention: patched ScheduleBatch.prepare_for_decode")

    # 5. Patch ForwardBatch.init_new (Phase G — positions override)
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from triattention.sglang.input_patches import _patched_forward_batch_init_new

    original_init_new = ForwardBatch.init_new
    ForwardBatch.init_new = classmethod(
        _patched_forward_batch_init_new(original_init_new.__func__)
    )
    logger.info("TriAttention: patched ForwardBatch.init_new")
