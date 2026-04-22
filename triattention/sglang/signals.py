"""Compression signal schema for scheduler-to-worker communication.

In sglang, the scheduler and worker live in the same process, so
signals do not need serialisation.  They are plain Python objects
attached to the batch / result as extra attributes.

Re-exports the shared ``CompressionSignal`` dataclass from the vLLM
runtime signals module.  The same frozen dataclass works in sglang
without modification because sglang's single-process architecture
does not require serialisation across process boundaries.
"""

# sglang uses the same CompressionSignal as vLLM (via planner.build_signal);
# re-export here to provide an explicit interface contract.
from triattention.vllm.runtime.signals import CompressionSignal, TriggerReason

__all__ = ["CompressionSignal", "TriggerReason"]
