"""Signal schema exchanged between scheduler and model runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TriggerReason = Literal["none", "length_threshold", "kv_usage_threshold"]


@dataclass(frozen=True)
class CompressionSignal:
    """Per-request compression signal for one scheduler step."""

    req_id: str
    should_compress: bool
    reason: TriggerReason
    estimated_cache_len: int
    step: int
    kv_usage: float | None
    protect_prefill: bool
    prefill_len: int
