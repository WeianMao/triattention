"""Compression trigger planner for TriAttention v2."""

from __future__ import annotations

from .config import TriAttentionV2Config
from .signals import CompressionSignal


class CompressionPlanner:
    """Builds per-request trigger decisions from scheduler-side signals."""

    def __init__(self, config: TriAttentionV2Config) -> None:
        self.config = config
        self._kv_pressure_armed = False

    def _check_kv_usage(self, kv_usage: float | None) -> bool:
        if not self.config.enable_kv_usage_trigger or kv_usage is None:
            return False

        if kv_usage >= self.config.kv_usage_trigger:
            self._kv_pressure_armed = True
        elif kv_usage <= self.config.kv_usage_release:
            self._kv_pressure_armed = False

        return self._kv_pressure_armed

    def build_signal(
        self,
        req_id: str,
        estimated_cache_len: int,
        prefill_len: int,
        step: int,
        kv_usage: float | None = None,
    ) -> CompressionSignal:
        length_threshold = self.config.kv_budget + self.config.divide_length
        # Keep scheduler trigger boundary consistent with effective budget
        # semantics used by compaction path.
        if self.config.protect_prefill and not self.config.include_prefill_in_budget:
            length_threshold += max(prefill_len, 0)
        length_triggered = estimated_cache_len >= length_threshold
        kv_triggered = self._check_kv_usage(kv_usage)
        should_compress = length_triggered or kv_triggered

        if kv_triggered:
            reason = "kv_usage_threshold"
        elif length_triggered:
            reason = "length_threshold"
        else:
            reason = "none"

        return CompressionSignal(
            req_id=req_id,
            should_compress=should_compress,
            reason=reason,
            estimated_cache_len=estimated_cache_len,
            step=step,
            kv_usage=kv_usage,
            protect_prefill=self.config.protect_prefill,
            prefill_len=prefill_len,
        )
