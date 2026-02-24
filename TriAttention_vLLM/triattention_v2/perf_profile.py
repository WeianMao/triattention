"""Lightweight aggregated perf profiling for TriAttention V2 (env gated)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _env_enabled(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


@dataclass
class TriAttentionPerfProfile:
    """Aggregates per-step timing counters with sparse logging."""

    logger: logging.Logger
    enabled: bool = False
    log_every_steps: int = 200
    total_steps: int = 0
    steps_with_overrides: int = 0
    steps_with_trigger: int = 0
    compress_calls: int = 0
    compress_applied: int = 0
    compress_skipped: int = 0
    compress_errors: int = 0
    t_state_ms: float = 0.0
    t_compress_ms: float = 0.0
    t_reclaim_ms: float = 0.0
    t_override_prep_ms: float = 0.0
    t_base_exec_ms: float = 0.0
    t_total_exec_ms: float = 0.0
    skip_reasons: Counter[str] = field(default_factory=Counter)
    cudagraph_modes: Counter[str] = field(default_factory=Counter)
    sink_dir: str | None = None

    @classmethod
    def from_env(cls, logger: logging.Logger) -> "TriAttentionPerfProfile":
        return cls(
            logger=logger,
            enabled=_env_enabled("TRIATTN_V2_PERF_PROFILE", "0"),
            log_every_steps=max(1, _env_int("TRIATTN_V2_PERF_LOG_EVERY", 200)),
            sink_dir=os.environ.get("TRIATTN_V2_PERF_SINK_DIR"),
        )

    def timer(self) -> "_Timer":
        return _Timer()

    def record_step(
        self,
        *,
        has_trigger: bool,
        uses_overrides: bool,
        t_state_ms: float,
        t_compress_ms: float,
        t_reclaim_ms: float,
        t_override_prep_ms: float,
        t_base_exec_ms: float,
        t_total_exec_ms: float,
    ) -> None:
        if not self.enabled:
            return
        self.total_steps += 1
        if has_trigger:
            self.steps_with_trigger += 1
        if uses_overrides:
            self.steps_with_overrides += 1
        self.t_state_ms += t_state_ms
        self.t_compress_ms += t_compress_ms
        self.t_reclaim_ms += t_reclaim_ms
        self.t_override_prep_ms += t_override_prep_ms
        self.t_base_exec_ms += t_base_exec_ms
        self.t_total_exec_ms += t_total_exec_ms
        if self.total_steps % self.log_every_steps == 0:
            self._log_summary()

    def record_compression_events(self, events: list[dict[str, Any]] | None) -> None:
        if not self.enabled or not isinstance(events, list):
            return
        for event in events:
            if not isinstance(event, dict):
                continue
            self.compress_calls += 1
            status = str(event.get("status", ""))
            if status == "applied":
                self.compress_applied += 1
            elif status == "skipped":
                self.compress_skipped += 1
                reason = event.get("reason")
                if isinstance(reason, str):
                    self.skip_reasons[reason] += 1
            elif status == "error":
                self.compress_errors += 1

    def _log_summary(self) -> None:
        steps = max(1, self.total_steps)
        top_skips = ",".join(
            f"{reason}:{count}" for reason, count in self.skip_reasons.most_common(5)
        )
        line = (
            "TRIATTN_PERF "
            f"steps={self.total_steps} trig_steps={self.steps_with_trigger} "
            f"override_steps={self.steps_with_overrides} "
            f"compress_calls={self.compress_calls} applied={self.compress_applied} "
            f"skipped={self.compress_skipped} errors={self.compress_errors} "
            f"avg_ms(total={self.t_total_exec_ms / steps:.2f} "
            f"state={self.t_state_ms / steps:.2f} "
            f"compress={self.t_compress_ms / steps:.2f} "
            f"reclaim={self.t_reclaim_ms / steps:.2f} "
            f"override_prep={self.t_override_prep_ms / steps:.2f} "
            f"base_exec={self.t_base_exec_ms / steps:.2f}) "
            f"top_skip_reasons={top_skips or 'none'} "
            f"cudagraph_modes={dict(self.cudagraph_modes)}"
        )
        self.logger.info("%s", line)
        if self.sink_dir:
            try:
                sink_dir = Path(self.sink_dir)
                sink_dir.mkdir(parents=True, exist_ok=True)
                sink_path = sink_dir / f"triattn_perf_{os.getpid()}.log"
                with sink_path.open("a", encoding="utf-8") as fp:
                    fp.write(line + "\n")
            except Exception:
                pass

    def record_model_output(self, output: Any) -> None:
        if not self.enabled or output is None:
            return
        try:
            stats = getattr(output, "cudagraph_stats", None)
        except Exception:
            return
        if stats is None:
            return
        runtime_mode = None
        try:
            runtime_mode = getattr(stats, "runtime_mode", None)
        except Exception:
            runtime_mode = None
        if runtime_mode is not None:
            self.cudagraph_modes[str(runtime_mode)] += 1


class _Timer:
    __slots__ = ("_t0",)

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
