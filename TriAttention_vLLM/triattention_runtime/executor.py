"""Compression executor abstractions for TriAttention v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .signals import CompressionSignal


@dataclass(frozen=True)
class CompressionExecutionResult:
    applied: bool
    reason: str
    cache_len_after: int | None = None
    details: dict[str, Any] | None = None


class CompressionExecutor:
    """Executor interface for request-level compression actions."""

    def execute(
        self,
        req_id: str,
        signal: CompressionSignal,
        scheduler_output: Any,
    ) -> CompressionExecutionResult:
        raise NotImplementedError


class RunnerHookCompressionExecutor(CompressionExecutor):
    """Default executor that delegates to an optional base-runner hook.

    Hook contract:
    - hook name: ``triattention_apply_compression``
    - input args: ``req_id``, ``signal``, ``scheduler_output``
    - output:
      1) bool: True means applied;
      2) dict: {"applied": bool, "reason": str, "cache_len_after": int|None}
    """

    hook_name = "triattention_apply_compression"

    def __init__(self, base_runner: Any):
        self._base_runner = base_runner

    def execute(
        self,
        req_id: str,
        signal: CompressionSignal,
        scheduler_output: Any,
    ) -> CompressionExecutionResult:
        hook = getattr(self._base_runner, self.hook_name, None)
        if not callable(hook):
            return CompressionExecutionResult(
                applied=False,
                reason="runner_hook_missing",
                cache_len_after=None,
            )

        hook_result = hook(
            req_id=req_id,
            signal=signal,
            scheduler_output=scheduler_output,
        )
        if isinstance(hook_result, bool):
            return CompressionExecutionResult(
                applied=hook_result,
                reason="applied" if hook_result else "runner_hook_rejected",
            )

        if isinstance(hook_result, dict):
            details = dict(hook_result)
            return CompressionExecutionResult(
                applied=bool(details.get("applied", False)),
                reason=str(details.get("reason", "runner_hook_result")),
                cache_len_after=details.get("cache_len_after"),
                details=details,
            )

        return CompressionExecutionResult(
            applied=False,
            reason=f"unsupported_hook_result:{type(hook_result).__name__}",
        )
