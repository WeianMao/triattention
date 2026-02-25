"""Legacy V0 monkeypatch integration entry (retired).

The active implementation is the runner/dispatcher integration in `triattention_runtime`.
This module remains only as a compatibility stub for stale imports.
"""

from __future__ import annotations


def __getattr__(name: str):
    raise RuntimeError(
        "triattention.vllm_integration (V0/V1 path) is retired. "
        "Use `evaluation/runner/vllm_triattention_runner.py` or dispatcher configs "
        "for the current implementation."
    )
