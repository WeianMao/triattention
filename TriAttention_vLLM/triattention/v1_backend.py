"""Legacy V1 custom attention backend entry (retired).

This module is intentionally kept as a compatibility stub so stale imports fail with a
clear message instead of silently using obsolete code paths.
"""

from __future__ import annotations


def __getattr__(name: str):
    raise RuntimeError(
        "triattention.v1_backend is retired. "
        "Use the current runner-side integration path "
        "(`evaluation/runner/vllm_triattention_runner.py`) instead."
    )

