"""Legacy backend package (retired).

The custom V1 backend route is no longer the active implementation path.
This package is kept only as a compatibility stub so stale imports fail clearly.
"""

from __future__ import annotations


def __getattr__(name: str):
    raise RuntimeError(
        "triattention.backends is retired. "
        "Use the current runner/dispatcher integration path instead of the legacy "
        "custom attention backend registration flow."
    )

