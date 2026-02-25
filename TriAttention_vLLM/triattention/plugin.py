"""Compatibility vLLM plugin entrypoint (legacy V1 backend registration retired).

The package entrypoint is kept to avoid import-time failures in environments that
auto-load installed vLLM plugins. Current TriAttention integration uses the runner-
side monkeypatch path, not the historical V1 custom attention backend plugin.
"""

from __future__ import annotations

import os


def register_triattention_backend():
    """No-op compatibility hook for retired V0/V1 backend plugin registration."""
    if os.environ.get("TRIATTENTION_QUIET", "0") != "1":
        print(
            "[TriAttention] Legacy V1 backend plugin registration is retired; "
            "using current runner/dispatcher integration path."
        )
