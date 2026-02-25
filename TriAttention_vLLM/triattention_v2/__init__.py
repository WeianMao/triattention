"""Compatibility package alias for the current TriAttention runtime implementation.

The active implementation moved to `triattention_runtime/` for clearer naming.
This package remains as a thin import-path compatibility layer so existing scripts
and tests importing `triattention_v2.*` continue to work.
"""

from __future__ import annotations

import triattention_runtime as _runtime  # type: ignore
from triattention_runtime import *  # type: ignore # noqa: F401,F403

# Forward submodule discovery (`import triattention_v2.foo`) to the runtime package
# directory so we do not need to duplicate wrapper files for each module.
__path__ = _runtime.__path__  # type: ignore[attr-defined]

try:
    __all__ = _runtime.__all__  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - defensive
    __all__ = []

