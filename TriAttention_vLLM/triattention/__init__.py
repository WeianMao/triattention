"""TriAttention shared core package (current default import path).

This package now serves two purposes:
1. Shared scoring/compressor utilities used by the current runtime implementation.
2. A stable public import path (`triattention`) while the current vLLM runtime
   implementation remains in the internal compatibility package `triattention_v2`.

Legacy V0/V1 vLLM integrations are intentionally retired from the active package
surface. Compatibility symbols remain as explicit error stubs to avoid silent misuse.
"""

from __future__ import annotations

from .compressor import TriAttentionCompressor
from .config import TriAttentionConfig
from .state import CompressionState
from .utils import load_frequency_stats, normalize_scores


def _legacy_api_removed(*_args, **_kwargs):
    raise RuntimeError(
        "Legacy TriAttention V0/V1 vLLM integrations were removed from the active package. "
        "Use the current runtime via "
        "`evaluation/runner/vllm_triattention_runner.py` (or the dispatcher configs), "
        "and keep legacy code under repository_archive if needed."
    )


# Explicit legacy stubs (kept only to fail clearly for stale imports).
TriAttentionWrapper = _legacy_api_removed
PagedKVCacheCompressor = _legacy_api_removed
create_triattention_wrapper = _legacy_api_removed
patch_vllm_attention = _legacy_api_removed


def _get_v1_backend():
    _legacy_api_removed()


__version__ = "0.1.0"
__all__ = [
    # Shared scoring/compression primitives
    "TriAttentionCompressor",
    "TriAttentionConfig",
    "CompressionState",
    "normalize_scores",
    "load_frequency_stats",
    # Current/default runtime exports
    "TriAttentionRuntimeConfig",
    "TriAttentionV2Config",
    "install_vllm_integration_monkeypatches",
    "install_runner_compression_hook",
    # Legacy stub exports (explicit fail-fast)
    "TriAttentionWrapper",
    "PagedKVCacheCompressor",
    "create_triattention_wrapper",
    "patch_vllm_attention",
    "_get_v1_backend",
]


def __getattr__(name: str):
    """Lazy-export current runtime symbols to avoid import cycles with triattention_v2."""
    if name in {
        "TriAttentionRuntimeConfig",
        "TriAttentionV2Config",
        "install_runner_compression_hook",
    }:
        from triattention_v2 import (  # type: ignore
            TriAttentionConfig as _TriAttentionRuntimeConfig,
        )
        from triattention_v2 import (  # type: ignore
            TriAttentionV2Config as _TriAttentionV2Config,
            install_runner_compression_hook as _install_runner_compression_hook,
        )

        mapping = {
            "TriAttentionRuntimeConfig": _TriAttentionRuntimeConfig,
            "TriAttentionV2Config": _TriAttentionV2Config,
            "install_runner_compression_hook": _install_runner_compression_hook,
        }
        return mapping[name]
    if name == "install_vllm_integration_monkeypatches":
        from triattention_v2.integration_monkeypatch import (  # type: ignore
            install_vllm_integration_monkeypatches as _install,
        )

        return _install
    raise AttributeError(name)
