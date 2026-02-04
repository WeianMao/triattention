"""TriAttention backend for vLLM.

This module provides attention backends that integrate TriAttention
KV cache compression with vLLM's FlashAttention backend.

Supports both V0 and V1 APIs:
- V0: Use triattention_backend.py / triattention_impl.py
- V1: Use ../v1_backend.py (registered via plugin entry point)

Usage (V0):
    from triattention.backends import register_triattention_backend, setup_triattention
    from triattention import TriAttentionConfig

    config = TriAttentionConfig(stats_path="stats.pt", kv_budget=2048)
    setup_triattention(config)
    register_triattention_backend()

    from vllm import LLM
    llm = LLM(model="...", enforce_eager=True)

Usage (V1):
    # Install package: pip install -e .
    # Set env vars: TRIATTENTION_STATS_PATH=/path/to/stats.pt
    # Run: python -m vllm ... --attention-backend TRIATTENTION
"""

from .triattention_backend import TriAttentionBackend
from .triattention_impl import TriAttentionImpl

__all__ = [
    "TriAttentionBackend",
    "TriAttentionImpl",
    "register_triattention_backend",
    "setup_triattention",
    "get_triattention_config",
]

# Global configuration storage
_TRIATTENTION_CONFIG = None


def setup_triattention(config) -> None:
    """Configure TriAttention compression parameters.

    This must be called before using vLLM with TriAttention backend.

    Args:
        config: TriAttentionConfig instance

    Example:
        from triattention import TriAttentionConfig
        config = TriAttentionConfig(stats_path="stats.pt", kv_budget=2048)
        setup_triattention(config)
    """
    global _TRIATTENTION_CONFIG
    _TRIATTENTION_CONFIG = config


def get_triattention_config():
    """Get the current TriAttention configuration.

    Returns:
        TriAttentionConfig instance, or None if not configured
    """
    return _TRIATTENTION_CONFIG


def register_triattention_backend() -> None:
    """Register TriAttention backend with vLLM.

    This function patches vLLM's attention backend registry to use TriAttention.
    Must be called after setup_triattention() and before creating LLM instance.

    Raises:
        RuntimeError: If TriAttention config is not set up
        ImportError: If vLLM is not available or incompatible

    Example:
        from triattention.backends import setup_triattention, register_triattention_backend
        from triattention import TriAttentionConfig

        config = TriAttentionConfig(stats_path="stats.pt", kv_budget=2048)
        setup_triattention(config)
        register_triattention_backend()

        # Now vLLM will use TriAttention backend
        from vllm import LLM
        llm = LLM(model="...", enforce_eager=True)
    """
    if _TRIATTENTION_CONFIG is None:
        raise RuntimeError(
            "TriAttention config not set. Call setup_triattention(config) first."
        )

    try:
        from vllm.attention.backends.flash_attn import FlashAttentionBackend
    except ImportError as e:
        raise ImportError(
            "Failed to import vLLM FlashAttentionBackend. "
            "Make sure vLLM is installed and compatible."
        ) from e

    # Verify that TriAttentionBackend is a proper subclass
    if not issubclass(TriAttentionBackend, FlashAttentionBackend):
        raise RuntimeError(
            "TriAttentionBackend must inherit from FlashAttentionBackend"
        )

    print("[TriAttention] Backend registered successfully")
    print(f"[TriAttention] Config: kv_budget={_TRIATTENTION_CONFIG.kv_budget}, "
          f"divide_length={_TRIATTENTION_CONFIG.divide_length}")
