"""vLLM plugin to register TriAttention backend.

This module is automatically loaded by vLLM via the entry point defined
in pyproject.toml. It registers TriAttention as a V1 attention backend.

Usage:
    1. Install triattention package: pip install -e .
    2. Set environment variables for configuration:
       - TRIATTENTION_STATS_PATH: Path to frequency statistics file
       - TRIATTENTION_KV_BUDGET: Maximum KV tokens to retain (default: 2048)
       - TRIATTENTION_DIVIDE_LENGTH: Compression interval (default: 128)
    3. Run vLLM with: --attention-backend TRIATTENTION
"""

import os


def register_triattention_backend():
    """Register TriAttention as a vLLM V1 attention backend.

    This function is called automatically by vLLM's plugin system.
    It registers the TriAttention backend class path with vLLM's
    backend registry.
    """
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )

        # Register TriAttention backend with vLLM
        register_backend(
            AttentionBackendEnum.TRIATTENTION,
            "triattention.v1_backend.TriAttentionBackend"
        )

        # Print registration message if not suppressed
        if os.environ.get("TRIATTENTION_QUIET", "0") != "1":
            print("[TriAttention] V1 Backend registered successfully")

    except ImportError as e:
        # vLLM V1 not available, skip registration
        print(f"[TriAttention] Warning: Failed to register V1 backend: {e}")
        print("[TriAttention] This is expected if using vLLM V0 API")
    except Exception as e:
        print(f"[TriAttention] Error during backend registration: {e}")
        raise
