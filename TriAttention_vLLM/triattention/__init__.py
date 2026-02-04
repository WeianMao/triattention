"""TriAttention: Efficient KV cache compression for vLLM.

This package implements the TriAttention (SpeckV) algorithm for KV cache compression
with Triton kernel optimization and vLLM integration.

Supports both vLLM V0 and V1 APIs:
- V0: Use patch_vllm_attention() for monkey patching
- V1: Use --attention-backend TRIATTENTION with pip install -e .

Phase 1: Core implementation with Triton scoring + PyTorch TopK/Gather
Phase 2: Advanced optimizations and edge case handling
"""

from .compressor import TriAttentionCompressor
from .config import TriAttentionConfig
from .state import CompressionState
from .vllm_integration import (
    PagedKVCacheCompressor,
    TriAttentionWrapper,
    create_triattention_wrapper,
    patch_vllm_attention,
)

# V1 Backend exports (lazy import to avoid import errors if vLLM V1 not available)
def _get_v1_backend():
    """Lazy import of V1 backend to avoid import errors."""
    from .v1_backend import TriAttentionBackend, TriAttentionImpl
    return TriAttentionBackend, TriAttentionImpl


__version__ = "0.1.0"
__all__ = [
    # Core components
    "TriAttentionCompressor",
    "TriAttentionConfig",
    "CompressionState",
    # V0 integration
    "TriAttentionWrapper",
    "PagedKVCacheCompressor",
    "create_triattention_wrapper",
    "patch_vllm_attention",
    # V1 backend (lazy import)
    "_get_v1_backend",
]
