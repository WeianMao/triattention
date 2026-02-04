"""TriAttention backend class.

This module provides the TriAttentionBackend class that inherits from
vLLM's FlashAttentionBackend and integrates KV cache compression.
"""

from typing import List, Tuple, Type

import torch

try:
    from vllm.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionMetadata
    from vllm.attention.backends.abstract import AttentionMetadata
except ImportError as e:
    raise ImportError(
        "Failed to import vLLM attention backends. "
        "Make sure vLLM is installed correctly."
    ) from e


class TriAttentionBackend(FlashAttentionBackend):
    """TriAttention backend that extends FlashAttention with KV cache compression.

    This backend integrates TriAttention compression into vLLM's attention mechanism
    by wrapping FlashAttention and applying compression during decode steps.

    Architecture:
    - Inherits from FlashAttentionBackend to maintain compatibility
    - Overrides get_impl_cls() to return TriAttentionImpl
    - Reuses all other parent methods (KV cache shape, swap/copy, metadata)

    Design Philosophy:
    - Minimal code duplication - only override what's necessary
    - Transparent to vLLM - behaves like FlashAttention from vLLM's perspective
    - Compression happens inside forward() - invisible to scheduler/executor
    """

    @staticmethod
    def get_name() -> str:
        """Get backend name for vLLM registration.

        Returns:
            Backend identifier string
        """
        return "TRIATTENTION"

    @staticmethod
    def get_impl_cls() -> Type["TriAttentionImpl"]:
        """Get the implementation class for this backend.

        Returns:
            TriAttentionImpl class
        """
        from .triattention_impl import TriAttentionImpl
        return TriAttentionImpl

    # All other methods inherited from FlashAttentionBackend:
    # - get_supported_head_sizes()
    # - get_metadata_cls() -> FlashAttentionMetadata
    # - get_builder_cls() -> FlashAttentionMetadataBuilder
    # - get_state_cls() -> CommonAttentionState
    # - get_kv_cache_shape()
    # - swap_blocks()
    # - copy_blocks()
