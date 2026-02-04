"""TriAttention implementation class.

This module provides the TriAttentionImpl class that wraps FlashAttentionImpl
with KV cache compression logic.
"""

from typing import Optional

import torch

try:
    from vllm.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
    from vllm.attention.backends.abstract import AttentionLayer, AttentionType
except ImportError as e:
    raise ImportError(
        "Failed to import vLLM attention implementation. "
        "Make sure vLLM is installed correctly."
    ) from e


class TriAttentionImpl(FlashAttentionImpl):
    """TriAttention implementation that extends FlashAttention with compression.

    This class inherits from FlashAttentionImpl and adds KV cache compression
    logic that runs after the parent's forward pass during decode steps.

    Integration Strategy:
    - Call parent's forward() first to populate KV cache
    - Check if compression is needed (decode step + threshold exceeded)
    - Apply compression using existing vllm_integration._apply_triattention_compression
    - Return the same output (compression doesn't affect current token)

    State Management:
    - Uses wrapper from setup_triattention() for configuration
    - Wrapper handles per-request state isolation
    - Model info (block_size, num_kv_heads, head_dim) extracted from layer

    Design Philosophy:
    - Minimal changes to parent class - only wrap forward()
    - Reuse existing compression logic from vllm_integration
    - Transparent to vLLM - same interface, same outputs
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        """Initialize TriAttention implementation.

        Args:
            num_heads: Number of query heads
            head_size: Dimension of each head
            scale: Attention scaling factor
            num_kv_heads: Number of KV heads (for GQA)
            alibi_slopes: ALiBi slopes for position encoding
            sliding_window: Sliding window size
            kv_cache_dtype: KV cache data type
            blocksparse_params: Block-sparse attention parameters
            logits_soft_cap: Logits soft capping value
            attn_type: Attention type (decoder/encoder/cross-attention)
        """
        # Initialize parent class
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            blocksparse_params=blocksparse_params,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
        )

        # Store model info for compression
        self._model_info = {
            'num_kv_heads': num_kv_heads,
            'head_dim': head_size,
            'block_size': 16,  # Default vLLM block size
        }

        # Cache wrapper reference (loaded lazily)
        self._wrapper = None
        self._wrapper_loaded = False

    def _get_wrapper(self):
        """Get TriAttention wrapper from global config.

        Returns:
            TriAttentionWrapper instance, or None if not configured
        """
        if self._wrapper_loaded:
            return self._wrapper

        try:
            from triattention.backends import get_triattention_config
            from triattention.vllm_integration import TriAttentionWrapper

            config = get_triattention_config()
            if config is None:
                print("[TriAttention] Warning: Config not set. Call setup_triattention() first.")
                self._wrapper = None
            else:
                # Create wrapper with config
                self._wrapper = TriAttentionWrapper(config)
                print(f"[TriAttention] Wrapper initialized: "
                      f"kv_budget={config.kv_budget}, divide_length={config.divide_length}")
        except Exception as e:
            print(f"[TriAttention] Failed to load wrapper: {e}")
            self._wrapper = None

        self._wrapper_loaded = True
        return self._wrapper

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with TriAttention compression.

        This method:
        1. Calls parent's forward() to populate KV cache
        2. Checks if compression should be applied (decode step + threshold)
        3. Applies compression to KV cache if needed
        4. Returns the output (unchanged by compression)

        Args:
            layer: Attention layer with KV cache parameters
            query: Query tensor [num_tokens, num_heads, head_size]
            key: Key tensor [num_tokens, num_kv_heads, head_size]
            value: Value tensor [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: FlashAttentionMetadata with sequence info
            output: Output tensor [num_tokens, num_heads, head_size]

        Returns:
            Output tensor with attention results
        """
        # Step 1: Call parent's forward to populate KV cache
        result = super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )

        # Step 2: Apply compression if conditions are met
        # Get wrapper (lazily loaded)
        wrapper = self._get_wrapper()
        if wrapper is None:
            # No compression configured
            return result

        # Check if this is a decode step
        # V1 API uses max_query_len (decode: 1, prefill: >1)
        # V0 API uses decode_metadata
        is_decode = False
        if hasattr(attn_metadata, 'max_query_len'):
            is_decode = attn_metadata.max_query_len == 1
        elif hasattr(attn_metadata, 'decode_metadata'):
            is_decode = attn_metadata.decode_metadata is not None

        # Only compress during decode when cache is populated
        if not is_decode or kv_cache.numel() == 0:
            return result

        # Step 3: Apply compression
        try:
            # Import compression function
            from triattention.vllm_integration import _apply_triattention_compression

            # Get layer index from attention layer if available
            # Note: vLLM doesn't always expose layer_idx, so we use a heuristic
            # In practice, this should be set by the model or passed via metadata
            layer_idx = getattr(layer, 'layer_idx', 0)

            # Extract block_size from cache_config if available
            # Otherwise use default
            if hasattr(layer, 'cache_config') and hasattr(layer.cache_config, 'block_size'):
                self._model_info['block_size'] = layer.cache_config.block_size

            # Apply compression
            _apply_triattention_compression(
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                tri_wrapper=wrapper,
                layer_idx=layer_idx,
                model_info=self._model_info,
            )

        except Exception as e:
            # Log but don't crash on compression errors
            # This ensures graceful degradation if compression fails
            print(f"[TriAttention] Compression failed: {e}")
            import traceback
            traceback.print_exc()

        # Step 4: Return the output (unchanged)
        return result
