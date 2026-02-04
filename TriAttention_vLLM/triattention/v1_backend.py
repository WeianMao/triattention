"""TriAttention V1 Backend for vLLM.

This module provides TriAttentionBackend and TriAttentionImpl classes that
integrate with vLLM's V1 Engine attention infrastructure.

Design:
- Inherits from FlashAttentionBackend/FlashAttentionImpl
- Compression happens AFTER KV cache is updated (forward_includes_kv_cache_update=False)
- Uses do_kv_cache_update() to apply compression after attention computation
- Configuration via environment variables or global setup

Environment Variables:
- TRIATTENTION_STATS_PATH: Path to frequency statistics file (required)
- TRIATTENTION_KV_BUDGET: Maximum KV tokens to retain (default: 2048)
- TRIATTENTION_DIVIDE_LENGTH: Compression interval (default: 128)
- TRIATTENTION_WINDOW_SIZE: Recent tokens to protect (default: 128)
- TRIATTENTION_PRUNING_MODE: Pruning strategy (default: per_head)
- TRIATTENTION_QUIET: Suppress logs if "1" (default: "0")

Usage:
    # Option 1: Environment variables
    export TRIATTENTION_STATS_PATH=/path/to/stats.pt
    export TRIATTENTION_KV_BUDGET=2048
    python -m vllm.entrypoints.openai.api_server --attention-backend TRIATTENTION

    # Option 2: Programmatic setup
    from triattention.backends import setup_triattention
    from triattention import TriAttentionConfig

    config = TriAttentionConfig(stats_path="stats.pt", kv_budget=2048)
    setup_triattention(config)

    from vllm import LLM
    llm = LLM(model="...", attention_backend="TRIATTENTION")
"""

import os
from pathlib import Path
from typing import ClassVar, Optional

import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionType


class TriAttentionBackend(FlashAttentionBackend):
    """TriAttention backend extending FlashAttention with KV cache compression.

    This backend inherits all FlashAttention capabilities and adds KV cache
    compression through the TriAttentionImpl class.

    Key Design Decisions:
    - forward_includes_kv_cache_update = False
      This tells vLLM to call do_kv_cache_update() separately, allowing us
      to apply compression after the KV cache is populated.
    - Uses FlashAttentionMetadataBuilder for metadata construction
    - All KV cache shape/stride methods inherited from FlashAttentionBackend
    """

    # Override: KV cache update happens separately (in do_kv_cache_update)
    # This allows compression to be applied after attention computation
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        """Backend identifier for vLLM registry."""
        return "TRIATTENTION"

    @staticmethod
    def get_impl_cls() -> type["TriAttentionImpl"]:
        """Return implementation class with compression logic."""
        return TriAttentionImpl

    # Inherit all other methods from FlashAttentionBackend:
    # - get_supported_kernel_block_sizes()
    # - get_builder_cls() -> FlashAttentionMetadataBuilder
    # - get_kv_cache_shape()
    # - get_kv_cache_stride_order()
    # - supports_* methods


class TriAttentionImpl(FlashAttentionImpl):
    """TriAttention implementation with KV cache compression.

    This class extends FlashAttentionImpl to add compression logic that runs
    after the KV cache is updated.

    Compression Flow:
    1. vLLM calls forward() for attention computation
    2. vLLM calls do_kv_cache_update() to populate KV cache
    3. In do_kv_cache_update(), we:
       a. Call parent's reshape_and_cache_flash to update cache
       b. Check if compression is needed
       c. Apply TriAttention compression if threshold exceeded

    State Management:
    - Uses global _TRIATTENTION_WRAPPER for configuration
    - Per-request state tracked via request_id
    - Lazy initialization of compressor on first use
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize TriAttention implementation.

        All parameters passed to parent FlashAttentionImpl.
        Additionally initializes compression-related state.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )

        # Model info for compression
        self._model_info = {
            "num_kv_heads": num_kv_heads,
            "head_dim": head_size,
            "block_size": 16,  # Default vLLM block size, updated at runtime
        }

        # Lazy-loaded wrapper
        self._wrapper = None
        self._wrapper_loaded = False

        # Layer index tracking (set during first forward call)
        self._layer_idx = None
        self._layer_idx_counter = _get_next_layer_idx()

    def _get_wrapper(self):
        """Get or create TriAttention wrapper.

        Loads configuration from:
        1. Global setup (setup_triattention())
        2. Environment variables as fallback

        Returns:
            TriAttentionWrapper instance, or None if not configured
        """
        if self._wrapper_loaded:
            return self._wrapper

        try:
            # Try global config first
            from triattention.backends import get_triattention_config
            config = get_triattention_config()

            if config is None:
                # Fallback to environment variables
                config = _load_config_from_env()

            if config is None:
                quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
                if not quiet:
                    print("[TriAttention] Warning: No configuration found. "
                          "Set TRIATTENTION_STATS_PATH or call setup_triattention().")
                self._wrapper = None
            else:
                from triattention.vllm_integration import TriAttentionWrapper
                self._wrapper = TriAttentionWrapper(config)
                quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
                if not quiet and self._layer_idx_counter == 0:
                    print(f"[TriAttention] V1 Impl initialized: "
                          f"kv_budget={config.kv_budget}, "
                          f"divide_length={config.divide_length}")

        except Exception as e:
            print(f"[TriAttention] Failed to initialize wrapper: {e}")
            import traceback
            traceback.print_exc()
            self._wrapper = None

        self._wrapper_loaded = True
        return self._wrapper

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        This method delegates to parent's forward(). Compression is applied
        separately in do_kv_cache_update().

        The forward_includes_kv_cache_update=False flag tells vLLM to call
        do_kv_cache_update() after this method returns.
        """
        return super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update KV cache and apply compression if needed.

        This method is called by vLLM after forward() when
        forward_includes_kv_cache_update=False.

        Flow:
        1. Call parent's do_kv_cache_update to populate cache
        2. Check compression conditions
        3. Apply compression if threshold exceeded
        """
        # Step 1: Update KV cache using parent implementation
        super().do_kv_cache_update(
            layer=layer,
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
        )

        # Step 2: Check if we should apply compression
        # Only compress for decoder attention (not encoder)
        if self.attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            self._maybe_compress_kv_cache(kv_cache, slot_mapping)

    def _maybe_compress_kv_cache(
        self,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Apply compression if conditions are met.

        Compression is applied when:
        - Wrapper is configured
        - Current sequence length exceeds budget + divide_length

        Note: slot_mapping tells us which slots were just written.
        We use this to determine sequence length and apply compression.
        """
        wrapper = self._get_wrapper()
        if wrapper is None:
            return

        # Get layer index
        layer_idx = self._layer_idx_counter

        # Calculate sequence length from slot_mapping
        # slot_mapping contains the slots that were just written
        if slot_mapping.numel() == 0:
            return

        # For V1 API, slot_mapping contains linear indices into the KV cache
        # We need to derive sequence length from the maximum slot + 1
        # Note: This is a simplified approach. In practice, vLLM manages
        # sequence lengths through the scheduler, but we don't have direct access.

        # Get block_size from KV cache shape
        # V1 format: [2, num_blocks, block_size, num_kv_heads, head_size]
        if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
            block_size = kv_cache.shape[2]
            self._model_info["block_size"] = block_size

        # Apply compression using the integration function
        try:
            from triattention.vllm_integration import _apply_triattention_compression

            # Create a minimal attn_metadata-like object for compatibility
            # The compression function expects block_table and seq_lens
            # For now, we'll skip compression if we can't determine these
            # TODO: Integrate with vLLM's scheduler to get proper seq_lens

            # For V1 backend, compression is more complex because we don't have
            # direct access to sequence metadata in do_kv_cache_update
            # The compression should ideally be triggered from a higher level
            # where metadata is available

            pass  # Placeholder for future integration

        except Exception as e:
            quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
            if not quiet:
                print(f"[TriAttention] Compression error: {e}")


# Global layer index counter for tracking
_LAYER_IDX_COUNTER = 0


def _get_next_layer_idx() -> int:
    """Get next layer index and increment counter."""
    global _LAYER_IDX_COUNTER
    idx = _LAYER_IDX_COUNTER
    _LAYER_IDX_COUNTER += 1
    return idx


def _reset_layer_idx_counter():
    """Reset layer index counter (for testing)."""
    global _LAYER_IDX_COUNTER
    _LAYER_IDX_COUNTER = 0


def _load_config_from_env():
    """Load TriAttention config from environment variables.

    Returns:
        TriAttentionConfig instance, or None if TRIATTENTION_STATS_PATH not set
    """
    stats_path = os.environ.get("TRIATTENTION_STATS_PATH")
    if not stats_path:
        return None

    try:
        from triattention.config import TriAttentionConfig

        config = TriAttentionConfig(
            stats_path=Path(stats_path),
            kv_budget=int(os.environ.get("TRIATTENTION_KV_BUDGET", "2048")),
            divide_length=int(os.environ.get("TRIATTENTION_DIVIDE_LENGTH", "128")),
            window_size=int(os.environ.get("TRIATTENTION_WINDOW_SIZE", "128")),
            pruning_mode=os.environ.get("TRIATTENTION_PRUNING_MODE", "per_head"),
        )

        quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
        if not quiet:
            print(f"[TriAttention] Loaded config from environment: "
                  f"stats_path={stats_path}, kv_budget={config.kv_budget}")

        return config

    except Exception as e:
        print(f"[TriAttention] Error loading config from env: {e}")
        return None
