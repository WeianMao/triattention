"""TriAttention V1 Backend for vLLM.

This module provides TriAttentionBackend and TriAttentionImpl classes that
integrate with vLLM's V1 Engine attention infrastructure.

Design:
- Inherits from FlashAttentionBackend/FlashAttentionImpl
- Compression happens in forward() AFTER attention computation
- forward_includes_kv_cache_update=False means vLLM calls do_kv_cache_update()
  before forward(), so the KV cache is fully populated when forward() runs
- Execution order: do_kv_cache_update() -> forward() -> compress

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
    python -m vllm.entrypoints.openai.api_server --attention-backend CUSTOM

    # Option 2: Programmatic setup
    from triattention.backends import setup_triattention
    from triattention import TriAttentionConfig

    config = TriAttentionConfig(stats_path="stats.pt", kv_budget=2048)
    setup_triattention(config)

    from vllm import LLM
    llm = LLM(model="...", attention_backend="CUSTOM")
"""

import os
from pathlib import Path
from typing import Optional

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
    - forward_includes_kv_cache_update = False (inherited from FlashAttentionBackend)
      This tells vLLM to call do_kv_cache_update() BEFORE forward().
      So the KV cache is already populated when forward() runs.
    - Compression runs in forward() AFTER super().forward() completes attention
    - Uses FlashAttentionMetadataBuilder for metadata construction
    - All KV cache shape/stride methods inherited from FlashAttentionBackend
    """

    # Inherited from FlashAttentionBackend: forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        """Backend identifier for vLLM registry.

        Must match the AttentionBackendEnum member name used for registration.
        Since we register under CUSTOM, this must return "CUSTOM".
        """
        return "CUSTOM"

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
    after attention computation in forward().

    Execution Order (when forward_includes_kv_cache_update=False):
    1. vLLM calls do_kv_cache_update() - writes new K/V to cache
    2. vLLM calls forward() - we run attention, then compress

    Compression Flow (inside forward):
    1. Call super().forward() for attention computation
    2. Check if compression is needed (using attn_metadata.seq_lens)
    3. If needed: gather KV from paged cache, compress, scatter back

    State Management:
    - Lazy initialization of TriAttentionWrapper on first use
    - Per-request state tracked via request_id (batch_idx proxy)
    """

    # Throttle compression logging
    _compress_log_count = 0
    _reshape_logged = False

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
                # Propagate model info to config for proper GQA handling
                # in stats loading (mirrors V0's patch_vllm_attention logic)
                if config.num_kv_heads is None:
                    config.num_kv_heads = self._model_info["num_kv_heads"]
                if config.head_dim is None:
                    config.head_dim = self._model_info["head_dim"]

                from triattention.vllm_integration import TriAttentionWrapper
                self._wrapper = TriAttentionWrapper(config)
                quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
                if not quiet and self._layer_idx_counter == 0:
                    print(f"[TriAttention] V1 Impl initialized: "
                          f"kv_budget={config.kv_budget}, "
                          f"divide_length={config.divide_length}, "
                          f"num_kv_heads={config.num_kv_heads}")

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
        """Forward pass with FlashAttention + post-attention compression.

        Execution order (guaranteed by vLLM when forward_includes_kv_cache_update=False):
        1. do_kv_cache_update() already ran - KV cache has all tokens
        2. super().forward() - attention computation on full cache
        3. _maybe_compress_kv_cache() - compress cache for next step
        """
        # Step 1: Run attention computation
        result = super().forward(
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

        # Step 2: Apply compression after attention (decoder only)
        if self.attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            self._maybe_compress_kv_cache(kv_cache, attn_metadata)

        return result

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update KV cache (no compression here).

        Compression has been moved to forward() where attn_metadata is available.
        This method simply delegates to the parent implementation.
        """
        super().do_kv_cache_update(
            layer=layer,
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
        )

    def _maybe_compress_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> None:
        """Apply KV cache compression if conditions are met.

        This mirrors _apply_triattention_compression() from vllm_integration.py
        but adapted for the V1 backend where attn_metadata is available in forward().

        Args:
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
            attn_metadata: FlashAttentionMetadata with seq_lens and block_table
        """
        wrapper = self._get_wrapper()
        if wrapper is None:
            return

        # attn_metadata can be None during profile/dummy runs
        if attn_metadata is None:
            return

        layer_idx = self._layer_idx_counter

        # V1 API: FlashAttentionMetadata has block_table and seq_lens directly
        block_tables = getattr(attn_metadata, 'block_table', None)
        seq_lens_tensor = getattr(attn_metadata, 'seq_lens', None)

        if block_tables is None or seq_lens_tensor is None:
            return

        # Extract model config
        block_size = self._model_info["block_size"]
        num_kv_heads = self._model_info["num_kv_heads"]
        head_dim = self._model_info["head_dim"]

        # Update block_size from actual KV cache shape
        if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
            block_size = kv_cache.shape[2]
            self._model_info["block_size"] = block_size

        # Handle flattened cache format: [2, num_blocks, block_size * num_kv_heads * head_dim]
        was_reshaped = False
        if kv_cache.dim() == 3 and kv_cache.shape[0] == 2:
            num_blocks = kv_cache.shape[1]
            flattened_size = kv_cache.shape[2]
            expected_size = block_size * num_kv_heads * head_dim

            if flattened_size == expected_size:
                if not TriAttentionImpl._reshape_logged:
                    print("[TriAttention] Detected flattened cache format, "
                          "reshaping to block format")
                    TriAttentionImpl._reshape_logged = True
                kv_cache = kv_cache.view(
                    2, num_blocks, block_size, num_kv_heads, head_dim
                )
                was_reshaped = True
            else:
                quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
                if not quiet:
                    print(f"[TriAttention] Warning: Flattened cache size mismatch. "
                          f"Expected {expected_size}, got {flattened_size}. "
                          f"Skipping compression.")
                return

        # Split cache into key and value
        if isinstance(kv_cache, tuple):
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
        elif kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
        else:
            quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
            if not quiet:
                print(f"[TriAttention] Warning: Unexpected kv_cache format: "
                      f"shape={kv_cache.shape}")
            return

        # Process each sequence in the batch
        batch_size = block_tables.shape[0]

        for batch_idx in range(batch_size):
            seq_len = seq_lens_tensor[batch_idx].item()
            block_table = block_tables[batch_idx]

            # Use batch_idx as request identifier
            request_id = f"decode_{batch_idx}"

            # Detect request transition: if seq_len dropped below the tracked
            # absolute_position, a new (shorter) request has started on this
            # slot. Reset compressor state so it doesn't carry over stale
            # position/cache-length from the previous request.
            compressor = wrapper.get_compressor(layer_idx, request_id)
            if (compressor.state.absolute_position > 0
                    and seq_len < compressor.state.absolute_position):
                quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
                if not quiet and layer_idx == 0:
                    print(f"[TriAttention] New request detected on slot "
                          f"{batch_idx}: seq_len={seq_len} < "
                          f"prev_pos={compressor.state.absolute_position}, "
                          f"resetting state")
                compressor.reset()

            # Check if compression is needed
            if not wrapper.should_compress(layer_idx, seq_len, request_id):
                continue

            try:
                # Log compression start (throttled, layer 0 only)
                if layer_idx == 0:
                    TriAttentionImpl._compress_log_count += 1
                    if TriAttentionImpl._compress_log_count <= 5:
                        print(f"[TriAttention] Compressing: seq_len={seq_len} "
                              f"-> budget={wrapper.config.kv_budget}")

                # Gather KV from paged cache to dense format
                from triattention.vllm_integration import (
                    _gather_kv_from_paged_cache,
                    _scatter_kv_to_paged_cache,
                )

                keys, values = _gather_kv_from_paged_cache(
                    key_cache, value_cache, block_table, seq_len, block_size
                )

                # Create position indices
                cache_positions = torch.arange(
                    seq_len, device=key_cache.device, dtype=torch.int32
                )

                # Compress using the compressor (shared state + stats)
                compressed_keys, compressed_values, new_positions = (
                    compressor.compress(
                        key_states=keys,
                        value_states=values,
                        cache_positions=cache_positions,
                        layer_idx=layer_idx,
                    )
                )

                # Scatter compressed data back to paged cache
                _scatter_kv_to_paged_cache(
                    compressed_keys, compressed_values,
                    key_cache, value_cache,
                    block_table, new_positions, block_size,
                )

                new_seq_len = compressed_keys.shape[2]

                # Log compression result (throttled, layer 0 only)
                if layer_idx == 0:
                    if TriAttentionImpl._compress_log_count <= 5:
                        print(f"[TriAttention] Compressed: "
                              f"{seq_len} -> {new_seq_len} tokens")

            except Exception as e:
                quiet = os.environ.get("TRIATTENTION_QUIET", "0") == "1"
                if not quiet:
                    import traceback
                    print(f"[TriAttention] Compression error for batch "
                          f"{batch_idx}, layer {layer_idx}: {e}")
                    traceback.print_exc()


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
