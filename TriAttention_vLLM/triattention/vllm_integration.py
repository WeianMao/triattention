"""vLLM integration layer for TriAttention.

This module provides non-invasive integration with vLLM's PagedAttention.
The integration wraps the attention mechanism to add KV cache compression.

Integration Point: FlashAttentionImpl.forward()
Strategy: Wrap reshape_and_cache_flash to intercept KV cache updates

Usage:
    from triattention.vllm_integration import TriAttentionWrapper

    # Initialize wrapper with config
    wrapper = TriAttentionWrapper(config)

    # Patch vLLM attention
    wrapper.patch_attention_impl(model)
"""
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from .compressor import TriAttentionCompressor
from .config import TriAttentionConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import Attention


class TriAttentionWrapper:
    """Wrapper for integrating TriAttention with vLLM.

    This wrapper intercepts KV cache operations to apply compression
    without modifying vLLM source code.

    Per-Request State Isolation:
    - Each request maintains its own compressor state per layer
    - State includes: absolute_position, prefill_length, compression_count
    - Request lifecycle: register() -> compress() calls -> unregister()

    Request Lifecycle Management:

    1. Request Start:
        wrapper.register_request(request_id)

    2. During Generation:
        - Compressor automatically tracks state per request
        - Compression triggered when cache reaches (budget + divide_length)
        - State persists across decode steps

    3. Request Complete/Cancel:
        wrapper.unregister_request(request_id)

    CRITICAL: You MUST unregister requests when:
    - Request completes successfully
    - Request is cancelled or fails
    - KV cache slot is reused for new request

    Failure to unregister causes:
    - Memory leaks (state accumulation)
    - State pollution (wrong compression for new requests)
    - Incorrect compression decisions

    Integration with vLLM Scheduler:
    - Hook into scheduler._free_seq() to call unregister_request()
    - Hook into request completion callbacks
    - For slot reuse: unregister before assigning new request to slot
    """

    def __init__(
        self,
        config: TriAttentionConfig,
        enabled_layers: Optional[set] = None,
    ):
        """Initialize TriAttention wrapper.

        Args:
            config: TriAttention configuration
            enabled_layers: Set of layer indices to apply compression.
                           If None, compression is applied to all layers.
        """
        self.config = config
        self.enabled_layers = enabled_layers

        # Per-request compressor storage: {request_id: {layer_idx: compressor}}
        self.request_compressors: Dict[str, Dict[int, TriAttentionCompressor]] = {}

        # Default request ID for backward compatibility
        self._default_request_id = "__default__"

        self._patched = False

    def register_request(self, request_id: str) -> None:
        """Register a new request and initialize its compressor state.

        This should be called when a new request starts processing.

        Args:
            request_id: Unique identifier for the request
        """
        if request_id in self.request_compressors:
            # Request already registered, reset its state instead
            self._reset_request(request_id)
        else:
            self.request_compressors[request_id] = {}

    def unregister_request(self, request_id: str) -> None:
        """Unregister a request and cleanup its state.

        This should be called when:
        1. Request completes successfully
        2. Request is cancelled or fails
        3. KV cache slot is being reused for a new request

        Args:
            request_id: Unique identifier for the request
        """
        if request_id in self.request_compressors:
            # Reset all compressors for this request
            for compressor in self.request_compressors[request_id].values():
                compressor.reset()
            # Remove request entry
            del self.request_compressors[request_id]

    def _reset_request(self, request_id: str) -> None:
        """Reset all compressor states for a request without unregistering.

        Args:
            request_id: Unique identifier for the request
        """
        if request_id in self.request_compressors:
            for compressor in self.request_compressors[request_id].values():
                compressor.reset()

    def get_compressor(
        self,
        layer_idx: int,
        request_id: Optional[str] = None,
    ) -> TriAttentionCompressor:
        """Get or create compressor for a specific request and layer.

        Args:
            layer_idx: Layer index
            request_id: Request identifier. If None, uses default request.

        Returns:
            TriAttentionCompressor instance for the request-layer pair

        Raises:
            ValueError: If layer_idx is invalid
        """
        # Validate layer_idx
        if self.config.num_layers is not None:
            if layer_idx < 0 or layer_idx >= self.config.num_layers:
                raise ValueError(
                    f"Invalid layer_idx {layer_idx}. Expected 0 <= layer_idx < {self.config.num_layers}"
                )

        # Use default request ID for backward compatibility
        if request_id is None:
            request_id = self._default_request_id

        # Auto-register request if not already registered
        if request_id not in self.request_compressors:
            self.register_request(request_id)

        # Get or create compressor for this request-layer pair
        if layer_idx not in self.request_compressors[request_id]:
            self.request_compressors[request_id][layer_idx] = TriAttentionCompressor(
                self.config
            )

        return self.request_compressors[request_id][layer_idx]

    def should_compress(
        self,
        layer_idx: int,
        seq_len: int,
        request_id: Optional[str] = None,
    ) -> bool:
        """Check if compression should be applied.

        Args:
            layer_idx: Current layer index
            seq_len: Current sequence length
            request_id: Request identifier. If None, uses default request.

        Returns:
            True if compression should be applied
        """
        if self.enabled_layers is not None and layer_idx not in self.enabled_layers:
            return False

        compressor = self.get_compressor(layer_idx, request_id)
        return compressor.state.should_compress(seq_len)

    def compress_kv_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_positions: torch.Tensor,
        layer_idx: int,
        request_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache using TriAttention.

        Args:
            key_cache: Key cache [batch, num_kv_heads, seq_len, head_dim]
            value_cache: Value cache [batch, num_kv_heads, seq_len, head_dim]
            cache_positions: Position indices
            layer_idx: Layer index
            request_id: Request identifier. If None, uses default request.

        Returns:
            Tuple of (compressed_keys, compressed_values, new_positions)
        """
        compressor = self.get_compressor(layer_idx, request_id)
        return compressor.compress(
            key_states=key_cache,
            value_states=value_cache,
            cache_positions=cache_positions,
            layer_idx=layer_idx,
        )

    def reset_all(self, request_id: Optional[str] = None):
        """Reset all compressor states.

        Args:
            request_id: If provided, reset only this request's state.
                       If None, reset all requests (for backward compatibility).
        """
        if request_id is not None:
            self._reset_request(request_id)
        else:
            # Reset all requests
            for req_id in list(self.request_compressors.keys()):
                self._reset_request(req_id)

    def get_active_requests(self) -> list:
        """Get list of currently registered request IDs.

        Returns:
            List of active request IDs
        """
        return [
            req_id for req_id in self.request_compressors.keys()
            if req_id != self._default_request_id
        ]

    def get_request_state_summary(self, request_id: str) -> Optional[dict]:
        """Get state summary for a specific request.

        Args:
            request_id: Request identifier

        Returns:
            Dictionary with state summary per layer, or None if request not found
        """
        if request_id not in self.request_compressors:
            return None

        summary = {}
        for layer_idx, compressor in self.request_compressors[request_id].items():
            summary[layer_idx] = compressor.state.to_dict()
        return summary


def _extract_model_info(model, model_config=None, cache_config=None) -> dict:
    """Extract model configuration info from vLLM model.

    Args:
        model: vLLM model instance
        model_config: vLLM ModelConfig (optional)
        cache_config: vLLM CacheConfig (optional)

    Returns:
        Dictionary with block_size, num_kv_heads, head_dim
    """
    info = {}

    # Try to get HF config from model if not provided
    hf_config = None
    if model_config is None and hasattr(model, 'config'):
        hf_config = model.config
    elif model_config is not None and hasattr(model_config, 'hf_config'):
        hf_config = model_config.hf_config
    elif hasattr(model, 'config'):
        hf_config = model.config

    # Extract block_size
    if cache_config is not None and hasattr(cache_config, 'block_size'):
        info['block_size'] = cache_config.block_size
    else:
        info['block_size'] = 16  # Default vLLM block size

    # Extract num_kv_heads
    # Try HuggingFace config first (more reliable)
    if hf_config is not None:
        if hasattr(hf_config, 'num_key_value_heads'):
            info['num_kv_heads'] = hf_config.num_key_value_heads
        elif hasattr(hf_config, 'num_attention_heads'):
            # Fallback for models without GQA (like OPT)
            info['num_kv_heads'] = hf_config.num_attention_heads
        else:
            raise ValueError("Cannot determine num_kv_heads from HF config")
    else:
        raise ValueError("model.config (HF config) is required to determine num_kv_heads")

    # Extract head_dim
    if hf_config is not None:
        if hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
            # Calculate head_dim from hidden_size / num_attention_heads
            info['head_dim'] = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise ValueError("Cannot determine head_dim from HF config")
    else:
        raise ValueError("model.config (HF config) is required to determine head_dim")

    return info


class PagedKVCacheCompressor:
    """Compressor for vLLM's paged KV cache.

    This class handles the block-based KV cache format used by vLLM.

    vLLM KV Cache Format:
        - key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        - value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        - block_tables: [batch_size, max_num_blocks_per_seq]
        - slot_mapping: [num_tokens] - maps tokens to slots in the cache

    Per-Request State:
        - Each request maintains its own compressor state
        - Use register_request() before processing
        - Use unregister_request() when request completes
    """

    def __init__(
        self,
        config: TriAttentionConfig,
        block_size: int = 16,
    ):
        """Initialize paged KV cache compressor.

        Args:
            config: TriAttention configuration
            block_size: vLLM block size (default: 16)
        """
        self.config = config
        self.block_size = block_size

        # Per-request compressor storage
        self.request_compressors: Dict[str, TriAttentionCompressor] = {}
        self._default_request_id = "__default__"

    def register_request(self, request_id: str) -> None:
        """Register a new request and initialize its compressor.

        Args:
            request_id: Unique identifier for the request
        """
        if request_id not in self.request_compressors:
            self.request_compressors[request_id] = TriAttentionCompressor(self.config)
        else:
            # Reset existing compressor
            self.request_compressors[request_id].reset()

    def unregister_request(self, request_id: str) -> None:
        """Unregister a request and cleanup its state.

        Args:
            request_id: Unique identifier for the request
        """
        if request_id in self.request_compressors:
            self.request_compressors[request_id].reset()
            del self.request_compressors[request_id]

    def _get_compressor(self, request_id: Optional[str] = None) -> TriAttentionCompressor:
        """Get compressor for a request.

        Args:
            request_id: Request identifier. If None, uses default.

        Returns:
            TriAttentionCompressor instance
        """
        if request_id is None:
            request_id = self._default_request_id

        if request_id not in self.request_compressors:
            self.register_request(request_id)

        return self.request_compressors[request_id]

    def gather_kv_from_paged_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather KV states from paged cache for a single sequence.

        Args:
            key_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim]
            block_table: Block indices for this sequence [num_seq_blocks]
            seq_len: Actual sequence length

        Returns:
            Tuple of (keys, values) in dense format:
            - keys: [1, num_kv_heads, seq_len, head_dim]
            - values: [1, num_kv_heads, seq_len, head_dim]
        """
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]

        # Calculate number of full blocks and remaining tokens
        num_full_blocks = seq_len // self.block_size
        remaining = seq_len % self.block_size

        # Gather tokens from full blocks
        gathered_keys = []
        gathered_values = []

        for block_idx in range(num_full_blocks):
            physical_block = block_table[block_idx].item()
            gathered_keys.append(key_cache[physical_block])  # [block_size, num_kv_heads, head_dim]
            gathered_values.append(value_cache[physical_block])

        # Gather remaining tokens from last partial block
        if remaining > 0:
            physical_block = block_table[num_full_blocks].item()
            gathered_keys.append(key_cache[physical_block, :remaining])
            gathered_values.append(value_cache[physical_block, :remaining])

        # Concatenate all tokens
        keys = torch.cat(gathered_keys, dim=0)  # [seq_len, num_kv_heads, head_dim]
        values = torch.cat(gathered_values, dim=0)

        # Transpose to [1, num_kv_heads, seq_len, head_dim]
        keys = keys.transpose(0, 1).unsqueeze(0)
        values = values.transpose(0, 1).unsqueeze(0)

        return keys, values

    def scatter_kv_to_paged_cache(
        self,
        compressed_keys: torch.Tensor,
        compressed_values: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        new_positions: torch.Tensor,
    ) -> None:
        """Scatter compressed KV states back to paged cache.

        Args:
            compressed_keys: [1, num_kv_heads, budget, head_dim]
            compressed_values: [1, num_kv_heads, budget, head_dim]
            key_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim]
            block_table: Block indices for this sequence [num_seq_blocks]
            new_positions: Position indices for compressed tokens [budget]
        """
        # Remove batch dimension and transpose
        # [1, num_kv_heads, budget, head_dim] -> [budget, num_kv_heads, head_dim]
        keys = compressed_keys.squeeze(0).transpose(0, 1)
        values = compressed_values.squeeze(0).transpose(0, 1)
        budget = keys.shape[0]

        # Scatter to blocks
        for token_idx in range(budget):
            block_idx = token_idx // self.block_size
            slot_in_block = token_idx % self.block_size
            physical_block = block_table[block_idx].item()

            key_cache[physical_block, slot_in_block] = keys[token_idx]
            value_cache[physical_block, slot_in_block] = values[token_idx]

    def compress_paged_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        layer_idx: int,
        request_id: Optional[str] = None,
    ) -> Tuple[int, torch.Tensor]:
        """Compress paged KV cache for a sequence.

        This is the main entry point for compressing paged cache.

        Args:
            key_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim]
            block_table: Block indices for this sequence
            seq_len: Current sequence length
            layer_idx: Layer index
            request_id: Request identifier. If None, uses default request.

        Returns:
            Tuple of (new_seq_len, new_positions)
        """
        compressor = self._get_compressor(request_id)

        # Check if compression is needed
        if not compressor.state.should_compress(seq_len):
            positions = torch.arange(seq_len, device=key_cache.device, dtype=torch.int32)
            return seq_len, positions

        # Log compression event (first layer only to reduce noise)
        if layer_idx == 0:
            print(f"[TriAttention] Compressing: seq_len={seq_len} -> budget={compressor.config.kv_budget}")

        # Gather KV from paged cache to dense format
        keys, values = self.gather_kv_from_paged_cache(
            key_cache, value_cache, block_table, seq_len
        )

        # Create position indices
        cache_positions = torch.arange(seq_len, device=key_cache.device, dtype=torch.int32)

        # Compress
        compressed_keys, compressed_values, new_positions = compressor.compress(
            key_states=keys,
            value_states=values,
            cache_positions=cache_positions,
            layer_idx=layer_idx,
        )

        # Scatter back to paged cache
        self.scatter_kv_to_paged_cache(
            compressed_keys, compressed_values,
            key_cache, value_cache,
            block_table, new_positions
        )

        new_seq_len = compressed_keys.shape[2]
        return new_seq_len, new_positions


def patch_vllm_attention(
    model,
    tri_wrapper: TriAttentionWrapper,
    layer_name_pattern: str = "model.layers",
    model_config=None,
    cache_config=None,
) -> None:
    """Patch vLLM model's attention layers to use TriAttention compression.

    This function wraps the FlashAttentionImpl.forward() method to apply
    KV cache compression after the cache is updated during decode steps.

    Integration Strategy:
    - Non-invasive: All code in TriAttention module, no vLLM source changes
    - Hook point: After reshape_and_cache_flash() updates KV cache
    - Timing: Apply compression during decode when cache exceeds threshold
    - Request isolation: Use request_id from attn_metadata.seq_data

    Args:
        model: vLLM model with attention layers (e.g., llm.llm_engine.model_executor.driver_worker.model_runner.model)
        tri_wrapper: TriAttention wrapper instance
        layer_name_pattern: Pattern to match layer names (default: "model.layers")
        model_config: vLLM ModelConfig (optional, will try to auto-detect from model)
        cache_config: vLLM CacheConfig (optional, will try to auto-detect from model)

    Example:
        llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        tri_wrapper = TriAttentionWrapper(config)

        # Access the model from vLLM engine
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        patch_vllm_attention(model, tri_wrapper)

        # Now compression will be applied automatically during generation
        outputs = llm.generate(prompts, sampling_params)
    """
    import functools
    import types
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from vllm.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata

    print(f"[TriAttention] Patching vLLM attention layers...")

    # Extract model config info if not provided
    model_info = _extract_model_info(model, model_config, cache_config)
    print(f"[TriAttention] Model info: block_size={model_info['block_size']}, "
          f"num_kv_heads={model_info['num_kv_heads']}, head_dim={model_info['head_dim']}")

    # Set num_kv_heads on wrapper config for proper GQA handling in stats loading
    if tri_wrapper.config.num_kv_heads is None:
        tri_wrapper.config.num_kv_heads = model_info['num_kv_heads']
    if tri_wrapper.config.head_dim is None:
        tri_wrapper.config.head_dim = model_info['head_dim']

    # Find all transformer layers (try different model structures)
    layers = None
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            layers = model.model.layers  # Llama, Qwen, etc.
        elif hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            layers = model.model.decoder.layers  # OPT, BART decoder
        elif hasattr(model.model, "encoder") and hasattr(model.model.encoder, "layers"):
            layers = model.model.encoder.layers  # BART encoder
    elif hasattr(model, "layers"):
        layers = model.layers  # Direct access
    elif hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        layers = model.decoder.layers  # Direct decoder access

    if layers is None:
        raise ValueError(
            f"Cannot find transformer layers in model. "
            f"Tried: model.model.layers, model.model.decoder.layers, model.layers, model.decoder.layers. "
            f"Available attributes: {dir(model)}"
        )

    num_patched = 0

    for layer_idx, layer in enumerate(layers):
        # Find attention layer - handle different model structures
        attn_impl = None

        # Try different attention layer patterns
        if hasattr(layer, "self_attn"):
            attn_layer = layer.self_attn
            # Try direct impl access (Llama, Qwen, etc.)
            if hasattr(attn_layer, "impl"):
                attn_impl = attn_layer.impl
            # Try attn.impl access (OPT, etc.)
            elif hasattr(attn_layer, "attn") and hasattr(attn_layer.attn, "impl"):
                attn_impl = attn_layer.attn.impl

        if attn_impl is None:
            continue

        # Store original forward method using __func__ to get unbound function
        original_forward_func = attn_impl.forward.__func__

        # Create wrapped forward with compression
        # NOTE: Must use a factory function to capture layer_idx correctly
        def make_wrapped_forward(orig_func, layer_index):
            def wrapped_forward(
                self,
                layer,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                kv_cache: torch.Tensor,
                attn_metadata,
                output: Optional[torch.Tensor] = None,
            ):
                # Call original attention forward
                result = orig_func(
                    self, layer, query, key, value, kv_cache, attn_metadata, output
                )

                # Apply compression after decode step if cache is populated
                # V1 API uses max_query_len (decode: 1, prefill: >1)
                # V0 API uses num_decode_tokens
                is_decode = getattr(attn_metadata, 'max_query_len', None) == 1 or \
                            getattr(attn_metadata, 'num_decode_tokens', 0) > 0
                if kv_cache.numel() > 0 and is_decode:
                    try:
                        _apply_triattention_compression(
                            kv_cache=kv_cache,
                            attn_metadata=attn_metadata,
                            tri_wrapper=tri_wrapper,
                            layer_idx=layer_index,
                            model_info=model_info,
                        )
                    except Exception as e:
                        # Log but don't crash on compression errors
                        print(f"[TriAttention] Warning: Compression failed at layer {layer_index}: {e}")

                return result
            return wrapped_forward

        # Replace forward method with properly captured closure
        # Use types.MethodType to bind the function to the instance
        attn_impl.forward = types.MethodType(
            make_wrapped_forward(original_forward_func, layer_idx),
            attn_impl
        )
        num_patched += 1

    print(f"[TriAttention] Successfully patched {num_patched} attention layers")
    tri_wrapper._patched = True


def _gather_kv_from_paged_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather KV states from paged cache for a single sequence.

    Args:
        key_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: Block indices for this sequence [num_seq_blocks]
        seq_len: Actual sequence length
        block_size: vLLM block size

    Returns:
        Tuple of (keys, values) in dense format:
        - keys: [1, num_kv_heads, seq_len, head_dim]
        - values: [1, num_kv_heads, seq_len, head_dim]
    """
    # Calculate number of full blocks and remaining tokens
    num_full_blocks = seq_len // block_size
    remaining = seq_len % block_size

    # Gather tokens from full blocks
    gathered_keys = []
    gathered_values = []

    for block_idx in range(num_full_blocks):
        physical_block = block_table[block_idx].item()
        gathered_keys.append(key_cache[physical_block])
        gathered_values.append(value_cache[physical_block])

    # Gather remaining tokens from last partial block
    if remaining > 0:
        physical_block = block_table[num_full_blocks].item()
        gathered_keys.append(key_cache[physical_block, :remaining])
        gathered_values.append(value_cache[physical_block, :remaining])

    # Concatenate all tokens
    keys = torch.cat(gathered_keys, dim=0)  # [seq_len, num_kv_heads, head_dim]
    values = torch.cat(gathered_values, dim=0)

    # Transpose to [1, num_kv_heads, seq_len, head_dim]
    keys = keys.transpose(0, 1).unsqueeze(0)
    values = values.transpose(0, 1).unsqueeze(0)

    return keys, values


def _scatter_kv_to_paged_cache(
    compressed_keys: torch.Tensor,
    compressed_values: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    new_positions: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter compressed KV states back to paged cache.

    Args:
        compressed_keys: [1, num_kv_heads, budget, head_dim]
        compressed_values: [1, num_kv_heads, budget, head_dim]
        key_cache: Paged key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Paged value cache [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: Block indices for this sequence [num_seq_blocks]
        new_positions: Position indices for compressed tokens [budget]
        block_size: vLLM block size
    """
    # Remove batch dimension and transpose
    # [1, num_kv_heads, budget, head_dim] -> [budget, num_kv_heads, head_dim]
    keys = compressed_keys.squeeze(0).transpose(0, 1)
    values = compressed_values.squeeze(0).transpose(0, 1)
    budget = keys.shape[0]

    # Scatter to blocks
    for token_idx in range(budget):
        block_idx = token_idx // block_size
        slot_in_block = token_idx % block_size
        physical_block = block_table[block_idx].item()

        key_cache[physical_block, slot_in_block] = keys[token_idx]
        value_cache[physical_block, slot_in_block] = values[token_idx]


def _apply_triattention_compression(
    kv_cache: torch.Tensor,
    attn_metadata,
    tri_wrapper: TriAttentionWrapper,
    layer_idx: int,
    model_info: dict,
) -> None:
    """Apply TriAttention compression to vLLM's paged KV cache.

    This function is called during decode steps to compress the KV cache
    when it exceeds the budget + divide_length threshold.

    Args:
        kv_cache: Paged KV cache - format varies by backend:
            - Expected: [2, num_blocks, block_size, num_kv_heads, head_dim]
            - XFormers: [2, num_blocks, block_size * num_kv_heads * head_dim] (flattened)
        attn_metadata: FlashAttentionMetadata with block_tables and seq_lens
        tri_wrapper: TriAttention wrapper instance
        layer_idx: Current layer index
        model_info: Dictionary with block_size, num_kv_heads, head_dim

    Implementation Notes:
    - vLLM KV cache format varies by backend (FlashAttention vs XFormers)
    - We reshape flattened formats to the expected block format
    - During decode, each request has a separate block_table
    - We compress each request independently using its request_id
    - After compression, we update the cache in-place
    """
    # Entry point debug logging (disabled for production)
    # if layer_idx == 0:
    #     print(f"\n[DEBUG HOOK] _apply_triattention_compression called:")
    #     print(f"  layer_idx: {layer_idx}")
    #     print(f"  kv_cache.shape: {kv_cache.shape}")
    #     print(f"  attn_metadata type: {type(attn_metadata).__name__}")

    # Extract decode metadata - support both V0 and V1 APIs
    # V0: Uses attn_metadata.decode_metadata with block_tables and seq_lens_tensor
    # V1: Uses attn_metadata directly with block_table and seq_lens
    decode_meta = getattr(attn_metadata, 'decode_metadata', None)

    if decode_meta is not None:
        # V0 API path
        block_tables = decode_meta.block_tables
        seq_lens_tensor = decode_meta.seq_lens_tensor
    else:
        # V1 API path - FlashAttentionMetadata has block_table and seq_lens directly
        block_tables = getattr(attn_metadata, 'block_table', None)
        seq_lens_tensor = getattr(attn_metadata, 'seq_lens', None)

    if block_tables is None or seq_lens_tensor is None:
        return

    # Extract model config
    block_size = model_info['block_size']
    num_kv_heads = model_info['num_kv_heads']
    head_dim = model_info['head_dim']

    # Reshape cache if needed (flattened -> block format)
    original_shape = kv_cache.shape
    was_reshaped = False

    # Flattened format: [2, num_blocks, block_size * num_kv_heads * head_dim]
    if kv_cache.dim() == 3 and kv_cache.shape[0] == 2:
        num_blocks = kv_cache.shape[1]
        flattened_size = kv_cache.shape[2]
        expected_size = block_size * num_kv_heads * head_dim

        if flattened_size == expected_size:
            # Reshape in-place to [2, num_blocks, block_size, num_kv_heads, head_dim]
            # Note: Only log once per layer to avoid spam
            if not hasattr(_apply_triattention_compression, '_reshape_logged'):
                print(f"[TriAttention] Detected flattened cache format, reshaping to block format")
                _apply_triattention_compression._reshape_logged = True
            kv_cache = kv_cache.view(2, num_blocks, block_size, num_kv_heads, head_dim)
            was_reshaped = True
        else:
            print(f"[TriAttention] Warning: Flattened cache size mismatch. "
                  f"Expected {expected_size}, got {flattened_size}. Skipping compression.")
            return

    # Split cache into key and value
    if isinstance(kv_cache, tuple):
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
    elif kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
    else:
        print(f"[TriAttention] Warning: Unexpected kv_cache format after reshape: shape={kv_cache.shape}")
        return

    # Process each sequence in the decode batch
    batch_size = block_tables.shape[0]

    for batch_idx in range(batch_size):
        seq_len = seq_lens_tensor[batch_idx].item()
        block_table = block_tables[batch_idx]

        # Generate request_id (use batch_idx as identifier in this decode step)
        # Note: In vLLM 0.7.0, we don't have direct access to sequence IDs in attn_metadata
        # We use batch_idx as a proxy for request identification
        request_id = f"decode_{batch_idx}"

        # Check if compression is needed
        should_compress_result = tri_wrapper.should_compress(layer_idx, seq_len, request_id)

        # Debug logging (disabled for production)
        # if layer_idx == 0:  # Only log for first layer to reduce clutter
        #     compressor = tri_wrapper.get_compressor(layer_idx, request_id)
        #     state = compressor.state
        #     print(f"\n[DEBUG] Layer {layer_idx}, Request {request_id}:")
        #     print(f"  seq_len: {seq_len}")
        #     print(f"  prefill_length: {state.prefill_length}")
        #     print(f"  current_cache_len: {state.current_cache_len}")
        #     print(f"  compression_count: {state.compression_count}")
        #     print(f"  config.kv_budget: {tri_wrapper.config.kv_budget}")
        #     print(f"  config.divide_length: {tri_wrapper.config.divide_length}")
        #     print(f"  config.protect_prefill: {tri_wrapper.config.protect_prefill}")
        #
        #     # Calculate effective size
        #     if tri_wrapper.config.protect_prefill:
        #         effective_size = max(0, seq_len - state.prefill_length)
        #     else:
        #         effective_size = seq_len
        #     trigger_threshold = tri_wrapper.config.kv_budget + tri_wrapper.config.divide_length
        #     print(f"  effective_size: {effective_size}")
        #     print(f"  trigger_threshold: {trigger_threshold}")
        #     print(f"  should_compress: {should_compress_result}")

        if not should_compress_result:
            continue

        # Apply compression using tri_wrapper's compressor (shares state and loaded stats)
        try:
            # Get or create compressor from wrapper (reuses loaded stats!)
            compressor = tri_wrapper.get_compressor(layer_idx, request_id)

            # Log compression start (throttled)
            if layer_idx == 0:
                if not hasattr(_apply_triattention_compression, '_compress_log_count'):
                    _apply_triattention_compression._compress_log_count = 0
                _apply_triattention_compression._compress_log_count += 1
                if _apply_triattention_compression._compress_log_count <= 5:
                    print(f"[TriAttention] Compressing: seq_len={seq_len} -> budget={tri_wrapper.config.kv_budget}")

            # Gather KV from paged cache to dense format
            keys, values = _gather_kv_from_paged_cache(
                key_cache, value_cache, block_table, seq_len, block_size
            )

            # Create position indices
            cache_positions = torch.arange(seq_len, device=key_cache.device, dtype=torch.int32)

            # Compress using the wrapper's compressor (shared state + stats)
            compressed_keys, compressed_values, new_positions = compressor.compress(
                key_states=keys,
                value_states=values,
                cache_positions=cache_positions,
                layer_idx=layer_idx,
            )

            # Scatter back to paged cache
            _scatter_kv_to_paged_cache(
                compressed_keys, compressed_values,
                key_cache, value_cache,
                block_table, new_positions, block_size
            )

            new_seq_len = compressed_keys.shape[2]

            # Log compression result (throttled, layer 0 only)
            if layer_idx == 0:
                if _apply_triattention_compression._compress_log_count <= 5:
                    print(f"[TriAttention] Compressed: {seq_len} -> {new_seq_len} tokens")

            # Note: Cache was modified in-place
            # If we reshaped from flattened format, the original tensor is automatically updated
            # because view() shares underlying storage

        except Exception as e:
            # Log but don't crash
            import traceback
            print(f"[TriAttention] Compression error for batch {batch_idx}, layer {layer_idx}: {e}")
            traceback.print_exc()

    # Note: No need to reshape back - view() shares storage with original tensor


def create_triattention_wrapper(
    stats_path: str,
    kv_budget: int = 2048,
    divide_length: int = 128,
    pruning_mode: str = "per_head",
    **kwargs,
) -> TriAttentionWrapper:
    """Factory function to create TriAttention wrapper.

    Args:
        stats_path: Path to precomputed frequency statistics
        kv_budget: Maximum KV tokens to retain
        divide_length: Compression interval
        pruning_mode: Pruning strategy ("per_head", "per_layer")
        **kwargs: Additional config parameters

    Returns:
        Configured TriAttentionWrapper instance
    """
    from pathlib import Path

    config = TriAttentionConfig(
        stats_path=Path(stats_path),
        kv_budget=kv_budget,
        divide_length=divide_length,
        pruning_mode=pruning_mode,
        **kwargs,
    )

    return TriAttentionWrapper(config)
