"""TriAttention KV cache compressor.

This module implements the main TriAttentionCompressor class that orchestrates
KV cache compression using frequency-based scoring and TopK selection.

Design Alignment:
- Follows Phase 1 architecture with Triton scoring + PyTorch TopK/Gather
- Supports per_head, per_layer pruning modes
- Integrates with vLLM's PagedAttention infrastructure
"""
from typing import Dict, Optional, Tuple

import torch

from .config import TriAttentionConfig
from .state import CompressionState
from .scoring import compute_scores
from .utils import (
    compute_rope_frequencies,
    gather_kv_by_indices,
    load_frequency_stats,
    normalize_scores,
    protect_window_tokens,
)


class TriAttentionCompressor:
    """Main KV cache compressor using TriAttention (SpeckV) algorithm.

    This class implements the core compression pipeline:
    1. Load precomputed frequency statistics
    2. Compute importance scores for each KV token
    3. Select top-k tokens to keep
    4. Gather compressed KV cache

    Phase 1 Implementation:
    - Scoring: Triton kernel (kernels/triton_scoring.py)
    - TopK/Gather: PyTorch (torch.topk + torch.gather)
    """

    def __init__(self, config: TriAttentionConfig):
        """Initialize TriAttention compressor.

        Args:
            config: TriAttention configuration
        """
        self.config = config
        self.state = CompressionState(config)

        # Stats will be loaded lazily on first compress call
        self.metadata: Optional[Dict] = None
        self.head_stats: Optional[Dict] = None

        # RoPE frequencies
        self.inv_freq: Optional[torch.Tensor] = None
        self.omega: Optional[torch.Tensor] = None

        # Frequency scaling factors
        self.freq_scale_sq: Optional[torch.Tensor] = None

        # Precomputed offsets for scoring
        self.offsets: Optional[torch.Tensor] = None

        # Optional precomputed trig cache for faster scoring
        self.trig_cache = None

        # Initialization flags
        self._initialized = False

    def _lazy_init(self):
        """Lazily initialize stats and RoPE parameters on first use."""
        if self._initialized:
            return

        # Load frequency statistics
        if self.config.stats_path is None:
            raise ValueError(
                "stats_path must be specified in TriAttentionConfig. "
                "Generate statistics using: python -m triattention.tools.generate_stats <model_path> <output_path>"
            )

        try:
            self.metadata, self.head_stats = load_frequency_stats(
                self.config.stats_path,
                device=self.config.device,
                dtype=self.config.compute_dtype,
                num_kv_heads=self.config.num_kv_heads,  # Pass for GQA mapping
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Frequency statistics file not found: {self.config.stats_path}. "
                f"Generate it using: python -m triattention.tools.generate_stats <model_path> {self.config.stats_path}"
            )

        # Auto-detect model parameters from stats if not specified
        if self.config.head_dim is None:
            self.config.head_dim = self.metadata["head_dim"]
        if self.config.num_kv_heads is None:
            self.config.num_kv_heads = self.metadata["num_kv_heads"]
        if self.config.num_layers is None:
            self.config.num_layers = self.metadata["num_layers"]
        if "rope_style" in self.metadata:
            self.config.rope_style = str(self.metadata["rope_style"])

        # Initialize RoPE frequencies
        self._init_rope()

        # Precompute frequency scaling factors
        self._precompute_freq_scale()

        # Initialize scoring offsets
        self._init_offsets()

        # Initialize optional trig cache for aligned rounds
        self._init_trig_cache()

        self._initialized = True

    def _init_rope(self):
        """Initialize RoPE frequencies from stats metadata.

        Preference order:
        1) metadata.inv_freq (if present) for model-exact rotary semantics;
        2) derive inv_freq from the real model config when model_path is available;
        3) fallback to metadata/legacy rope_theta-based construction.
        """
        inv_freq_raw = self.metadata.get("inv_freq")
        if isinstance(inv_freq_raw, torch.Tensor):
            inv_freq = inv_freq_raw.to(device=self.config.device, dtype=torch.float32)
            expected_freq_count = int(self.config.head_dim) // 2
            if inv_freq.numel() < expected_freq_count:
                raise ValueError(
                    "metadata.inv_freq has fewer elements than required by "
                    f"head_dim={self.config.head_dim}: got {inv_freq.numel()}, "
                    f"expected at least {expected_freq_count}"
                )
            self.inv_freq = inv_freq[:expected_freq_count].contiguous()
        elif isinstance(inv_freq_raw, (list, tuple)):
            inv_freq = torch.tensor(
                inv_freq_raw,
                device=self.config.device,
                dtype=torch.float32,
            )
            expected_freq_count = int(self.config.head_dim) // 2
            if inv_freq.numel() < expected_freq_count:
                raise ValueError(
                    "metadata.inv_freq has fewer elements than required by "
                    f"head_dim={self.config.head_dim}: got {inv_freq.numel()}, "
                    f"expected at least {expected_freq_count}"
                )
            self.inv_freq = inv_freq[:expected_freq_count].contiguous()
        else:
            derived_inv_freq: torch.Tensor | None = None
            model_path = getattr(self.config, "model_path", None)
            if model_path is not None:
                try:
                    from transformers import AutoConfig

                    from weian_development.speckv.round_pruning_utils import build_rotary

                    model_config = AutoConfig.from_pretrained(
                        str(model_path),
                        trust_remote_code=True,
                    )
                    rotary = build_rotary(
                        cache_device=self.config.device,
                        model_path=model_path,
                        dtype=self.config.compute_dtype,
                        config=model_config,
                    )
                    inv_freq = getattr(rotary, "inv_freq", None)
                    if isinstance(inv_freq, torch.Tensor):
                        derived_inv_freq = inv_freq.to(
                            device=self.config.device,
                            dtype=torch.float32,
                        )[: int(self.config.head_dim) // 2].contiguous()
                        self.config.rope_style = str(
                            getattr(rotary, "_rope_style", self.config.rope_style)
                        )
                except Exception:
                    derived_inv_freq = None

            if derived_inv_freq is not None:
                self.inv_freq = derived_inv_freq
            else:
                rope_theta = self.metadata.get("rope_theta", 10000.0)
                self.inv_freq = compute_rope_frequencies(
                    self.config.head_dim,
                    rope_theta=rope_theta,
                    device=self.config.device,
                )
        # Match HF/R-KV reference scoring semantics: use rotary inv_freq directly.
        # The scoring formula expects the same frequency basis as the model's RoPE.
        self.omega = self.inv_freq

    def _precompute_freq_scale(self):
        """Precompute frequency scaling factors from stats.

        This extracts freq_scale_sq from head_stats for efficient scoring.
        Shape: [num_layers, num_kv_heads, freq_count]
        """
        num_layers = self.config.num_layers
        num_kv_heads = self.config.num_kv_heads
        freq_count = self.config.head_dim // 2

        # Allocate freq_scale_sq tensor
        self.freq_scale_sq = torch.zeros(
            num_layers,
            num_kv_heads,
            freq_count,
            device=self.config.device,
            dtype=self.config.compute_dtype,
        )

        # Fill from head_stats
        for layer_idx in range(num_layers):
            if layer_idx in self.head_stats:
                layer_data = self.head_stats[layer_idx]
                if "freq_scale_sq" in layer_data:
                    self.freq_scale_sq[layer_idx] = layer_data["freq_scale_sq"]

    def _init_offsets(self):
        """Initialize scoring offsets.

        Offsets define multiple reference positions for scoring robustness.
        Match HF SpeckV helper: geometric offsets [1, 2, 4, ..., offset_max_length].
        """
        max_length = int(self.config.offset_max_length)
        if max_length < 1:
            raise ValueError(
                f"offset_max_length must be >= 1, got {self.config.offset_max_length}"
            )
        offsets: list[float] = []
        value = 1
        while value <= max_length:
            offsets.append(float(value))
            value *= 2
        self.offsets = torch.tensor(
            offsets,
            device=self.config.device,
            dtype=torch.float32,
        )

    def _init_trig_cache(self):
        """Initialize optional precomputed trig cache for scoring."""
        self.trig_cache = None
        if not self.config.use_triton_scoring or not self.config.use_trig_cache:
            return
        try:
            from .kernels.triton_scoring import create_trig_cache
        except Exception:
            return

        if self.offsets is None or self.omega is None:
            return

        max_seq_len = self.config.trig_cache_max_seq_len
        if max_seq_len is None:
            max_seq_len = max(
                int(self.config.offset_max_length),
                int(self.config.kv_budget + self.config.divide_length),
            )
        max_seq_len = max(int(max_seq_len), int(self.config.divide_length))
        if max_seq_len <= 0:
            return

        try:
            self.trig_cache = create_trig_cache(
                max_seq_len=max_seq_len,
                compress_interval=int(self.config.divide_length),
                offsets=self.offsets,
                omega=self.omega,
                device=self.config.device,
                warn_threshold_mb=float(self.config.trig_cache_warn_threshold_mb),
            )
        except Exception:
            self.trig_cache = None

    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_positions: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache using TriAttention.

        Args:
            key_states: K cache [batch, num_kv_heads, seq_len, head_dim]
            value_states: V cache [batch, num_kv_heads, seq_len, head_dim]
            cache_positions: DEPRECATED - no longer used for scoring.
                Kept for API compatibility. Pass None to save memory.
            layer_idx: Current layer index (for layer-specific stats)

        Returns:
            Tuple of (compressed_keys, compressed_values, keep_indices)
            - compressed_keys: [batch, num_kv_heads, budget, head_dim]
            - compressed_values: [batch, num_kv_heads, budget, head_dim]
            - keep_indices: [budget] or [num_kv_heads, budget] - indices of kept tokens
        """
        # Lazy initialization
        self._lazy_init()

        batch_size, num_kv_heads, seq_len, head_dim = key_states.shape

        # Batch size validation
        if batch_size > 1:
            raise ValueError(
                f"TriAttention currently only supports batch_size=1 for compression. "
                f"Got batch_size={batch_size}. For batch inference, process requests sequentially "
                f"or use separate compressor instances per request with proper request_id isolation."
            )

        # Check if compression is needed
        # NOTE: should_compress() will auto-initialize state on first call and track incremental updates
        if not self.state.should_compress(seq_len):
            # No compression needed, return as-is with identity indices
            keep_indices = torch.arange(seq_len, device=key_states.device)
            return key_states, value_states, keep_indices

        # Step 1: Compute importance scores
        scores = self._compute_scores(
            key_states=key_states,
            layer_idx=layer_idx,
        )

        # Step 2: Apply normalization if enabled
        if self.config.sparse_normalize_scores:
            scores = normalize_scores(scores)

        # Step 3: Protect window tokens if specified
        if self.config.window_size > 0:
            scores = protect_window_tokens(scores, self.config.window_size)

        # Step 4: Select top-k tokens
        keep_indices = self._select_topk(scores, self.config.kv_budget)

        # Step 5: Gather compressed KV cache
        compressed_keys = gather_kv_by_indices(key_states, keep_indices, dim=2)
        compressed_values = gather_kv_by_indices(value_states, keep_indices, dim=2)

        # Step 6: Update compression state (new_cache_len = budget)
        new_cache_len = compressed_keys.shape[2]
        self.state.update_after_compression(new_cache_len)

        return compressed_keys, compressed_values, keep_indices

    def _compute_scores(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute importance scores for KV tokens.

        Args:
            key_states: K cache [batch, num_kv_heads, seq_len, head_dim]
                NOTE: In vLLM, keys are stored AFTER RoPE rotation
            layer_idx: Layer index for layer-specific stats

        Returns:
            Scores tensor [batch, num_kv_heads, seq_len] or [batch, seq_len]
        """
        # Get layer-specific stats
        if layer_idx not in self.head_stats:
            raise ValueError(f"No stats found for layer {layer_idx}")

        layer_stats = self.head_stats[layer_idx]

        # Get round_start from state (absolute_position)
        round_start = self.state.get_round_start()

        # Call the scoring module (dispatches to Triton or PyTorch)
        # NOTE: cache_positions is None - not needed for scoring anymore
        scores = compute_scores(
            key_states=key_states,
            cache_positions=None,  # Not used in scoring
            head_stats=layer_stats,
            omega=self.omega,
            offsets=self.offsets,
            freq_scale_sq=self.freq_scale_sq[layer_idx],
            config=self.config,
            round_start=round_start,
            trig_cache=self.trig_cache,
        )

        return scores

    def _select_topk(
        self,
        scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Select top-k tokens based on scores.

        Phase 1: Uses PyTorch torch.topk
        Phase 2: May switch to Triton kernel for efficiency

        Args:
            scores: Importance scores
                [batch, num_kv_heads, seq_len] for per_head
                [batch, seq_len] for per_layer
            k: Number of tokens to keep

        Returns:
            Indices of selected tokens (same shape as scores, last dim = k)
        """
        # Use torch.topk (largest=True to keep highest scores)
        topk_result = torch.topk(scores, k=k, dim=-1, largest=True, sorted=False)
        return topk_result.indices

    def reset(self):
        """Reset compressor state for a new sequence."""
        self.state.reset()

    def get_state(self) -> Dict:
        """Get current compression state for debugging.

        Returns:
            State dictionary
        """
        return self.state.to_dict()
