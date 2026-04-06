"""TriAttention configuration class.

This module defines the configuration parameters for TriAttention KV cache compression.
Based on Phase 1 design specifications and aligned with R-KV parameter conventions.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch


@dataclass
class TriAttentionConfig:
    """Configuration for TriAttention KV cache compression.

    This configuration class defines all parameters needed for TriAttention compression,
    including budget constraints, pruning modes, scoring parameters, and optimization flags.

    Design Alignment:
    - Budget and divide_length align with R-KV conventions
    - TopK dtype follows Phase 1 precision requirements (fp32 for accuracy)
    - Pruning modes support per_head, per_layer, and per_layer_per_head strategies
    """

    # ===== Core Parameters =====
    kv_budget: int = 2048
    """Maximum number of KV tokens to retain (budget constraint)."""

    divide_length: int = 128
    """Compression interval - trigger compression every N tokens."""

    sparse_round_window: int = 32
    """Window size for sparse round compression (R-KV alignment)."""

    # ===== Pruning Strategy =====
    pruning_mode: Literal["per_head", "per_layer", "per_layer_per_head"] = "per_head"
    """Token selection strategy:
    - per_head: Each head selects its own tokens independently
    - per_layer: All heads in a layer share the same token selection
    - per_layer_per_head: Alias for per_head (for R-KV compatibility)
    """

    # ===== Scoring Parameters =====
    score_aggregation: Literal["mean", "max"] = "mean"
    """How to aggregate scores across multiple offsets."""

    offset_max_length: int = 65536
    """Maximum offset length for position-dependent scoring."""

    disable_top_n_high_freq: int = 0
    """Disable top N high-frequency components (0 = use all frequencies)."""

    disable_mlr: bool = False
    """Disable magnitude-LR term in scoring."""

    disable_trig: bool = False
    """Disable trigonometric (position-dependent) term in scoring."""

    # ===== Budget Management =====
    include_prefill_in_budget: bool = True
    """Whether prefill tokens count toward the budget."""

    protect_prefill: bool = True
    """Protect prefill tokens from being pruned (R-KV alignment)."""

    window_size: int = 128
    """Number of most recent tokens to always protect (default: 128, aligned with HF baseline)."""

    # ===== Normalization =====
    sparse_normalize_scores: bool = True
    """Apply z-score normalization to scores before TopK selection."""

    # ===== Precision Control =====
    topk_dtype: torch.dtype = torch.float32
    """Data type for TopK operation (fp32 for numerical stability)."""

    compute_dtype: torch.dtype = torch.bfloat16
    """Data type for general computation (bf16/fp16 for efficiency)."""

    position_indices_dtype: torch.dtype = torch.int32
    """Data type for position_indices storage (int32 recommended, bf16 also supported)."""

    # ===== Device Configuration =====
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    """Device for computation (default: CUDA)."""

    # ===== Triton Kernel Parameters =====
    triton_block_size: int = 128
    """Block size for Triton kernels (tuned via autotune)."""

    use_triton_scoring: bool = True
    """Use Triton kernel for scoring (Phase 1: always True)."""

    use_trig_cache: bool = True
    """Use precomputed trig cache when round_start aligns with divide_length."""

    trig_cache_max_seq_len: Optional[int] = None
    """Optional max sequence length for trig cache table construction."""

    trig_cache_warn_threshold_mb: float = 100.0
    """Warn threshold for trig cache memory footprint."""

    # ===== Stats and Model Paths =====
    stats_path: Optional[Path] = None
    """Path to precomputed frequency statistics file."""

    model_path: Optional[Path] = None
    """Path to model for RoPE configuration detection."""

    # ===== RoPE Configuration (Auto-detected) =====
    rope_style: Literal["half", "interleaved"] = "half"
    """RoPE frequency pairing style (detected from model)."""

    head_dim: Optional[int] = None
    """Head dimension (auto-detected from model if not specified)."""

    num_kv_heads: Optional[int] = None
    """Number of KV heads (auto-detected from model if not specified)."""

    num_layers: Optional[int] = None
    """Number of transformer layers (auto-detected from model if not specified)."""

    # ===== Optional Features =====
    seed: Optional[int] = None
    """Random seed for reproducibility (None = no seeding)."""

    enable_debug_logging: bool = False
    """Enable detailed debug logging."""

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Convert string paths to Path objects
        if isinstance(self.stats_path, str):
            self.stats_path = Path(self.stats_path)
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)

        # Validate budget constraints
        if self.kv_budget <= 0:
            raise ValueError(f"kv_budget must be positive, got {self.kv_budget}")

        if self.divide_length <= 0:
            raise ValueError(f"divide_length must be positive, got {self.divide_length}")

        # Validate window_size
        if self.window_size < 0:
            raise ValueError(f"window_size cannot be negative, got {self.window_size}")

        if self.window_size >= self.kv_budget:
            raise ValueError(
                f"window_size ({self.window_size}) must be less than kv_budget ({self.kv_budget})"
            )

        # Normalize pruning_mode aliases
        if self.pruning_mode == "per_layer_per_head":
            self.pruning_mode = "per_head"  # Alias for R-KV compatibility

        # Validate dtype choices
        valid_compute_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if self.compute_dtype not in valid_compute_dtypes:
            raise ValueError(
                f"compute_dtype must be one of {valid_compute_dtypes}, got {self.compute_dtype}"
            )

        valid_position_dtypes = {torch.int32, torch.int64}
        if self.position_indices_dtype not in valid_position_dtypes:
            raise ValueError(
                f"position_indices_dtype must be one of {valid_position_dtypes}, "
                f"got {self.position_indices_dtype}"
            )

        # NOTE: stats_path validation is deferred to lazy loading in compressor._lazy_init()
        # This allows config creation without requiring the stats file immediately

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "kv_budget": self.kv_budget,
            "divide_length": self.divide_length,
            "sparse_round_window": self.sparse_round_window,
            "pruning_mode": self.pruning_mode,
            "score_aggregation": self.score_aggregation,
            "offset_max_length": self.offset_max_length,
            "disable_top_n_high_freq": self.disable_top_n_high_freq,
            "disable_mlr": self.disable_mlr,
            "disable_trig": self.disable_trig,
            "include_prefill_in_budget": self.include_prefill_in_budget,
            "protect_prefill": self.protect_prefill,
            "window_size": self.window_size,
            "sparse_normalize_scores": self.sparse_normalize_scores,
            "topk_dtype": str(self.topk_dtype),
            "compute_dtype": str(self.compute_dtype),
            "position_indices_dtype": str(self.position_indices_dtype),
            "triton_block_size": self.triton_block_size,
            "use_triton_scoring": self.use_triton_scoring,
            "use_trig_cache": self.use_trig_cache,
            "trig_cache_max_seq_len": self.trig_cache_max_seq_len,
            "trig_cache_warn_threshold_mb": self.trig_cache_warn_threshold_mb,
            "rope_style": self.rope_style,
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads,
            "num_layers": self.num_layers,
            "seed": self.seed,
            "enable_debug_logging": self.enable_debug_logging,
        }
