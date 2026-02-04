"""Utility functions for TriAttention.

This module provides helper functions for:
- Frequency statistics loading
- RoPE alignment verification
- Position index management
- Debugging utilities
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def load_frequency_stats(
    stats_path: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    num_kv_heads: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Load precomputed frequency statistics from file.

    Args:
        stats_path: Path to stats file (.pt or .pth format)
        device: Device to load tensors onto
        dtype: Data type for loaded tensors

    Returns:
        Tuple of (metadata, head_stats)
        - metadata: Dict with model configuration info
        - head_stats: Dict with per-layer, per-head frequency statistics

    Supports two file formats:
    1. TriAttention format:
        {
            'metadata': {
                'num_attention_heads': int,
                'num_kv_heads': int,
                'head_dim': int,
                'num_layers': int,
                'rope_theta': float,
                'rope_style': str ('half' or 'interleaved'),
            },
            'layer_stats': {
                layer_idx: {
                    'q_mean_complex': Tensor [num_kv_heads, freq_count, 2],
                    'freq_scale_sq': Tensor [num_kv_heads, freq_count],
                    'extra_coef': Tensor [num_kv_heads, freq_count],
                    'sampled_heads': List[int] (optional),
                }
            }
        }

    2. R-KV format (auto-converted):
        {
            'metadata': {
                'head_dim': int,
                'rope_style': str,
                'sampled_heads': List[List[int]],  # [[layer, head], ...]
                ...
            },
            'stats': {
                'layer00_head00': dict,
                'layer00_head01': dict,
                ...
            }
        }
    """
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    # Load stats file
    stats = torch.load(stats_path, map_location="cpu")

    # Extract metadata
    metadata = stats.get("metadata", {})

    # Check if this is R-KV format (has 'stats' key with 'layerXX_headYY' keys)
    rkv_stats = stats.get("stats", {})
    is_rkv_format = rkv_stats and any(
        k.startswith("layer") and "_head" in k for k in rkv_stats.keys()
    )

    if is_rkv_format:
        # Convert R-KV format to TriAttention format
        metadata, head_stats = _convert_rkv_stats(stats, device, dtype, num_kv_heads)
        return metadata, head_stats

    # TriAttention format - validate required keys
    required_metadata_keys = [
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
        "num_layers",
    ]
    missing_keys = [k for k in required_metadata_keys if k not in metadata]
    if missing_keys:
        raise ValueError(
            f"Stats file missing required metadata keys: {missing_keys}. "
            f"Found keys: {list(metadata.keys())}"
        )

    # Extract and move layer stats to device
    layer_stats = stats.get("layer_stats", {})
    if not layer_stats:
        raise ValueError("Stats file does not contain 'layer_stats'")

    # Move all tensors to target device and dtype
    head_stats = {}
    for layer_idx, layer_data in layer_stats.items():
        head_stats[layer_idx] = {}
        for key, value in layer_data.items():
            if isinstance(value, torch.Tensor):
                head_stats[layer_idx][key] = value.to(device=device, dtype=dtype)
            else:
                head_stats[layer_idx][key] = value

    return metadata, head_stats


def _convert_rkv_stats(
    stats: Dict,
    device: torch.device,
    dtype: torch.dtype,
    num_kv_heads: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Convert R-KV stats format to TriAttention format.

    R-KV format: per-head flat structure with 'layerXX_headYY' keys
    TriAttention format: per-layer structure with stacked head tensors

    Handles GQA (Grouped Query Attention) by averaging Q head stats
    to match the number of KV heads when num_kv_heads < num_attention_heads.

    Args:
        stats: Raw stats dict in R-KV format
        device: Target device
        dtype: Target dtype
        num_kv_heads: Number of KV heads (for GQA). If None, uses num_attention_heads.

    Returns:
        Tuple of (metadata, head_stats) in TriAttention format
    """
    rkv_metadata = stats.get("metadata", {})
    rkv_stats = stats.get("stats", {})

    # Infer num_layers and num_heads from stats keys
    layer_nums = set()
    head_nums = set()
    for key in rkv_stats.keys():
        if key.startswith("layer") and "_head" in key:
            parts = key.split("_")
            if len(parts) == 2:
                layer_nums.add(int(parts[0].replace("layer", "")))
                head_nums.add(int(parts[1].replace("head", "")))

    num_layers = len(layer_nums)
    num_attention_heads = len(head_nums)
    head_dim = rkv_metadata.get("head_dim", 128)
    freq_count = head_dim // 2

    # Handle GQA: if num_kv_heads specified and < num_attention_heads
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads

    gqa_ratio = num_attention_heads // num_kv_heads if num_kv_heads > 0 else 1

    # Build TriAttention metadata
    metadata = {
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "rope_style": rkv_metadata.get("rope_style", "half"),
        "rope_theta": rkv_metadata.get("rope_theta", 10000.0),
        "gqa_ratio": gqa_ratio,
        # Preserve original metadata
        "rkv_metadata": rkv_metadata,
    }

    # Convert to per-layer format with stacked tensors
    # Expected: head_stats[layer_idx] = {
    #     "freq_scale_sq": [num_kv_heads, freq_count],
    #     "q_mean_complex": [num_kv_heads, freq_count, 2],  # optional
    # }
    head_stats = {}
    for layer_idx in sorted(layer_nums):
        # Collect all attention heads' data for this layer
        all_freq_scale_sq = []
        all_q_mean_real = []
        all_q_mean_imag = []
        all_q_abs_mean = []

        for head_idx in sorted(head_nums):
            key = f"layer{layer_idx:02d}_head{head_idx:02d}"
            if key in rkv_stats:
                head_data = rkv_stats[key]
                # R-KV stores q_abs_mean which we use as freq_scale_sq
                if "q_abs_mean" in head_data:
                    q_abs = head_data["q_abs_mean"].to(dtype=dtype)
                    all_freq_scale_sq.append(q_abs ** 2)
                    all_q_abs_mean.append(q_abs)
                elif "freq_scale_sq" in head_data:
                    all_freq_scale_sq.append(
                        head_data["freq_scale_sq"].to(dtype=dtype)
                    )
                    # Compute q_abs_mean from freq_scale_sq
                    all_q_abs_mean.append(
                        torch.sqrt(head_data["freq_scale_sq"].to(dtype=dtype))
                    )
                else:
                    all_freq_scale_sq.append(
                        torch.ones(freq_count, dtype=dtype)
                    )
                    all_q_abs_mean.append(
                        torch.ones(freq_count, dtype=dtype)
                    )

                if "q_mean_real" in head_data and "q_mean_imag" in head_data:
                    all_q_mean_real.append(
                        head_data["q_mean_real"].to(dtype=dtype)
                    )
                    all_q_mean_imag.append(
                        head_data["q_mean_imag"].to(dtype=dtype)
                    )

        # Stack all attention heads: [num_attention_heads, freq_count]
        all_freq_scale_sq = torch.stack(all_freq_scale_sq, dim=0)
        all_q_abs_mean = torch.stack(all_q_abs_mean, dim=0)

        # Apply GQA mapping: average Q heads that share each KV head
        if gqa_ratio > 1:
            # Reshape and average: [num_kv_heads, gqa_ratio, freq_count] -> [num_kv_heads, freq_count]
            freq_scale_sq = all_freq_scale_sq.reshape(
                num_kv_heads, gqa_ratio, freq_count
            ).mean(dim=1)
            q_abs_mean = all_q_abs_mean.reshape(
                num_kv_heads, gqa_ratio, freq_count
            ).mean(dim=1)
        else:
            freq_scale_sq = all_freq_scale_sq
            q_abs_mean = all_q_abs_mean

        head_stats[layer_idx] = {
            "freq_scale_sq": freq_scale_sq.to(device),
            "q_abs_mean": q_abs_mean.to(device),
        }

        # Add q_mean_complex if available
        if all_q_mean_real and all_q_mean_imag:
            all_q_mean_real = torch.stack(all_q_mean_real, dim=0)
            all_q_mean_imag = torch.stack(all_q_mean_imag, dim=0)

            # Apply GQA mapping
            if gqa_ratio > 1:
                q_mean_real = all_q_mean_real.reshape(
                    num_kv_heads, gqa_ratio, freq_count
                ).mean(dim=1)
                q_mean_imag = all_q_mean_imag.reshape(
                    num_kv_heads, gqa_ratio, freq_count
                ).mean(dim=1)
            else:
                q_mean_real = all_q_mean_real
                q_mean_imag = all_q_mean_imag

            q_mean_complex = torch.stack([q_mean_real, q_mean_imag], dim=-1)
            head_stats[layer_idx]["q_mean_complex"] = q_mean_complex.to(device)

    return metadata, head_stats


def verify_rope_alignment(
    pruner_rotary: torch.nn.Module,
    model_rotary: torch.nn.Module,
    tolerance: float = 1e-5,
) -> None:
    """Verify that RoPE configurations match between pruner and model.

    This is a critical safety check adapted from R-KV's verification logic.
    Mismatched RoPE frequencies will cause silent failures (GIGO - garbage in, garbage out).

    Args:
        pruner_rotary: Rotary embedding used by the pruner
        model_rotary: Rotary embedding from the model
        tolerance: Numerical tolerance for frequency comparison

    Raises:
        ValueError: If RoPE configurations do not match
    """
    # Extract inv_freq from both rotary embeddings
    pruner_inv_freq = getattr(pruner_rotary, "inv_freq", None)
    model_inv_freq = getattr(model_rotary, "inv_freq", None)

    if pruner_inv_freq is None:
        raise ValueError("pruner_rotary does not have inv_freq attribute")
    if model_inv_freq is None:
        raise ValueError("model_rotary does not have inv_freq attribute")

    # Check shape
    if pruner_inv_freq.shape != model_inv_freq.shape:
        raise ValueError(
            f"RoPE inv_freq shape mismatch: "
            f"pruner={pruner_inv_freq.shape}, model={model_inv_freq.shape}"
        )

    # Check numerical values
    max_diff = (pruner_inv_freq - model_inv_freq).abs().max().item()
    if max_diff > tolerance:
        raise ValueError(
            f"RoPE inv_freq mismatch (max diff={max_diff:.2e} > tolerance={tolerance:.2e}). "
            f"This indicates a config mismatch (e.g., 'attn_factor' vs 'factor' in Llama/Qwen). "
            f"CRITICAL: Using mismatched RoPE will cause silent failures and random token eviction."
        )


def create_position_indices(
    seq_len: int,
    start_position: int = 0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Create position indices for a sequence.

    Args:
        seq_len: Sequence length
        start_position: Starting position (default: 0)
        device: Device to create tensor on
        dtype: Data type for position indices

    Returns:
        Position indices tensor [seq_len]
    """
    return torch.arange(
        start_position,
        start_position + seq_len,
        device=device,
        dtype=dtype,
    )


def gather_kv_by_indices(
    kv_tensor: torch.Tensor,
    indices: torch.Tensor,
    dim: int = 2,
) -> torch.Tensor:
    """Gather KV cache by indices.

    Args:
        kv_tensor: KV cache tensor [batch, num_kv_heads, seq_len, head_dim]
        indices: Indices to gather [batch, num_kv_heads, budget] or [batch, budget]
        dim: Dimension to gather along (default: 2, the seq_len dimension)

    Returns:
        Gathered KV tensor [batch, num_kv_heads, budget, head_dim]
    """
    # Expand indices to match KV tensor dimensions
    if indices.ndim == 2:
        # [batch, budget] -> [batch, num_kv_heads, budget, 1]
        batch_size, budget = indices.shape
        num_kv_heads = kv_tensor.shape[1]
        head_dim = kv_tensor.shape[3]
        indices = indices.unsqueeze(1).unsqueeze(-1)
        indices = indices.expand(batch_size, num_kv_heads, budget, head_dim)
    elif indices.ndim == 3:
        # [batch, num_kv_heads, budget] -> [batch, num_kv_heads, budget, head_dim]
        head_dim = kv_tensor.shape[3]
        indices = indices.unsqueeze(-1).expand(*indices.shape, head_dim)
    else:
        raise ValueError(f"Unsupported indices shape: {indices.shape}")

    return torch.gather(kv_tensor, dim=dim, index=indices)


def normalize_scores(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply z-score normalization to scores.

    Args:
        scores: Score tensor [..., seq_len]
        eps: Small constant for numerical stability

    Returns:
        Normalized scores with mean=0, std=1
        If std is close to zero, returns the original scores unchanged.
    """
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True)
    std_safe = torch.where(std < eps, torch.ones_like(std), std)
    return (scores - mean) / std_safe


def protect_window_tokens(
    scores: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Protect the last window_size tokens by setting their scores to infinity.

    Args:
        scores: Score tensor [..., seq_len]
        window_size: Number of most recent tokens to protect

    Returns:
        Scores with protected tokens set to inf
    """
    if window_size <= 0:
        return scores

    # Set last window_size tokens to inf (they will be selected by topk)
    protected_scores = scores.clone()
    protected_scores[..., -window_size:] = float("inf")
    return protected_scores


def debug_log_state(
    state_dict: dict,
    prefix: str = "CompressionState",
) -> None:
    """Log compression state for debugging.

    Args:
        state_dict: State dictionary from CompressionState.to_dict()
        prefix: Log message prefix
    """
    print(f"[{prefix}]")
    for key, value in state_dict.items():
        print(f"  {key}: {value}")


def compute_rope_frequencies(
    head_dim: int,
    rope_theta: float = 10000.0,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Compute RoPE inverse frequencies.

    Args:
        head_dim: Head dimension (must be even)
        rope_theta: RoPE theta parameter (default: 10000.0)
        device: Device to create tensor on

    Returns:
        Inverse frequencies tensor [head_dim // 2]
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    freq_count = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    return inv_freq


def detect_rope_style(model_config: dict) -> str:
    """Detect RoPE style from model config.

    Args:
        model_config: Model configuration dictionary

    Returns:
        RoPE style: 'half' or 'interleaved'
    """
    # Default to 'half' for most models (Llama, Qwen, etc.)
    rope_style = model_config.get("rope_style", "half")

    # Some models may specify this differently
    if "rotary_emb_style" in model_config:
        rope_style = model_config["rotary_emb_style"]

    return rope_style


def format_memory_usage(num_bytes: int) -> str:
    """Format memory usage in human-readable format.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "8.5 KB", "230 MB")
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.2f} GB"
