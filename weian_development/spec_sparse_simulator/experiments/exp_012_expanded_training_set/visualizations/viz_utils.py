"""Shared utility functions for exp_006 failure case visualization.

This module provides utilities for:
1. Loading miss case analysis data
2. Loading and processing QK trace data
3. Converting tensors to complex representations
4. Frequency analysis
5. Matplotlib configuration
6. Color scheme management
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def load_miss_cases(path: Path) -> dict:
    """Load miss_case_analysis.json containing 248 failure cases.

    Args:
        path: Path to miss_case_analysis.json

    Returns:
        dict with structure: {
            "summary": {...},
            "miss_cases": [
                {
                    "query_idx": int,
                    "argmax_key": int,
                    "selected_bin": int,
                    "best_bin": int,
                    "rank_in_selected": int,
                    "best_rank": int,
                    "miss_type": "A" or "B",
                    "num_historical": int
                },
                ...
            ]
        }
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_qk_data(trace_path: Path, layer: int, head: int) -> torch.Tensor:
    """Load qk.pt and extract single [layer, head] K tensor slice.

    CRITICAL: K in qk.pt is ALREADY post-RoPE rotated.
    DO NOT apply invert_rope on the returned tensor.

    Args:
        trace_path: Path to qk.pt or qk_test.pt
        layer: Layer index (0-35 for 36-layer model)
        head: Head index (0-31 for 32-head model)

    Returns:
        K tensor [seq_len, head_dim=128], post-RoPE rotated
    """
    data = torch.load(trace_path, map_location='cpu')
    return data['k'][layer, head]


def load_query_data(trace_path: Path, layer: int, head: int) -> torch.Tensor:
    """Load qk.pt and extract single [layer, head] Q tensor slice.

    CRITICAL: Q in qk.pt is ALREADY post-RoPE rotated.
    DO NOT apply invert_rope on the returned tensor.

    Args:
        trace_path: Path to qk.pt or qk_test.pt
        layer: Layer index (0-35 for 36-layer model)
        head: Head index (0-31 for 32-head model)

    Returns:
        Q tensor [seq_len, head_dim=128], post-RoPE rotated
    """
    data = torch.load(trace_path, map_location='cpu')
    return data['q'][layer, head]


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [seq, head_dim=128] to [seq, freq_count=64] complex representation.

    Converts real-valued tensor to complex pairs using first half as real part,
    second half as imaginary part.

    Args:
        tensor: [seq_len, head_dim] tensor where head_dim=128

    Returns:
        complex64 tensor [seq_len, freq_count=64]

    Raises:
        ValueError: if head_dim is not even
    """
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")

    seq_len, head_dim = tensor.shape
    freq_count = head_dim // 2

    # Convert to float32 for complex operations if needed
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)

    # Split into real and imaginary parts
    real = tensor_real[:, :freq_count].contiguous()
    imag = tensor_real[:, freq_count:].contiguous()

    return torch.complex(real, imag)


def get_top_k_frequencies(k_complex: torch.Tensor, k: int = 6) -> list[int]:
    """Find top-k frequencies by mean magnitude across sequence.

    Args:
        k_complex: [seq_len, freq_count=64] complex tensor
        k: number of top frequencies to return (default: 6)

    Returns:
        list of frequency indices (0-63) sorted by magnitude (highest first)
    """
    # Compute mean complex value across sequence
    k_mean = k_complex.mean(dim=0)  # [freq_count]

    # Compute magnitude
    k_magnitude = torch.abs(k_mean)

    # Get top-k indices
    _, top_indices = torch.topk(k_magnitude, k=k)

    return top_indices.tolist()


def setup_matplotlib() -> dict[str, str]:
    """Setup matplotlib with Agg backend and configure defaults.

    Sets figure DPI, font size, and returns color scheme dictionary.

    Returns:
        dict with color mapping: {
            'success': 'blue',
            'failure': 'red',
            'unassigned': 'gray'
        }
    """
    plt.rcParams.update({
        'font.size': 10,
        'figure.dpi': 100,
    })

    return {
        'success': 'blue',
        'failure': 'red',
        'unassigned': 'gray'
    }


def get_color_for_bin(bin_id: int, top_bins: list[int], cmap_name: str = 'tab10') -> str:
    """Return color for bin visualization.

    Args:
        bin_id: Bin identifier
        top_bins: List of top bin IDs (e.g., top-10 most utilized bins)
        cmap_name: Matplotlib colormap name (default: 'tab10')

    Returns:
        Color string ('gray' if not in top_bins, otherwise distinct color from cmap)
    """
    if bin_id not in top_bins:
        return 'gray'

    cmap = plt.cm.get_cmap(cmap_name)
    # Use index in top_bins list to get consistent colors
    return cmap(top_bins.index(bin_id) % 10)
