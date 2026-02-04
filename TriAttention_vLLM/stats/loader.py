"""Stats file loader for TriAttention.

This module provides utilities to load precomputed frequency statistics
from R-KV or TriAttention stats files.
"""
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_head_frequency_stats(
    stats_path: Path,
    device: torch.device,
) -> Tuple[Dict, Dict]:
    """Load frequency statistics from file.

    This is an alias for the utils.load_frequency_stats function,
    provided for backward compatibility with R-KV naming conventions.

    Args:
        stats_path: Path to stats file
        device: Device to load tensors onto

    Returns:
        Tuple of (metadata, head_stats)
    """
    from triattention.utils import load_frequency_stats

    return load_frequency_stats(stats_path, device)
