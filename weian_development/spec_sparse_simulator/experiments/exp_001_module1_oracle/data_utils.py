"""Data loading utilities for Module 1 Key Pruning experiments.

Reuses patterns from reference implementation:
weian_development/spec_sparse_simulator/attention_pruning_case_study_hybrid_rounds_xtrace.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoConfig

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.hf_offline_runner_sparse.round_pruning_utils import (
    build_rotary,
    compute_rotary_tables,
    invert_rope,
)


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Default head indices file path
DEFAULT_HEAD_INDICES_PATH = Path(__file__).resolve().parents[2] / "hybrid_sample_heads_lowret_top10.json"

# Default model path for RoPE parameters
DEFAULT_MODEL_PATH = Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B")


def load_trace_data(
    trace_dir: Path,
    device: torch.device = torch.device("cpu"),
    dtype: str = "float32",
) -> Dict:
    """
    Load QK trace data from a trace directory.

    Args:
        trace_dir: Path to trace directory containing qk.pt and metadata.json.
        device: Device to load tensors to.
        dtype: Data type for computation ("float32", "float16", "bfloat16").

    Returns:
        Dict with keys:
            - q: Q tensor (layers, heads, seq_len, head_dim)
            - k: K tensor (layers, heads, seq_len, head_dim)
            - seq_len: Sequence length
            - metadata: Full metadata dict (if available)
    """
    trace_dir = Path(trace_dir)
    qk_path = trace_dir / "qk.pt"

    if not qk_path.exists():
        raise FileNotFoundError(f"QK trace file not found: {qk_path}")

    # Load QK tensors
    qk_data = torch.load(qk_path, map_location=device, weights_only=True)
    compute_dtype = DTYPE_MAP.get(dtype, torch.float32)

    q = qk_data["q"].to(device=device, dtype=compute_dtype)
    k = qk_data["k"].to(device=device, dtype=compute_dtype)

    # Get sequence length
    seq_len = q.shape[2]

    # Load metadata if available
    metadata_path = trace_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        # Override seq_len from metadata if available
        if "seq_len" in metadata:
            seq_len = metadata["seq_len"]

    return {
        "q": q,
        "k": k,
        "seq_len": seq_len,
        "metadata": metadata,
    }


def load_trace_metadata(trace_dir: Path) -> Dict:
    """
    Load only metadata from trace directory (memory efficient).

    Args:
        trace_dir: Path to trace directory.

    Returns:
        Dict with seq_len and metadata.
    """
    trace_dir = Path(trace_dir)
    qk_path = trace_dir / "qk.pt"

    if not qk_path.exists():
        raise FileNotFoundError(f"QK trace file not found: {qk_path}")

    # Load only to get shape info (load to CPU, don't convert dtype yet)
    qk_data = torch.load(qk_path, map_location="cpu", weights_only=True)
    q_shape = qk_data["q"].shape
    seq_len = q_shape[2]

    # Clean up to free memory
    del qk_data

    # Load metadata if available
    metadata_path = trace_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if "seq_len" in metadata:
            seq_len = metadata["seq_len"]

    return {
        "seq_len": seq_len,
        "q_shape": q_shape,
        "metadata": metadata,
    }


def load_head_data(
    trace_dir: Path,
    layer: int,
    head: int,
    device: torch.device = torch.device("cpu"),
    dtype: str = "float32",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load Q and K data for a single (layer, head) pair.

    Memory-efficient: only keeps one head's data in memory at a time.

    Args:
        trace_dir: Path to trace directory.
        layer: Layer index.
        head: Head index.
        device: Device to load tensors to.
        dtype: Data type for computation.

    Returns:
        Tuple of (q_head, k_head, seq_len):
            - q_head: (seq_len, head_dim) Query tensor for this head
            - k_head: (seq_len, head_dim) Key tensor for this head
            - seq_len: Sequence length
    """
    trace_dir = Path(trace_dir)
    qk_path = trace_dir / "qk.pt"

    if not qk_path.exists():
        raise FileNotFoundError(f"QK trace file not found: {qk_path}")

    compute_dtype = DTYPE_MAP.get(dtype, torch.float32)

    # Load to CPU first, then extract only the needed head
    qk_data = torch.load(qk_path, map_location="cpu", weights_only=True)

    q_head = qk_data["q"][layer, head].to(device=device, dtype=compute_dtype)
    k_head = qk_data["k"][layer, head].to(device=device, dtype=compute_dtype)
    seq_len = q_head.shape[0]

    # Clean up to free memory
    del qk_data

    # Load metadata for seq_len if available
    metadata_path = trace_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if "seq_len" in metadata:
            seq_len = metadata["seq_len"]

    return q_head, k_head, seq_len


def load_head_indices(
    head_indices_path: Optional[Path] = None,
) -> List[Tuple[int, int]]:
    """
    Load head indices from JSON file.

    Args:
        head_indices_path: Path to JSON file with (layer, head) pairs.
                          Defaults to hybrid_sample_heads_lowret_top10.json.

    Returns:
        List of (layer, head) tuples.
    """
    if head_indices_path is None:
        head_indices_path = DEFAULT_HEAD_INDICES_PATH

    head_indices_path = Path(head_indices_path)

    if not head_indices_path.exists():
        raise FileNotFoundError(f"Head indices file not found: {head_indices_path}")

    with head_indices_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [(int(item[0]), int(item[1])) for item in data]


def compute_attention_matrix(
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    apply_causal_mask: bool = True,
) -> torch.Tensor:
    """
    Compute attention weights matrix from Q and K tensors.

    Args:
        q_head: (seq_len, head_dim) Query tensor for a single head.
        k_head: (seq_len, head_dim) Key tensor for a single head.
        apply_causal_mask: Whether to apply causal mask (default True).

    Returns:
        attention: (seq_len, seq_len) Attention weights matrix.
                   attention[q, k] = softmax attention weight from query q to key k.
    """
    seq_len, head_dim = q_head.shape
    scale = head_dim ** -0.5

    # Compute attention logits: (seq_len, seq_len)
    # logits[q, k] = q[q] @ k[k].T
    logits = torch.matmul(q_head, k_head.T) * scale

    if apply_causal_mask:
        # Create causal mask: positions where key > query should be masked
        positions = torch.arange(seq_len, device=q_head.device)
        causal_mask = positions.unsqueeze(0) > positions.unsqueeze(1)  # (seq_len, seq_len)
        logits = logits.masked_fill(causal_mask, float("-inf"))

    # Apply softmax along key dimension
    attention = torch.softmax(logits, dim=-1)

    return attention


def compute_attention_with_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    layer: int,
    head: int,
    model_path: Path = DEFAULT_MODEL_PATH,
    apply_causal_mask: bool = True,
    use_rope_inversion: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute attention matrix for a specific layer and head, optionally with RoPE inversion.

    For Oracle experiment, we typically use the raw attention patterns directly
    without RoPE inversion (use_rope_inversion=False).

    Args:
        q: (layers, heads, seq_len, head_dim) Full Q tensor.
        k: (layers, heads, seq_len, head_dim) Full K tensor.
        layer: Layer index.
        head: Head index.
        model_path: Path to model for RoPE parameters.
        apply_causal_mask: Whether to apply causal mask.
        use_rope_inversion: Whether to invert RoPE before computing attention.
        device: Device for computation.

    Returns:
        attention: (seq_len, seq_len) Attention weights matrix.
    """
    if device is None:
        device = q.device

    q_head = q[layer, head].to(device)  # (seq_len, head_dim)
    k_head = k[layer, head].to(device)  # (seq_len, head_dim)

    if use_rope_inversion:
        seq_len = q_head.shape[0]
        head_dim = q_head.shape[1]
        dtype = q_head.dtype

        # Build rotary embedding and tables
        rotary = build_rotary(device, model_path, dtype)
        cos_table, sin_table, _ = compute_rotary_tables(
            rotary, seq_len, head_dim, dtype, device
        )

        # Compute attention scale (for inversion)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        attention_scale = 1.0
        if hasattr(config, "rope_scaling") and config.rope_scaling:
            attention_scale = config.rope_scaling.get("attention_factor", 1.0)
            if "attn_factor" in config.rope_scaling:
                attention_scale = config.rope_scaling["attn_factor"]

        # Invert RoPE
        q_head = invert_rope(q_head, cos_table, sin_table, attention_scale)
        k_head = invert_rope(k_head, cos_table, sin_table, attention_scale)

    return compute_attention_matrix(q_head, k_head, apply_causal_mask)


def get_query_argmax_info(
    attention: torch.Tensor,
    round_start: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get argmax key indices and history flags for queries starting from round_start.

    Args:
        attention: (seq_len, seq_len) Full attention matrix.
        round_start: Starting position of the current round.

    Returns:
        query_argmax_indices: (num_queries,) Argmax key index for each query.
        argmax_in_history: (num_queries,) Boolean tensor, True if argmax is in history
                          (position < round_start), False if in current round.
    """
    seq_len = attention.shape[0]
    num_queries = seq_len - round_start

    if num_queries <= 0:
        return (
            torch.empty(0, dtype=torch.long, device=attention.device),
            torch.empty(0, dtype=torch.bool, device=attention.device),
        )

    # Get queries from round_start onwards
    query_attention = attention[round_start:seq_len]  # (num_queries, seq_len)

    # Find argmax for each query (among all valid keys, including current round)
    query_argmax_indices = query_attention.argmax(dim=1)  # (num_queries,)

    # Check if argmax is in history (position < round_start)
    argmax_in_history = query_argmax_indices < round_start

    return query_argmax_indices, argmax_in_history
