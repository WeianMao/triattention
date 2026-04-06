"""Helpers for TriAttention stats metadata validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def normalize_dtype_name(value: Any) -> str:
    if isinstance(value, torch.dtype):
        name = str(value)
    else:
        name = str(value)
    name = name.lower().replace("torch.", "")
    if name == "bf16":
        return "bfloat16"
    if name == "fp16":
        return "float16"
    if name == "fp32":
        return "float32"
    return name


def _require(metadata: Mapping[str, Any], key: str, stats_path: Path) -> Any:
    if key not in metadata:
        raise ValueError(f"Stats file {stats_path} missing required metadata field '{key}'. Regenerate stats.")
    return metadata[key]


def validate_stats_metadata(
    metadata: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    stats_path: Path,
) -> None:
    # Validate rope_style if expected
    expected_rope_style = expected.get("rope_style")
    if expected_rope_style is not None:
        stats_rope_style = metadata.get("rope_style")
        if stats_rope_style is not None and str(stats_rope_style) != str(expected_rope_style):
            raise ValueError(
                f"rope_style mismatch for stats {stats_path}: expected {expected_rope_style}, found {stats_rope_style}."
            )

    # Validate head_dim if expected
    expected_head_dim = expected.get("head_dim")
    if expected_head_dim is not None:
        stats_head_dim = metadata.get("head_dim")
        if stats_head_dim is not None and int(stats_head_dim) != int(expected_head_dim):
            raise ValueError(
                f"head_dim mismatch for stats {stats_path}: expected {expected_head_dim}, found {stats_head_dim}."
            )
