"""Helpers for SpeckV stats metadata validation."""
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
    prompt_template = _require(metadata, "prompt_template", stats_path)
    expected_template = expected.get("prompt_template")
    if expected_template and prompt_template != expected_template:
        raise ValueError(
            f"Prompt template in stats ({stats_path}) does not match runtime template. "
            "Regenerate stats with the same template used for inference."
        )

    use_chat_template = _require(metadata, "use_chat_template", stats_path)
    expected_chat = expected.get("use_chat_template")
    if expected_chat is not None and bool(use_chat_template) != bool(expected_chat):
        raise ValueError(
            f"use_chat_template mismatch for stats {stats_path}: expected {expected_chat}, found {use_chat_template}."
        )

    if use_chat_template:
        stats_system = _require(metadata, "system_prompt", stats_path)
        expected_system = expected.get("system_prompt")
        if expected_system is not None and stats_system != expected_system:
            raise ValueError(
                f"system_prompt mismatch for stats {stats_path}: expected '{expected_system}', found '{stats_system}'."
            )

    stats_attn = _require(metadata, "attn_implementation", stats_path)
    expected_attn = expected.get("attn_implementation")
    if expected_attn and str(stats_attn).lower() != str(expected_attn).lower():
        raise ValueError(
            f"attn_implementation mismatch for stats {stats_path}: expected {expected_attn}, found {stats_attn}."
        )

    stats_dtype = _require(metadata, "dtype", stats_path)
    expected_dtype = expected.get("dtype")
    if expected_dtype and normalize_dtype_name(stats_dtype) != normalize_dtype_name(expected_dtype):
        raise ValueError(
            f"dtype mismatch for stats {stats_path}: expected {expected_dtype}, found {stats_dtype}."
        )

    expected_kv = expected.get("kv_budget")
    if expected_kv is not None:
        stats_kv = _require(metadata, "kv_budget", stats_path)
        if int(stats_kv) != int(expected_kv):
            raise ValueError(
                f"kv_budget mismatch for stats {stats_path}: expected {expected_kv}, found {stats_kv}."
            )
