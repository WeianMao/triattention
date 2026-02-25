"""Installer for vLLM runtime input patch hooks used by TriAttention V2."""
from __future__ import annotations

from typing import Any, Callable

from .input_patch_vllm_backend import (
    make_patched_compute_slot_mappings,
    make_patched_prepare_pos_seq_lens,
)

_PATCH_INSTALLED = False
_ORIGINAL_PREPARE_POS_SEQ_LENS: Callable[..., Any] | None = None
_ORIGINAL_COMPUTE_SLOT_MAPPINGS: Callable[..., Any] | None = None


def install_runtime_input_patch_hooks() -> bool:
    """Patch vLLM GPU input prep once.

    Returns True when the patch is active (including repeated calls).
    """
    global _PATCH_INSTALLED, _ORIGINAL_PREPARE_POS_SEQ_LENS, _ORIGINAL_COMPUTE_SLOT_MAPPINGS
    if _PATCH_INSTALLED:
        return True

    try:
        import vllm.v1.worker.gpu.block_table as gpu_block_table
        import vllm.v1.worker.gpu.model_runner as gpu_model_runner
    except Exception:
        return False

    original = getattr(gpu_model_runner, "prepare_pos_seq_lens", None)
    if original is None:
        return False

    compute_slot_mappings = getattr(gpu_block_table.BlockTables, "compute_slot_mappings", None)
    if compute_slot_mappings is None:
        return False

    _ORIGINAL_PREPARE_POS_SEQ_LENS = original
    _ORIGINAL_COMPUTE_SLOT_MAPPINGS = compute_slot_mappings

    gpu_model_runner.prepare_pos_seq_lens = make_patched_prepare_pos_seq_lens(
        _ORIGINAL_PREPARE_POS_SEQ_LENS
    )
    gpu_block_table.BlockTables.compute_slot_mappings = make_patched_compute_slot_mappings(
        _ORIGINAL_COMPUTE_SLOT_MAPPINGS
    )
    _PATCH_INSTALLED = True
    return True
