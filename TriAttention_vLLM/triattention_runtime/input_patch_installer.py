"""Installer for vLLM runtime input patch hooks used by TriAttention runtime."""
from __future__ import annotations

import os
from typing import Any, Callable

from .input_patch_vllm_backend import (
    make_patched_compute_slot_mappings,
    make_patched_prepare_pos_seq_lens,
)
from .input_patch_vllm_v1_backend import make_patched_v1_prepare_inputs

_PATCH_INSTALLED = False
_ORIGINAL_PREPARE_POS_SEQ_LENS: Callable[..., Any] | None = None
_ORIGINAL_COMPUTE_SLOT_MAPPINGS: Callable[..., Any] | None = None
_ORIGINAL_V1_PREPARE_INPUTS: Callable[..., Any] | None = None


def install_runtime_input_patch_hooks() -> bool:
    """Patch vLLM GPU input prep once.

    Returns True when the patch is active (including repeated calls).
    """
    global _PATCH_INSTALLED, _ORIGINAL_PREPARE_POS_SEQ_LENS, _ORIGINAL_COMPUTE_SLOT_MAPPINGS
    global _ORIGINAL_V1_PREPARE_INPUTS
    if _PATCH_INSTALLED:
        return True

    patched_any = False

    try:
        import vllm.v1.worker.gpu.block_table as gpu_block_table
        import vllm.v1.worker.gpu.model_runner as gpu_model_runner
    except Exception:
        gpu_block_table = None
        gpu_model_runner = None

    if gpu_block_table is not None and gpu_model_runner is not None:
        original = getattr(gpu_model_runner, "prepare_pos_seq_lens", None)
        compute_slot_mappings = getattr(gpu_block_table.BlockTables, "compute_slot_mappings", None)
        if original is not None and compute_slot_mappings is not None:
            _ORIGINAL_PREPARE_POS_SEQ_LENS = original
            _ORIGINAL_COMPUTE_SLOT_MAPPINGS = compute_slot_mappings
            gpu_model_runner.prepare_pos_seq_lens = make_patched_prepare_pos_seq_lens(
                _ORIGINAL_PREPARE_POS_SEQ_LENS
            )
            gpu_block_table.BlockTables.compute_slot_mappings = make_patched_compute_slot_mappings(
                _ORIGINAL_COMPUTE_SLOT_MAPPINGS
            )
            patched_any = True

    if os.environ.get("TRIATTN_DEBUG_ENABLE_V1_OVERRIDE_PATH", "0") == "1":
        try:
            import vllm.v1.worker.gpu_model_runner as gpu_model_runner_v1
        except Exception:
            gpu_model_runner_v1 = None
        if gpu_model_runner_v1 is not None:
            original_v1_prepare_inputs = getattr(gpu_model_runner_v1.GPUModelRunner, "_prepare_inputs", None)
            if original_v1_prepare_inputs is not None:
                _ORIGINAL_V1_PREPARE_INPUTS = original_v1_prepare_inputs
                gpu_model_runner_v1.GPUModelRunner._prepare_inputs = make_patched_v1_prepare_inputs(
                    _ORIGINAL_V1_PREPARE_INPUTS
                )
                patched_any = True

    _PATCH_INSTALLED = patched_any
    return patched_any
