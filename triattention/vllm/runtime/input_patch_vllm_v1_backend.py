"""Debug-only V1 GPUModelRunner input patch helpers.

This module provides the smallest possible compatibility layer needed to
validate effective override semantics on the legacy/default vLLM V1 runner
path (`vllm.v1.worker.gpu_model_runner.GPUModelRunner`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import input_patch_state as _patch_state


def _debug_drop_pos_delta() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_POS_DELTA", "0") == "1"


def _debug_drop_seq_base() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_SEQ_BASE", "0") == "1"


def _debug_preserve_rope_positions() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_PRESERVE_ROPE_POSITIONS", "0") == "1"


def _debug_positions_dump_path() -> str:
    return os.environ.get("TRIATTN_DEBUG_V1_DUMP_POSITIONS_PATH", "").strip()


def _debug_positions_dump_limit() -> int:
    raw = os.environ.get("TRIATTN_DEBUG_V1_DUMP_POSITIONS_LIMIT", "8").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 8


def _debug_dump_positions(
    *,
    original_positions_np: np.ndarray,
    slot_positions_np: np.ndarray | None,
    req_indices: np.ndarray,
    total_num_scheduled_tokens: int,
    forward_positions_np: np.ndarray,
) -> None:
    dump_path = _debug_positions_dump_path()
    if not dump_path:
        return

    limit = min(
        int(total_num_scheduled_tokens),
        int(original_positions_np.size),
        int(forward_positions_np.size),
        _debug_positions_dump_limit(),
    )
    if slot_positions_np is None:
        slot_slice: list[int | None] = [None] * limit
    else:
        slot_slice = [int(x) for x in slot_positions_np[:limit].tolist()]

    rows = []
    for idx in range(limit):
        rows.append(
            {
                "token_index_in_batch": int(idx),
                "req_index": int(req_indices[idx]) if idx < int(req_indices.size) else None,
                "logical_position": int(original_positions_np[idx]),
                "kv_write_position": slot_slice[idx],
                "model_forward_position": int(forward_positions_np[idx]),
            }
        )

    payload = {
        "event": "v1_positions_after_prepare_inputs",
        "rows": rows,
    }
    path = Path(dump_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _build_effective_slot_positions(
    *,
    positions_np: np.ndarray,
    req_indices: np.ndarray,
) -> np.ndarray | None:
    if _debug_drop_pos_delta():
        return None
    if (
        int(req_indices.size) == 0
        or int(positions_np.size) == 0
    ):
        return None

    # Slot positions may follow the compacted KV layout, but decode-time
    # RoPE positions must stay in the original logical sequence space.
    out = positions_np.copy()

    if int(req_indices.max(initial=-1)) + 1 == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0:
        out += int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA)
        return out

    sparse_pos_deltas = _patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX
    if not sparse_pos_deltas:
        return None

    row_deltas = np.zeros(int(req_indices.max()) + 1, dtype=positions_np.dtype)
    for req_idx, delta in sparse_pos_deltas.items():
        if 0 <= int(req_idx) < row_deltas.shape[0]:
            row_deltas[int(req_idx)] = int(delta)
    out += row_deltas[req_indices]
    return out


def _apply_sparse_seq_len_overrides_in_place(
    *,
    seq_lens_np: np.ndarray,
    num_computed_tokens_cpu: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_reqs: int,
) -> bool:
    if _debug_drop_seq_base():
        return False
    if num_reqs <= 0:
        return False

    applied = False
    if num_reqs == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None:
        seq_lens_np[0] = int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE) + int(num_scheduled_tokens[0])
        return True

    sparse_bases = _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX
    if not sparse_bases:
        return False

    seq_lens_np[:num_reqs] = num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens[:num_reqs]
    for req_idx, effective_base in sparse_bases.items():
        idx = int(req_idx)
        if 0 <= idx < num_reqs:
            seq_lens_np[idx] = int(effective_base) + int(num_scheduled_tokens[idx])
            applied = True
    return applied


def make_patched_v1_prepare_inputs(
    original_prepare_inputs: Callable[..., Any],
) -> Callable[..., Any]:
    def _patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        out = original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)

        if not _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED:
            return out

        _patch_state.mark_active_effective_overrides_consumed()

        total_num_scheduled_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
        num_reqs = int(getattr(self.input_batch, "num_reqs", 0))
        if total_num_scheduled_tokens <= 0 or num_reqs <= 0:
            return out

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        original_positions_np = positions_np.copy()

        slot_positions_np = _build_effective_slot_positions(
            positions_np=positions_np,
            req_indices=req_indices,
        )
        if slot_positions_np is not None:
            self.input_batch.block_table.compute_slot_mapping(req_indices, slot_positions_np)
            self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
        forward_positions_np = self.positions.np[:total_num_scheduled_tokens]
        _debug_dump_positions(
            original_positions_np=original_positions_np,
            slot_positions_np=slot_positions_np,
            req_indices=req_indices,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            forward_positions_np=forward_positions_np,
        )

        seq_applied = _apply_sparse_seq_len_overrides_in_place(
            seq_lens_np=self.seq_lens.np,
            num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu,
            num_scheduled_tokens=num_scheduled_tokens,
            num_reqs=num_reqs,
        )
        if seq_applied:
            self.seq_lens.np[num_reqs:].fill(0)
            self.seq_lens.copy_to_gpu()

        return out

    return _patched_prepare_inputs
