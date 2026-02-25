"""Runtime input adapter entry points for TriAttention runtime.

This module centralizes runner-side preparation of effective-length/slot
semantics before worker input prep. It currently wraps the existing
`gpu_seq_len_patch` sparse override path and serves as the migration point
toward a patch-light/native adapter implementation (D-017).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from .effective_overrides import build_effective_sparse_overrides
from .request_key_compat import get_scheduled_token_items
from .input_patch_backend import (
    activate_effective_sparse_overrides,
    clear_effective_overrides,
)


@dataclass(frozen=True)
class EffectiveInputOverrides:
    seq_base_map: dict[int, int] | None
    pos_delta_map: dict[int, int] | None
    single_seq_base: int | None
    single_pos_delta: int
    expected_req_row_indices: tuple[int, ...] | None = None
    expected_query_lens: tuple[int, ...] | None = None

    def activate(self) -> None:
        activate_effective_sparse_overrides(
            seq_base_map=self.seq_base_map,
            pos_delta_map=self.pos_delta_map,
            single_seq_base=self.single_seq_base,
            single_pos_delta=self.single_pos_delta,
            expected_req_row_indices=self.expected_req_row_indices,
            expected_query_lens=self.expected_query_lens,
        )

    @staticmethod
    def clear() -> None:
        clear_effective_overrides()


@contextmanager
def active_effective_input_overrides(
    overrides: EffectiveInputOverrides,
) -> Iterator[EffectiveInputOverrides]:
    """Context manager boundary for runtime override activation/cleanup."""
    overrides.activate()
    try:
        yield overrides
    finally:
        EffectiveInputOverrides.clear()


def prepare_effective_input_overrides(
    *,
    base_runner: Any,
    state_store: Any,
    scheduler_output: Any,
) -> EffectiveInputOverrides:
    seq_base_map, pos_delta_map, single_seq_base, single_pos_delta = build_effective_sparse_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        compression_events=None,
    )
    expected_req_row_indices: tuple[int, ...] | None = None
    expected_query_lens: tuple[int, ...] | None = None
    req_states = getattr(base_runner, "req_states", None)
    req_id_to_index = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    scheduled_items = get_scheduled_token_items(scheduler_output)
    if isinstance(req_id_to_index, dict):
        row_indices: list[int] = []
        q_lens: list[int] = []
        all_ok = True
        for _raw_key, req_id, scheduled_tokens in scheduled_items:
            req_idx = req_id_to_index.get(req_id)
            if not isinstance(req_idx, int):
                all_ok = False
                break
            row_indices.append(int(req_idx))
            q_lens.append(int(scheduled_tokens))
        if all_ok and row_indices:
            expected_req_row_indices = tuple(row_indices)
            expected_query_lens = tuple(q_lens)
        elif (seq_base_map or pos_delta_map) and not all_ok:
            raise RuntimeError(
                "TRIATTN_EXPECTED_REQ_ROW_INDEX_BUILD_FAILED:"
                "scheduled_req_missing_req_state_index_while_overrides_active"
            )
    elif seq_base_map or pos_delta_map:
        raise RuntimeError(
            "TRIATTN_EXPECTED_REQ_ROW_INDEX_UNAVAILABLE:"
            "req_id_to_index_missing_while_overrides_active"
        )

    return EffectiveInputOverrides(
        seq_base_map=seq_base_map,
        pos_delta_map=pos_delta_map,
        single_seq_base=single_seq_base,
        single_pos_delta=single_pos_delta,
        expected_req_row_indices=expected_req_row_indices,
        expected_query_lens=expected_query_lens,
    )
