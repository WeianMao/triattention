from __future__ import annotations

from types import SimpleNamespace

from triattention_runtime.config import TriAttentionRuntimeConfig
from triattention_runtime.hook_group_pipeline import (
    GroupPipelineOutcome,
    finalize_hook_placement_result,
    normalize_mutable_block_ids_by_group,
)
from triattention_runtime.plan_models import ReclaimGroup


def test_normalize_mutable_block_ids_by_group_handles_mixed_groups():
    out = normalize_mutable_block_ids_by_group(([0, 1], None, (2, 3)))
    assert out == [[0, 1], None, [2, 3]]
    assert normalize_mutable_block_ids_by_group(None) is None
    assert normalize_mutable_block_ids_by_group("bad") is None


def test_finalize_hook_placement_result_updates_req_state_block_ids_and_payload():
    req_state = SimpleNamespace(block_ids=([0, 1, 2], [5, 6]))
    cfg = TriAttentionRuntimeConfig(enable_experimental_block_reclaim=True)
    outcome = GroupPipelineOutcome(
        cache_len_after=16,
        selection_mode="shared",
        block_reclaim_groups=[
            ReclaimGroup(
                gid=0,
                block_ids_before=[0, 1, 2],
                block_ids_after=[0],
                block_ids_removed=[1, 2],
            )
        ],
        mutable_block_ids_by_group=[[0], [5, 6]],
    )
    result = finalize_hook_placement_result(
        req_state=req_state,
        original_block_ids_by_group=req_state.block_ids,
        config=cfg,
        selector_status="enabled",
        outcome=outcome,
        effective_tokens=32,
        budget_total=16,
        recent_unabsorbed_tokens=4,
    )
    assert result["applied"] is True
    assert result["cache_len_after"] == 16
    assert result["reclaimed_block_count"] == 2
    assert isinstance(result["block_reclaim"], dict)
    assert req_state.block_ids[0] == [0]
    assert req_state.block_ids[1] == [5, 6]
