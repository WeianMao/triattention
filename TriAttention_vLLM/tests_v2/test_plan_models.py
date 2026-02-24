from __future__ import annotations

import torch

from triattention_v2.plan_models import KeepPlan, PlacementPlan, ReclaimEvent, ReclaimGroup


def test_keep_plan_selection_mode_and_count_shared_and_per_head():
    shared = KeepPlan.from_selector_result({"mode": "shared", "indices": [1, 3, 5]})
    assert shared.selection_mode_label == "shared"
    assert shared.keep_count() == 3

    per_head = KeepPlan.from_selector_result(
        {
            "mode": "per_head",
            "indices": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "semantic": "hf_aligned_global_per_head",
        }
    )
    assert per_head.selection_mode_label == "per_head:hf_aligned_global_per_head"
    assert per_head.keep_count() == 2


def test_placement_plan_to_hook_result_dict_serializes_reclaim_event():
    reclaim = ReclaimEvent(
        mode="truncate_tail",
        groups=[
            ReclaimGroup(
                gid=0,
                block_ids_before=[1, 2, 3],
                block_ids_after=[1, 2],
                block_ids_removed=[3],
            )
        ],
    )
    placement = PlacementPlan(
        cache_len_after=2048,
        selector_status="enabled",
        selection_mode="per_layer",
        effective_tokens_before=2176,
        budget_total=2048,
        recent_unabsorbed_tokens=128,
        block_reclaim=reclaim,
    )
    out = placement.to_hook_result_dict()
    assert out["applied"] is True
    assert out["reason"] == "kv_compacted:per_layer"
    assert out["reclaimed_block_count"] == 1
    assert out["block_reclaim"]["mode"] == "truncate_tail"
    assert out["block_reclaim"]["groups"][0]["gid"] == 0
