from __future__ import annotations

import torch

from triattention_v2.config import TriAttentionV2Config
from triattention_v2.selection_planner import prepare_group_layer_compactions


def _layer_tensors():
    kv = torch.zeros((2, 2, 16, 1, 2), dtype=torch.float32)
    return [(0, kv)]


def test_prepare_group_layer_compactions_fallback_shared():
    cfg = TriAttentionV2Config(
        kv_budget=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )
    plan = prepare_group_layer_compactions(
        req_id="r1",
        gid=0,
        layer_tensors=_layer_tensors(),
        normalized_block_ids=[0, 1],
        block_size=16,
        group_total_tokens=8,
        group_prefill_len=2,
        protect_prefill=True,
        round_start=8,
        group_budget_total=4,
        config=cfg,
        strict_triton_required=False,
        select_keep_indices=None,
        select_keep_indices_for_group=None,
    )
    assert len(plan.tasks) == 1
    assert plan.selection_mode == "fallback"
    assert plan.tasks[0].keep_plan.mode == "shared"
    assert plan.tasks[0].keep_plan.keep_count() == 4


def test_prepare_group_layer_compactions_selector_path_builds_tasks():
    cfg = TriAttentionV2Config(
        kv_budget=4,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )

    def _selector(**kwargs):
        del kwargs
        return {"mode": "shared", "indices": [0, 3, 5, 7], "semantic": "unit"}

    plan = prepare_group_layer_compactions(
        req_id="r1",
        gid=0,
        layer_tensors=_layer_tensors(),
        normalized_block_ids=[0, 1],
        block_size=16,
        group_total_tokens=8,
        group_prefill_len=0,
        protect_prefill=False,
        round_start=8,
        group_budget_total=4,
        config=cfg,
        strict_triton_required=False,
        select_keep_indices=_selector,
        select_keep_indices_for_group=None,
    )
    assert len(plan.tasks) == 1
    assert plan.selection_mode == "shared:unit"
    assert plan.tasks[0].keep_plan.keep_count() == 4


def test_prepare_group_layer_compactions_maps_prefill_exceeds_budget_to_value_error():
    cfg = TriAttentionV2Config(
        kv_budget=2,
        include_prefill_in_budget=True,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )
    try:
        prepare_group_layer_compactions(
            req_id="r1",
            gid=0,
            layer_tensors=_layer_tensors(),
            normalized_block_ids=[0, 1],
            block_size=16,
            group_total_tokens=8,
            group_prefill_len=4,
            protect_prefill=True,
            round_start=8,
            group_budget_total=2,
            config=cfg,
            strict_triton_required=False,
            select_keep_indices=None,
            select_keep_indices_for_group=None,
        )
    except ValueError as exc:
        assert str(exc) == "prefill_exceeds_budget"
    else:
        raise AssertionError("expected ValueError('prefill_exceeds_budget')")
