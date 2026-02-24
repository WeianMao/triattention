from __future__ import annotations

import torch

from triattention_v2.config import TriAttentionV2Config
from triattention_v2.selection_planner import prepare_group_layer_compactions


def _layer_tensors():
    kv = torch.zeros((2, 2, 16, 1, 2), dtype=torch.float32)
    return [(0, kv)]


def _two_layer_tensors():
    kv0 = torch.zeros((2, 2, 16, 1, 2), dtype=torch.float32)
    kv1 = torch.zeros((2, 2, 16, 1, 2), dtype=torch.float32)
    return [(0, kv0), (1, kv1)]


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


def test_per_layer_per_head_does_not_use_group_global_per_head_selector():
    cfg = TriAttentionV2Config(
        kv_budget=4,
        pruning_mode="per_layer_per_head",
        per_head_selection_semantics="hf_aligned_global_per_head",
        enable_experimental_kv_compaction=True,
        require_triton_scoring=False,
    )
    calls = {"group": 0, "layer": 0}

    def _group_selector(**kwargs):
        del kwargs
        calls["group"] += 1
        return {
            "mode": "per_head",
            "indices": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
            "semantic": "hf_aligned_global_per_head",
        }

    def _layer_selector(**kwargs):
        layer_idx = int(kwargs["layer_idx"])
        calls["layer"] += 1
        base = 0 if layer_idx == 0 else 4
        return {
            "mode": "per_head",
            "indices": torch.tensor(
                [[base + 0, base + 1, base + 2, base + 3],
                 [base + 0, base + 1, base + 2, base + 3]],
                dtype=torch.long,
            ),
            "semantic": "hf_aligned_global_per_head",
        }

    plan = prepare_group_layer_compactions(
        req_id="r1",
        gid=0,
        layer_tensors=_two_layer_tensors(),
        normalized_block_ids=[0, 1],
        block_size=16,
        group_total_tokens=8,
        group_prefill_len=0,
        protect_prefill=False,
        round_start=8,
        group_budget_total=4,
        config=cfg,
        strict_triton_required=False,
        select_keep_indices=_layer_selector,
        select_keep_indices_for_group=_group_selector,
    )

    assert calls["group"] == 0
    assert calls["layer"] == 2
    assert len(plan.tasks) == 2
    assert plan.tasks[0].keep_plan.mode == "per_head"
    assert plan.tasks[1].keep_plan.mode == "per_head"
    t0 = plan.tasks[0].keep_plan.indices
    t1 = plan.tasks[1].keep_plan.indices
    assert isinstance(t0, torch.Tensor) and isinstance(t1, torch.Tensor)
    assert torch.equal(t0, torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long))
    assert torch.equal(t1, torch.tensor([[4, 5, 6, 7], [4, 5, 6, 7]], dtype=torch.long))
