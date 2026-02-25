from __future__ import annotations

import torch

from triattention_runtime.layout_engine import PreparedLayerCompaction, execute_group_compaction
from triattention_runtime.plan_models import KeepPlan


def _task(layer_idx: int = 0) -> PreparedLayerCompaction:
    kv_cache = torch.zeros((2, 4, 16, 1, 2), dtype=torch.float32)
    block_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    keep_plan = KeepPlan(mode="shared", indices=[0, 1, 2, 3], semantic=None)
    return PreparedLayerCompaction(
        layer_idx=layer_idx,
        kv_cache=kv_cache,
        block_ids=block_ids,
        keep_plan=keep_plan,
    )


def test_execute_group_compaction_emits_reclaim_group():
    outcome = execute_group_compaction(
        req_id="r1",
        gid=0,
        normalized_block_ids=[0, 1, 2, 3],
        tasks=[_task(0)],
        block_size=16,
        total_tokens=64,
        enable_experimental_block_reclaim=True,
        require_physical_reclaim=True,
        shared_compact_fn=lambda **kwargs: 16,
    )
    assert outcome.cache_len_after == 16
    assert outcome.kept_block_ids == [0]
    assert outcome.removed_block_ids == [1, 2, 3]
    assert outcome.reclaim_group is not None
    assert outcome.reclaim_group.gid == 0


def test_execute_group_compaction_raises_on_inconsistent_layer_lengths():
    calls = {"n": 0}

    def _fake_shared_compact(**kwargs):
        del kwargs
        calls["n"] += 1
        return 16 if calls["n"] == 1 else 15

    try:
        execute_group_compaction(
            req_id="r1",
            gid=0,
            normalized_block_ids=[0, 1],
            tasks=[_task(0), _task(1)],
            block_size=16,
            total_tokens=32,
            enable_experimental_block_reclaim=False,
            require_physical_reclaim=False,
            shared_compact_fn=_fake_shared_compact,
        )
    except RuntimeError as exc:
        assert "inconsistent_cache_len_after" in str(exc)
    else:
        raise AssertionError("expected inconsistent_cache_len_after RuntimeError")
