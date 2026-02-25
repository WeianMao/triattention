from types import SimpleNamespace

from triattention_runtime.scheduler import TriAttentionScheduler


class _Tracker:
    def __init__(self) -> None:
        self.calls = []

    def apply_compression(self, req_id, cache_len_after, num_computed_tokens):
        self.calls.append(
            (req_id, int(cache_len_after), int(num_computed_tokens))
        )


class _FakeBlock:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id


class _FakeBlockPool:
    def __init__(self, size: int) -> None:
        self.blocks = [_FakeBlock(i) for i in range(size)]
        self.freed_block_ids: list[list[int]] = []

    def free_blocks(self, ordered_blocks):
        blocks = list(ordered_blocks)
        self.freed_block_ids.append([block.block_id for block in blocks])


def _make_scheduler(enable_reclaim: bool):
    scheduler = TriAttentionScheduler.__new__(TriAttentionScheduler)
    scheduler.triattention_config = SimpleNamespace(
        enable_experimental_block_reclaim=enable_reclaim,
        require_physical_reclaim=True,
    )
    scheduler.block_size = 16
    scheduler._effective_len_tracker = _Tracker()
    scheduler.requests = {"r1": SimpleNamespace(num_computed_tokens=77)}

    pool = _FakeBlockPool(size=8)
    manager = SimpleNamespace(
        req_to_blocks={"r1": [pool.blocks[0], pool.blocks[1], pool.blocks[2], pool.blocks[3]]},
        num_cached_block={"r1": 4},
        block_pool=pool,
    )
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=(manager,),
            block_pool=pool,
        )
    )
    return scheduler, manager, pool


def test_scheduler_applies_block_reclaim_event():
    scheduler, manager, pool = _make_scheduler(enable_reclaim=True)
    scheduler._apply_compression_events(
        [
            {
                "status": "applied",
                "req_id": "r1",
                "cache_len_after": 32,
                "block_reclaim": {
                    "groups": [
                        {
                            "gid": 0,
                            "block_ids_after": [0, 1],
                        }
                    ]
                },
            }
        ]
    )

    assert scheduler._effective_len_tracker.calls == [("r1", 32, 77)]
    assert [blk.block_id for blk in manager.req_to_blocks["r1"]] == [0, 1]
    assert manager.num_cached_block["r1"] == 2
    # Removed tail blocks are freed in tail-first order.
    assert pool.freed_block_ids == [[3, 2]]


def test_scheduler_skips_reclaim_when_disabled():
    scheduler, manager, pool = _make_scheduler(enable_reclaim=False)
    scheduler._apply_compression_events(
        [
            {
                "status": "applied",
                "req_id": "r1",
                "cache_len_after": 16,
                "block_reclaim": {
                    "groups": [
                        {
                            "gid": 0,
                            "block_ids_after": [0],
                        }
                    ]
                },
            }
        ]
    )

    assert scheduler._effective_len_tracker.calls == [("r1", 16, 77)]
    assert [blk.block_id for blk in manager.req_to_blocks["r1"]] == [0, 1, 2, 3]
    assert pool.freed_block_ids == []


def test_scheduler_reclaim_prefix_mismatch_raises():
    scheduler, _manager, _pool = _make_scheduler(enable_reclaim=True)

    try:
        scheduler._apply_compression_events(
            [
                {
                    "status": "applied",
                    "req_id": "r1",
                    "cache_len_after": 16,
                    "block_reclaim": {
                        "groups": [
                            {
                                "gid": 0,
                                "block_ids_after": [1, 0],
                            }
                        ]
                    },
                }
            ]
        )
    except RuntimeError as exc:
        assert "prefix mismatch" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for reclaim prefix mismatch")


def test_scheduler_missing_block_reclaim_raises_when_shrink_expected():
    scheduler, _manager, _pool = _make_scheduler(enable_reclaim=True)

    try:
        scheduler._apply_compression_events(
            [
                {
                    "status": "applied",
                    "req_id": "r1",
                    "cache_len_after": 16,
                }
            ]
        )
    except RuntimeError as exc:
        assert "missing while shrink expected" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing reclaim payload")
