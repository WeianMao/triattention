import torch

from triattention_v2.kv_compaction import (
    build_keep_token_indices,
    compact_request_kv_in_place,
    compact_request_kv_in_place_per_head,
    gather_request_k_dense,
    gather_request_k_dense_range,
    gather_request_kv_dense,
)


def test_build_keep_indices_protect_prefill():
    keep = build_keep_token_indices(
        total_tokens=12,
        kv_budget=8,
        prefill_len=3,
        protect_prefill=True,
    )
    assert keep == [0, 1, 2, 7, 8, 9, 10, 11]


def test_build_keep_indices_prefill_overflow():
    keep = build_keep_token_indices(
        total_tokens=20,
        kv_budget=8,
        prefill_len=9,
        protect_prefill=True,
    )
    assert keep is None


def test_build_keep_indices_trim_prefill_mode():
    keep = build_keep_token_indices(
        total_tokens=12,
        kv_budget=8,
        prefill_len=5,
        protect_prefill=False,
    )
    assert keep == [4, 5, 6, 7, 8, 9, 10, 11]


def test_build_keep_indices_prefill_outside_budget():
    keep = build_keep_token_indices(
        total_tokens=20,
        kv_budget=8,
        prefill_len=6,
        protect_prefill=True,
        include_prefill_in_budget=False,
    )
    assert keep == [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19]


def test_compact_request_kv_in_place_layout0():
    # Layout: [2, num_blocks, block_size, H, D]
    kv = torch.arange(2 * 4 * 4 * 1 * 2, dtype=torch.float32).view(2, 4, 4, 1, 2)
    block_ids = [0, 1, 2]
    total_tokens = 10  # consume slots in blocks 0,1 and half of 2
    keep = [0, 1, 7, 8]

    kept = compact_request_kv_in_place(
        kv_cache=kv,
        block_ids=block_ids,
        block_size=4,
        keep_token_indices=keep,
        total_tokens=total_tokens,
    )
    assert kept == 4

    # destination token 0 should equal source token 0
    assert torch.equal(kv[0, 0, 0], torch.tensor([[0.0, 1.0]]))
    # destination token 2 should equal original source token 7 (block1,off3)
    assert torch.equal(kv[0, 0, 2], torch.tensor([[14.0, 15.0]]))
    # trailing area keeps reordered non-selected tokens (no zero tail)
    assert torch.equal(kv[0, 1, 2], torch.tensor([[8.0, 9.0]]))


def test_gather_request_kv_dense_layout0():
    kv = torch.arange(2 * 3 * 4 * 1 * 2, dtype=torch.float32).view(2, 3, 4, 1, 2)
    keys, values = gather_request_kv_dense(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        total_tokens=6,
    )
    assert keys.shape == (1, 1, 6, 2)
    assert values.shape == (1, 1, 6, 2)
    # token 0
    assert torch.equal(keys[0, 0, 0], torch.tensor([0.0, 1.0]))
    # token 5 -> block1 offset1
    assert torch.equal(keys[0, 0, 5], torch.tensor([10.0, 11.0]))


def test_gather_request_k_dense_layout0():
    kv = torch.arange(2 * 3 * 4 * 1 * 2, dtype=torch.float32).view(2, 3, 4, 1, 2)
    keys = gather_request_k_dense(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        total_tokens=6,
    )
    assert keys.shape == (1, 1, 6, 2)
    assert torch.equal(keys[0, 0, 0], torch.tensor([0.0, 1.0]))
    assert torch.equal(keys[0, 0, 5], torch.tensor([10.0, 11.0]))


def test_gather_request_k_dense_non_consecutive_blocks():
    kv = torch.arange(2 * 4 * 4 * 1 * 2, dtype=torch.float32).view(2, 4, 4, 1, 2)
    keys = gather_request_k_dense(
        kv_cache=kv,
        block_ids=[0, 2],
        block_size=4,
        total_tokens=6,
    )
    assert keys.shape == (1, 1, 6, 2)
    # token 5 -> block2 offset1
    assert torch.equal(keys[0, 0, 5], torch.tensor([18.0, 19.0]))


def test_gather_request_kv_dense_non_consecutive_blocks():
    kv = torch.arange(2 * 4 * 4 * 1 * 2, dtype=torch.float32).view(2, 4, 4, 1, 2)
    keys, values = gather_request_kv_dense(
        kv_cache=kv,
        block_ids=[1, 3],
        block_size=4,
        total_tokens=7,
    )
    assert keys.shape == (1, 1, 7, 2)
    assert values.shape == (1, 1, 7, 2)
    # token 6 -> block3 offset2
    assert torch.equal(keys[0, 0, 6], torch.tensor([28.0, 29.0]))
    assert torch.equal(values[0, 0, 6], torch.tensor([60.0, 61.0]))


def test_gather_request_k_dense_range_layout0():
    kv = torch.arange(2 * 4 * 4 * 1 * 2, dtype=torch.float32).view(2, 4, 4, 1, 2)
    keys = gather_request_k_dense_range(
        kv_cache=kv,
        block_ids=[0, 1, 2],
        block_size=4,
        start_token=3,
        num_tokens=5,
    )
    assert keys.shape == (1, 1, 5, 2)
    assert torch.equal(keys[0, 0, 0], torch.tensor([6.0, 7.0]))
    assert torch.equal(keys[0, 0, 4], torch.tensor([14.0, 15.0]))


def test_gather_request_k_dense_range_non_consecutive_blocks():
    kv = torch.arange(2 * 4 * 4 * 1 * 2, dtype=torch.float32).view(2, 4, 4, 1, 2)
    keys = gather_request_k_dense_range(
        kv_cache=kv,
        block_ids=[0, 2, 3],
        block_size=4,
        start_token=4,
        num_tokens=4,
    )
    assert keys.shape == (1, 1, 4, 2)
    # logical token 4 maps to block 2 offset 0
    assert torch.equal(keys[0, 0, 0], torch.tensor([16.0, 17.0]))
    assert torch.equal(keys[0, 0, 3], torch.tensor([22.0, 23.0]))


def test_gather_request_k_dense_range_with_tensor_block_ids():
    kv = torch.arange(2 * 4 * 4 * 1 * 2, dtype=torch.float32).view(2, 4, 4, 1, 2)
    block_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    keys = gather_request_k_dense_range(
        kv_cache=kv,
        block_ids=block_ids,
        block_size=4,
        start_token=2,
        num_tokens=6,
    )
    assert keys.shape == (1, 1, 6, 2)
    assert torch.equal(keys[0, 0, 0], torch.tensor([4.0, 5.0]))
    assert torch.equal(keys[0, 0, 5], torch.tensor([14.0, 15.0]))


def test_compact_request_kv_in_place_per_head_layout0():
    kv = torch.arange(2 * 3 * 4 * 2 * 1, dtype=torch.float32).view(2, 3, 4, 2, 1)
    # total tokens: 8 (block0 + block1)
    kept = compact_request_kv_in_place_per_head(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        keep_token_indices_per_head=[
            [0, 7],  # head 0 keeps token 0 and token 7
            [1, 6],  # head 1 keeps token 1 and token 6
        ],
        total_tokens=8,
    )
    assert kept == 2
    # dst token 0: head0 from token0, head1 from token1
    assert torch.equal(kv[0, 0, 0, 0], torch.tensor([0.0]))
    assert torch.equal(kv[0, 0, 0, 1], torch.tensor([3.0]))
    # dst token 1: head0 from token7, head1 from token6
    assert torch.equal(kv[0, 0, 1, 0], torch.tensor([14.0]))
    assert torch.equal(kv[0, 0, 1, 1], torch.tensor([13.0]))
    # tail stores reordered non-selected tokens per head (no zero tail)
    assert torch.equal(kv[0, 0, 2], torch.tensor([[2.0], [1.0]]))


def test_compact_request_kv_in_place_keep_only_fast_path():
    kv = torch.arange(2 * 4 * 4 * 1 * 2, dtype=torch.float32).view(2, 4, 4, 1, 2)
    block_ids = [0, 1, 2]
    total_tokens = 10
    keep = [0, 1, 7, 8]

    kept = compact_request_kv_in_place(
        kv_cache=kv,
        block_ids=block_ids,
        block_size=4,
        keep_token_indices=keep,
        total_tokens=total_tokens,
        preserve_dropped_tokens=False,
    )
    assert kept == 4
    assert torch.equal(kv[0, 0, 0], torch.tensor([[0.0, 1.0]]))
    assert torch.equal(kv[0, 0, 1], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(kv[0, 0, 2], torch.tensor([[14.0, 15.0]]))
    assert torch.equal(kv[0, 0, 3], torch.tensor([[16.0, 17.0]]))


def test_compact_request_kv_in_place_per_head_keep_only_fast_path():
    kv = torch.arange(2 * 3 * 4 * 2 * 1, dtype=torch.float32).view(2, 3, 4, 2, 1)
    kept = compact_request_kv_in_place_per_head(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        keep_token_indices_per_head=[
            [0, 7],
            [1, 6],
        ],
        total_tokens=8,
        preserve_dropped_tokens=False,
    )
    assert kept == 2
    assert torch.equal(kv[0, 0, 0, 0], torch.tensor([0.0]))
    assert torch.equal(kv[0, 0, 0, 1], torch.tensor([13.0]))
    assert torch.equal(kv[0, 0, 1, 0], torch.tensor([14.0]))
    assert torch.equal(kv[0, 0, 1, 1], torch.tensor([3.0]))


def test_compact_request_kv_in_place_keep_only_prefix_noop():
    kv = torch.arange(2 * 3 * 4 * 1 * 2, dtype=torch.float32).view(2, 3, 4, 1, 2)
    before = kv.clone()
    kept = compact_request_kv_in_place(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        keep_token_indices=[0, 1, 2, 3],
        total_tokens=8,
        preserve_dropped_tokens=False,
    )
    assert kept == 4
    assert torch.equal(kv, before)


def test_compact_request_kv_in_place_per_head_keep_only_prefix_noop():
    kv = torch.arange(2 * 3 * 4 * 2 * 1, dtype=torch.float32).view(2, 3, 4, 2, 1)
    before = kv.clone()
    kept = compact_request_kv_in_place_per_head(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        keep_token_indices_per_head=[
            [0, 1, 2],
            [0, 1, 2],
        ],
        total_tokens=8,
        preserve_dropped_tokens=False,
    )
    assert kept == 3
    assert torch.equal(kv, before)


def test_compact_request_kv_in_place_keep_only_fill_holes_not_shift_prefix_kept():
    kv = torch.arange(2 * 3 * 4 * 1 * 1, dtype=torch.float32).view(2, 3, 4, 1, 1)
    # tokens 0..7 mapped on blocks [0,1]
    kept = compact_request_kv_in_place(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        keep_token_indices=[0, 2, 3, 4],  # hole at dst slot 1, tail survivor token 4
        total_tokens=8,
        preserve_dropped_tokens=False,
    )
    assert kept == 4
    # slot0 keeps token0
    assert torch.equal(kv[0, 0, 0, 0], torch.tensor([0.0]))
    # slot1 filled by tail survivor token4
    assert torch.equal(kv[0, 0, 1, 0], torch.tensor([4.0]))
    # slot2/3 remain token2/token3 (no unnecessary shift)
    assert torch.equal(kv[0, 0, 2, 0], torch.tensor([2.0]))
    assert torch.equal(kv[0, 0, 3, 0], torch.tensor([3.0]))


def test_compact_request_kv_in_place_per_head_keep_only_fill_holes():
    kv = torch.arange(2 * 3 * 4 * 2 * 1, dtype=torch.float32).view(2, 3, 4, 2, 1)
    kept = compact_request_kv_in_place_per_head(
        kv_cache=kv,
        block_ids=[0, 1],
        block_size=4,
        keep_token_indices_per_head=[
            [0, 2, 4],  # head0 hole@1 <- token4
            [1, 2, 5],  # head1 hole@0 <- token5
        ],
        total_tokens=8,
        preserve_dropped_tokens=False,
    )
    assert kept == 3
    # head0: slot0 token0, slot1 token4, slot2 token2
    assert torch.equal(kv[0, 0, 0, 0], torch.tensor([0.0]))
    assert torch.equal(kv[0, 0, 1, 0], torch.tensor([8.0]))
    assert torch.equal(kv[0, 0, 2, 0], torch.tensor([4.0]))
    # head1: slot0 token5, slot1 token1, slot2 token2
    assert torch.equal(kv[0, 0, 0, 1], torch.tensor([11.0]))
    assert torch.equal(kv[0, 0, 1, 1], torch.tensor([3.0]))
    assert torch.equal(kv[0, 0, 2, 1], torch.tensor([5.0]))
