from evaluation.dispatch.triattention_sharded_dispatch import compute_local_runs


def test_compute_local_runs_even_split():
    assert compute_local_runs(8, 8, 0) == (0, 1)
    assert compute_local_runs(8, 8, 7) == (7, 1)


def test_compute_local_runs_uneven_split():
    # 8 runs on 3 shards -> [3, 3, 2]
    assert compute_local_runs(8, 3, 0) == (0, 3)
    assert compute_local_runs(8, 3, 1) == (3, 3)
    assert compute_local_runs(8, 3, 2) == (6, 2)


def test_compute_local_runs_more_shards_than_runs():
    # 3 runs on 8 shards -> first 3 shards get 1 run, others get 0
    assert compute_local_runs(3, 8, 0) == (0, 1)
    assert compute_local_runs(3, 8, 2) == (2, 1)
    assert compute_local_runs(3, 8, 3) == (3, 0)
    assert compute_local_runs(3, 8, 7) == (3, 0)
