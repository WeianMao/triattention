import json

from evaluation.dispatch.triattention_sharded_dispatch import (
    compute_local_runs,
    validate_merged_output_completeness,
)


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


def test_validate_merged_output_completeness_passes(tmp_path):
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    merged_file = merged_dir / "merged.jsonl"
    rows = [
        {"sample_idx": 0, "draw_idx": 0, "output": "a"},
        {"sample_idx": 0, "draw_idx": 1, "output": "b"},
        {"sample_idx": 1, "draw_idx": 0, "output": "c"},
        {"sample_idx": 1, "draw_idx": 1, "output": "d"},
    ]
    with merged_file.open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")

    validate_merged_output_completeness(merged_dir, expected_num_samples=2)


def test_validate_merged_output_completeness_raises_on_incomplete(tmp_path):
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    merged_file = merged_dir / "merged.jsonl"
    rows = [
        {"sample_idx": 0, "draw_idx": 0, "output": "a"},
        {"sample_idx": 1, "draw_idx": 0, "output": "c"},
        {"sample_idx": 1, "draw_idx": 1, "output": "d"},
    ]
    with merged_file.open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")

    try:
        validate_merged_output_completeness(merged_dir, expected_num_samples=2)
    except RuntimeError as exc:
        assert "incomplete or duplicate" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for incomplete merged outputs")


def test_validate_merged_output_completeness_raises_on_empty_file(tmp_path):
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "merged.jsonl").write_text("")

    try:
        validate_merged_output_completeness(merged_dir, expected_num_samples=2)
    except RuntimeError as exc:
        assert "no valid records" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for empty merged outputs")
