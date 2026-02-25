import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from benchmarks.reasoning.compare_results import (
    calculate_accuracy,
    generate_comparison_report,
    load_results,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_results_keeps_index_zero_distinct():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "rows.jsonl"
        _write_jsonl(
            path,
            [
                {"index": 0, "question": "q0", "answer": "1", "output": "ans 1"},
                {"index": 1, "question": "q1", "answer": "2", "output": "ans 2"},
            ],
        )
        rows = load_results(str(path))
        assert len(rows) == 2
        rows_by_index = {row["index"]: row for row in rows}
        assert rows_by_index[0]["generated_answers"] == ["ans 1"]
        assert rows_by_index[1]["generated_answers"] == ["ans 2"]


def test_calculate_accuracy_fallback_boxed_extract():
    rows = [
        {
            "index": 0,
            "ground_truth": "42",
            "generated_answers": ["reasoning ... \\boxed{42}"],
        }
    ]
    acc, correct, total = calculate_accuracy(
        rows,
        dataset_name="aime24",
        extract_answer_fn=None,
        parse_ground_truth_fn=None,
        math_equal_fn=None,
    )
    assert total == 1
    assert correct == 1
    assert acc == 1.0


def test_generate_report_handles_empty_generated_answers():
    args = SimpleNamespace(dataset_name="aime24", detailed=True)
    report = generate_comparison_report(
        hf_results=[{"index": 0, "ground_truth": "1", "generated_answers": []}],
        vllm_results=[{"index": 0, "ground_truth": "1", "generated_answers": []}],
        args=args,
        helper_status="fallback simple extractor",
        extract_answer_fn=None,
        parse_ground_truth_fn=None,
        math_equal_fn=None,
    )
    assert "Common Questions: 1" in report
