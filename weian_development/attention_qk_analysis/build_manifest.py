"""构建离线 trace Q/K 捕获的抽样清单。"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


@dataclass
class TraceEntry:
    qid: int
    source_path: str
    trace_index: int
    question: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采样离线 reasoning traces 构建 manifest")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="存放 offline reasoning JSON 的目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weian_development/attention_qk_analysis/trace_manifest.json"),
        help="输出 manifest 路径",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="抽样问题数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20250107,
        help="随机种子",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出调试信息",
    )
    return parser.parse_args()


def load_trace_candidates(json_path: Path) -> Tuple[int, str, List[int]]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"文件 {json_path} 内容为空列表")
        payload = payload[0]

    qid = int(payload["qid"])
    question = payload.get("question", "").strip()
    traces = payload.get("traces", [])
    indices = [int(trace.get("index", idx + 1)) for idx, trace in enumerate(traces)]
    return qid, question, indices


def build_manifest(entries: Dict[int, List[TraceEntry]]) -> List[Dict[str, object]]:
    manifest: List[Dict[str, object]] = []
    for qid, choices in entries.items():
        for entry in choices:
            manifest.append(asdict(entry))
    return manifest


def main() -> None:
    mask_process_command("PD-L1_binder_sel")
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"输入目录不存在: {args.input_dir}")

    json_files = sorted(
        p for p in args.input_dir.glob("*.json") if p.name.startswith("deepthink_offline_qid")
    )
    if not json_files:
        raise SystemExit(f"在 {args.input_dir} 下没有找到 JSON 文件")

    qid_to_file: Dict[int, Path] = {}
    qid_to_question: Dict[int, str] = {}
    qid_to_trace_indices: Dict[int, List[int]] = {}

    for json_path in json_files:
        qid, question, indices = load_trace_candidates(json_path)
        qid_to_file[qid] = json_path
        qid_to_question[qid] = question
        qid_to_trace_indices[qid] = indices

    available_qids = sorted(qid_to_file.keys())
    target_count = args.count
    if len(available_qids) < target_count:
        target_count = len(available_qids)
        if args.verbose:
            print(
                f"警告: 仅找到 {target_count} 个问题，将按可用数量构建 manifest",
                flush=True,
            )

    rng = random.Random(args.seed)
    chosen_qids = sorted(rng.sample(available_qids, target_count))

    sampled_entries: Dict[int, List[TraceEntry]] = {}
    for qid in chosen_qids:
        trace_indices = qid_to_trace_indices[qid]
        if not trace_indices:
            raise SystemExit(f"问题 {qid} 在 {qid_to_file[qid]} 中没有 trace")
        chosen_index = rng.choice(trace_indices)
        entry = TraceEntry(
            qid=qid,
            source_path=str(qid_to_file[qid].resolve()),
            trace_index=chosen_index,
            question=qid_to_question[qid],
        )
        sampled_entries[qid] = [entry]

    manifest_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": args.seed,
        "requested_count": args.count,
        "actual_count": target_count,
        "input_dir": str(args.input_dir.resolve()),
        "entries": build_manifest(sampled_entries),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(manifest_payload, fh, ensure_ascii=False, indent=2)

    if args.verbose:
        print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
