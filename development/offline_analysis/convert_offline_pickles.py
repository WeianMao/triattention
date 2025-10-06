"""Convert offline DeepThink pickle outputs to lightweight artifacts."""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np


def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python objects."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def normalize_token_id(value: Any) -> Union[int, str]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def to_int_default(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def serialize_logprob_candidates(per_token: Dict[Any, Any]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    if not per_token:
        return serialized

    for cand_token_id, entry in per_token.items():
        serialized.append({
            'token_id': normalize_token_id(cand_token_id),
            'logprob': float(entry.logprob),
            'rank': getattr(entry, 'rank', None),
            'decoded_token': getattr(entry, 'decoded_token', None),
        })
    return serialized


def extract_selected_logprobs(token_ids: List[int], token_logprobs: Iterable[Dict[Any, Any]]) -> List[float]:
    """Return per-token logprob for the actually generated token."""
    if not token_ids:
        return []

    if not token_logprobs:
        return [math.nan] * len(token_ids)

    values: List[float] = []
    for idx, token_id in enumerate(token_ids):
        per_token = token_logprobs[idx] if idx < len(token_logprobs) else None
        entry = None
        if per_token:
            entry = per_token.get(token_id)
            if entry is None:
                entry = per_token.get(str(token_id))
            if entry is None and per_token:
                entry = next(iter(per_token.values()))
        values.append(float(entry.logprob) if entry is not None else math.nan)
    return values


def convert_file(pkl_path: Path, output_root: Path, overwrite: bool = False) -> None:
    target_dir = output_root / pkl_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = target_dir / "metadata.json"
    traces_dir = target_dir / "traces"
    sequences_path = target_dir / "trace_sequences.npz"

    if metadata_path.exists() and traces_dir.exists() and sequences_path.exists() and not overwrite:
        print(f"[skip] {pkl_path.name} already converted")
        return

    if traces_dir.exists() and overwrite:
        shutil.rmtree(traces_dir)

    traces_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {pkl_path}")
    try:
        with pkl_path.open('rb') as fh:
            data = pickle.load(fh)
    except EOFError:
        print(f"[warn] {pkl_path.name} appears to be corrupted (EOFError). Skipping.")
        return
    except Exception as exc:
        print(f"[warn] Failed to load {pkl_path.name}: {exc}. Skipping.")
        return

    traces = data.get('all_traces') or []
    ground_truth = str(data.get('ground_truth', '')).strip()

    # Serialize metadata first.
    metadata = {
        'source_file': pkl_path.name,
        'qid': data.get('qid'),
        'run_id': data.get('run_id'),
        'question': data.get('question'),
        'ground_truth': ground_truth,
        'final_answer': data.get('final_answer'),
        'voted_answer': data.get('voted_answer'),
        'store_logprobs': data.get('store_logprobs'),
        'config': data.get('config'),
        'token_stats': data.get('token_stats'),
        'timing_stats': data.get('timing_stats'),
        'evaluation': to_serializable(data.get('evaluation')),
        'voting_results': to_serializable(data.get('voting_results')),
        'total_traces_count': data.get('total_traces_count'),
        'timestamp': data.get('timestamp'),
    }

    with metadata_path.open('w', encoding='utf-8') as f_meta:
        json.dump(metadata, f_meta, ensure_ascii=False, indent=2)

    # Prepare flattened arrays.
    logprob_values: List[float] = []
    conf_values: List[float] = []
    token_id_values: List[int] = []
    offsets: List[int] = [0]

    for idx, trace in enumerate(traces):
            token_ids = trace.get('token_ids') or []
            token_logprobs = trace.get('logprobs') or []
            confs = trace.get('confs') or []
            stop_reason = trace.get('stop_reason')
            extracted_answer = trace.get('extracted_answer')
            full_text = trace.get('text') or ''

            selected_logprobs = extract_selected_logprobs(token_ids, token_logprobs)

            if not confs or len(confs) != len(selected_logprobs):
                confs = [math.nan] * len(selected_logprobs)

            confs_array = np.asarray([
                float(c) if c is not None and not (isinstance(c, float) and math.isnan(c)) else np.nan
                for c in confs
            ], dtype=np.float32) if confs else np.asarray([], dtype=np.float32)

            logprob_values.extend(selected_logprobs)
            conf_values.extend(float(c) if not math.isnan(c) else math.nan for c in confs_array)
            token_id_values.extend(to_int_default(t) for t in token_ids[:len(selected_logprobs)])
            offsets.append(offsets[-1] + len(selected_logprobs))

            mean_conf = float(np.nanmean(confs_array)) if confs_array.size else math.nan
            min_conf = float(np.nanmin(confs_array)) if confs_array.size else math.nan
            answer_str = str(extracted_answer).strip() if extracted_answer is not None else ''
            is_correct = bool(ground_truth) and answer_str == ground_truth

            serialized_confs = [
                None if (isinstance(val, float) and math.isnan(val)) else float(val)
                for val in confs_array
            ]

            serialized_selected = [
                None if (isinstance(val, float) and math.isnan(val)) else float(val)
                for val in selected_logprobs
            ]

            serialized_logprobs = [serialize_logprob_candidates(per_token) for per_token in token_logprobs]

            serialized_token_ids = [normalize_token_id(t) for t in token_ids]

            trace_record = {
                'trace_index': idx,
                'num_tokens': len(selected_logprobs),
                'stop_reason': stop_reason,
                'extracted_answer': extracted_answer,
                'normalized_answer': answer_str,
                'is_correct': is_correct,
                'mean_confidence': mean_conf,
                'min_confidence': min_conf,
                'text': full_text,
                'token_ids': serialized_token_ids,
                'confs': serialized_confs,
                'selected_logprobs': serialized_selected,
                'candidate_logprobs': serialized_logprobs,
            }

            trace_path = traces_dir / f"trace_{idx:04d}.json.gz"
            with gzip.open(trace_path, 'wt', encoding='utf-8') as trace_fp:
                json.dump(trace_record, trace_fp, ensure_ascii=False)

            # Drop large fields from trace to free memory
            trace.pop('logprobs', None)
            trace.pop('confs', None)
            trace.pop('token_ids', None)
            trace.pop('text', None)

    logprob_arr = np.asarray(logprob_values, dtype=np.float32)
    conf_arr = np.asarray(conf_values, dtype=np.float32)
    token_arr = np.asarray(token_id_values, dtype=np.int32)
    offsets_arr = np.asarray(offsets, dtype=np.int64)

    np.savez_compressed(
        sequences_path,
        logprobs=logprob_arr,
        confs=conf_arr,
        token_ids=token_arr,
        offsets=offsets_arr,
    )

    # Help GC before processing next file
    del data
    gc.collect()
    print(f"[done] {pkl_path.name} -> {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeepThink offline pickles to lightweight format.")
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory containing *.pkl files')
    parser.add_argument('--output-dir', type=Path, required=True, help='Destination directory for converted files')
    parser.add_argument('--overwrite', action='store_true', help='Recreate outputs even if they already exist')
    args = parser.parse_args()

    pkl_files = sorted(args.input_dir.glob('*.pkl'), key=lambda p: p.stat().st_size)
    if not pkl_files:
        raise SystemExit(f"No pickle files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pkl_path in pkl_files:
        convert_file(pkl_path, args.output_dir, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
