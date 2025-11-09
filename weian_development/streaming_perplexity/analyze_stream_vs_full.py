#!/usr/bin/env python3
"""Compare full-attention and streaming perplexity outputs sentence-by-sentence."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from transformers import AutoTokenizer

MODEL_PATH = Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B")
FULL_ROOT = Path("outputs/deepseek_r1_qwen3_8b/perplexity_full")
STREAM_ROOT = Path("outputs/deepseek_r1_qwen3_8b/perplexity_stream")
OUTPUT_PATH = Path("outputs/deepseek_r1_qwen3_8b/perplexity_stream_sentence_stats.json")
TRACE_OFFSET = 0  # pick the first trace for analysis


@dataclass
class TracePayload:
    token_ids: torch.Tensor
    log_probs: torch.Tensor
    trace_index: int


_sentence_regex = re.compile(r"[^.!?\n]+(?:[.!?\n]+|$)")


def load_payload(path: Path, trace_idx: int) -> TracePayload:
    payload = torch.load(path, map_location="cpu")
    trace_entry = payload["trace_data"][trace_idx]
    return TracePayload(
        token_ids=trace_entry["generated_token_ids"],
        log_probs=trace_entry["log_probs"],
        trace_index=int(trace_entry["trace_index"]),
    )


def build_token_spans(tokenizer, token_ids: Sequence[int]) -> Dict[str, object]:
    pieces: List[str] = []
    spans: List[Dict[str, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        pieces.append(piece)
        start = cursor
        cursor += len(piece)
        spans.append({"start": start, "end": cursor})
    text = "".join(pieces)
    return {"text": text, "spans": spans}


def sentence_spans(text: str) -> List[Dict[str, int]]:
    spans: List[Dict[str, int]] = []
    for match in _sentence_regex.finditer(text):
        start, end = match.span()
        snippet = text[start:end].strip()
        if not snippet:
            continue
        spans.append({"start": start, "end": end, "text": snippet})
    return spans


def tokens_for_span(spans: List[Dict[str, int]], start: int, end: int) -> List[int]:
    token_indices: List[int] = []
    for idx, span in enumerate(spans):
        if span["end"] <= start:
            continue
        if span["start"] >= end:
            break
        token_indices.append(idx)
    return token_indices


def compute_sentence_metrics(
    tokenizer,
    token_ids: torch.Tensor,
    log_probs_stream: torch.Tensor,
    log_probs_full: torch.Tensor,
) -> List[Dict[str, object]]:
    decoded = build_token_spans(tokenizer, token_ids.tolist())
    token_spans = decoded["spans"]  # type: ignore[assignment]
    sentences = sentence_spans(decoded["text"])  # type: ignore[arg-type]
    metrics: List[Dict[str, object]] = []

    for sentence in sentences:
        idxs = tokens_for_span(token_spans, sentence["start"], sentence["end"])
        if not idxs:
            continue
        stream_values = log_probs_stream[idxs]
        full_values = log_probs_full[idxs]
        avg_stream = float(stream_values.mean().item())
        avg_full = float(full_values.mean().item())
        metrics.append(
            {
                "sentence": sentence["text"],
                "token_count": len(idxs),
                "avg_log_prob_stream": avg_stream,
                "perplexity_stream": math.exp(-avg_stream),
                "avg_log_prob_full": avg_full,
                "perplexity_full": math.exp(-avg_full),
            }
        )
    return metrics


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    meta_dir = FULL_ROOT / "metadata"
    full_tensor_dir = FULL_ROOT / "tensors"
    stream_tensor_dir = STREAM_ROOT / "tensors"

    summaries: List[Dict[str, object]] = []
    kl_values: List[Dict[str, float]] = []

    for meta_path in sorted(meta_dir.glob("*.json")):
        stem = meta_path.stem
        stream_tensor_path = stream_tensor_dir / f"{stem}.pt"
        full_tensor_path = full_tensor_dir / f"{stem}.pt"
        if not stream_tensor_path.exists():
            continue

        with meta_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)

        full_payload = load_payload(full_tensor_path, TRACE_OFFSET)
        stream_payload = load_payload(stream_tensor_path, TRACE_OFFSET)

        if full_payload.token_ids.numel() != stream_payload.token_ids.numel():
            raise RuntimeError(f"Token length mismatch for {stem}")

        if full_payload.trace_index != stream_payload.trace_index:
            raise RuntimeError(f"Trace index mismatch for {stem}")

        sentence_metrics = compute_sentence_metrics(
            tokenizer,
            stream_payload.token_ids,
            stream_payload.log_probs,
            full_payload.log_probs,
        )

        log_prob_diff = full_payload.log_probs - stream_payload.log_probs
        mean_kl = float(log_prob_diff.mean().item())
        token_count = int(full_payload.log_probs.numel())
        kl_values.append({"kl": mean_kl, "tokens": token_count})

        summaries.append(
            {
                "source": stem,
                "question": meta.get("question"),
                "trace_index": full_payload.trace_index,
                "token_count": token_count,
                "trace_kl": mean_kl,
                "sentences": sentence_metrics,
            }
        )

    total_tokens = sum(item["tokens"] for item in kl_values)
    token_weighted = 0.0
    if total_tokens:
        token_weighted = sum(item["kl"] * item["tokens"] for item in kl_values) / total_tokens

    aggregate = {
        "trace_count": len(kl_values),
        "mean_kl": sum(item["kl"] for item in kl_values) / max(len(kl_values), 1),
        "token_weighted_kl": token_weighted,
        "max_kl": max((item["kl"] for item in kl_values), default=0.0),
        "min_kl": min((item["kl"] for item in kl_values), default=0.0),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump({"aggregate": aggregate, "questions": summaries}, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
