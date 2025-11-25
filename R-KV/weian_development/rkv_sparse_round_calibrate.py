#!/usr/bin/env python3
"""Calibrate SparseRound (prefill-keep) stats for DeepSeek-R1 on R-KV traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weian_development.attention_qk_analysis.capture_qk_distributed import (
    LayerCaptureBuffer,
    QKCollector,
)
from weian_development.hf_offline_runner_sparse.round_pruning_utils import (
    HeadFrequencyStats,
    build_rotary,
    compute_rotary_tables,
    invert_rope,
    load_or_create_sample,
    save_head_frequency_stats,
    to_complex_pairs,
)
from weian_development.process_utils import mask_process_command

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS_OUT = (
    PROJECT_ROOT
    / "R-KV"
    / "outputs"
    / "sample8_fullkv_aime24_official"
    / "stats"
    / "deepseek_r1_llama8b_chat_stats.pt"
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=PROJECT_ROOT
        / "R-KV"
        / "outputs"
        / "sample8_fullkv_aime24_official",
        help="Directory containing merged.jsonl or shard outputs used as trace text.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B"),
        help="Path to the HuggingFace model directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_STATS_OUT,
        help="Destination for the serialized stats (.pt).",
    )
    parser.add_argument(
        "--head-sample-file",
        type=Path,
        default=PROJECT_ROOT
        / "weian_development"
        / "hf_offline_runner_sparse"
        / "stats"
        / "deepseek_r1_llama8b_heads.json",
        help="JSON file describing sampled heads; created if missing.",
    )
    parser.add_argument("--sample-count", type=int, default=100, help="Number of heads to sample.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for head sampling.")
    parser.add_argument("--num-traces", type=int, default=3, help="Number of traces to average.")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt for chat template.")
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        default=True,
        help="Always on: apply tokenizer chat template to question before appending trace text.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype for forward capture.",
    )
    return parser.parse_args()


def load_records(trace_root: Path, limit: int) -> List[Dict]:
    candidates: List[Path] = []
    if trace_root.is_file():
        candidates = [trace_root]
    else:
        merged = trace_root / "merged" / "merged.jsonl"
        if merged.exists():
            candidates = [merged]
        else:
            shards_dir = trace_root / "shards"
            if shards_dir.exists():
                candidates = sorted(shards_dir.glob("*.jsonl"))
            else:
                candidates = sorted(trace_root.glob("*.jsonl"))
    records: List[Dict] = []
    for path in candidates:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))
                if len(records) >= limit:
                    return records
    return records


def format_full_text(
    tokenizer: AutoTokenizer,
    question: str,
    response: str,
    system_prompt: str,
    use_chat_template: bool,
) -> str:
    user_prompt = (
        "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. "
        "First, you need to provide the chain-of-thought, then provide the final answer.\n\n "
        "Provide the final answer in the format: Final answer:  \\boxed{{}}"
    ).format(question=question)
    if use_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = user_prompt
    return prompt + response


def capture_qk_single(
    model,
    tokenizer: AutoTokenizer,
    text: str,
    precision: torch.dtype,
) -> LayerCaptureBuffer:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    text_config = model.config.get_text_config()
    num_layers = text_config.num_hidden_layers
    num_heads = text_config.num_attention_heads
    head_dim = (
        text_config.head_dim
        if hasattr(text_config, "head_dim")
        else text_config.hidden_size // text_config.num_attention_heads
    )
    buffer = LayerCaptureBuffer(num_layers, num_heads, input_ids.shape[1], head_dim, dtype=precision)
    collector = QKCollector(model, buffer)
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        collector.remove()
    return buffer


def aggregate_head_means(
    buffers: Sequence[LayerCaptureBuffer],
    sampled_heads: Sequence[Tuple[int, int]],
    model_path: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[Tuple[int, int], HeadFrequencyStats]:
    if not buffers:
        raise ValueError("No traces captured; cannot compute stats.")

    seq_len = max(buf.q.shape[2] for buf in buffers)
    head_dim = buffers[0].q.shape[-1]
    rotary = build_rotary(device, model_path, dtype)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))
    cos_table, sin_table, inv_freq, freq_scale = compute_rotary_tables(rotary, seq_len, head_dim, dtype, device)
    omega = inv_freq[: head_dim // 2]
    freq_scale_sq = freq_scale.pow(2)

    accum: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

    for buffer in buffers:
        local_seq_len = buffer.q.shape[2]
        cos_local = cos_table[:local_seq_len]
        sin_local = sin_table[:local_seq_len]

        for layer, head in sampled_heads:
            q_head = buffer.q[layer, head, :local_seq_len, :].to(device=device)
            q_unrot = invert_rope(q_head, cos_local, sin_local, attention_scale)
            q_complex = to_complex_pairs(q_unrot)
            q_mean_complex = q_complex.mean(dim=0).to(torch.complex64)
            q_abs_mean = torch.abs(q_complex).mean(dim=0).to(torch.float32)

            entry = accum.setdefault(
                (layer, head),
                {
                    "q_mean_sum": torch.zeros_like(q_mean_complex, device=device),
                    "q_abs_sum": torch.zeros_like(q_abs_mean, device=device),
                    "count": torch.tensor(0, device=device, dtype=torch.float32),
                },
            )
            entry["q_mean_sum"] += q_mean_complex
            entry["q_abs_sum"] += q_abs_mean
            entry["count"] += 1

    stats: Dict[Tuple[int, int], HeadFrequencyStats] = {}
    for head, payload in accum.items():
        count = max(1.0, float(payload["count"].item()))
        stats[head] = HeadFrequencyStats(
            q_mean_complex=payload["q_mean_sum"] / count,
            q_abs_mean=payload["q_abs_sum"] / count,
        )
    return stats


def main() -> None:
    mask_process_command("PD-L1_binder")
    args = parse_args()

    records = load_records(args.trace_root, args.num_traces)
    if not records:
        raise SystemExit(f"No trace records found under {args.trace_root}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    precision = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=precision,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        use_cache=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text_config = model.config.get_text_config()
    sampled_heads = load_or_create_sample(
        args.head_sample_file,
        args.sample_count,
        args.sample_seed,
        text_config.num_hidden_layers,
        text_config.num_attention_heads,
    )

    buffers: List[LayerCaptureBuffer] = []
    for idx, record in enumerate(records):
        question = record.get("question") or record.get("problem") or ""
        response = record.get("output") or record.get("model_output") or ""
        full_text = format_full_text(
            tokenizer,
            question,
            response,
            system_prompt=args.system_prompt,
            use_chat_template=args.use_chat_template,
        )
        buffer = capture_qk_single(model, tokenizer, full_text, precision)
        buffers.append(buffer)
        torch.cuda.empty_cache()

    stats_map = aggregate_head_means(
        buffers,
        sampled_heads,
        args.model_path,
        dtype=torch.float32,
        device=device,
    )

    metadata = {
        "model_path": str(args.model_path),
        "num_traces": len(buffers),
        "head_dim": buffers[0].q.shape[-1],
        "dtype": str(precision),
        "trace_root": str(args.trace_root),
        "use_chat_template": True,
    }
    save_head_frequency_stats(args.output_path, sampled_heads, stats_map, metadata)
    print(f"Saved sparse round stats to {args.output_path}")


if __name__ == "__main__":
    main()
