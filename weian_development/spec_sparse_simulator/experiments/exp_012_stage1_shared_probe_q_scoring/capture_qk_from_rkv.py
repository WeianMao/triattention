"""Capture Q/K tensors from R-KV inference outputs (multi-GPU distributed).

Adapted from weian_development/attention_qk_analysis/capture_qk_distributed.py
for use with R-KV project outputs and DeepSeek-R1-Distill-Qwen-7B model.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# Add R-KV evaluation path for answer checking
RKV_EVAL_PATH = ROOT / "R-KV" / "HuggingFace" / "evaluation"
if str(RKV_EVAL_PATH) not in sys.path:
    sys.path.insert(0, str(RKV_EVAL_PATH))

from parser import extract_answer, strip_string
from grader import math_equal

# Qwen2 model utilities (for RoPE and GQA expansion)
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


DEFAULT_MODEL_PATH = Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B")


@dataclass(frozen=True)
class TraceTask:
    sample_idx: int
    draw_idx: int
    question: str
    prompt: str
    output: str
    answer: str
    prefill_tokens: int  # From R-KV inference, exact prompt token count


@dataclass
class CaptureResult:
    sample_idx: int
    draw_idx: int
    output_pt: Path
    output_json: Path
    tokens: int
    prompt_tokens: int
    device: str


class LayerCaptureBuffer:
    """Pre-allocate CPU tensors for storing Q/K per layer."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.q = torch.empty((num_layers, num_heads, seq_len, head_dim), dtype=dtype)
        self.k = torch.empty((num_layers, num_heads, seq_len, head_dim), dtype=dtype)

    def store(self, layer_idx: int, q_tensor: torch.Tensor, k_tensor: torch.Tensor) -> None:
        q_cpu = q_tensor.squeeze(0).to(device="cpu", dtype=self.dtype)
        k_cpu = k_tensor.squeeze(0).to(device="cpu", dtype=self.dtype)
        self.q[layer_idx].copy_(q_cpu)
        self.k[layer_idx].copy_(k_cpu)
        del q_cpu, k_cpu


class QKCollector:
    """Forward pre-hook to capture Q/K after RoPE application."""

    def __init__(self, model, buffer: LayerCaptureBuffer) -> None:
        self.model = model
        self.buffer = buffer
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        text_config = model.config.get_text_config()
        self.num_heads = text_config.num_attention_heads
        self.kv_heads = text_config.num_key_value_heads
        self.head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)

        for layer_idx, layer in enumerate(model.model.layers):
            handle = layer.self_attn.register_forward_pre_hook(
                self._make_hook(layer_idx),
                with_kwargs=True,
            )
            self.handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                position_embeddings = args[1]
            if hidden_states is None or position_embeddings is None:
                raise RuntimeError("Cannot get hidden_states or position_embeddings in pre-hook")

            cos, sin = position_embeddings
            batch, seq_len, _ = hidden_states.shape
            if batch != 1:
                raise RuntimeError(f"Expected batch=1, got {batch}")

            # Q projection and reshape
            q_proj = module.q_proj(hidden_states)
            q_proj = q_proj.view(batch, seq_len, self.num_heads, self.head_dim)
            # Qwen2 doesn't have q_norm, but keep check for compatibility
            if hasattr(module, "q_norm") and module.q_norm is not None:
                q_proj = module.q_norm(q_proj)
            q_states = q_proj.permute(0, 2, 1, 3).contiguous()

            # K projection and reshape
            k_proj = module.k_proj(hidden_states)
            k_proj = k_proj.view(batch, seq_len, self.kv_heads, self.head_dim)
            if hasattr(module, "k_norm") and module.k_norm is not None:
                k_proj = module.k_norm(k_proj)
            k_states = k_proj.permute(0, 2, 1, 3).contiguous()

            # Apply RoPE
            q_rot, k_rot = apply_rotary_pos_emb(q_states, k_states, cos, sin)

            # Expand K for GQA (4 KV heads -> 28 attention heads)
            k_rot = repeat_kv(k_rot, module.num_key_value_groups)

            self.buffer.store(layer_idx, q_rot.detach(), k_rot.detach())

            del q_proj, q_states, k_proj, k_states, q_rot, k_rot

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Q/K tensors from R-KV inference outputs")
    parser.add_argument("jsonl_path", type=Path, help="R-KV merged JSONL file path")
    parser.add_argument("output_dir", type=Path, help="Output directory for QK tensors")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU IDs or 'auto' to detect")
    parser.add_argument("--reserve-gpus", type=int, default=1, help="Number of GPUs to reserve (not use)")
    parser.add_argument("--precision", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-traces", type=int, default=16, help="Maximum traces to capture")
    parser.add_argument("--dataset-name", default="aime24", help="Dataset name for answer extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trace selection")
    parser.add_argument("--max-workers", type=int, default=None, help="Limit parallel workers")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def detect_available_gpus(reserve: int = 1) -> List[str]:
    """Detect available GPUs and reserve some for other users."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        gpu_info = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(",")
            gpu_id = parts[0].strip()
            mem_used = int(parts[1].strip())
            gpu_info.append((gpu_id, mem_used))

        # Sort by memory usage (prefer less used GPUs)
        gpu_info.sort(key=lambda x: x[1])

        # Select all but reserved
        available = [g[0] for g in gpu_info]
        if len(available) > reserve:
            available = available[:-reserve]

        print(f"Detected {len(gpu_info)} GPUs, using {len(available)}: {available}")
        return available
    except Exception as e:
        print(f"GPU detection failed: {e}, falling back to GPU 0")
        return ["0"]


def load_rkv_outputs_with_diversity(
    jsonl_path: Path,
    max_traces: int = 16,
    dataset_name: str = "aime24",
    seed: int = 42,
) -> List[TraceTask]:
    """Load R-KV output JSONL, filter correct answers, select with max question diversity."""
    random.seed(seed)

    # 1. Load all records and filter correct answers
    correct_by_question = defaultdict(list)
    total_records = 0
    correct_records = 0

    with jsonl_path.open() as f:
        for line in f:
            total_records += 1
            record = json.loads(line)
            pred = extract_answer(record["output"], dataset_name)
            gt = strip_string(record["answer"])
            if math_equal(pred, gt):
                correct_records += 1
                sample_idx = record.get("sample_idx", record.get("index", 0))
                correct_by_question[sample_idx].append(record)

    print(f"Loaded {total_records} records, {correct_records} correct ({100*correct_records/total_records:.1f}%)")
    print(f"Correct answers from {len(correct_by_question)} different questions")

    # 2. Shuffle questions for randomness
    question_ids = list(correct_by_question.keys())
    random.shuffle(question_ids)

    # 3. Round-robin selection: 1 trace per question until we have max_traces
    selected = []
    question_idx = 0
    while len(selected) < max_traces and question_ids:
        qid = question_ids[question_idx % len(question_ids)]
        traces = correct_by_question[qid]
        if traces:
            trace = random.choice(traces)
            selected.append(trace)
            traces.remove(trace)
            if not traces:
                correct_by_question.pop(qid)
                question_ids.remove(qid)
        question_idx += 1
        if question_idx >= len(question_ids) and question_ids:
            question_idx = 0

    # Convert to TraceTask objects
    tasks = []
    for record in selected:
        tasks.append(TraceTask(
            sample_idx=record.get("sample_idx", record.get("index", 0)),
            draw_idx=record.get("draw_idx", 0),
            question=record.get("question", ""),
            prompt=record["prompt"],
            output=record["output"],
            answer=record["answer"],
            prefill_tokens=record["prefill_tokens"],
        ))

    unique_questions = len(set(t.sample_idx for t in tasks))
    print(f"Selected {len(tasks)} traces from {unique_questions} questions")
    return tasks


def split_tasks(tasks: List[TraceTask], gpus: List[str]) -> List[List[TraceTask]]:
    batches: List[List[TraceTask]] = [[] for _ in gpus]
    for idx, task in enumerate(tasks):
        batches[idx % len(gpus)].append(task)
    return batches


def prepare_input_text(task: TraceTask) -> str:
    """Construct full text for QK capture: prompt + output."""
    return task.prompt + task.output


def capture_single_trace(
    task: TraceTask,
    model,
    tokenizer,
    output_dir: Path,
    precision: torch.dtype,
    verbose: bool = False,
) -> CaptureResult:
    full_text = prepare_input_text(task)

    tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)
    seq_len = input_ids.shape[1]

    text_config = model.config.get_text_config()
    num_layers = text_config.num_hidden_layers
    num_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)

    buffer = LayerCaptureBuffer(num_layers, num_heads, seq_len, head_dim, dtype=precision)
    collector = QKCollector(model, buffer)

    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        collector.remove()

    tokens_total = int(attention_mask.sum().item())

    trace_dir = output_dir / f"sample{task.sample_idx:04d}_draw{task.draw_idx:02d}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"

    # Use new zipfile serialization for mmap support when loading
    torch.save({"q": buffer.q, "k": buffer.k}, tensor_path)

    meta_payload = {
        "sample_idx": task.sample_idx,
        "draw_idx": task.draw_idx,
        "question": task.question,
        "answer": task.answer,
        "token_count": tokens_total,
        "prompt_tokens": task.prefill_tokens,
        "sequence_length": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "precision": str(buffer.q.dtype),
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    del buffer.q, buffer.k, buffer
    torch.cuda.empty_cache()

    if verbose:
        print(f"Done sample={task.sample_idx} draw={task.draw_idx} tokens={tokens_total}")

    return CaptureResult(
        sample_idx=task.sample_idx,
        draw_idx=task.draw_idx,
        output_pt=tensor_path,
        output_json=meta_path,
        tokens=tokens_total,
        prompt_tokens=task.prefill_tokens,
        device=str(model.device),
    )


def worker_main(
    worker_id: int,
    gpu_id: str,
    tasks: List[TraceTask],
    args: argparse.Namespace,
    report_queue: "mp.Queue[Dict[str, str]]",
) -> None:
    alias = f"PD-L1_binder_{gpu_id}"[:15]
    mask_process_command(alias)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        device_index = int(gpu_id)
    except ValueError as exc:
        raise SystemExit(f"Cannot parse GPU ID {gpu_id}: {exc}")

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    load_dtype = dtype_map[args.precision]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=load_dtype,
        trust_remote_code=True,
        device_map=None,
    )
    model.to(device)
    model.eval()

    for task in tasks:
        try:
            result = capture_single_trace(
                task,
                model,
                tokenizer,
                args.output_dir,
                precision=load_dtype,
                verbose=args.verbose,
            )
            report_queue.put(
                {
                    "status": "ok",
                    "gpu": gpu_id,
                    "sample_idx": str(result.sample_idx),
                    "draw_idx": str(result.draw_idx),
                    "tokens": str(result.tokens),
                    "pt": result.output_pt.name,
                }
            )
        except Exception as exc:
            report_queue.put(
                {
                    "status": "error",
                    "gpu": gpu_id,
                    "sample_idx": str(task.sample_idx),
                    "draw_idx": str(task.draw_idx),
                    "message": str(exc),
                }
            )
            raise

    report_queue.put({"status": "done", "gpu": gpu_id})


def main() -> None:
    mask_process_command("PD-L1_binder_ctl")
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    # Load and filter traces
    tasks = load_rkv_outputs_with_diversity(
        args.jsonl_path,
        max_traces=args.max_traces,
        dataset_name=args.dataset_name,
        seed=args.seed,
    )
    if not tasks:
        raise SystemExit("No valid traces found after filtering")

    # Determine GPUs
    if args.gpus == "auto":
        gpus = detect_available_gpus(reserve=args.reserve_gpus)
    else:
        gpus = [token.strip() for token in args.gpus.split(",") if token.strip()]
    if not gpus:
        raise SystemExit("No GPUs available")

    if args.max_workers is not None:
        gpus = gpus[: args.max_workers]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Dry run: Would process {len(tasks)} traces on GPU {gpus}")
        for task in tasks:
            print(f"  sample={task.sample_idx} draw={task.draw_idx} question_len={len(task.question)}")
        return

    batches = split_tasks(tasks, gpus)

    report_queue: mp.Queue[Dict[str, str]] = mp.Queue()
    workers: List[mp.Process] = []

    for worker_id, (gpu_id, batch) in enumerate(zip(gpus, batches)):
        if not batch:
            continue
        proc = mp.Process(
            target=worker_main,
            args=(worker_id, gpu_id, batch, args, report_queue),
            daemon=False,
        )
        proc.start()
        workers.append(proc)

    finished = 0
    total_workers = len(workers)

    while finished < total_workers:
        try:
            report = report_queue.get(timeout=30)
        except queue.Empty:
            continue

        state = report.get("status")
        if state == "ok":
            print(
                f"GPU {report['gpu']} -> sample {report['sample_idx']} draw {report['draw_idx']} tokens {report['tokens']}"
            )
        elif state == "error":
            print(
                f"[ERROR] GPU {report['gpu']} sample {report['sample_idx']} draw {report['draw_idx']}: {report['message']}",
                flush=True,
            )
        elif state == "done":
            finished += 1

    for proc in workers:
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(f"Worker exited with code {proc.exitcode}")

    print(f"Done! Captured {len(tasks)} traces to {args.output_dir}")


if __name__ == "__main__":
    main()
