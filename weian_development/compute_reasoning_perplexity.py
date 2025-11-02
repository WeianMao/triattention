"""Compute per-token log probabilities for DeepConf reasoning traces."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from process_utils import mask_process_command


_DEEPSEEK_SYSTEM_PROMPT = (
    "该助手为DeepSeek-R1，由深度求索公司创造。\n"
    "今天是2025年5月28日，星期一。\n"
)


@dataclass
class PerplexityConfig:
    model_path: Path
    precision: str = "float16"
    chunk_size: int = 256
    model_type: str = "deepseek"
    reasoning_effort: str = "high"
    limit_traces: Optional[int] = None
    device_override: Optional[str] = None


@dataclass
class TraceMetrics:
    trace_index: int
    generated_token_ids: torch.Tensor
    log_probs: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.generated_token_ids.numel())

    @property
    def average_log_prob(self) -> float:
        if self.token_count == 0:
            return float("nan")
        return float(self.log_probs.mean().item())

    @property
    def perplexity(self) -> float:
        if self.token_count == 0:
            return float("nan")
        return float(math.exp(-self.average_log_prob))


class PerplexityEvaluator:
    def __init__(self, config: PerplexityConfig) -> None:
        self.config = config
        mask_process_command()

        self.device = torch.device(
            config.device_override
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.device.type == "cpu" and config.precision != "float32":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = dtype_map[config.precision]

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ prompt helpers --

    def _build_messages(self, question: str) -> List[Dict[str, str]]:
        if self.config.model_type == "deepseek":
            return [
                {"role": "system", "content": _DEEPSEEK_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        if self.config.model_type == "gpt":
            return [
                {"role": "user", "content": question},
            ]
        raise ValueError(f"Unsupported model_type: {self.config.model_type}")

    def _build_prompt(self, question: str) -> str:
        messages = self._build_messages(question)
        if self.config.model_type == "gpt":
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=self.config.reasoning_effort,
            )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ----------------------------------------------------------------- evaluation core --

    def evaluate_file(self, file_path: Path) -> Dict[str, object]:
        with file_path.open("r", encoding="utf-8") as f:
            record = json.load(f)

        question = record.get("question")
        if not question:
            raise ValueError(f"Missing question text in {file_path}")

        prompt_text = self._build_prompt(question)
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        prompt_token_ids = prompt_ids.squeeze(0).to(torch.int32).cpu()
        prompt_len = prompt_token_ids.numel()
        drop = max(prompt_len - 1, 0)

        traces = record.get("traces", [])
        if self.config.limit_traces is not None:
            traces = traces[: self.config.limit_traces]

        trace_metrics: List[TraceMetrics] = []
        per_trace_meta: List[Dict[str, object]] = []
        total_tokens = 0
        total_logprob = 0.0

        for idx, trace in enumerate(traces, start=1):
            trace_index = int(trace.get("index", idx))
            generated_text = trace.get("text", "")
            full_text = prompt_text + generated_text

            full_ids = self.tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(self.device)

            if full_ids.size(1) <= prompt_len:
                generated_ids_cpu = torch.empty(0, dtype=torch.int32)
                log_probs_tensor = torch.empty(0, dtype=torch.float32)
            else:
                if not torch.equal(full_ids[0, :prompt_len], prompt_ids[0]):
                    raise ValueError(
                        f"Prompt tokenization mismatch for {file_path} trace {trace_index}"
                    )
                log_probs_sequence = self._compute_log_probs(full_ids)
                generated_log_probs = log_probs_sequence[drop:]
                generated_ids = full_ids[0, prompt_len:]
                generated_ids_cpu = generated_ids.to(torch.int32).cpu()
                log_probs_tensor = generated_log_probs.cpu()

            metrics = TraceMetrics(
                trace_index=trace_index,
                generated_token_ids=generated_ids_cpu,
                log_probs=log_probs_tensor,
            )
            trace_metrics.append(metrics)

            total_tokens += metrics.token_count
            if metrics.token_count:
                total_logprob += float(metrics.log_probs.sum().item())

            per_trace_meta.append(
                {
                    "trace_index": trace_index,
                    "token_count": metrics.token_count,
                    "average_log_prob": metrics.average_log_prob,
                    "perplexity": metrics.perplexity,
                    "text_preview": generated_text[:160],
                }
            )

        weighted_avg_logprob = (
            total_logprob / total_tokens if total_tokens else float("nan")
        )
        overall_perplexity = (
            math.exp(-weighted_avg_logprob) if total_tokens else float("nan")
        )

        summary = {
            "source_file": record.get("source_file", file_path.name),
            "qid": record.get("qid"),
            "question": question,
            "prompt_token_count": int(prompt_len),
            "trace_count": len(trace_metrics),
            "token_count": total_tokens,
            "weighted_average_log_prob": weighted_avg_logprob,
            "overall_perplexity": overall_perplexity,
            "chunk_size": self.config.chunk_size,
            "precision": self.config.precision,
            "model_type": self.config.model_type,
            "reasoning_effort": self.config.reasoning_effort,
            "per_trace": per_trace_meta,
        }

        tensor_payload = {
            "prompt_token_ids": prompt_token_ids,
            "trace_data": [
                {
                    "trace_index": m.trace_index,
                    "generated_token_ids": m.generated_token_ids,
                    "log_probs": m.log_probs,
                }
                for m in trace_metrics
            ],
        }

        return {
            "summary": summary,
            "tensor_payload": tensor_payload,
        }

    # ----------------------------------------------------------------- model utility --

    def _compute_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        chunk = self.config.chunk_size
        if chunk <= 0:
            raise ValueError("chunk_size must be positive")

        seq_len = input_ids.size(1)
        past_key_values = None
        prev_tail: Optional[torch.Tensor] = None
        collected: List[torch.Tensor] = []
        position = 0

        with torch.no_grad():
            while position < seq_len:
                end = min(position + chunk, seq_len)
                input_chunk = input_ids[:, position:end]
                attention_chunk = torch.ones_like(input_chunk, device=self.device)

                outputs = self.model(
                    input_ids=input_chunk,
                    attention_mask=attention_chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
                log_probs_chunk = torch.log_softmax(logits, dim=-1)
                chunk_len = input_chunk.size(1)

                if position == 0:
                    start_local = 1
                else:
                    start_local = 0
                    if prev_tail is not None:
                        token_id = int(input_chunk[0, 0])
                        collected.append(prev_tail[token_id].unsqueeze(0))

                for offset in range(start_local, chunk_len):
                    if offset == 0:
                        continue
                    token_id = int(input_chunk[0, offset])
                    log_prob = log_probs_chunk[0, offset - 1, token_id]
                    collected.append(log_prob.unsqueeze(0))

                prev_tail = log_probs_chunk[0, chunk_len - 1]
                position = end

        if collected:
            return torch.cat(collected).to(torch.float32)
        return torch.empty(0, dtype=torch.float32)

    # ----------------------------------------------------------------- lifecycle --

    def shutdown(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- helpers --

def list_json_files(
    json_dir: Path,
    limit_files: Optional[int] = None,
    file_list: Optional[Path] = None,
) -> List[Path]:
    if file_list is not None:
        files: List[Path] = []
        with file_list.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                path = Path(name)
                if not path.is_absolute():
                    path = json_dir / path
                files.append(path)
    else:
        files = sorted(json_dir.glob("*.json"))

    files = [p for p in files if p.is_file() and p.name != "extraction_stats.json"]
    if limit_files is not None:
        files = files[:limit_files]
    return files


def save_perplexity_artifacts(record: Dict[str, object], output_root: Path) -> Path:
    tensors_dir = output_root / "tensors"
    meta_dir = output_root / "metadata"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    summary = record["summary"]  # type: ignore[index]
    tensor_payload = record["tensor_payload"]  # type: ignore[index]

    stem = Path(summary["source_file"]).stem  # type: ignore[index]
    tensor_path = tensors_dir / f"{stem}.pt"
    meta_path = meta_dir / f"{stem}.json"

    torch.save(tensor_payload, tensor_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return tensor_path


# ------------------------------------------------------------------------- CLI entry --

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute per-token log probabilities for reasoning traces")
    parser.add_argument("json_dir", type=Path, help="Directory containing reasoning JSON files")
    parser.add_argument("output_dir", type=Path, help="Directory to store tensor outputs")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
    )
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--model-type", choices=["deepseek", "gpt"], default="deepseek")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-traces", type=int, default=None)
    parser.add_argument("--file-list", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    files = list_json_files(args.json_dir, args.limit_files, args.file_list)
    if not files:
        raise SystemExit(f"No JSON files found in {args.json_dir}")

    config = PerplexityConfig(
        model_path=args.model_path,
        precision=args.precision,
        chunk_size=args.chunk_size,
        model_type=args.model_type,
        reasoning_effort=args.reasoning_effort,
        limit_traces=args.limit_traces,
        device_override=args.device,
    )

    evaluator = PerplexityEvaluator(config)
    try:
        for file_path in files:
            record = evaluator.evaluate_file(file_path)
            save_perplexity_artifacts(record, args.output_dir)
    finally:
        evaluator.shutdown()


if __name__ == "__main__":
    main()
