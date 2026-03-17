"""多 GPU 分发离线 trace 的 Q/K 捕获任务。"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RKV_ROOT = Path(__file__).resolve().parents[3]
if str(RKV_ROOT) not in sys.path:
    sys.path.insert(0, str(RKV_ROOT))

from weian_development.process_utils import mask_process_command

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as apply_qwen_rotary, repeat_kv as repeat_qwen_kv
except Exception:
    apply_qwen_rotary = None  # type: ignore[assignment]
    repeat_qwen_kv = None  # type: ignore[assignment]

try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as apply_qwen2_rotary, repeat_kv as repeat_qwen2_kv
except Exception:
    apply_qwen2_rotary = None  # type: ignore[assignment]
    repeat_qwen2_kv = None  # type: ignore[assignment]

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_llama_rotary, repeat_kv as repeat_llama_kv
except Exception:
    apply_llama_rotary = None  # type: ignore[assignment]
    repeat_llama_kv = None  # type: ignore[assignment]


DEFAULT_SYSTEM_PROMPT = "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"


@dataclass(frozen=True)
class TraceTask:
    qid: int
    source_path: Path
    trace_index: int
    question: str


@dataclass
class CaptureResult:
    qid: int
    trace_index: int
    output_pt: Path
    output_json: Path
    tokens: int
    prompt_tokens: int
    device: str


class LayerCaptureBuffer:
    """预先分配 CPU Tensor，用于逐层写入 Q/K 结果。"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.q = torch.empty((num_layers, num_heads, seq_len, head_dim), dtype=dtype)
        self.k = torch.empty((num_layers, num_heads, seq_len, head_dim), dtype=dtype)

    def store(self, layer_idx: int, q_tensor: torch.Tensor, k_tensor: torch.Tensor) -> None:
        # q_tensor/k_tensor shape: (batch, num_heads, seq_len, head_dim)
        q_cpu = q_tensor.squeeze(0).to(device="cpu", dtype=self.dtype)
        k_cpu = k_tensor.squeeze(0).to(device="cpu", dtype=self.dtype)
        self.q[layer_idx].copy_(q_cpu)
        self.k[layer_idx].copy_(k_cpu)
        del q_cpu, k_cpu


class QKCollector:
    """注册到模型上的前向 pre-hook，用来捕获 RoPE 之后的 Q/K。"""

    def __init__(self, model, buffer: LayerCaptureBuffer) -> None:
        self.model = model
        self.buffer = buffer
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        text_config = model.config.get_text_config()
        self.num_heads = text_config.num_attention_heads
        self.kv_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim if hasattr(text_config, "head_dim") else text_config.hidden_size // text_config.num_attention_heads
        self.apply_rotary, self.repeat_kv = self._select_rotary_impl(text_config.model_type)

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
                raise RuntimeError("无法在 pre-hook 中获取 hidden_states 或 position_embeddings")

            cos, sin = position_embeddings
            batch, seq_len, _ = hidden_states.shape
            if batch != 1:
                raise RuntimeError(f"预期 batch=1，但得到 {batch}")

            # 复制 Qwen3Attention.forward 中的投影/归一化逻辑
            q_proj = module.q_proj(hidden_states)
            q_proj = q_proj.view(batch, seq_len, self.num_heads, self.head_dim)
            if hasattr(module, "q_norm") and module.q_norm is not None:
                q_proj = module.q_norm(q_proj)
            q_states = q_proj.permute(0, 2, 1, 3).contiguous()

            k_proj = module.k_proj(hidden_states)
            k_proj = k_proj.view(batch, seq_len, self.kv_heads, self.head_dim)
            if hasattr(module, "k_norm") and module.k_norm is not None:
                k_proj = module.k_norm(k_proj)
            k_states = k_proj.permute(0, 2, 1, 3).contiguous()

            q_rot, k_rot = self.apply_rotary(q_states, k_states, cos, sin)
            k_rot = self.repeat_kv(k_rot, module.num_key_value_groups)

            self.buffer.store(layer_idx, q_rot.detach(), k_rot.detach())

            # 释放 GPU 临时变量
            del q_proj, q_states, k_proj, k_states, q_rot, k_rot

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _select_rotary_impl(self, model_type: str):
        """Choose apply_rotary_pos_emb/repeat_kv per model type to match runtime."""
        model_type = (model_type or "").lower()
        if "llama" in model_type and apply_llama_rotary is not None and repeat_llama_kv is not None:
            return apply_llama_rotary, repeat_llama_kv
        if "qwen3" in model_type and apply_qwen_rotary is not None and repeat_qwen_kv is not None:
            return apply_qwen_rotary, repeat_qwen_kv
        if "qwen" in model_type and apply_qwen2_rotary is not None and repeat_qwen2_kv is not None:
            return apply_qwen2_rotary, repeat_qwen2_kv
        raise SystemExit(f"未找到匹配模型类型 {model_type} 的 apply_rotary_pos_emb/repeat_kv 实现")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="捕获离线 reasoning trace 的 Q/K 张量")
    parser.add_argument("manifest", type=Path, help="trace manifest JSON 路径")
    parser.add_argument("output_dir", type=Path, help="输出目录（每条 trace 两个文件）")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
    )
    parser.add_argument("--gpus", default="0", help="逗号分隔的 GPU 编号列表")
    parser.add_argument("--precision", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-workers", type=int, default=None, help="限制并行 worker 数量（默认与 GPU 数相同）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> List[TraceTask]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks: List[TraceTask] = []
    for entry in payload.get("entries", []):
        tasks.append(
            TraceTask(
                qid=int(entry["qid"]),
                source_path=Path(entry["source_path"]),
                trace_index=int(entry["trace_index"]),
                question=entry.get("question", ""),
            )
        )
    return tasks


def split_tasks(tasks: List[TraceTask], gpus: List[str]) -> List[List[TraceTask]]:
    batches: List[List[TraceTask]] = [[] for _ in gpus]
    for idx, task in enumerate(tasks):
        batches[idx % len(gpus)].append(task)
    return batches


def load_trace_payload(task: TraceTask) -> Dict:
    data = json.loads(task.source_path.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    for candidate in traces:
        if int(candidate.get("index", -1)) == task.trace_index:
            return {
                "question": data.get("question", ""),
                "trace_text": candidate.get("text", ""),
                "source": data,
                "system_prompt": data.get("system_prompt"),
                "prompt_override": candidate.get("prompt") or data.get("prompt"),
            }
    raise RuntimeError(f"未找到 qid={task.qid} trace_index={task.trace_index} 对应的文本")


def prepare_input_text(
    tokenizer,
    question: str,
    trace_text: str,
    *,
    system_prompt: str | None = None,
    prompt_override: str | None = None,
) -> str:
    if prompt_override:
        return prompt_override + trace_text

    prompt_system = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + trace_text


def capture_single_trace(
    task: TraceTask,
    model,
    tokenizer,
    output_dir: Path,
    precision: torch.dtype,
    verbose: bool = False,
) -> CaptureResult:
    payload = load_trace_payload(task)
    full_text = prepare_input_text(
        tokenizer,
        payload["question"],
        payload["trace_text"],
        system_prompt=payload.get("system_prompt"),
        prompt_override=payload.get("prompt_override"),
    )

    tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)
    seq_len = input_ids.shape[1]

    text_config = model.config.get_text_config()
    num_layers = text_config.num_hidden_layers
    num_heads = text_config.num_attention_heads
    head_dim = text_config.head_dim if hasattr(text_config, "head_dim") else text_config.hidden_size // text_config.num_attention_heads

    buffer = LayerCaptureBuffer(num_layers, num_heads, seq_len, head_dim, dtype=precision)
    collector = QKCollector(model, buffer)

    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        collector.remove()

    tokens_total = int(attention_mask.sum().item())
    prompt_tokens = tokenizer(
        prepare_input_text(
            tokenizer,
            payload["question"],
            "",
            system_prompt=payload.get("system_prompt"),
            prompt_override=payload.get("prompt_override"),
        ),
        add_special_tokens=False,
        return_length=True,
    )["length"][0]

    trace_dir = output_dir / f"qid{task.qid:04d}_trace{task.trace_index:02d}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"

    torch.save({"q": buffer.q, "k": buffer.k}, tensor_path, _use_new_zipfile_serialization=False)

    meta_payload = {
        "qid": task.qid,
        "trace_index": task.trace_index,
        "source_json": str(task.source_path),
        "question": payload["question"],
        "token_count": tokens_total,
        "prompt_tokens": int(prompt_tokens),
        "sequence_length": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "precision": str(buffer.q.dtype),
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 清理内存
    del buffer.q, buffer.k, buffer
    torch.cuda.empty_cache()

    if verbose:
        print(f"完成 qid={task.qid} trace={task.trace_index} tokens={tokens_total}")

    return CaptureResult(
        qid=task.qid,
        trace_index=task.trace_index,
        output_pt=tensor_path,
        output_json=meta_path,
        tokens=tokens_total,
        prompt_tokens=int(prompt_tokens),
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
        raise SystemExit(f"无法解析 GPU 编号 {gpu_id}: {exc}")

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    load_dtype = dtype_map[args.precision]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
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
                    "qid": str(result.qid),
                    "trace": str(result.trace_index),
                    "tokens": str(result.tokens),
                    "pt": result.output_pt.name,
                }
            )
        except Exception as exc:  # pragma: no cover
            report_queue.put(
                {
                    "status": "error",
                    "gpu": gpu_id,
                    "qid": str(task.qid),
                    "trace": str(task.trace_index),
                    "message": str(exc),
                }
            )
            raise

    report_queue.put({"status": "done", "gpu": gpu_id})


def main() -> None:
    mask_process_command("PD-L1_binder_ctl")
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    tasks = load_manifest(args.manifest)
    if not tasks:
        raise SystemExit("manifest 中没有任务")

    gpus = [token.strip() for token in args.gpus.split(",") if token.strip()]
    if not gpus:
        raise SystemExit("需要至少指定一个 GPU")

    if args.max_workers is not None:
        gpus = gpus[: args.max_workers]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Dry run: 将在 GPU {gpus} 上处理 {len(tasks)} 条 trace")
        return

    batches = split_tasks(tasks, gpus)

    report_queue: mp.Queue[Dict[str, str]] = mp.Queue()
    workers: List[mp.Process] = []

    for worker_id, (gpu_id, batch) in enumerate(zip(gpus, batches)):
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
            report = report_queue.get(timeout=10)
        except queue.Empty:
            continue

        state = report.get("status")
        if state == "ok" and args.verbose:
            print(
                f"GPU {report['gpu']} -> qid {report['qid']} trace {report['trace']} tokens {report['tokens']}"
            )
        elif state == "error":
            print(
                f"[ERROR] GPU {report['gpu']} qid {report['qid']} trace {report['trace']}: {report['message']}",
                flush=True,
            )
        elif state == "done":
            finished += 1

    for proc in workers:
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(f"Worker 退出码 {proc.exitcode}")


if __name__ == "__main__":
    main()
