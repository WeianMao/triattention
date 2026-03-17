#!/usr/bin/env python3
"""
重放单条 AIME24 样本（默认 id=61），按 R-KV fullkv 配置捕获 prefill Q/K 张量。

- Prompt 构造/seed/温度/Top-p 与 `run_fullkv_aime24_official_sampled8.sh` 对齐。
- 只跑指定样本，单卡即可；不依赖已有 shard 结果。
- Q/K 在每层 forward pre-hook 中计算（复制 LlamaAttention 投影 + RoPE），保存原始张量。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

REPO_ROOT = Path(__file__).resolve().parents[2]
RKV_ROOT = REPO_ROOT / "R-KV"
for path in (REPO_ROOT, RKV_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from weian_development.speckv.prompt_utils import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    build_prompt,
    build_prompt_with_response,
    extract_question_from_record,
)
from weian_development.rkv_cache_utils import reset_model_cache  # noqa: E402
from weian_development.rkv_sharded_eval import str2bool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qk_capture_fullkv"))
    parser.add_argument("--sample-id", type=int, default=61, help="Dataset field `id` to replay.")
    parser.add_argument("--shard-id", type=int, default=0, help="输出目录中的 shard 编号（仅命名用）。")
    parser.add_argument("--run-id", type=int, default=0, help="输出目录中的 run 编号（仅命名用）。")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--load-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
    )
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--use-chat-template", type=str2bool, default=False)
    parser.add_argument("--chat-system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--reset-cache-each-batch", type=str2bool, default=False)
    parser.add_argument("--max-examples", type=int, default=None, help="可选：限制读取数据条数（默认读取全部）。")
    parser.add_argument(
        "--response-jsonl",
        type=Path,
        default=None,
        help="可选：R-KV shard 输出 jsonl，若提供则按 sample-id 匹配记录并将其 output 追加到 prompt 以捕获 reasoning trace 的 Q/K。",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def load_sample(dataset_path: Path, target_id: int, max_examples: int | None = None) -> Tuple[Dict, int]:
    with dataset_path.open() as fp:
        for idx, line in enumerate(fp):
            if max_examples is not None and idx >= max_examples:
                break
            record = json.loads(line)
            if int(record.get("id", -1)) == target_id:
                return record, idx
    raise ValueError(f"sample id {target_id} not found in {dataset_path}")


def load_response_from_jsonl(path: Path, target_id: int) -> str | None:
    if not path:
        return None
    with path.open() as fp:
        for line in fp:
            record = json.loads(line)
            if int(record.get("id", -1)) == target_id:
                return record.get("output") or record.get("reasoning") or record.get("answer")
    return None


class LayerCaptureBuffer:
    """预分配 CPU Tensor，逐层写入 Q/K。"""

    def __init__(self, num_layers: int, num_heads: int, seq_len: int, head_dim: int, dtype: torch.dtype):
        self.q = torch.empty((num_layers, num_heads, seq_len, head_dim), dtype=dtype)
        self.k = torch.empty((num_layers, num_heads, seq_len, head_dim), dtype=dtype)

    def store(self, layer_idx: int, q_tensor: torch.Tensor, k_tensor: torch.Tensor) -> None:
        # 期望形状 (1, heads, seq, dim)
        self.q[layer_idx].copy_(q_tensor.squeeze(0).to("cpu"))
        self.k[layer_idx].copy_(k_tensor.squeeze(0).to("cpu"))


class LlamaQKCollector:
    """注册 pre-hook，复制 LlamaAttention 的投影 + RoPE 计算，捕获 prefill Q/K。"""

    def __init__(self, model, buffer: LayerCaptureBuffer) -> None:
        self.model = model
        self.buffer = buffer
        self.handles = []

        text_config = model.config
        self.num_heads = text_config.num_attention_heads
        self.kv_heads = text_config.num_key_value_heads
        self.head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)

        for layer_idx, layer in enumerate(model.model.layers):
            handle = layer.self_attn.register_forward_pre_hook(self._make_hook(layer_idx), with_kwargs=True)
            self.handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
            position_embeddings = kwargs.get("position_embeddings") if "position_embeddings" in kwargs else args[1]
            cos, sin = position_embeddings
            batch, seq_len, _ = hidden_states.shape
            if batch != 1:
                raise RuntimeError(f"期望 batch=1，得到 {batch}")

            q_proj = module.q_proj(hidden_states)
            q_proj = q_proj.view(batch, seq_len, self.num_heads, self.head_dim)
            q_states = q_proj.permute(0, 2, 1, 3).contiguous()

            k_proj = module.k_proj(hidden_states)
            k_proj = k_proj.view(batch, seq_len, self.kv_heads, self.head_dim)
            k_states = k_proj.permute(0, 2, 1, 3).contiguous()

            q_rot, k_rot = apply_rotary_pos_emb(q_states, k_states, cos, sin)
            k_rot = repeat_kv(k_rot, module.num_key_value_groups)

            self.buffer.store(layer_idx, q_rot.detach(), k_rot.detach())

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    record, dataset_index = load_sample(args.dataset_path, args.sample_id, args.max_examples)
    question = extract_question_from_record(record, fallback_keys=["question", "problem"])
    response_text = load_response_from_jsonl(args.response_jsonl, args.sample_id)
    prompt = build_prompt_with_response(
        tokenizer,
        question,
        response=response_text or "",
        use_chat_template=args.use_chat_template,
        system_prompt=args.chat_system_prompt if args.use_chat_template else "",
    ) if response_text else build_prompt(
        tokenizer,
        question,
        use_chat_template=args.use_chat_template,
        system_prompt=args.chat_system_prompt if args.use_chat_template else "",
    )

    tokenized = tokenizer(
        [prompt],
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True,
    ).to("cuda")
    prefill_length = int(tokenized["attention_mask"].sum().item())

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=resolve_dtype(args.load_dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    if args.reset_cache_each_batch:
        reset_model_cache(model)

    text_config = model.config
    num_layers = text_config.num_hidden_layers
    num_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
    seq_len = tokenized["input_ids"].shape[1]
    buffer = LayerCaptureBuffer(num_layers, num_heads, seq_len, head_dim, dtype=torch.float32)
    collector = LlamaQKCollector(model, buffer)

    try:
        with torch.no_grad():
            _ = model(**tokenized, use_cache=True)
    finally:
        collector.remove()

    out_dir = args.output_dir / f"shard{args.shard_id:02d}" / f"run{args.run_id:03d}_sample{args.sample_id:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"q": buffer.q, "k": buffer.k}, out_dir / "qk.pt", _use_new_zipfile_serialization=False)
    meta = {
        "sample_id": args.sample_id,
        "dataset_index": dataset_index,
        "prefill_length": prefill_length,
        "sequence_length": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "attn_implementation": args.attn_implementation,
        "load_dtype": args.load_dtype,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "prompt": prompt,
        "response_jsonl": str(args.response_jsonl) if args.response_jsonl else None,
        "response_included": bool(response_text),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    del buffer.q, buffer.k, buffer
    torch.cuda.empty_cache()

    print(f"[qk_capture] 完成 sample id {args.sample_id} (dataset idx {dataset_index}), 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
