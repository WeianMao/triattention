from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RKV_ROOT = PROJECT_ROOT / "R-KV"
HF_RKV_ROOT = RKV_ROOT / "HuggingFace"
for path in (RKV_ROOT, HF_RKV_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if TYPE_CHECKING:
    from weian_development.speckv.speckv_rkv_style import SpeckVRKVStyle

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ProbeResult:
    mode: str
    model_path: str
    stats_path: str
    dataset_path: str
    kv_budget: int
    prompt_tokens: int
    compressed_prompt_tokens: int
    output_tokens: int
    total_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    max_same_ws_run: int
    max_same_char_run: int
    max_same_char: str
    eos_reached: bool
    output: str
    prompt_preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HF-side isolated prefill compression probe for long-context SpeckV diagnosis."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/data/rbg/users/weian/env/huggingface/hub/models--JunHowie--Qwen3-32B-GPTQ-Int4/"
            "snapshots/275d13ed8617787bde259624a8ab2f5527266465"
        ),
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=PROJECT_ROOT / "demo/openclaw-demo/stats/qwen3_32b_int4_speckv_stats.pt",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=PROJECT_ROOT / "demo/openclaw-demo/fixtures/openclaw_like_dataset.jsonl",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/hf_prefill_probe"),
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "compress_once", "compress_iterative", "both"),
        default="both",
    )
    parser.add_argument("--kv-budget", type=int, default=7000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compression-seed",
        type=int,
        default=None,
        help="Optional seed for HF-side compression tie-break noise. Default disables compressor noise.",
    )
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--load-dtype", type=str, default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt-key", type=str, default="question")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional raw prompt file. When provided, bypass dataset/template building.",
    )
    parser.add_argument(
        "--dump-keep-indices-out",
        type=Path,
        default=None,
        help="Optional JSON path for the first compression keep indices dump.",
    )
    parser.add_argument(
        "--dump-k-sample-out",
        type=Path,
        default=None,
        help="Optional JSON path for sampled K vectors before manual compression.",
    )
    parser.add_argument(
        "--dump-compression-round-dir",
        type=Path,
        default=None,
        help="Optional directory to dump every compression round's keep indices.",
    )
    parser.add_argument("--score-aggregation", type=str, default="mean")
    parser.add_argument("--normalize-scores", action="store_true")
    parser.add_argument("--use-rank-aggregation", action="store_true")
    parser.add_argument("--per-head-pruning", action="store_true")
    parser.add_argument("--per-layer-perhead-pruning", action="store_true")
    parser.add_argument("--per-layer-pruning", action="store_true")
    parser.add_argument("--layer-perhead-aggregation", type=str, default="max")
    parser.add_argument("--per-layer-aggregation", type=str, default="max")
    parser.add_argument("--disable-top-n-high-freq", type=int, default=0)
    parser.add_argument("--disable-mlr", action="store_true")
    parser.add_argument("--disable-trig", action="store_true")
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Protect the most recent N tokens during compression, aligned with vLLM runtime semantics.",
    )
    parser.add_argument(
        "--divide-length",
        type=int,
        default=128,
        help="Overflow chunk length used by iterative compression trigger.",
    )
    parser.add_argument(
        "--compare-vllm-step-semantics",
        action="store_true",
        help=(
            "Align iterative decode compression timing with vLLM runtime semantics: "
            "compress on the pre-step cache when current_len + scheduled_tokens reaches "
            "budget + divide_length."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def patch_qwen3_logits_to_keep() -> None:
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    except Exception:
        return

    orig_forward = Qwen3ForCausalLM.forward
    if getattr(orig_forward, "_hf_probe_logits_to_keep_patch", False):
        return

    def patched_forward(self, *args, **kwargs):
        kwargs.setdefault("logits_to_keep", 1)
        return orig_forward(self, *args, **kwargs)

    patched_forward._hf_probe_logits_to_keep_patch = True  # type: ignore[attr-defined]
    Qwen3ForCausalLM.forward = patched_forward


def resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[normalized]


def load_record(dataset_path: Path, sample_index: int) -> Dict[str, Any]:
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    if sample_index < 0 or sample_index >= len(lines):
        raise IndexError(f"sample-index {sample_index} out of range for {dataset_path}")
    return json.loads(lines[sample_index])


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    patch_qwen3_logits_to_keep()

    print(f"[hf-probe] loading tokenizer from {args.model_path}", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"[hf-probe] loading model attn={args.attn_implementation} dtype={args.load_dtype}",
        file=sys.stderr,
        flush=True,
    )
    model_path_lower = str(args.model_path).lower()
    native_gptq_available = False
    if "gptq" in model_path_lower:
        try:
            import gptqmodel  # noqa: F401

            native_gptq_available = True
        except Exception:
            native_gptq_available = False

    if "gptq" not in model_path_lower or native_gptq_available:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_path),
            torch_dtype=resolve_torch_dtype(args.load_dtype),
            low_cpu_mem_usage=True,
            device_map=args.device,
            use_cache=True,
            attn_implementation=args.attn_implementation,
            trust_remote_code=bool(args.trust_remote_code),
        )
    else:
        from auto_gptq import AutoGPTQForCausalLM

        device = "cuda:0" if str(args.device).startswith("cuda") else str(args.device)
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            str(args.model_path),
            device=device,
            use_triton=True,
            trust_remote_code=bool(args.trust_remote_code),
        )
        model = model_wrapper.model
    model.eval()
    print("[hf-probe] model ready", file=sys.stderr, flush=True)
    return model, tokenizer


def normalize_past_key_values(past_key_values: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    from transformers.cache_utils import Cache

    if isinstance(past_key_values, Cache):
        return tuple(past_key_values.to_legacy_cache())
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def current_cache_len(past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> int:
    if not past_key_values:
        return 0
    return int(past_key_values[0][0].shape[2])


def build_probe_prompt(
    tokenizer,
    record: Dict[str, Any],
    *,
    use_chat_template: bool,
    system_prompt: str,
    prompt_key: str,
) -> str:
    from weian_development.speckv.prompt_utils import build_prompt, extract_question_from_record

    question = extract_question_from_record(record, fallback_keys=[prompt_key] if prompt_key else None)
    return build_prompt(
        tokenizer,
        question,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove = cumulative_probs > top_p
    if sorted_remove.numel() > 0:
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
    return logits.masked_fill(remove_mask, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    logits = logits.float()
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        topk_vals, _ = torch.topk(logits, k=top_k, dim=-1)
        cutoff = topk_vals[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    logits = top_p_filtering(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def create_compressor(args: argparse.Namespace, device: torch.device) -> "SpeckVRKVStyle":
    try:
        from weian_development.speckv.speckv_rkv_style import SpeckVRKVStyle, SpeckVRKVStyleConfig
    except ModuleNotFoundError:
        speckv_dir = RKV_ROOT / "weian_development/speckv"
        _load_module_from_path(
            "weian_development.speckv.round_pruning_utils",
            speckv_dir / "round_pruning_utils.py",
        )
        _load_module_from_path(
            "weian_development.speckv.stats_utils",
            speckv_dir / "stats_utils.py",
        )
        module = _load_module_from_path(
            "weian_development.speckv.speckv_rkv_style",
            speckv_dir / "speckv_rkv_style.py",
        )
        SpeckVRKVStyle = module.SpeckVRKVStyle
        SpeckVRKVStyleConfig = module.SpeckVRKVStyleConfig

    config = SpeckVRKVStyleConfig(
        stats_path=args.stats_path,
        model_path=args.model_path,
        device=device,
        dtype=torch.float32,
        budget=int(args.kv_budget),
        offset_max_length=65536,
        score_aggregation=args.score_aggregation,
        seed=args.compression_seed,
        head_limit=None,
        metadata_expectations=None,
        normalize_scores=bool(args.normalize_scores),
        use_rank_aggregation=bool(args.use_rank_aggregation),
        include_prefill_in_budget=True,
        allow_prefill_compression=True,
        divide_length=int(args.divide_length),
        use_slack_trigger=False,
        per_head_pruning=bool(args.per_head_pruning),
        per_layer_perhead_pruning=bool(args.per_layer_perhead_pruning),
        layer_perhead_aggregation=args.layer_perhead_aggregation,
        per_layer_pruning=bool(args.per_layer_pruning),
        per_layer_aggregation=args.per_layer_aggregation,
        disable_top_n_high_freq=args.disable_top_n_high_freq,
        disable_mlr=bool(args.disable_mlr),
        disable_trig=bool(args.disable_trig),
    )
    comp = SpeckVRKVStyle(config)
    setattr(comp.config, "window_size", int(args.window_size))
    return comp


def _compute_keep_indices_with_window(
    comp: "SpeckVRKVStyle",
    pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    *,
    recent_unabsorbed_tokens: int | None,
) -> torch.Tensor:
    """Debug-only keep-index path that mirrors vLLM recent-window protection.

    This keeps the HF-side scoring/selection semantics intact while adding the
    runtime-only recent-token guard used by the vLLM path. It is intentionally
    scoped to the probe script so we can compare like-for-like without touching
    production code.
    """
    if not pkv_tuple:
        return torch.arange(0, device=comp.config.device)

    kv_cache_len = pkv_tuple[0][0].shape[-2]
    if kv_cache_len <= comp.budget:
        return torch.arange(kv_cache_len, device=comp.config.device)

    prefix_length = 0 if comp.allow_prefill_compression else comp.prefix_length
    decode_start = min(prefix_length, kv_cache_len)
    decode_count = max(0, kv_cache_len - decode_start)
    if decode_count == 0:
        return torch.arange(min(comp.budget, kv_cache_len), device=comp.config.device)

    decode_budget = max(0, comp.budget - decode_start)
    if decode_budget == 0:
        return torch.arange(min(comp.budget, decode_start), device=comp.config.device)

    decode_positions = torch.tensor(
        comp.cache_positions[decode_start:kv_cache_len],
        device=comp.config.device,
        dtype=torch.long,
    )
    positions_per_kv_head: List[torch.Tensor] | None = None
    if comp.cache_positions_per_head is not None:
        positions_per_kv_head = [
            torch.tensor(
                head_positions[decode_start:kv_cache_len],
                device=comp.config.device,
                dtype=torch.long,
            )
            for head_positions in comp.cache_positions_per_head
        ]

    all_head_scores: List[torch.Tensor] = []
    for layer_idx, (key_states, _) in enumerate(pkv_tuple):
        layer_scores = comp._compute_layer_head_scores(
            key_states,
            decode_positions,
            layer_idx,
            start_index=decode_start,
            positions_per_kv_head=positions_per_kv_head,
        )
        if layer_scores is not None:
            all_head_scores.append(layer_scores)

    if not all_head_scores:
        prefill_indices = torch.arange(decode_start, device=comp.config.device)
        decode_indices = torch.arange(
            decode_start,
            min(decode_start + decode_budget, kv_cache_len),
            device=comp.config.device,
        )
        return torch.cat([prefill_indices, decode_indices])

    head_matrix = torch.cat(all_head_scores, dim=0)
    if comp.use_rank_aggregation:
        ranks = torch.argsort(torch.argsort(head_matrix, dim=1, descending=True), dim=1)
        head_matrix = ranks.float()
    elif comp.normalize_scores and head_matrix.numel() > 0:
        mean = head_matrix.mean(dim=1, keepdim=True)
        std = head_matrix.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        head_matrix = (head_matrix - mean) / std

    if comp.generator is not None and head_matrix.numel() > 0:
        noise = torch.rand(
            head_matrix.shape,
            device=head_matrix.device,
            generator=comp.generator,
        ) * 1e-6
        head_matrix = head_matrix + noise

    if comp.config.window_size > 0 and head_matrix.numel() > 0:
        if recent_unabsorbed_tokens is None:
            recent_count = min(int(comp.config.window_size), int(head_matrix.shape[-1]))
        else:
            recent_count = min(
                int(comp.config.window_size),
                max(0, min(int(head_matrix.shape[-1]), int(recent_unabsorbed_tokens))),
            )
        if recent_count > 0:
            head_matrix = head_matrix.clone()
            head_matrix[:, -recent_count:] = float("inf")

    keep_count = min(decode_budget, decode_count)
    if comp.per_head_pruning:
        return comp._select_per_head_independent(head_matrix, keep_count, decode_start)

    if comp.use_rank_aggregation:
        combined = -head_matrix.min(dim=0).values
    else:
        combined = head_matrix.max(dim=0).values
    decode_keep_indices = comp._select_union_based(head_matrix, combined, keep_count)
    prefill_indices = torch.arange(decode_start, device=comp.config.device)
    decode_keep_absolute = decode_keep_indices + decode_start
    keep_indices = torch.cat([prefill_indices, decode_keep_absolute])
    return torch.sort(keep_indices).values


def apply_manual_compression(
    comp: "SpeckVRKVStyle",
    pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    *,
    dump_keep_indices_out: Path | None = None,
    dump_k_sample_out: Path | None = None,
    recent_unabsorbed_tokens: int | None = None,
) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], int]:
    seq_len = current_cache_len(pkv_tuple)
    comp.cache_positions = list(range(seq_len))
    comp.absolute_position = seq_len
    comp.prefix_length = seq_len

    if dump_k_sample_out is not None and pkv_tuple:
        dump_k_sample_out.parent.mkdir(parents=True, exist_ok=True)
        sample_positions = sorted(
            set(
                [0, 1, 2, 3, 4]
                + [max(0, seq_len // 4 - 1), seq_len // 4, min(seq_len - 1, seq_len // 4 + 1)]
                + [max(0, seq_len // 2 - 1), seq_len // 2, min(seq_len - 1, seq_len // 2 + 1)]
                + [max(0, (3 * seq_len) // 4 - 1), (3 * seq_len) // 4, min(seq_len - 1, (3 * seq_len) // 4 + 1)]
                + [max(0, seq_len - 5), max(0, seq_len - 4), max(0, seq_len - 3), max(0, seq_len - 2), seq_len - 1]
            )
        )
        head_limit = min(2, int(pkv_tuple[0][0].shape[1]))
        preferred_layers = [idx for idx in (0, 1, 16, 32, 48, 63) if idx < len(pkv_tuple)]
        if not preferred_layers:
            preferred_layers = list(range(min(2, len(pkv_tuple))))
        payload = {
            "seq_len": int(seq_len),
            "sample_positions": [int(x) for x in sample_positions],
            "layers": {},
        }
        for layer_idx in preferred_layers:
            k, _ = pkv_tuple[layer_idx]
            layer_payload = {}
            for head_idx in range(head_limit):
                vectors = []
                for pos in sample_positions:
                    vectors.append(
                        k[0, head_idx, pos]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .tolist()
                    )
                layer_payload[str(head_idx)] = vectors
            payload["layers"][str(layer_idx)] = layer_payload
        dump_k_sample_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    keep_indices = _compute_keep_indices_with_window(
        comp,
        pkv_tuple,
        recent_unabsorbed_tokens=recent_unabsorbed_tokens,
    )
    if dump_keep_indices_out is not None:
        dump_keep_indices_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seq_len": int(seq_len),
            "budget": int(comp.budget),
            "prefix_length": int(comp.prefix_length),
            "window_size": int(comp.config.window_size),
            "recent_unabsorbed_tokens": (
                int(recent_unabsorbed_tokens)
                if recent_unabsorbed_tokens is not None
                else None
            ),
            "shape": list(keep_indices.shape),
            "indices": keep_indices.detach().to(device="cpu", dtype=torch.long).tolist(),
        }
        dump_keep_indices_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    reclaimed = seq_len - (keep_indices.size(-1) if keep_indices.dim() > 1 else keep_indices.numel())

    if keep_indices.dim() == 3:
        num_layers = keep_indices.size(0)
        num_kv_heads = keep_indices.size(1)
        budget = keep_indices.size(2)
        new_pkv = []
        for layer_idx, (k, v) in enumerate(pkv_tuple):
            batch_size = k.size(0)
            head_dim = k.size(3)
            layer_indices = keep_indices[layer_idx]
            expanded_indices = layer_indices.unsqueeze(0).unsqueeze(-1).expand(
                batch_size, num_kv_heads, budget, head_dim
            )
            k_new = k.gather(dim=2, index=expanded_indices)
            v_new = v.gather(dim=2, index=expanded_indices)
            new_pkv.append((k_new.contiguous(), v_new.contiguous()))
        pkv_tuple = tuple(new_pkv)
        comp.cache_positions_per_layer_perhead = {
            (layer_idx, kv_head): [comp.cache_positions[idx] for idx in keep_indices[layer_idx, kv_head].tolist()]
            for layer_idx in range(num_layers)
            for kv_head in range(num_kv_heads)
        }
        comp.cache_positions = comp.cache_positions_per_layer_perhead[(0, 0)].copy()
    elif keep_indices.dim() == 2 and comp.per_layer_pruning:
        num_layers = keep_indices.size(0)
        budget = keep_indices.size(1)
        new_pkv = []
        for layer_idx, (k, v) in enumerate(pkv_tuple):
            batch_size = k.size(0)
            num_kv_heads = k.size(1)
            head_dim = k.size(3)
            layer_indices = keep_indices[layer_idx]
            expanded_indices = layer_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
                batch_size, num_kv_heads, budget, head_dim
            )
            k_new = k.gather(dim=2, index=expanded_indices)
            v_new = v.gather(dim=2, index=expanded_indices)
            new_pkv.append((k_new.contiguous(), v_new.contiguous()))
        pkv_tuple = tuple(new_pkv)
        comp.cache_positions_per_layer = {
            layer_idx: [comp.cache_positions[idx] for idx in keep_indices[layer_idx].tolist()]
            for layer_idx in range(num_layers)
        }
        comp.cache_positions = comp.cache_positions_per_layer[0].copy()
    elif keep_indices.dim() == 2:
        new_pkv = []
        for k, v in pkv_tuple:
            batch_size = k.size(0)
            num_kv_heads = k.size(1)
            budget = keep_indices.size(1)
            head_dim = k.size(3)
            expanded_indices = keep_indices.unsqueeze(0).unsqueeze(-1).expand(
                batch_size, num_kv_heads, budget, head_dim
            )
            k_new = k.gather(dim=2, index=expanded_indices)
            v_new = v.gather(dim=2, index=expanded_indices)
            new_pkv.append((k_new.contiguous(), v_new.contiguous()))
        pkv_tuple = tuple(new_pkv)
        comp.cache_positions_per_head = [
            [comp.cache_positions[idx] for idx in keep_indices[kv_head].tolist()]
            for kv_head in range(keep_indices.size(0))
        ]
        comp.cache_positions = comp.cache_positions_per_head[0].copy()
    else:
        new_pkv = []
        for k, v in pkv_tuple:
            new_pkv.append((k.index_select(2, keep_indices), v.index_select(2, keep_indices)))
        pkv_tuple = tuple(new_pkv)
        comp.cache_positions = [comp.cache_positions[i] for i in keep_indices.tolist()]

    return pkv_tuple, reclaimed


def append_generated_position(comp: "SpeckVRKVStyle", absolute_position: int, pkv_tuple) -> None:
    """Append one newly generated token's absolute position to compressor state.

    This mirrors the runtime semantics after a single decode token has been appended
    to the KV cache. We keep the position bookkeeping in sync without resetting the
    compressor, so later-round iterative compression uses the true absolute indices.
    """
    if comp.cache_positions_per_layer_perhead is not None:
        num_layers = len(pkv_tuple)
        num_kv_heads = int(pkv_tuple[0][0].shape[1])
        for layer_idx in range(num_layers):
            for kv_head_idx in range(num_kv_heads):
                comp.cache_positions_per_layer_perhead[(layer_idx, kv_head_idx)].append(int(absolute_position))
        comp.cache_positions = comp.cache_positions_per_layer_perhead[(0, 0)].copy()
        return

    if comp.cache_positions_per_layer is not None:
        num_layers = len(pkv_tuple)
        for layer_idx in range(num_layers):
            comp.cache_positions_per_layer[layer_idx].append(int(absolute_position))
        comp.cache_positions = comp.cache_positions_per_layer[0].copy()
        return

    if comp.cache_positions_per_head is not None:
        num_kv_heads = int(pkv_tuple[0][0].shape[1])
        for kv_head_idx in range(num_kv_heads):
            comp.cache_positions_per_head[kv_head_idx].append(int(absolute_position))
        comp.cache_positions = comp.cache_positions_per_head[0].copy()
        return

    comp.cache_positions.append(int(absolute_position))


def apply_iterative_compression(
    comp: "SpeckVRKVStyle",
    pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    *,
    dump_keep_indices_out: Path | None = None,
    dump_k_sample_out: Path | None = None,
    recent_unabsorbed_tokens: int | None = None,
) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], int]:
    """Compress using existing compressor state instead of rebuilding prefill state.

    Unlike apply_manual_compression(), this preserves cache_positions / absolute_position
    accumulated across previous compression rounds and decode steps.
    """
    seq_len = current_cache_len(pkv_tuple)
    if len(comp.cache_positions) != seq_len:
        raise ValueError(
            f"Compressor state/cache length mismatch: state={len(comp.cache_positions)} cache={seq_len}"
        )

    if dump_k_sample_out is not None and pkv_tuple:
        dump_k_sample_out.parent.mkdir(parents=True, exist_ok=True)
        sample_positions = sorted(
            set(
                [0, 1, 2, 3, 4]
                + [max(0, seq_len // 4 - 1), seq_len // 4, min(seq_len - 1, seq_len // 4 + 1)]
                + [max(0, seq_len // 2 - 1), seq_len // 2, min(seq_len - 1, seq_len // 2 + 1)]
                + [max(0, (3 * seq_len) // 4 - 1), (3 * seq_len) // 4, min(seq_len - 1, (3 * seq_len) // 4 + 1)]
                + [max(0, seq_len - 5), max(0, seq_len - 4), max(0, seq_len - 3), max(0, seq_len - 2), seq_len - 1]
            )
        )
        head_limit = min(2, int(pkv_tuple[0][0].shape[1]))
        preferred_layers = [idx for idx in (0, 1, 16, 32, 48, 63) if idx < len(pkv_tuple)]
        if not preferred_layers:
            preferred_layers = list(range(min(2, len(pkv_tuple))))
        payload = {
            "seq_len": int(seq_len),
            "sample_positions": [int(x) for x in sample_positions],
            "layers": {},
        }
        for layer_idx in preferred_layers:
            k, _ = pkv_tuple[layer_idx]
            layer_payload = {}
            for head_idx in range(head_limit):
                vectors = []
                for pos in sample_positions:
                    vectors.append(
                        k[0, head_idx, pos]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .tolist()
                    )
                layer_payload[str(head_idx)] = vectors
            payload["layers"][str(layer_idx)] = layer_payload
        dump_k_sample_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    keep_indices = _compute_keep_indices_with_window(
        comp,
        pkv_tuple,
        recent_unabsorbed_tokens=recent_unabsorbed_tokens,
    )
    if dump_keep_indices_out is not None:
        dump_keep_indices_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seq_len": int(seq_len),
            "budget": int(comp.budget),
            "prefix_length": int(comp.prefix_length),
            "absolute_position": int(comp.absolute_position),
            "window_size": int(comp.config.window_size),
            "recent_unabsorbed_tokens": (
                int(recent_unabsorbed_tokens)
                if recent_unabsorbed_tokens is not None
                else None
            ),
            "shape": list(keep_indices.shape),
            "indices": keep_indices.detach().to(device="cpu", dtype=torch.long).tolist(),
        }
        dump_keep_indices_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    reclaimed = seq_len - (keep_indices.size(-1) if keep_indices.dim() > 1 else keep_indices.numel())

    if keep_indices.dim() == 3:
        num_layers = keep_indices.size(0)
        num_kv_heads = keep_indices.size(1)
        budget = keep_indices.size(2)
        new_pkv = []
        for layer_idx, (k, v) in enumerate(pkv_tuple):
            batch_size = k.size(0)
            head_dim = k.size(3)
            layer_indices = keep_indices[layer_idx]
            expanded_indices = layer_indices.unsqueeze(0).unsqueeze(-1).expand(
                batch_size, num_kv_heads, budget, head_dim
            )
            k_new = k.gather(dim=2, index=expanded_indices)
            v_new = v.gather(dim=2, index=expanded_indices)
            new_pkv.append((k_new.contiguous(), v_new.contiguous()))
        pkv_tuple = tuple(new_pkv)
        comp.cache_positions_per_layer_perhead = {
            (layer_idx, kv_head): [comp.cache_positions[idx] for idx in keep_indices[layer_idx, kv_head].tolist()]
            for layer_idx in range(num_layers)
            for kv_head in range(num_kv_heads)
        }
        comp.cache_positions = comp.cache_positions_per_layer_perhead[(0, 0)].copy()
    elif keep_indices.dim() == 2 and comp.per_layer_pruning:
        num_layers = keep_indices.size(0)
        budget = keep_indices.size(1)
        new_pkv = []
        for layer_idx, (k, v) in enumerate(pkv_tuple):
            batch_size = k.size(0)
            num_kv_heads = k.size(1)
            head_dim = k.size(3)
            layer_indices = keep_indices[layer_idx]
            expanded_indices = layer_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
                batch_size, num_kv_heads, budget, head_dim
            )
            k_new = k.gather(dim=2, index=expanded_indices)
            v_new = v.gather(dim=2, index=expanded_indices)
            new_pkv.append((k_new.contiguous(), v_new.contiguous()))
        pkv_tuple = tuple(new_pkv)
        comp.cache_positions_per_layer = {
            layer_idx: [comp.cache_positions[idx] for idx in keep_indices[layer_idx].tolist()]
            for layer_idx in range(num_layers)
        }
        comp.cache_positions = comp.cache_positions_per_layer[0].copy()
    elif keep_indices.dim() == 2:
        new_pkv = []
        for k, v in pkv_tuple:
            batch_size = k.size(0)
            num_kv_heads = k.size(1)
            budget = keep_indices.size(1)
            head_dim = k.size(3)
            expanded_indices = keep_indices.unsqueeze(0).unsqueeze(-1).expand(
                batch_size, num_kv_heads, budget, head_dim
            )
            k_new = k.gather(dim=2, index=expanded_indices)
            v_new = v.gather(dim=2, index=expanded_indices)
            new_pkv.append((k_new.contiguous(), v_new.contiguous()))
        pkv_tuple = tuple(new_pkv)
        comp.cache_positions_per_head = [
            [comp.cache_positions[idx] for idx in keep_indices[kv_head].tolist()]
            for kv_head in range(keep_indices.size(0))
        ]
        comp.cache_positions = comp.cache_positions_per_head[0].copy()
    else:
        new_pkv = []
        for k, v in pkv_tuple:
            new_pkv.append((k.index_select(2, keep_indices), v.index_select(2, keep_indices)))
        pkv_tuple = tuple(new_pkv)
        comp.cache_positions = [comp.cache_positions[i] for i in keep_indices.tolist()]

    return pkv_tuple, reclaimed


def compute_repetition_metrics(text: str) -> Tuple[int, int, str]:
    ws = [tok for tok in re.split(r"\s+", text) if tok]
    max_ws = 1
    cur = 1
    prev = None
    for tok in ws:
        if tok == prev:
            cur += 1
        else:
            cur = 1
            prev = tok
        max_ws = max(max_ws, cur)

    max_char = 1
    max_char_c = ""
    for c, group in itertools.groupby(text):
        count = sum(1 for _ in group)
        if count > max_char:
            max_char = count
            max_char_c = c

    return max_ws, max_char, max_char_c


def run_mode(
    *,
    mode: str,
    model,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
) -> ProbeResult:
    from transformers.cache_utils import DynamicCache

    print(f"[hf-probe] mode={mode} prefill start", file=sys.stderr, flush=True)
    device = next(model.parameters()).device
    sample_generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    sample_generator.manual_seed(args.seed)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    prompt_tokens = int(input_ids.shape[1])

    with torch.no_grad():
        prefill_outputs = model(input_ids=input_ids, use_cache=True)
    pkv_tuple = normalize_past_key_values(prefill_outputs.past_key_values)
    compressed_prompt_tokens = prompt_tokens
    print(
        f"[hf-probe] mode={mode} prefill done prompt_tokens={prompt_tokens}",
        file=sys.stderr,
        flush=True,
    )

    if mode == "compress_once":
        print("[hf-probe] mode=compress_once manual compression start", file=sys.stderr, flush=True)
        compressor = create_compressor(args, device)
        pkv_tuple, reclaimed = apply_manual_compression(
            compressor,
            pkv_tuple,
            dump_keep_indices_out=args.dump_keep_indices_out,
            dump_k_sample_out=args.dump_k_sample_out,
            recent_unabsorbed_tokens=None,
        )
        compressed_prompt_tokens = current_cache_len(pkv_tuple)
        sys.stderr.write(
            f"[hf-probe] manual prefill compression applied: before={prompt_tokens} "
            f"after={compressed_prompt_tokens} reclaimed={reclaimed}\n"
        )
        absolute_position = prompt_tokens
    elif mode == "compress_iterative":
        print("[hf-probe] mode=compress_iterative setup", file=sys.stderr, flush=True)
        compressor = create_compressor(args, device)
        compressor.cache_positions = list(range(prompt_tokens))
        compressor.absolute_position = prompt_tokens
        compressor.prefix_length = prompt_tokens
        compression_round = 0
        if prompt_tokens > args.kv_budget:
            print("[hf-probe] mode=compress_iterative prefill compression start", file=sys.stderr, flush=True)
            round_dump_path = None
            if args.dump_compression_round_dir is not None:
                round_dump_path = args.dump_compression_round_dir / f"round_{compression_round:03d}_prefill.json"
            pkv_tuple, reclaimed = apply_iterative_compression(
                compressor,
                pkv_tuple,
                dump_keep_indices_out=round_dump_path or args.dump_keep_indices_out,
                dump_k_sample_out=args.dump_k_sample_out,
            )
            compressed_prompt_tokens = current_cache_len(pkv_tuple)
            sys.stderr.write(
                f"[hf-probe] iterative prefill compression applied: before={prompt_tokens} "
                f"after={compressed_prompt_tokens} reclaimed={reclaimed}\n"
            )
            compression_round += 1
        absolute_position = prompt_tokens
    else:
        absolute_position = prompt_tokens

    first_token = sample_next_token(
        prefill_outputs.logits[:, -1, :],
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        generator=sample_generator,
    )
    generated: List[int] = [int(first_token.item())]
    if tokenizer.eos_token_id is not None and generated[0] == int(tokenizer.eos_token_id):
        output_text = tokenizer.decode(generated, skip_special_tokens=False)
        max_ws, max_char, max_char_c = compute_repetition_metrics(output_text)
        return ProbeResult(
            mode=mode,
            model_path=str(args.model_path),
            stats_path=str(args.stats_path),
            dataset_path=str(args.dataset_path),
            kv_budget=int(args.kv_budget),
            prompt_tokens=prompt_tokens,
            compressed_prompt_tokens=compressed_prompt_tokens,
            output_tokens=len(generated),
            total_tokens=prompt_tokens + len(generated),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            seed=int(args.seed),
            max_same_ws_run=max_ws,
            max_same_char_run=max_char,
            max_same_char=max_char_c,
            eos_reached=True,
            output=output_text,
            prompt_preview=prompt[:1000],
        )

    eos_reached = False
    next_input_ids = first_token.view(1, 1)
    past_key_values = pkv_tuple

    print(f"[hf-probe] mode={mode} decode start", file=sys.stderr, flush=True)
    for _ in range(max(0, args.max_new_tokens - 1)):
        current_len = current_cache_len(past_key_values)
        if mode == "compress_iterative" and args.compare_vllm_step_semantics:
            trigger_len = int(args.kv_budget) + int(args.divide_length)
            # vLLM scheduler triggers on estimated length that already includes the
            # currently scheduled decode token, while the selector sees the effective
            # pre-step cache. Mirror that timing here for keep-index comparisons.
            if (current_len + 1) >= trigger_len:
                round_dump_path = None
                if args.dump_compression_round_dir is not None:
                    round_dump_path = (
                        args.dump_compression_round_dir
                        / f"round_{compression_round:03d}_decode.json"
                    )
                past_key_values, reclaimed = apply_iterative_compression(
                    compressor,
                    past_key_values,
                    dump_keep_indices_out=round_dump_path,
                    dump_k_sample_out=None,
                    recent_unabsorbed_tokens=max(0, current_len - int(args.kv_budget)),
                )
                sys.stderr.write(
                    f"[hf-probe] iterative decode compression applied (vllm-step): "
                    f"before={current_len} after={current_cache_len(past_key_values)} "
                    f"reclaimed={reclaimed} absolute_position={absolute_position}\n"
                )
                compression_round += 1
                current_len = current_cache_len(past_key_values)
        position_ids = torch.tensor(
            [[absolute_position]],
            device=device,
            dtype=torch.long,
        )
        cache_position = torch.tensor(
            [current_len],
            device=device,
            dtype=torch.long,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_input_ids,
                past_key_values=DynamicCache.from_legacy_cache(past_key_values),
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        next_token = sample_next_token(
            logits,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            generator=sample_generator,
        )
        token_id = int(next_token.item())
        generated.append(token_id)
        if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
            eos_reached = True
            break
        next_input_ids = next_token.view(1, 1)
        absolute_position += 1
        past_key_values = normalize_past_key_values(outputs.past_key_values)
        if mode == "compress_iterative":
            append_generated_position(compressor, absolute_position - 1, past_key_values)
            compressor.absolute_position = absolute_position
            current_total = current_cache_len(past_key_values)
            trigger_len = int(args.kv_budget) + int(args.divide_length)
            if (not args.compare_vllm_step_semantics) and current_total >= trigger_len:
                round_dump_path = None
                if args.dump_compression_round_dir is not None:
                    round_dump_path = args.dump_compression_round_dir / f"round_{compression_round:03d}_decode.json"
                past_key_values, reclaimed = apply_iterative_compression(
                    compressor,
                    past_key_values,
                    dump_keep_indices_out=round_dump_path,
                    dump_k_sample_out=None,
                    recent_unabsorbed_tokens=max(0, current_total - int(args.kv_budget)),
                )
                sys.stderr.write(
                    f"[hf-probe] iterative decode compression applied: before={current_total} "
                    f"after={current_cache_len(past_key_values)} reclaimed={reclaimed} "
                    f"absolute_position={absolute_position}\n"
                )
                compression_round += 1

    output_text = tokenizer.decode(generated, skip_special_tokens=False)
    max_ws, max_char, max_char_c = compute_repetition_metrics(output_text)

    return ProbeResult(
        mode=mode,
        model_path=str(args.model_path),
        stats_path=str(args.stats_path),
        dataset_path=str(args.dataset_path),
        kv_budget=int(args.kv_budget),
        prompt_tokens=prompt_tokens,
        compressed_prompt_tokens=compressed_prompt_tokens,
        output_tokens=len(generated),
        total_tokens=prompt_tokens + len(generated),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        seed=int(args.seed),
        max_same_ws_run=max_ws,
        max_same_char_run=max_char,
        max_same_char=max_char_c,
        eos_reached=eos_reached,
        output=output_text,
        prompt_preview=prompt[:1000],
    )


def write_result(output_dir: Path, result: ProbeResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.mode}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return path


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
        print(f"[hf-probe] using raw prompt file {args.prompt_file}", file=sys.stderr, flush=True)
    else:
        print("[hf-probe] loading dataset", file=sys.stderr, flush=True)
        record = load_record(args.dataset_path, args.sample_index)
        prompt = build_probe_prompt(
            tokenizer,
            record,
            use_chat_template=bool(args.use_chat_template),
            system_prompt=args.system_prompt,
            prompt_key=args.prompt_key,
        )

    modes: Sequence[str]
    if args.mode == "both":
        modes = ("baseline", "compress_once")
    else:
        modes = (args.mode,)

    written: List[Path] = []
    for mode in modes:
        result = run_mode(
            mode=mode,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            args=args,
        )
        path = write_result(args.output_dir, result)
        written.append(path)
        print(
            json.dumps(
                {
                    "mode": result.mode,
                    "path": str(path),
                    "prompt_tokens": result.prompt_tokens,
                    "compressed_prompt_tokens": result.compressed_prompt_tokens,
                    "output_tokens": result.output_tokens,
                    "max_same_ws_run": result.max_same_ws_run,
                    "max_same_char_run": result.max_same_char_run,
                    "eos_reached": result.eos_reached,
                },
                ensure_ascii=False,
            )
        )

    print(json.dumps({"written": [str(p) for p in written]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
