#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from TriAttention_vLLM.triattention_runtime.config import TriAttentionRuntimeConfig
from TriAttention_vLLM.triattention_runtime.selector_hf import build_speckv_selector
from weian_development.demo_debug.hf_prefill_manual_compression_probe import (
    load_model_and_tokenizer,
    normalize_past_key_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the vLLM selector offline on HF-generated dense keys."
    )
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--hf-keep-json", type=Path, default=None)
    parser.add_argument("--vllm-keep-json", type=Path, default=None)
    parser.add_argument("--kv-budget", type=int, default=7000)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--load-dtype", type=str, default="float16")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    return parser.parse_args()


def _compare_keep_lists(lhs: list[list[int]], rhs: list[list[int]]) -> dict[str, Any]:
    jaccards: list[float] = []
    symdiffs: list[int] = []
    per_head: list[dict[str, Any]] = []
    for head_idx, (left, right) in enumerate(zip(lhs, rhs)):
        lset = set(int(x) for x in left)
        rset = set(int(x) for x in right)
        union = len(lset | rset)
        inter = len(lset & rset)
        jaccard = float(inter / union) if union else 1.0
        symdiff = len(lset ^ rset)
        jaccards.append(jaccard)
        symdiffs.append(symdiff)
        per_head.append(
            {
                "head_idx": head_idx,
                "jaccard": jaccard,
                "symdiff": symdiff,
                "lhs_only_sample": sorted(lset - rset)[:20],
                "rhs_only_sample": sorted(rset - lset)[:20],
            }
        )
    return {
        "mean_jaccard": mean(jaccards) if jaccards else 1.0,
        "min_jaccard": min(jaccards) if jaccards else 1.0,
        "max_jaccard": max(jaccards) if jaccards else 1.0,
        "mean_symdiff": mean(symdiffs) if symdiffs else 0.0,
        "per_head": per_head,
    }


def _load_keep_indices(path: Path | None) -> list[list[int]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("indices")


def main() -> None:
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    probe_args = argparse.Namespace(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        load_dtype=args.load_dtype,
        trust_remote_code=False,
        device="cuda:0",
    )
    model, tokenizer = load_model_and_tokenizer(probe_args)
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    import torch

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    pkv_tuple = normalize_past_key_values(outputs.past_key_values)
    seq_len = int(pkv_tuple[0][0].shape[2]) if pkv_tuple else 0
    layer_inputs = [(layer_idx, layer_k) for layer_idx, (layer_k, _) in enumerate(pkv_tuple)]

    config = TriAttentionRuntimeConfig(
        model_path=str(args.model_path),
        kv_budget=int(args.kv_budget),
        divide_length=128,
        protect_prefill=False,
        window_size=int(args.window_size),
        sparse_stats_path=str(args.stats_path),
        pruning_mode="per_head",
        sparse_normalize_scores=True,
        include_prefill_in_budget=True,
        per_head_selection_semantics="hf_aligned_global_per_head",
        enable_experimental_kv_compaction=True,
    )
    _select_keep_indices, select_keep_indices_for_group_per_head, status = (
        build_speckv_selector(config=config, base_runner=None)
    )
    result = select_keep_indices_for_group_per_head(
        layer_inputs=layer_inputs,
        total_tokens=seq_len,
        prefill_len=seq_len,
        protect_prefill=False,
        round_start=seq_len,
        budget_total=int(args.kv_budget),
    )
    if result is None:
        raise RuntimeError(f"selector returned None, status={status}")

    keep_cpu = result["indices"].detach().to(device="cpu").tolist()
    payload: dict[str, Any] = {
        "seq_len": seq_len,
        "selector_status": status,
        "shape": list(result["indices"].shape),
        "group_agg_mode": result.get("group_agg_mode"),
        "indices": keep_cpu,
    }

    hf_keep = _load_keep_indices(args.hf_keep_json)
    if hf_keep is not None:
        payload["compare_hf"] = _compare_keep_lists(keep_cpu, hf_keep)
    vllm_keep = _load_keep_indices(args.vllm_keep_json)
    if vllm_keep is not None:
        payload["compare_vllm"] = _compare_keep_lists(keep_cpu, vllm_keep)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output_json)


if __name__ == "__main__":
    main()
