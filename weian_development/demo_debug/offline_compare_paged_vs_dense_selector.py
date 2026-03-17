#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from TriAttention_vLLM.triattention_runtime.config import TriAttentionRuntimeConfig
from TriAttention_vLLM.triattention_runtime.selector_hf import build_speckv_selector
from weian_development.demo_debug.hf_prefill_manual_compression_probe import (
    load_model_and_tokenizer,
    normalize_past_key_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vLLM selector dense path and paged path on identical HF-generated KV."
    )
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--hf-keep-json", type=Path, default=None)
    parser.add_argument("--kv-budget", type=int, default=7000)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--active-recent-unabsorbed-tokens", type=int, default=None)
    parser.add_argument("--load-dtype", type=str, default="float16")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    return parser.parse_args()


def _compare_keep_lists(lhs: list[list[int]], rhs: list[list[int]]) -> dict[str, Any]:
    jaccards: list[float] = []
    symdiffs: list[int] = []
    for left, right in zip(lhs, rhs):
        lset = set(int(x) for x in left)
        rset = set(int(x) for x in right)
        union = len(lset | rset)
        inter = len(lset & rset)
        jaccards.append(float(inter / union) if union else 1.0)
        symdiffs.append(len(lset ^ rset))
    return {
        "mean_jaccard": mean(jaccards) if jaccards else 1.0,
        "min_jaccard": min(jaccards) if jaccards else 1.0,
        "max_jaccard": max(jaccards) if jaccards else 1.0,
        "mean_symdiff": mean(symdiffs) if symdiffs else 0.0,
        "max_symdiff": max(symdiffs) if symdiffs else 0,
    }


def _pack_paged_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, list[int]]:
    # key_states/value_states are [H, T, D]
    heads, total_tokens, head_dim = key_states.shape
    pad = (-total_tokens) % block_size
    if pad > 0:
        key_states = torch.cat(
            [
                key_states,
                torch.zeros(
                    heads,
                    pad,
                    head_dim,
                    device=key_states.device,
                    dtype=key_states.dtype,
                ),
            ],
            dim=1,
        )
        value_states = torch.cat(
            [
                value_states,
                torch.zeros(
                    heads,
                    pad,
                    head_dim,
                    device=value_states.device,
                    dtype=value_states.dtype,
                ),
            ],
            dim=1,
        )
    padded_tokens = key_states.shape[1]
    num_blocks = padded_tokens // block_size
    key_blocks = (
        key_states.permute(1, 0, 2)
        .contiguous()
        .view(num_blocks, block_size, heads, head_dim)
    )
    value_blocks = (
        value_states.permute(1, 0, 2)
        .contiguous()
        .view(num_blocks, block_size, heads, head_dim)
    )
    kv_cache = torch.stack([key_blocks, value_blocks], dim=0).contiguous()
    block_ids = list(range(num_blocks))
    return kv_cache, block_ids


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

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    pkv_tuple = normalize_past_key_values(outputs.past_key_values)
    seq_len = int(pkv_tuple[0][0].shape[2]) if pkv_tuple else 0

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
        layer_perhead_aggregation="max",
        per_layer_aggregation="max",
        enable_experimental_kv_compaction=True,
    )
    class _DummyRunner:
        pass

    base_runner = _DummyRunner()
    if args.active_recent_unabsorbed_tokens is not None:
        setattr(
            base_runner,
            "_triattention_active_recent_unabsorbed_tokens",
            int(args.active_recent_unabsorbed_tokens),
        )
    else:
        setattr(base_runner, "_triattention_active_recent_unabsorbed_tokens", None)
    setattr(base_runner, "_triattention_active_req_id", "offline")
    setattr(base_runner, "_triattention_state_store", None)

    _select_keep_indices, group_selector, status = build_speckv_selector(
        config=config,
        base_runner=base_runner,
    )

    layer_inputs: list[tuple[int, torch.Tensor]] = []
    layer_kv_entries: list[tuple[int, torch.Tensor, list[int], int]] = []
    for layer_idx, (layer_k, layer_v) in enumerate(pkv_tuple):
        key_states = layer_k[0]  # [H, T, D]
        value_states = layer_v[0]
        layer_inputs.append((layer_idx, layer_k))
        kv_cache, block_ids = _pack_paged_kv(
            key_states,
            value_states,
            block_size=int(args.block_size),
        )
        layer_kv_entries.append((layer_idx, kv_cache, block_ids, int(args.block_size)))

    dense_result = group_selector(
        layer_inputs=layer_inputs,
        total_tokens=seq_len,
        prefill_len=seq_len,
        protect_prefill=False,
        round_start=seq_len,
        budget_total=int(args.kv_budget),
    )
    paged_result = group_selector(
        layer_inputs=None,
        layer_kv_iter=lambda: iter(layer_kv_entries),
        total_tokens=seq_len,
        prefill_len=seq_len,
        protect_prefill=False,
        round_start=seq_len,
        budget_total=int(args.kv_budget),
    )
    if dense_result is None or paged_result is None:
        raise RuntimeError(f"selector returned None, status={status}")

    dense_keep = dense_result["indices"].detach().to(device="cpu").tolist()
    paged_keep = paged_result["indices"].detach().to(device="cpu").tolist()
    payload: dict[str, Any] = {
        "seq_len": seq_len,
        "selector_status": status,
        "dense_shape": list(dense_result["indices"].shape),
        "paged_shape": list(paged_result["indices"].shape),
        "dense_head0_first20": dense_keep[0][:20] if dense_keep else [],
        "paged_head0_first20": paged_keep[0][:20] if paged_keep else [],
        "paged_vs_dense": _compare_keep_lists(paged_keep, dense_keep),
        "active_recent_unabsorbed_tokens": args.active_recent_unabsorbed_tokens,
    }
    if args.hf_keep_json is not None:
        hf_keep = json.loads(args.hf_keep_json.read_text(encoding="utf-8")).get("indices")
        if hf_keep is not None:
            payload["dense_vs_hf"] = _compare_keep_lists(dense_keep, hf_keep)
            payload["paged_vs_hf"] = _compare_keep_lists(paged_keep, hf_keep)
            payload["hf_head0_first20"] = hf_keep[0][:20] if hf_keep else []

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output_json)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
