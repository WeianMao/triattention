#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import torch

from TriAttention_vLLM.triattention.config import TriAttentionConfig
from TriAttention_vLLM.triattention.compressor import TriAttentionCompressor
from TriAttention_vLLM.triattention.scoring import compute_scores_triton
from weian_development.demo_debug.hf_prefill_manual_compression_probe import (
    create_compressor as create_hf_compressor,
    load_model_and_tokenizer,
    normalize_past_key_values,
)
from weian_development.speckv.round_pruning_utils import (
    compute_frequency_statistics_from_means,
    invert_rope,
    score_keys_for_round,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF reference layer scores against vLLM optimized layer scores on identical dense keys."
    )
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--layers", type=str, default="0,16,32,48,63")
    parser.add_argument("--load-dtype", type=str, default="float16")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--score-aggregation", type=str, default="mean")
    return parser.parse_args()


def _compare_scores(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    diff = (lhs - rhs).abs().to(dtype=torch.float32)
    flat = diff.view(-1)
    return {
        "mean_abs_diff": float(flat.mean().item()),
        "max_abs_diff": float(flat.max().item()),
        "p99_abs_diff": float(torch.quantile(flat, 0.99).item()),
    }


def _compare_topk(lhs: torch.Tensor, rhs: torch.Tensor, k: int) -> dict[str, Any]:
    lhs_top = torch.topk(lhs, k=k, dim=-1, largest=True, sorted=False).indices.cpu().tolist()
    rhs_top = torch.topk(rhs, k=k, dim=-1, largest=True, sorted=False).indices.cpu().tolist()
    jaccards = []
    symdiffs = []
    for la, rb in zip(lhs_top, rhs_top):
        lset = set(int(x) for x in la)
        rset = set(int(x) for x in rb)
        jaccards.append(len(lset & rset) / len(lset | rset))
        symdiffs.append(len(lset ^ rset))
    return {
        "mean_jaccard": mean(jaccards),
        "min_jaccard": min(jaccards),
        "max_jaccard": max(jaccards),
        "mean_symdiff": mean(symdiffs),
    }


def _to_complex_pairs(values: torch.Tensor, style: str) -> torch.Tensor:
    half = values.shape[-1] // 2
    if style == "interleaved":
        real = values[..., 0::2]
        imag = values[..., 1::2]
    else:
        real = values[..., :half]
        imag = values[..., half:]
    return torch.complex(real.to(dtype=torch.float32), imag.to(dtype=torch.float32))


def _direct_formula_compare(
    hf_comp,
    tri_comp: TriAttentionCompressor,
    *,
    key_states: torch.Tensor,
    layer_idx: int,
    seq_len: int,
) -> dict[str, Any]:
    sampled_heads = [(l, h) for l, h in hf_comp.sampled_heads if l == layer_idx]
    if not sampled_heads:
        return {"available": False}
    _, sampled_head = sampled_heads[0]
    kv_head = sampled_head // max(1, hf_comp.num_key_value_groups)
    token_positions = [0, seq_len // 2, seq_len - 1]
    token_positions = sorted(set(int(x) for x in token_positions))

    k_rot = key_states[0, kv_head, token_positions].to(device=key_states.device, dtype=hf_comp.config.dtype)
    pos_tensor = torch.tensor(token_positions, device=key_states.device, dtype=torch.long)
    base = torch.zeros(1, len(token_positions), hf_comp.head_dim, device=key_states.device, dtype=hf_comp.config.dtype)
    cos, sin = hf_comp.rotary(base, pos_tensor.unsqueeze(0))
    k_unrot = invert_rope(k_rot, cos[0], sin[0], hf_comp.attention_scale, style=hf_comp.rope_style)

    hf_stats = hf_comp.head_stats[(layer_idx, sampled_head)]
    amp, phi, extra = compute_frequency_statistics_from_means(
        hf_stats.q_mean_complex,
        hf_stats.q_abs_mean,
        k_unrot,
        style=hf_comp.rope_style,
        disable_mlr=False,
    )
    hf_scores = score_keys_for_round(
        key_indices=pos_tensor,
        round_start=seq_len,
        amp=amp,
        phi=phi,
        omega=hf_comp.omega,
        extra=extra,
        offsets=hf_comp.offsets,
        aggregation="mean",
        freq_scale_sq=hf_comp.freq_scale_sq,
        disable_top_n_high_freq=0,
        disable_trig=False,
    ).to(dtype=torch.float32)

    tri_comp._lazy_init()
    tri_stats = tri_comp.head_stats[layer_idx]
    q_mean_complex = torch.complex(
        tri_stats["q_mean_complex"][sampled_head, :, 0].to(dtype=torch.float32),
        tri_stats["q_mean_complex"][sampled_head, :, 1].to(dtype=torch.float32),
    )
    q_abs_mean = tri_stats["q_abs_mean"][sampled_head].to(dtype=torch.float32)
    freq_scale_sq = tri_comp.freq_scale_sq[layer_idx][sampled_head].to(dtype=torch.float32)
    offsets = tri_comp.offsets.to(dtype=torch.float32)
    omega = tri_comp.omega.to(dtype=torch.float32)
    q_mean_abs = torch.abs(q_mean_complex)
    k_complex = _to_complex_pairs(k_rot, getattr(tri_comp.config, "rope_style", "half"))
    prod = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    prod_real = prod.real
    prod_imag = prod.imag
    k_abs = torch.abs(k_complex)
    extra_term = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs * freq_scale_sq.unsqueeze(0)
    additive = extra_term.sum(dim=-1)
    phases = (float(seq_len) + offsets[:, None]) * omega[None, :]
    cos_vals = torch.cos(phases)
    sin_vals = torch.sin(phases)

    def _aggregate(base_scores: torch.Tensor) -> torch.Tensor:
        return (base_scores + additive.unsqueeze(0)).mean(dim=0).to(dtype=torch.float32)

    tri_scores = _aggregate(
        (
            prod_real.unsqueeze(0) * freq_scale_sq.view(1, 1, -1) * cos_vals[:, None, :]
            - prod_imag.unsqueeze(0) * freq_scale_sq.view(1, 1, -1) * sin_vals[:, None, :]
        ).sum(dim=-1)
    )
    tri_scores_imag_flip = _aggregate(
        (
            prod_real.unsqueeze(0) * freq_scale_sq.view(1, 1, -1) * cos_vals[:, None, :]
            + prod_imag.unsqueeze(0) * freq_scale_sq.view(1, 1, -1) * sin_vals[:, None, :]
        ).sum(dim=-1)
    )
    raw_real = q_mean_complex.real.unsqueeze(0) * k_complex.real + q_mean_complex.imag.unsqueeze(0) * k_complex.imag
    raw_imag_alt = q_mean_complex.real.unsqueeze(0) * k_complex.imag - q_mean_complex.imag.unsqueeze(0) * k_complex.real
    tri_scores_conj_flip = _aggregate(
        (
            raw_real.unsqueeze(0) * freq_scale_sq.view(1, 1, -1) * cos_vals[:, None, :]
            - raw_imag_alt.unsqueeze(0) * freq_scale_sq.view(1, 1, -1) * sin_vals[:, None, :]
        ).sum(dim=-1)
    )

    diff = (tri_scores - hf_scores).abs()
    diff_imag_flip = (tri_scores_imag_flip - hf_scores).abs()
    diff_conj_flip = (tri_scores_conj_flip - hf_scores).abs()
    return {
        "available": True,
        "layer_idx": layer_idx,
        "sampled_head": int(sampled_head),
        "kv_head": int(kv_head),
        "token_positions": token_positions,
        "hf_scores": [float(x) for x in hf_scores.tolist()],
        "tri_scores": [float(x) for x in tri_scores.tolist()],
        "abs_diff": [float(x) for x in diff.tolist()],
        "max_abs_diff": float(diff.max().item()),
        "tri_scores_imag_flip": [float(x) for x in tri_scores_imag_flip.tolist()],
        "abs_diff_imag_flip": [float(x) for x in diff_imag_flip.tolist()],
        "max_abs_diff_imag_flip": float(diff_imag_flip.max().item()),
        "tri_scores_conj_flip": [float(x) for x in tri_scores_conj_flip.tolist()],
        "abs_diff_conj_flip": [float(x) for x in diff_conj_flip.tolist()],
        "max_abs_diff_conj_flip": float(diff_conj_flip.max().item()),
    }


def _normalize_per_head(scores: torch.Tensor) -> torch.Tensor:
    mean_scores = scores.mean(dim=1, keepdim=True)
    std_scores = scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (scores - mean_scores) / std_scores


def _hf_layer_scores(
    hf_comp,
    *,
    key_states: torch.Tensor,
    layer_idx: int,
    seq_len: int,
) -> torch.Tensor:
    decode_positions = torch.arange(seq_len, device=key_states.device, dtype=torch.long)
    layer_scores = hf_comp._compute_layer_head_scores(
        key_states,
        decode_positions,
        layer_idx,
        start_index=0,
        positions_per_kv_head=None,
    )
    if layer_scores is None:
        raise RuntimeError(f"no sampled heads for layer {layer_idx}")
    layer_scores = _normalize_per_head(layer_scores)

    layer_heads = [(l, h) for l, h in hf_comp.sampled_heads if l == layer_idx]
    kv_head_groups: dict[int, list[int]] = {}
    for i, (_, attn_head) in enumerate(layer_heads):
        kv_head = attn_head // max(1, hf_comp.num_key_value_groups)
        kv_head_groups.setdefault(kv_head, []).append(i)

    out = []
    for kv_head_idx in range(hf_comp.num_key_value_heads):
        if kv_head_idx in kv_head_groups:
            indices = kv_head_groups[kv_head_idx]
            out.append(layer_scores[indices].max(dim=0).values)
        else:
            out.append(layer_scores.mean(dim=0))
    return torch.stack(out, dim=0).to(dtype=torch.float32)


def _vllm_layer_scores(
    vllm_comp: TriAttentionCompressor,
    *,
    key_states: torch.Tensor,
    layer_idx: int,
    seq_len: int,
    score_aggregation: str,
) -> torch.Tensor:
    vllm_comp._lazy_init()
    runtime_heads = int(key_states.shape[1])
    layer_head_stats = vllm_comp.head_stats[layer_idx]
    layer_freq_scale_sq = vllm_comp.freq_scale_sq[layer_idx]
    stats_heads = int(layer_freq_scale_sq.shape[0])
    if stats_heads % runtime_heads != 0:
        raise RuntimeError(f"stats/runtime head mismatch: {stats_heads} vs {runtime_heads}")
    group_size = stats_heads // runtime_heads
    repeated_keys = key_states.repeat_interleave(group_size, dim=1).contiguous()
    cfg = TriAttentionConfig(
        stats_path=Path("/tmp/unused.pt"),
        model_path=Path("/tmp/unused-model"),
        kv_budget=7000,
        divide_length=128,
        pruning_mode="per_head",
        score_aggregation=score_aggregation,
        sparse_normalize_scores=True,
        window_size=0,
        include_prefill_in_budget=True,
        protect_prefill=False,
        disable_mlr=False,
        disable_trig=False,
        use_triton_scoring=True,
    )
    cfg.rope_style = getattr(vllm_comp.config, "rope_style", "half")
    raw = compute_scores_triton(
        key_states=repeated_keys,
        cache_positions=None,
        head_stats=layer_head_stats,
        omega=vllm_comp.omega,
        offsets=vllm_comp.offsets,
        freq_scale_sq=layer_freq_scale_sq,
        config=cfg,
        round_start=seq_len,
        trig_cache=getattr(vllm_comp, "trig_cache", None),
    )[0]
    raw = _normalize_per_head(raw)
    grouped = raw.view(runtime_heads, group_size, seq_len)
    return grouped.max(dim=1).values.to(dtype=torch.float32)


def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    prompt = args.prompt_file.read_text(encoding="utf-8")

    probe_args = SimpleNamespace(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        load_dtype=args.load_dtype,
        trust_remote_code=False,
        device="cuda:0",
        stats_path=args.stats_path,
        kv_budget=7000,
        compression_seed=None,
        normalize_scores=True,
        use_rank_aggregation=False,
        divide_length=128,
        per_head_pruning=True,
        per_layer_perhead_pruning=False,
        layer_perhead_aggregation="max",
        per_layer_pruning=False,
        per_layer_aggregation="max",
        disable_top_n_high_freq=0,
        disable_mlr=False,
        disable_trig=False,
        score_aggregation=args.score_aggregation,
        window_size=128,
    )
    model, tokenizer = load_model_and_tokenizer(probe_args)
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    pkv_tuple = normalize_past_key_values(outputs.past_key_values)
    seq_len = int(pkv_tuple[0][0].shape[2]) if pkv_tuple else 0

    hf_comp = create_hf_compressor(probe_args, device)
    tri_cfg = TriAttentionConfig(
        stats_path=args.stats_path,
        model_path=args.model_path,
        kv_budget=7000,
        divide_length=128,
        pruning_mode="per_head",
        score_aggregation=args.score_aggregation,
        sparse_normalize_scores=True,
        window_size=0,
        include_prefill_in_budget=True,
        protect_prefill=False,
        disable_mlr=False,
        disable_trig=False,
        use_triton_scoring=True,
    )
    vllm_comp = TriAttentionCompressor(tri_cfg)

    layer_results = []
    for layer_idx in layers:
        key_states = pkv_tuple[layer_idx][0]
        hf_scores = _hf_layer_scores(hf_comp, key_states=key_states, layer_idx=layer_idx, seq_len=seq_len)
        vllm_scores = _vllm_layer_scores(
            vllm_comp,
            key_states=key_states,
            layer_idx=layer_idx,
            seq_len=seq_len,
            score_aggregation=args.score_aggregation,
        )
        layer_results.append(
            {
                "layer_idx": layer_idx,
                "score_diff": _compare_scores(vllm_scores, hf_scores),
                "topk_7000": _compare_topk(vllm_scores, hf_scores, k=7000),
                "topk_256": _compare_topk(vllm_scores, hf_scores, k=256),
            }
        )

    payload = {
        "seq_len": seq_len,
        "input_stats_check": {
            "omega_maxdiff": float((hf_comp.omega - vllm_comp.omega).abs().max().item()),
            "freq_scale_sq_layer0_maxdiff": float(
                (hf_comp.freq_scale_sq - vllm_comp.freq_scale_sq[layers[0]]).abs().max().item()
            ),
            "q_mean_real_layer0_head0_maxdiff": float(
                (
                    hf_comp.head_stats[(layers[0], 0)].q_mean_complex.real
                    - vllm_comp.head_stats[layers[0]]["q_mean_complex"][0, :, 0]
                ).abs().max().item()
            ),
            "q_mean_imag_layer0_head0_maxdiff": float(
                (
                    hf_comp.head_stats[(layers[0], 0)].q_mean_complex.imag
                    - vllm_comp.head_stats[layers[0]]["q_mean_complex"][0, :, 1]
                ).abs().max().item()
            ),
            "q_abs_mean_layer0_head0_maxdiff": float(
                (
                    hf_comp.head_stats[(layers[0], 0)].q_abs_mean
                    - vllm_comp.head_stats[layers[0]]["q_abs_mean"][0]
                ).abs().max().item()
            ),
        },
        "layers": layer_results,
        "direct_formula_check": _direct_formula_compare(
            hf_comp,
            vllm_comp,
            key_states=pkv_tuple[layers[0]][0],
            layer_idx=layers[0],
            seq_len=seq_len,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
