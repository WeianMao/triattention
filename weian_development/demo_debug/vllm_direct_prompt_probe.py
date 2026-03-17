#!/usr/bin/env python3
"""Direct raw-prompt vLLM/TriAttention probe for bad-case debugging."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from TriAttention_vLLM.evaluation.runner.vllm_triattention_runtime_runner import (
    parse_arguments,
    setup_vllm_engine,
)


def parse_args() -> argparse.Namespace:
    probe_parser = argparse.ArgumentParser(add_help=False)
    probe_parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Raw prompt file to send directly to vLLM without dataset/template wrapping.",
    )
    probe_parser.add_argument(
        "--result-json",
        type=str,
        required=True,
        help="Where to write the compact JSON result.",
    )
    probe_args, remaining = probe_parser.parse_known_args()

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        runner_args = parse_arguments()
    finally:
        sys.argv = old_argv

    setattr(runner_args, "prompt_file", probe_args.prompt_file)
    setattr(runner_args, "result_json", probe_args.result_json)
    return runner_args


def repetition_metrics(text: str) -> dict[str, int | str]:
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

    return {
        "max_same_ws_run": max_ws,
        "max_same_char_run": max_char,
        "max_same_char": max_char_c,
    }


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompt_file)
    result_path = Path(args.result_json)
    prompt = prompt_path.read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = setup_vllm_engine(args)
    llm_tokenizer = llm.get_tokenizer()
    eos_token_id = getattr(llm_tokenizer, "eos_token_id", None)

    from vllm import SamplingParams

    prefill_length = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    max_new_tokens = max(1, args.max_length - prefill_length)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=max_new_tokens,
        n=1,
        seed=args.seed,
        stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
    )
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    out = outputs[0]
    text = out.outputs[0].text
    obj = {
        "prompt_file": str(prompt_path),
        "prompt_tokens": len(out.prompt_token_ids),
        "output_tokens": len(out.outputs[0].token_ids),
        "total_tokens": len(out.prompt_token_ids) + len(out.outputs[0].token_ids),
        "kv_budget": args.kv_budget,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "output": text,
    }
    obj.update(repetition_metrics(text))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result_path)
    print(
        json.dumps(
            {
                "prompt_tokens": obj["prompt_tokens"],
                "output_tokens": obj["output_tokens"],
                "kv_budget": obj["kv_budget"],
                "max_same_ws_run": obj["max_same_ws_run"],
                "max_same_char_run": obj["max_same_char_run"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
