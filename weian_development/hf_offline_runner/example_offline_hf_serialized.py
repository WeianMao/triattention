"""离线 HuggingFace 推理示例（保留 DeepConf 序列化接口）。"""
import argparse
import json
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from dynasor.core.evaluator import math_equal
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepconf.utils import extract_answer
from development.serialization_utils import SerializationError, dump_msgpack
from weian_development.process_utils import mask_process_command


# ============= PROMPT PREPARATION FUNCTIONS =============

def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    """准备 DeepSeek 风格的聊天提示。"""
    if model_type == "deepseek":
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "user", "content": question},
        ]

    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return full_prompt


def prepare_prompt_gpt(question: str, tokenizer, reasoning_effort: str = "high") -> str:
    """准备 GPT 风格的聊天提示。"""
    messages = [
        {"role": "user", "content": question},
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True,
    )
    return full_prompt


def quick_parse(text: str) -> str:
    """去除 LaTeX 中的 \text 标记方便比较。"""
    if "\\text{" in text and "}" in text:
        while "\\text{" in text:
            start = text.find("\\text{")
            if start == -1:
                break
            end = text.find("}", start)
            if end == -1:
                break
            content = text[start + 6 : end]
            text = text[:start] + content + text[end + 1 :]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """对推理答案进行容错比较。"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    return math_equal(answer, ground_truth)


def compute_token_logprobs(
    scores: List[torch.Tensor],
    generated_ids: torch.Tensor,
    tokenizer,
) -> List[Dict[str, Any]]:
    """把 HuggingFace 的 logits 序列转换成逐 token 的 logprob 列表。"""
    tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
    token_stats: List[Dict[str, Any]] = []

    for idx, token_id in enumerate(generated_ids.tolist()):
        score = scores[idx][0].float()
        logprobs = torch.log_softmax(score, dim=-1)
        token_logprob = float(logprobs[token_id].item())
        token_stats.append(
            {
                "token_id": int(token_id),
                "token": tokens[idx],
                "logprob": token_logprob,
                "prob": math.exp(token_logprob),
            }
        )
    return token_stats


def build_trace(
    prompt: str,
    output_text: str,
    generated_ids: torch.Tensor,
    tokenizer,
    store_logprobs: bool,
    scores: Optional[List[torch.Tensor]],
) -> Dict[str, Any]:
    """构造与原始 DeepConf 接口兼容的 trace 结构。"""
    trace: Dict[str, Any] = {
        "prompt": prompt,
        "response": output_text,
        "token_ids": [int(tok_id) for tok_id in generated_ids.tolist()],
        "tokens": tokenizer.convert_ids_to_tokens(generated_ids.tolist()),
    }

    extracted = extract_answer(output_text)
    if extracted:
        trace["extracted_answer"] = extracted

    if store_logprobs and scores is not None:
        trace["logprobs"] = compute_token_logprobs(scores, generated_ids, tokenizer)

    return trace


def prepare_evaluation(predicted: str, ground_truth: str) -> Optional[Dict[str, Any]]:
    if not ground_truth:
        return None
    comparison_answer = (predicted or "").strip()
    return {
        "prediction": comparison_answer,
        "ground_truth": ground_truth,
        "is_correct": equal_func(comparison_answer, ground_truth),
    }


def main() -> None:
    mask_process_command("PD-L1_binder")

    parser = argparse.ArgumentParser(description="HuggingFace 离线推理示例（单次回答）")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="模型路径或名称")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="占位参数，兼容旧配置")
    parser.add_argument("--dataset", type=str, default="aime25.jsonl", help="数据集文件路径")
    parser.add_argument("--qid", type=int, required=True, help="要处理的问题编号 (0-based)")
    parser.add_argument("--rid", type=str, default="offline_run", help="运行标识")
    parser.add_argument("--budget", type=int, default=1, help="保留的配置参数，当前始终生成一次")
    parser.add_argument("--window_size", type=int, default=2048, help="保留的配置参数")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=-1,
        help="最大生成 token 数；默认为 -1 自动匹配模型可接受的最大上下文",
    )
    parser.add_argument("--max_model_len", type=int, default=None, help="保留参数，HuggingFace 未使用")
    parser.add_argument("--model_type", type=str, default="gpt", choices=["deepseek", "gpt"], help="提示模板类型")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="GPT 模型推理强度")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p 采样")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k 采样")
    parser.add_argument("--output_dir", type=str, default="outputs", help="结果输出目录")
    parser.add_argument("--no_multiple_voting", action="store_true", help="兼容旧参数，无副作用")
    parser.add_argument("--no_store_logprobs", action="store_true", help="禁用 logprob 存储")
    parser.add_argument(
        "--serializer",
        choices=["pickle", "msgpack_gzip", "msgpack_zstd", "msgpack_plain"],
        default="msgpack_gzip",
        help="结果序列化方式",
    )
    parser.add_argument("--compression_level", type=int, default=3, help="压缩等级")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None, help="兼容旧参数")

    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset from {dataset_path} ...")
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = [json.loads(line.strip()) for line in handle]

    if not 0 <= args.qid < len(data):
        raise ValueError(f"Question ID {args.qid} is out of range (0-{len(data) - 1})")

    question_data = data[args.qid]
    question = question_data["question"]
    ground_truth = str(question_data.get("answer", "")).strip()

    print(f"Processing question {args.qid}: {question[:100]}...")

    # Prepare model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tensor_parallel_size != 1:
        print("[Warning] tensor_parallel_size is fixed to 1 for the HuggingFace backend.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading HuggingFace model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    # Prepare prompt
    if args.model_type == "gpt":
        prompt = prepare_prompt_gpt(question, tokenizer, args.reasoning_effort)
    else:
        prompt = prepare_prompt(question, tokenizer, args.model_type)

    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = prompt_inputs["input_ids"].to(device)
    attention_mask = prompt_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_length = int(input_ids.shape[-1])

    candidate_context_lengths: List[int] = []
    for attr in (
        "max_position_embeddings",
        "max_sequence_length",
        "n_positions",
        "max_seq_len",
        "window",
        "context_length",
    ):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            candidate_context_lengths.append(value)

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**7:
        candidate_context_lengths.append(tokenizer_limit)

    model_context_length = max(candidate_context_lengths) if candidate_context_lengths else 4096
    available_context = max(1, model_context_length - prompt_length)

    if 0 < args.max_tokens:
        max_new_tokens = min(args.max_tokens, available_context)
    else:
        max_new_tokens = available_context
    if max_new_tokens <= 0:
        raise ValueError(
            f"Prompt 长度 {prompt_length} 已经超过模型上下文限制 {model_context_length}，无法继续生成。"
        )

    do_sample = args.temperature > 0 or args.top_p < 1.0 or args.top_k > 0
    store_logprobs = not args.no_store_logprobs

    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
        "return_dict_in_generate": True,
        "output_scores": store_logprobs,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }

    if do_sample:
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
        if args.top_p <= 1.0:
            gen_kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    generation_start = time.time()
    with torch.inference_mode():
        generated = model.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})
    generation_time = time.time() - generation_start

    sequence = generated.sequences[0]
    generated_ids = sequence[prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    trace = build_trace(
        prompt=prompt,
        output_text=generated_text,
        generated_ids=generated_ids,
        tokenizer=tokenizer,
        store_logprobs=store_logprobs,
        scores=generated.scores if store_logprobs else None,
    )

    predicted_answer = trace.get("extracted_answer", generated_text.strip())
    evaluation = prepare_evaluation(predicted_answer, ground_truth)

    generated_token_count = int(generated_ids.shape[-1])
    prompt_token_count = int(prompt_length)
    total_tokens = prompt_token_count + generated_token_count
    throughput = (generated_token_count / generation_time) if generation_time > 0 else None

    print("\n=== HuggingFace Offline Summary ===")
    print(f"Prompt tokens: {prompt_token_count}")
    print(f"Generated tokens: {generated_token_count}")
    print(f"Generation time: {generation_time:.2f}s")
    if throughput is not None:
        print(f"Throughput: {throughput:.1f} tokens/s")
    print(f"Predicted answer: {predicted_answer}")
    if evaluation is not None:
        print(f"Correct: {evaluation['is_correct']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data: Dict[str, Any] = {
        "backend": "huggingface",
        "model": args.model,
        "question": question,
        "ground_truth": ground_truth,
        "qid": args.qid,
        "run_id": args.rid,
        "prompt": prompt,
        "response": generated_text,
        "predicted_answer": predicted_answer,
        "evaluation": evaluation,
        "metrics": {
            "generation_time": generation_time,
            "prompt_tokens": prompt_token_count,
            "generated_tokens": generated_token_count,
            "total_tokens": total_tokens,
            "throughput_tokens_per_sec": throughput,
        },
        "trace": trace,
        "all_traces": [trace],
        "total_traces_count": 1,
        "store_logprobs": store_logprobs,
        "config": {
            "model_type": args.model_type,
            "max_tokens": max_new_tokens,
            "context_limit": model_context_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "budget": args.budget,
            "window_size": args.window_size,
        },
        "timestamp": timestamp,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"deepthink_offline_qid{args.qid}_rid{args.rid}_{timestamp}"

    if args.serializer == "pickle":
        result_path = output_dir / f"{base_name}.pkl"
        with result_path.open("wb") as handle:
            pickle.dump(result_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        extension_map = {
            "msgpack_gzip": "msgpack.gz",
            "msgpack_zstd": "msgpack.zst",
            "msgpack_plain": "msgpack",
        }
        compression_map = {
            "msgpack_gzip": "gzip",
            "msgpack_zstd": "zstd",
            "msgpack_plain": "none",
        }
        extension = extension_map[args.serializer]
        result_path = output_dir / f"{base_name}.{extension}"
        try:
            dump_msgpack(
                result_data,
                str(result_path),
                compression=compression_map[args.serializer],
                compression_level=args.compression_level,
            )
        except SerializationError as exc:
            raise SystemExit(f"Failed to serialize result: {exc}")

    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
