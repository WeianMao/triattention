"""离线 StreamingLLM (HuggingFace) 推理示例（支持可选序列化格式）"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
try:  # Transformers 4.56+
    from transformers.utils import (
        is_flash_attn_2_available,
        is_flash_attn_3_available,
    )
except ImportError:  # pragma: no cover
    def is_flash_attn_2_available() -> bool:  # type: ignore
        return False

    def is_flash_attn_3_available() -> bool:  # type: ignore
        return False

from dynasor.core.evaluator import math_equal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepconf.outputs import DeepThinkOutput
from deepconf.utils import (
    compute_all_voting_results,
    extract_answer,
)
from development.serialization_utils import SerializationError, dump_msgpack
from streaming_llm.enable_streaming_llm import enable_streaming_llm


@dataclass
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int


def quick_parse(text: str) -> str:
    """Parse LaTeX-like boxed answers to plain text."""
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
    answer = quick_parse(answer)
    if (
        len(answer) == 1
        and answer.isalpha()
        and len(ground_truth) == 1
        and ground_truth.isalpha()
    ):
        return answer.lower() == ground_truth.lower()
    return math_equal(answer, ground_truth)


def evaluate_voting_results(voting_results, ground_truth):
    evaluation = {}
    for method, result in voting_results.items():
        if result and result.get("answer"):
            try:
                is_correct = equal_func(result["answer"], ground_truth)
            except Exception:  # noqa: BLE001
                is_correct = str(result["answer"]) == str(ground_truth)

            evaluation[method] = {
                "answer": result["answer"],
                "is_correct": is_correct,
                "confidence": result.get("confidence"),
                "num_votes": result.get("num_votes", 0),
            }
        else:
            evaluation[method] = {
                "answer": None,
                "is_correct": False,
                "confidence": None,
                "num_votes": 0,
            }
    return evaluation


def print_evaluation_report(question, ground_truth, evaluation, result: DeepThinkOutput):
    print("\n=== Evaluation Report ===")
    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Total traces generated: {result.total_traces_count}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Generation time: {result.generation_time:.2f}s")
    if result.generation_time > 0:
        print(f"Generation throughput: {result.total_tokens / result.generation_time:.1f} tokens/second")

    correct_traces = sum(
        1
        for trace in result.all_traces
        if trace.get("extracted_answer")
        and equal_func(trace["extracted_answer"], ground_truth)
    )
    total_valid_traces = sum(1 for trace in result.all_traces if trace.get("extracted_answer"))
    if total_valid_traces > 0:
        trace_accuracy = correct_traces / total_valid_traces
        print(f"Individual trace accuracy: {correct_traces}/{total_valid_traces} ({trace_accuracy:.1%})")

    print("\n=== Voting Method Results ===")
    print("-" * 80)
    print(f"{'Method':<25} {'Answer':<20} {'Correct':<8} {'Confidence':<12} {'Votes':<6}")
    print("-" * 80)

    correct_methods = []
    for method, eval_result in evaluation.items():
        answer = eval_result["answer"]
        answer_str = (
            str(answer)[:18] + "..." if answer is not None and len(str(answer)) > 20 else str(answer)
        )
        is_correct = eval_result["is_correct"]
        confidence = eval_result["confidence"]
        num_votes = eval_result["num_votes"]

        correct_symbol = "✓" if is_correct else "✗"
        conf_str = f"{confidence:.3f}" if confidence is not None else "-"

        print(
            f"{method:<25} {answer_str:<20} {correct_symbol:<8} {conf_str:<12} {num_votes:<6}"
        )
        if is_correct:
            correct_methods.append(method)

    print(f"\nCorrect voting methods: {correct_methods}")
    correct_evals = {k: v for k, v in evaluation.items() if v["is_correct"]}
    if correct_evals:
        best_method = max(
            correct_evals.items(),
            key=lambda item: item[1]["confidence"] if item[1]["confidence"] is not None else 0,
        )
        print(
            f"Best correct method: {best_method[0]} (confidence: {best_method[1]['confidence']:.3f})"
        )
    print(
        f"Method accuracy: {len(correct_methods)}/{len(evaluation)} ({len(correct_methods)/max(1, len(evaluation)):.1%})"
    )

def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    """Prepare prompt for DeepSeek-style models."""
    if model_type == "deepseek":
        messages = [
            {
                "role": "system",
                "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n",
            },
            {"role": "user", "content": question},
        ]
    else:
        messages = [{"role": "user", "content": question}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_prompt_gpt(question: str, tokenizer, reasoning_effort: str = "high") -> str:
    """Prepare prompt for GPT-style models with reasoning-effort control."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True,
    )


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Filter logits using top-k and/or top-p (nucleus) sampling constraints."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_mask = cumulative_probs <= top_p
        sorted_mask[..., 0] = True
        mask = torch.full_like(logits, False, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

    return logits


def sample_next_token(
    logits: torch.Tensor,
    sampling: SamplingConfig,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, float, Dict[str, Dict[str, float]]]:
    """Sample a token ID from logits under the given sampling config."""
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    elif logits.dim() == 1:
        logits = logits.unsqueeze(0)
    temperature = max(1e-6, sampling.temperature)
    scaled_logits = logits / temperature
    filtered_logits = top_k_top_p_filtering(scaled_logits, sampling.top_k, sampling.top_p)
    log_probs = torch.nn.functional.log_softmax(filtered_logits, dim=-1)
    probs = torch.exp(log_probs)

    token_id = torch.multinomial(probs, num_samples=1, generator=generator).item()
    logprob = log_probs[0, token_id].item()

    logprob_entry = {str(token_id): {"logprob": logprob}}
    return token_id, logprob, logprob_entry


def generate_streaming_trace(
    model,
    tokenizer,
    kv_cache,
    prompt: str,
    sampling: SamplingConfig,
    eos_token_id: int,
    generator: Optional[torch.Generator] = None,
    store_logprobs: bool = False,
) -> Dict:
    """Generate a single trace using HuggingFace + StreamingLLM."""
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    past_key_values = None
    generated_ids: List[int] = []
    logprob_records: List[Dict[str, Dict[str, float]]] = []
    confs: List[float] = []
    stop_reason = "length"

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        if kv_cache is not None:
            past_key_values = kv_cache(past_key_values)
        logits = outputs.logits[:, -1:, :]

        for _ in range(sampling.max_new_tokens):
            token_id, logprob, logprob_entry = sample_next_token(
                logits,
                sampling,
                generator=generator,
            )

            confs.append(float(-logprob))
            generated_ids.append(token_id)
            if store_logprobs:
                logprob_records.append(logprob_entry)

            if token_id == eos_token_id:
                stop_reason = "eos"
                break

            next_input = torch.tensor([[token_id]], device=device, dtype=input_ids.dtype)
            if kv_cache is not None:
                past_key_values = kv_cache.evict_for_space(past_key_values, 1)

            outputs = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
            logits = outputs.logits[:, -1:, :]

        else:
            stop_reason = "length"

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    token_ids = generated_ids

    trace = {
        "stop_reason": stop_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids),
        "confs": confs,
        "extracted_answer": extract_answer(text),
    }

    if store_logprobs:
        trace["logprobs"] = logprob_records

    return trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StreamingLLM (HuggingFace) Offline Example")
    parser.add_argument("--model", type=str, required=True, help="模型路径或名称")
    parser.add_argument("--dataset", type=str, default="aime25.jsonl", help="数据集 JSONL 文件")
    parser.add_argument("--qid", type=int, required=True, help="需要处理的问题 ID (0-based)")
    parser.add_argument("--rid", type=str, default="offline_streaming", help="运行标识")
    parser.add_argument("--budget", type=int, default=64, help="生成的 trace 数量")
    parser.add_argument("--window_size", type=int, default=2048, help="置信度窗口大小（保留字段）")
    parser.add_argument("--max_tokens", type=int, default=32000, help="最大生成 token 数")
    parser.add_argument("--model_type", type=str, default="deepseek", choices=["deepseek", "gpt"], help="prompt 模式")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="gpt prompt reasoning effort")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling 参数")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k 采样参数 (0 表示不启用)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="结果输出目录")
    parser.add_argument("--no_multiple_voting", action="store_true", help="是否跳过多投票分析")
    parser.add_argument("--no_store_logprobs", action="store_true", help="是否跳过保存原始 logprobs")
    parser.add_argument("--serializer", choices=["pickle", "msgpack_gzip", "msgpack_zstd", "msgpack_plain"], default="msgpack_gzip", help="序列化方式")
    parser.add_argument("--compression_level", type=int, default=3, help="压缩等级 (gzip/zstd)")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto", help="模型加载使用的精度")
    parser.add_argument("--device_map", default="auto", help="传递给 AutoModelForCausalLM.from_pretrained 的 device_map")
    parser.add_argument("--start_size", type=int, default=4, help="StreamingLLM 起始缓存保留长度")
    parser.add_argument("--recent_size", type=int, default=2048, help="StreamingLLM 最近缓存窗口长度")
    parser.add_argument("--seed", type=int, default=None, help="可选随机种子")
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="auto",
        choices=["auto", "flash_attn2", "flash_attn3", "sdpa", "eager"],
        help="指定 Transformers 注意力后端，auto 表示保持默认设置",
    )
    return parser.parse_args()


def load_tokenizer(model_name: str) -> Tuple[AutoTokenizer, float]:
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, time.time() - start


def load_model(model_name: str, torch_dtype: Optional[torch.dtype], device_map: str):
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    load_time = time.time() - start
    return model, load_time


def set_process_name(name: str) -> None:
    """Attempt to change the process name shown in system monitors."""
    if os.environ.get("PD_L1_AFFINITY_ALIAS") != "1":
        return
    try:
        from setproctitle import setproctitle

        setproctitle(name)
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode("utf-8"), 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    args = parse_args()

    total_start = time.time()

    set_process_name("PD_L1_affinity")

    if args.attention_backend != "auto":
        backend_alias = {
            "flash_attn2": "flash_attention_2",
            "flash_attn3": "flash_attention_3",
            "sdpa": "sdpa",
            "eager": "eager",
        }
        target_backend = backend_alias.get(args.attention_backend)
        if target_backend is None:
            raise ValueError(f"Unsupported attention backend: {args.attention_backend}")
        if target_backend == "flash_attention_2" and not is_flash_attn_2_available():
            raise RuntimeError(
                "flash_attention_2 backend requested，但当前环境未检测到 flash-attn v2，请先安装对应库。"
            )
        if target_backend == "flash_attention_3" and not is_flash_attn_3_available():
            raise RuntimeError(
                "flash_attention_3 backend requested，但当前环境未检测到 flash-attn v3，请先安装对应库。"
            )
        os.environ["TRANSFORMERS_ATTENTION_BACKEND"] = target_backend

    tokenizer, tokenizer_init = load_tokenizer(args.model)

    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]
    model, llm_init = load_model(args.model, torch_dtype, args.device_map)
    model.eval()

    kv_cache = enable_streaming_llm(model, start_size=args.start_size, recent_size=args.recent_size)

    store_logprobs = not args.no_store_logprobs

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens,
    )

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        data = [json.loads(line.strip()) for line in handle if line.strip()]

    if args.qid < 0 or args.qid >= len(data):
        raise ValueError(f"Question ID {args.qid} is out of range (0-{len(data) - 1})")

    question_entry = data[args.qid]
    question = question_entry["question"]
    ground_truth = str(question_entry.get("answer", "")).strip()

    if args.model_type == "gpt":
        prompt = prepare_prompt_gpt(question, tokenizer, args.reasoning_effort)
    else:
        prompt = prepare_prompt(question, tokenizer, args.model_type)

    traces: List[Dict] = []
    total_tokens = 0

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if eos_token_id is None:
        raise ValueError("无法确定模型的 EOS token id，请检查 tokenizer 配置。")

    gen_start = time.time()
    for trace_idx in trange(args.budget, desc="Streaming traces", unit="trace"):
        trace = generate_streaming_trace(
            model,
            tokenizer,
            kv_cache,
            prompt,
            sampling,
            eos_token_id=eos_token_id,
            generator=generator,
            store_logprobs=store_logprobs,
        )
        traces.append(trace)
        total_tokens += trace["num_tokens"]
    generation_time = time.time() - gen_start

    output = DeepThinkOutput()
    output.mode = "offline"
    output.llm_init_time = llm_init
    output.tokenizer_init_time = tokenizer_init
    output.generation_time = generation_time
    output.total_time = time.time() - total_start
    output.total_traces_count = len(traces)
    output.all_traces = traces
    output.final_traces = traces
    output.total_tokens = total_tokens
    output.avg_tokens_per_trace = float(total_tokens / max(1, len(traces)))
    output.warmup_tokens = 0
    output.final_tokens = total_tokens
    output.avg_tokens_per_warmup_trace = 0.0
    output.avg_tokens_per_final_trace = output.avg_tokens_per_trace
    output.processing_time = 0.0
    output.config = {
        "model": args.model,
        "mode": "offline",
        "budget": args.budget,
        "window_size": args.window_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "start_size": args.start_size,
        "recent_size": args.recent_size,
        "max_tokens": args.max_tokens,
        "model_type": args.model_type,
    }

    answers: List[str] = []
    weights: List[float] = []
    for trace in traces:
        ans = trace.get("extracted_answer")
        if ans:
            answers.append(ans)
            confs = trace.get("confs") or []
            weight = float(sum(confs) / len(confs)) if confs else 0.0
            weights.append(weight)
    output.voting_answers = answers
    output.voting_weights = weights

    if not args.no_multiple_voting:
        output.voting_results = compute_all_voting_results(traces)
        majority = output.voting_results.get("majority") if output.voting_results else None
        if majority and majority.get("answer"):
            output.voted_answer = majority["answer"]
            output.final_answer = output.voted_answer
        elif traces and traces[0].get("extracted_answer"):
            output.final_answer = traces[0]["extracted_answer"]
    elif traces and traces[0].get("extracted_answer"):
        output.final_answer = traces[0]["extracted_answer"]

    result_data = output.to_dict()
    result_data.update(
        {
            "question": question,
            "ground_truth": ground_truth,
            "qid": args.qid,
            "run_id": args.rid,
            "store_logprobs": store_logprobs,
        }
    )

    if ground_truth and output.voting_results:
        evaluation = evaluate_voting_results(output.voting_results, ground_truth)
        print_evaluation_report(question, ground_truth, evaluation, output)
        result_data["evaluation"] = evaluation

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"deepthink_offline_qid{args.qid}_rid{args.rid}_{timestamp}"

    if args.serializer == "pickle":
        result_path = output_dir / f"{base_name}.pkl"
        import pickle

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
        result_path = output_dir / f"{base_name}.{extension_map[args.serializer]}"
        try:
            dump_msgpack(
                result_data,
                str(result_path),
                compression=compression_map[args.serializer],
                compression_level=args.compression_level,
            )
        except SerializationError as exc:
            raise SystemExit(f"Failed to serialize result: {exc}") from exc

    print("\n=== 运行总结 ===")
    print(f"模型路径: {args.model}")
    print(f"问题编号: {args.qid}")
    print(f"生成 trace 数: {len(traces)}")
    print(f"总生成 token: {total_tokens}")
    if generation_time > 0:
        print(f"生成速率: {total_tokens / generation_time:.1f} token/s")
    print(f"结果文件: {result_path}")


if __name__ == "__main__":
    main()
