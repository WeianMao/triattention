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
import torch.nn.functional as F
try:
    from dynasor.core.evaluator import math_equal
except ImportError:  # fallback when dynasor is not installed
    def math_equal(lhs, rhs):
        return str(lhs).strip() == str(rhs).strip()
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from deepconf.utils import extract_answer
except ImportError:
    def extract_answer(text: str) -> Optional[str]:
        """Simplified boxed-answer extractor when DeepConf isn't installed."""
        if "boxed" not in text:
            return None
        ans = text.split("boxed")[-1]
        if not ans:
            return ""
        if ans[0] != "{":
            return ans.split("$")[0].strip()
        stack = 1
        collected = []
        for char in ans[1:]:
            if char == "{":
                stack += 1
                collected.append(char)
            elif char == "}":
                stack -= 1
                if stack == 0:
                    break
                collected.append(char)
            else:
                collected.append(char)
        return "".join(collected).strip()
try:
    from development.serialization_utils import SerializationError, dump_msgpack
except Exception:  # pragma: no cover - fallback when msgpack or deps missing
    class SerializationError(RuntimeError):
        """Raised when fallback serialization fails."""

    def dump_msgpack(data: Any, path: str, *, compression: str = "gzip", compression_level: int = 3) -> None:
        """Fallback serializer writing JSON when msgpack isn't available."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
from weian_development.process_utils import mask_process_command
from weian_development.hf_offline_runner_sparse.sparse_round_pruner import (
    SparsePruningConfig,
    SparseRoundPruner,
)


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


def build_sampling_components(
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[LogitsProcessorList, LogitsProcessorList]:
    processors = LogitsProcessorList([InfNanRemoveLogitsProcessor()])
    warpers = LogitsProcessorList()
    if not do_sample:
        return processors, warpers
    if temperature and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k and top_k > 0:
        warpers.append(TopKLogitsWarper(top_k))
    if 0 < top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p))
    return processors, warpers


def sample_next_token(
    logits: torch.Tensor,
    sequence: torch.Tensor,
    do_sample: bool,
    processors: LogitsProcessorList,
    warpers: LogitsProcessorList,
) -> torch.Tensor:
    processed = processors(sequence, logits)
    if not do_sample:
        return torch.argmax(processed, dim=-1, keepdim=True)
    warped = warpers(sequence, processed)
    probs = F.softmax(warped, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def run_sparse_generation(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    store_logprobs: bool,
    pruner: SparseRoundPruner,
) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    if max_new_tokens <= 0:
        return input_ids.clone(), [] if store_logprobs else None

    processors, warpers = build_sampling_components(do_sample, temperature, top_p, top_k)
    sequence = input_ids.clone()
    score_list: List[torch.Tensor] = []

    with torch.inference_mode():
        attention_mask = torch.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    pruner.attach_initial_cache(past_key_values)
    past_key_values = pruner.enforce_max_limit(past_key_values)
    past_key_values = pruner.ensure_capacity(past_key_values)

    generated: List[int] = []

    for _ in range(max_new_tokens):
        if store_logprobs:
            score_list.append(logits.detach().to(device="cpu", dtype=torch.float32))
        next_token = sample_next_token(logits, sequence, do_sample, processors, warpers)
        token_id = int(next_token.item())
        generated.append(token_id)
        next_token_tensor = torch.tensor([[token_id]], device=input_ids.device, dtype=torch.long)
        sequence = torch.cat([sequence, next_token_tensor], dim=1)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        with torch.inference_mode():
            position_ids = torch.tensor(
                [[pruner.absolute_position]],
                device=next_token_tensor.device,
                dtype=torch.long,
            )
            outputs = model(
                input_ids=next_token_tensor,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                position_ids=position_ids,
            )

        past_key_values = outputs.past_key_values
        pruner.on_token_appended()
        if pruner.should_start_next_round():
            past_key_values = pruner.start_next_round(past_key_values)

        logits = outputs.logits[:, -1, :]

    if generated:
        gen_tensor = torch.tensor(generated, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        full_sequence = torch.cat([input_ids, gen_tensor], dim=1)
    else:
        full_sequence = input_ids.clone()

    return full_sequence, (score_list if store_logprobs else None)


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
    parser.add_argument(
        "--disable_sparse_pruning",
        action="store_true",
        help="禁用轮次 KV 稀疏裁剪 (默认开启)",
    )
    parser.add_argument(
        "--sparse-stats-path",
        type=Path,
        default=Path("weian_development/hf_offline_runner_sparse/stats/qid0008_trace46_stats.pt"),
        help="export_round_pruning_stats.py 生成的统计文件路径",
    )
    parser.add_argument("--sparse-max-keys", type=int, default=3072, help="每层最多保留的 KV 数 M")
    parser.add_argument("--sparse-round-window", type=int, default=64, help="每轮解码 token 数 W")
    parser.add_argument(
        "--sparse-offset-max-length",
        type=int,
        default=65536,
        help="几何偏移最大长度",
    )
    parser.add_argument(
        "--sparse-score-aggregation",
        choices=["mean", "max"],
        default="mean",
        help="偏移得分聚合方式",
    )
    parser.add_argument(
        "--sparse-head-limit",
        type=int,
        default=None,
        help="可选：限制用于打分的 head 数",
    )
    parser.add_argument(
        "--sparse-seed",
        type=int,
        default=0,
        help="KV 裁剪打分扰动的随机种子",
    )

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
    cuda_available = torch.cuda.is_available()
    if args.tensor_parallel_size != 1:
        raise ValueError(
            "当前稀疏推理实现仅支持 tensor_parallel_size=1；请通过 CUDA_VISIBLE_DEVICES 选择单卡运行"
        )

    device = torch.device("cuda:0" if cuda_available else "cpu")

    sparse_pruner: Optional[SparseRoundPruner] = None
    sparse_enabled = not args.disable_sparse_pruning
    if sparse_enabled:
        if args.sparse_stats_path is None:
            raise ValueError("--sparse-stats-path is required when enabling sparse pruning")
        stats_path = args.sparse_stats_path
        if not stats_path.is_absolute():
            stats_path = (Path.cwd() / stats_path).resolve()
        if not stats_path.exists():
            raise FileNotFoundError(f"Sparse stats file not found: {stats_path}")
        sparse_config = SparsePruningConfig(
            stats_path=stats_path,
            model_path=Path(args.model),
            device=device,
            dtype=torch.float32,
            max_keys=args.sparse_max_keys,
            round_window=args.sparse_round_window,
            offset_max_length=args.sparse_offset_max_length,
            score_aggregation=args.sparse_score_aggregation,
            seed=args.sparse_seed,
            head_limit=args.sparse_head_limit,
        )
        sparse_pruner = SparseRoundPruner(sparse_config)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading HuggingFace model...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
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
    if sparse_pruner is not None:
        sequences, sparse_scores = run_sparse_generation(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            store_logprobs=store_logprobs,
            pruner=sparse_pruner,
        )
        generation_time = time.time() - generation_start
        sequence = sequences[0]
        generated_ids = sequence[prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        score_payload = sparse_scores
    else:
        with torch.inference_mode():
            generated = model.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})
        generation_time = time.time() - generation_start
        sequence = generated.sequences[0]
        generated_ids = sequence[prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        score_payload = generated.scores if store_logprobs else None

    trace = build_trace(
        prompt=prompt,
        output_text=generated_text,
        generated_ids=generated_ids,
        tokenizer=tokenizer,
        store_logprobs=store_logprobs,
        scores=score_payload,
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
        "sparse_pruning": sparse_enabled,
        "timestamp": timestamp,
    }

    if sparse_enabled:
        result_data["sparse_config"] = {
            "stats_path": str(args.sparse_stats_path),
            "max_keys": args.sparse_max_keys,
            "round_window": args.sparse_round_window,
            "offset_max_length": args.sparse_offset_max_length,
            "score_aggregation": args.sparse_score_aggregation,
            "head_limit": args.sparse_head_limit,
            "seed": args.sparse_seed,
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
