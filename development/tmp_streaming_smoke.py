import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from streaming_llm.enable_streaming_llm import enable_streaming_llm


@torch.no_grad()
def run_streaming_inference(model, tokenizer, prompt, kv_cache, max_new_tokens):
    model.eval()
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past_key_values = None

    if kv_cache is not None:
        space_needed = input_ids.shape[1] + max_new_tokens
        past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

    outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    generated = []

    next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    for _ in range(max_new_tokens):
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        token_id = next_token.item()
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break

    if generated:
        generated_tensor = torch.tensor([generated], device=device, dtype=input_ids.dtype)
        full = torch.cat([input_ids, generated_tensor], dim=-1)
    else:
        full = input_ids

    text = tokenizer.decode(full[0], skip_special_tokens=True)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="1+1=?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--start-size", type=int, default=4)
    parser.add_argument("--recent-size", type=int, default=2048)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])

    args = parser.parse_args()

    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )

    kv_cache = enable_streaming_llm(model, start_size=args.start_size, recent_size=args.recent_size)
    result = run_streaming_inference(model, tokenizer, args.prompt, kv_cache, args.max_new_tokens)
    print("\n=== 输出 ===\n")
    print(result)


if __name__ == "__main__":
    main()
