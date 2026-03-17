# launch the offline engine

import time
import torch
import sys
import random
# sys.path.append('/tmp/kewan/sglang-dev/sglang/python')
import numpy as np
import sglang as sgl
import asyncio
import os
import json

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    set_seed(42)
    
    compress_algorithm = "RKV" # "RKV"
    compress_max_window = 32
    compress_max_prompt = 128
    compress_divide_length = 64
    compress_divide_method = "newline" # "step_length", "newline"
    max_new_tokens = 16384

    
    llm = sgl.Engine(
        model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        compress_algorithm=compress_algorithm, 
        compress_max_window=compress_max_window,
        compress_max_prompt=compress_max_prompt,
        compress_divide_length=compress_divide_length,
        compress_divide_method=compress_divide_method,
        dtype="bfloat16",
    )

    prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"

    
    test_data = []
    prompts = []
    root_dir = os.path.dirname(os.path.realpath(__file__))
    file_dir = os.path.join(root_dir, "test.jsonl")
    with open(file_dir, 'r') as fp:
        for index, line in enumerate(fp):
            example = json.loads(line)
            prompt = prompt_template.format(**example)
            example["prompt"] = prompt
            prompts.append(prompt)
            test_data.append(example)

    prompts = prompts[:100]

    sampling_params = {"temperature": 0.85, "top_p": 0.95, "max_new_tokens": max_new_tokens}

    # Mesurer le temps avant la génération
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Mesurer le temps après la génération
    end_time = time.time()
    
    # Calculer le temps total
    total_time = end_time - start_time
    
    # Calculer tokens générés
    total_tokens_generated = sum(output["meta_info"]["completion_tokens"] for output in outputs)
    
    # Calculer throughput en tokens par seconde
    throughput_tokens = total_tokens_generated / total_time
    
    # Calculer throughput en requêtes par seconde
    throughput_requests = len(prompts) / total_time
    
    print(f"Temps total d'exécution: {total_time:.2f} secondes")
    print(f"Throughput (tokens/s): {throughput_tokens:.2f}")
    print(f"Throughput (requêtes/s): {throughput_requests:.2f}")

    with open("output.txt") as f:
        f.write(outputs)