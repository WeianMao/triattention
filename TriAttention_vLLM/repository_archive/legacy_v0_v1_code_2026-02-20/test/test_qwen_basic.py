#!/usr/bin/env python3
"""Basic test to verify Qwen model loads and generates with vLLM."""
import sys
from pathlib import Path

# Add vLLM fork first
VLLM_ROOT = Path(__file__).resolve().parents[2] / "R-KV" / "vLLM"
WEIAN_DEV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VLLM_ROOT))
sys.path.insert(1, str(WEIAN_DEV))

from vllm import LLM, SamplingParams

# Load Qwen model
print("Loading Qwen-7B model...")
llm = LLM(
    model='/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B',
    dtype='float16',
    gpu_memory_utilization=0.8,
    max_model_len=512,
    enforce_eager=True,
    trust_remote_code=True,
)

# Test generation
print("Running generation...")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)
outputs = llm.generate(['Hello, my name is'], sampling_params)
print(f'Generated: {outputs[0].outputs[0].text}')
print('Success!')
