#!/usr/bin/env python3
"""Simple test to debug vLLM patch execution error."""
import sys
from pathlib import Path

# Add vLLM fork first, then TriAttention project
VLLM_ROOT = Path(__file__).resolve().parents[2] / "R-KV" / "vLLM"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIAN_DEV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VLLM_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))
sys.path.insert(2, str(WEIAN_DEV))

from vllm import LLM, SamplingParams
from triattention import TriAttentionConfig, TriAttentionWrapper, patch_vllm_attention

# Create config with real stats
stats_path = Path("/data/rbg/users/weian/project/rl/dc/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt")
config = TriAttentionConfig(
    kv_budget=256,
    divide_length=32,
    pruning_mode='per_head',
    stats_path=stats_path,
)
wrapper = TriAttentionWrapper(config)

# Load Qwen model
print("Loading Qwen model...")
llm = LLM(
    model='/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B',
    dtype='float16',
    gpu_memory_utilization=0.95,  # Increased for KV cache
    max_model_len=256,  # Reduced to minimal for testing
    enforce_eager=True,
    trust_remote_code=True,
)

# Patch attention
print("Patching attention...")
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
patch_vllm_attention(model, wrapper)

# Test generation
print("Running generation...")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)
outputs = llm.generate(['Hello, my name is'], sampling_params)
print('Generated:', outputs[0].outputs[0].text)
print('Patched:', wrapper._patched)
print('Success!')
