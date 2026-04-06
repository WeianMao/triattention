<div align="center">

# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/TODO)
[![Project Page](https://img.shields.io/badge/Project-Page-teal)](https://TODO)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/downloads/)

*Compress KV cache by 10.7x and boost throughput by 2.5x on long reasoning tasks -- with no accuracy loss.*

[Weian Mao](https://scholar.google.com/citations?user=TODO)<sup>1*</sup>,
[Xi Lin](https://scholar.google.com/citations?user=TODO)<sup>3*</sup>,
[Wei Huang](https://aaron-weihuang.com/)<sup>2*</sup>,
[Yuxin Xie](https://scholar.google.com/citations?user=TODO)<sup>1</sup>,
[Tianfu Fu](https://scholar.google.com/citations?user=TODO)<sup>1</sup>,
[Bohan Zhuang](https://scholar.google.com/citations?user=TODO)<sup>3</sup>,
[Song Han](http://songhan.mit.edu/)<sup>1,2</sup>,
[Yukang Chen](https://yukangchen.com/)<sup>2</sup>

<sup>1</sup>MIT, <sup>2</sup>NVIDIA, <sup>3</sup>ZJU &nbsp;&nbsp; <sup>*</sup>Equal contribution

</div>




https://github.com/user-attachments/assets/1d51b186-edd4-443f-ad52-f4520f701a38





## Highlights

- **2.5x throughput** on AIME25 long reasoning while matching Full Attention accuracy (40.8 vs 40.8)
- **10.7x KV memory reduction** with trigonometric frequency-domain compression
- **4 model architectures** verified: Qwen3-8B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B, GPT-OSS-20B

<p align="center">
  <img src="docs/assets/tradeoff.png" width="80%">
</p>
<p align="center"><i>TriAttention achieves 2.5x higher throughput and 10.7x KV memory reduction on AIME25 while matching Full Attention accuracy.</i></p>

## How It Works

Pre-RoPE Q/K vectors in long reasoning models concentrate around fixed centers that determine distance preferences via a trigonometric series. TriAttention scores keys using these centers and norms instead of requiring representative query selection, enabling accurate KV cache compression without the overhead of existing attention-based methods.

## Installation

```bash
git clone https://github.com/TODO/triattention.git
cd triattention
pip install -e .
pip install flash-attn --no-build-isolation  # recommended
```

## Quick Start

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --budget 2048
```

## Supported Models

| Model | HuggingFace ID | Status |
|-------|---------------|--------|
| Qwen3-8B | `Qwen/Qwen3-8B` | Verified |
| DeepSeek-R1-Distill-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Verified |
| DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Verified |
| GPT-OSS-20B | <!-- TODO: Add HF ID --> | Coming soon |

## Results

### AIME24 / AIME25 (KV Budget = 2048, DS-Llama = 512)

| Method | Qwen3-8B | DS-Llama-8B | DS-Qwen-7B | GPT-OSS-20B |
|--------|----------|-------------|-------------|-------------|
| Full Attention | 57.1 / 40.8 | 50.4 / 31.4 | 43.8 / 34.2 | 69.2 / 60.0 |
| SnapKV | 34.6 / 20.0 | 5.0 / 6.7 | 34.6 / 25.0 | 48.3 / 36.7 |
| R-KV | 25.4 / 17.5 | 25.8 / 11.2 | 34.6 / 23.3 | 49.6 / 39.2 |
| **TriAttention** | **42.1 / 32.9** | **33.8 / 19.6** | **42.5 / 30.0** | **59.2 / 49.2** |

### Throughput (Qwen3-8B, tokens/sec)

| Benchmark | TriAttn Budget | Full Acc | TriAttn Acc | Full Throughput | TriAttn Throughput | Speedup |
|-----------|---------------|----------|-------------|-----------------|-------------------|---------|
| MATH-500 | 1024 | 69.6 | 68.4 | 222.8 | 1405.2 | **6.3x** |
| AIME24 | 4096 | 57.1 | 54.6 | 222.8 | 413.9 | **1.9x** |
| AIME25 | 3072 | 40.8 | 40.8 | 222.8 | 563.5 | **2.5x** |

See [docs/results.md](docs/results.md) for complete results including MATH-500 accuracy table, accuracy vs. budget curves, and DFS memory retention analysis.

## vLLM Integration

TriAttention includes a vLLM plugin that enables transparent KV cache compression for production deployment. After installation, vLLM automatically discovers and activates the plugin -- no code changes required.

### Server Mode (OpenAI-Compatible API)

```bash
# Set compression parameters
export TRIATTN_RUNTIME_KV_BUDGET=2048
export TRIATTN_RUNTIME_DIVIDE_LENGTH=128
export TRIATTN_RUNTIME_WINDOW_SIZE=128
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=triattention/vllm/stats/qwen3_32b_int4_stats.pt
export TRIATTN_RUNTIME_PRUNING_MODE=per_head
export TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION=true
export TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM=true

# Launch vLLM server -- TriAttention activates automatically
vllm serve <model_path> \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enforce-eager \
    --trust-remote-code

# Use the standard OpenAI-compatible API
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "<model_path>", "messages": [{"role": "user", "content": "Solve: ..."}]}'
```

### Python API

```python
from triattention.vllm.runtime.integration_monkeypatch import (
    install_vllm_integration_monkeypatches,
)

# Install patches before creating the LLM instance
install_vllm_integration_monkeypatches(patch_scheduler=True, patch_worker=True)

# Standard vLLM API -- compression happens transparently
from vllm import LLM, SamplingParams

llm = LLM(
    model="<model_path>",
    dtype="bfloat16",
    max_model_len=32768,
    enforce_eager=True,
    trust_remote_code=True,
)

outputs = llm.generate(["Your prompt here"], SamplingParams(temperature=0.6, top_p=0.95))
print(outputs[0].outputs[0].text)
```

### Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TRIATTN_RUNTIME_KV_BUDGET` | `2048` | Maximum tokens retained in KV cache per request |
| `TRIATTN_RUNTIME_DIVIDE_LENGTH` | `128` | Compression trigger interval (every N new tokens) |
| `TRIATTN_RUNTIME_WINDOW_SIZE` | `128` | Recent tokens always preserved |
| `TRIATTN_RUNTIME_PRUNING_MODE` | `per_head` | Token selection strategy (`per_head` or `per_layer_per_head`) |
| `TRIATTN_RUNTIME_SPARSE_STATS_PATH` | -- | Path to precomputed frequency statistics `.pt` file |
| `TRIATTN_RUNTIME_PROTECT_PREFILL` | `true` | Protect initial prompt tokens from eviction |
| `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION` | `false` | Enable in-place KV cache compaction |
| `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM` | `false` | Enable freed block reclamation |
| `ENABLE_TRIATTENTION` | `true` | Master switch to enable/disable the plugin |

### Precomputed Statistics

TriAttention requires precomputed Q/K frequency statistics for scoring. We provide pre-calibrated stats for supported models in `triattention/vllm/stats/`. See the [Calibration Guide](docs/calibration.md) for generating stats for custom models.

## Documentation

- [Reproduction Guide](docs/reproduction.md) -- full experiment commands for all benchmarks
- [Calibration Guide](docs/calibration.md) -- generating custom Q/K statistics
- [Full Results](docs/results.md) -- complete tables, figures, and analysis

## Roadmap

- [x] vLLM integration
- [ ] SGLang integration
- [ ] Ollama integration
- [ ] Support for more model architectures

## Citation

```bibtex
@article{mao2026triattention,
    title={TriAttention: Efficient Long Reasoning with Trigonometric KV Compression},
    author={Weian Mao and Xi Lin and Wei Huang and Yuxin Xie and Tianfu Fu and Bohan Zhuang and Song Han and Yukang Chen},
    year={2026},
    eprint={TODO},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Acknowledgements

We thank the following projects for their contributions and inspiration:
[R-KV](https://github.com/Microsoft/R-KV) | [SnapKV](https://github.com/FasterDecoding/SnapKV)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
