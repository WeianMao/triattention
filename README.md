<div align="center">

# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

<!-- TODO: Add paper link badge once arXiv is available -->
[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/TODO)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/downloads/)

*Compress KV cache by 10.7x and boost throughput by 2.5x on long reasoning tasks -- with no accuracy loss.*

[Weian Mao](https://scholar.google.com/citations?user=TODO)<sup>1*</sup>,
[Xi Lin](https://scholar.google.com/citations?user=TODO)<sup>2*</sup>,
[Wei Huang](https://aaron-weihuang.com/)<sup>3*</sup>,
[Yuxin Xie](https://scholar.google.com/citations?user=TODO)<sup>1</sup>,
[Tianfu Fu](https://scholar.google.com/citations?user=TODO)<sup>4</sup>,
[Bohan Zhuang](https://scholar.google.com/citations?user=TODO)<sup>2</sup>,
[Song Han](http://songhan.mit.edu/)<sup>1,3</sup>,
[Yukang Chen](https://yukangchen.com/)<sup>3</sup>

<sup>1</sup>MIT, <sup>2</sup>ZJU, <sup>3</sup>NVIDIA, <sup>4</sup>xAI &nbsp;&nbsp; <sup>*</sup>Equal contribution

[Paper](https://arxiv.org/abs/TODO) | [Code](https://github.com/TODO/triattention) | [Demo](https://TODO)
<!-- TODO: Add actual URLs when available -->

</div>

## Highlights

- **2.5x throughput** on AIME25 long reasoning while matching Full Attention accuracy (40.8 vs 40.8)
- **10.7x KV memory reduction** with trigonometric frequency-domain compression
- **4 model architectures** verified: Qwen3-8B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B, GPT-OSS-20B

<!-- TODO: Add Figure 4 (method overview) -->

## Table of Contents

1. [News](#news)
2. [Method](#method)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Supported Models](#supported-models)
6. [Results](#results)
7. [Reproduction](#reproduction)
8. [Calibration](#calibration)
9. [Roadmap](#roadmap)
10. [Citation](#citation)
11. [Acknowledgements](#acknowledgements)
12. [License](#license)

## News

- [x] [2026.4] Initial release of TriAttention codebase.

## Method

<!-- TODO: Add method figure (Figure 4 from paper) -->

Pre-RoPE Q/K vectors in long reasoning models concentrate around fixed centers. These centers determine distance preferences via a trigonometric series. TriAttention exploits this structure: it scores keys using the pre-RoPE centers and norms instead of requiring representative query selection, enabling accurate KV cache compression without the overhead of existing attention-based methods.

## Installation

```bash
git clone https://github.com/TODO/triattention.git
cd triattention
pip install -e .
```

> **Note:** For best performance, install [FlashAttention](https://github.com/Dao-AILab/flash-attention):
> ```bash
> pip install flash-attn --no-build-isolation
> ```

## Quick Start

**CLI**

```bash
python scripts/cli.py \
    --model Qwen/Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --kv-budget 2048
```

**Python API**

```python
from triattention import TriAttentionModel

model = TriAttentionModel("Qwen/Qwen3-8B", kv_budget=2048)
output = model.generate("Solve this step by step: ...")
print(output)
```

<!-- TODO: Verify Python API matches actual implementation -->

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

### MATH-500 (KV Budget = 512)

| Method | Qwen3-8B | DS-Llama-8B | DS-Qwen-7B | GPT-OSS-20B |
|--------|----------|-------------|-------------|-------------|
| Full Attention | 69.6 | 82.4 | 87.0 | 91.4 |
| SnapKV | 49.2 | 65.5 | 66.4 | 68.2 |
| R-KV | 46.4 | 76.9 | 71.6 | 77.4 |
| **TriAttention** | **56.0** | **80.6** | **79.6** | **81.2** |

### Throughput (Qwen3-8B, tokens/sec)

| Benchmark | TriAttn Budget | Full Acc | TriAttn Acc | Full Throughput | TriAttn Throughput | Speedup |
|-----------|---------------|----------|-------------|-----------------|-------------------|---------|
| MATH-500 | 1024 | 69.6 | 68.4 | 222.8 | 1405.2 | **6.3x** |
| AIME24 | 4096 | 57.1 | 54.6 | 222.8 | 413.9 | **1.9x** |
| AIME25 | 3072 | 40.8 | 40.8 | 222.8 | 563.5 | **2.5x** |

<!-- TODO: Add DFS memory benchmark results -->

## Reproduction

Experiment configs and scripts are in `scripts/experiments/`. Example:

```bash
# AIME24 with TriAttention on Qwen3-8B
python scripts/cli.py \
    --model Qwen/Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --kv-budget 2048

# MATH-500 with DeepSeek-R1-Distill-Qwen-7B
python scripts/cli.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset math500 \
    --method triattention \
    --kv-budget 512

# Compare against baselines
python scripts/cli.py \
    --model Qwen/Qwen3-8B \
    --dataset aime25 \
    --method full \
    --kv-budget 2048
```

See `scripts/experiments/` for full experiment configurations.

## Calibration

TriAttention uses pre-computed statistics (Q/K centers and norms) for each model. To generate stats for a custom model:

```bash
python scripts/calibrate.py \
    --model <your-model-id> \
    --calibration-data <your-data.jsonl> \
    --output-dir stats/
```

<!-- TODO: Verify calibration script path and arguments -->

## Roadmap

- [ ] vLLM integration
- [ ] SGLang integration
- [ ] Support for more model architectures
- [ ] GPT-OSS-20B public release

## Citation

```bibtex
<!-- TODO: Add BibTeX once paper is on arXiv -->
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

- [R-KV](https://github.com/Microsoft/R-KV)
- [SnapKV](https://github.com/FasterDecoding/SnapKV)
- [StreamingLLM](https://github.com/mit-han-lab/streaming-llm)
- [ToRA](https://github.com/microsoft/ToRA)
- [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
