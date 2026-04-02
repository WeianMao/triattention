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
[Tianfu Fu](https://scholar.google.com/citations?user=TODO)<sup>4</sup>,
[Bohan Zhuang](https://scholar.google.com/citations?user=TODO)<sup>3</sup>,
[Song Han](http://songhan.mit.edu/)<sup>1,2</sup>,
[Yukang Chen](https://yukangchen.com/)<sup>2</sup>

<sup>1</sup>MIT, <sup>2</sup>NVIDIA, <sup>3</sup>ZJU, <sup>4</sup>xAI &nbsp;&nbsp; <sup>*</sup>Equal contribution

[Paper](https://arxiv.org/abs/TODO) | [Project Page](https://TODO) | [Code](https://github.com/TODO/triattention)

</div>

## Highlights

- **2.5x throughput** on AIME25 long reasoning while matching Full Attention accuracy (40.8 vs 40.8)
- **10.7x KV memory reduction** with trigonometric frequency-domain compression
- **4 model architectures** verified: Qwen3-8B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B, GPT-OSS-20B

<p align="center">
  <img src="assets/tradeoff.png" width="80%">
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

## Documentation

- [Reproduction Guide](docs/reproduction.md) -- full experiment commands for all benchmarks
- [Calibration Guide](docs/calibration.md) -- generating custom Q/K statistics
- [Full Results](docs/results.md) -- complete tables, figures, and analysis

## Roadmap

- [ ] vLLM integration
- [ ] SGLang integration
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
[R-KV](https://github.com/Microsoft/R-KV) | [SnapKV](https://github.com/FasterDecoding/SnapKV) | [StreamingLLM](https://github.com/mit-han-lab/streaming-llm)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
