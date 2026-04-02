# TriAttention

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/downloads/)

<!-- TODO: Add paper link -->
<!-- TODO: Add project page link -->
<!-- TODO: Add demo link -->

## TL;DR

TriAttention compresses KV cache using frequency domain analysis of attention patterns, enabling efficient long-context LLM reasoning with minimal accuracy loss. It identifies and preserves the most important key-value pairs by analyzing attention frequency spectra, achieving significant memory savings while maintaining reasoning quality.

<!-- TODO: Add demo video -->

## Method Overview

<!-- TODO: Add method figure -->

## News

- **2025-XX**: Initial release of TriAttention.

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

Run inference with TriAttention KV cache compression:

```bash
python scripts/cli.py \
    --model Qwen/Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --kv-budget 2048
```

## Supported Models

| Model | HuggingFace ID | Verified |
|-------|---------------|----------|
| DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Yes |
| DeepSeek-R1-Distill-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Yes |
| Qwen3-8B | `Qwen/Qwen3-8B` | Yes |

## Results

<!-- TODO: Add result tables from paper -->

## Reproduction

Experiment configs and scripts are in `scripts/experiments/`. Example:

```bash
# Run AIME24 benchmark with TriAttention on Qwen3-8B
python scripts/cli.py \
    --model Qwen/Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --kv-budget 2048

# Run with a different model
python scripts/cli.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset aime24 \
    --method triattention \
    --kv-budget 2048
```

See `scripts/experiments/` for full experiment configurations.

## Roadmap

- [ ] vLLM integration
- [ ] SGLang integration
- [ ] Support for more model architectures

## Citation

<!-- TODO: Add BibTeX -->

## Acknowledgements

We thank the following projects for their contributions and inspiration:

- [ToRA](https://github.com/microsoft/ToRA)
- [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)
- [R-KV](https://github.com/Microsoft/R-KV)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
