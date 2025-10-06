#!/usr/bin/env bash
set -euo pipefail

python development/pkl_read_benchmark.py \
  outputs/deepseek_r1_qwen3_8b/offline \
  --samples 1
