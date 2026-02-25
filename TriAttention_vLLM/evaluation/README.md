# TriAttention vLLM Evaluation

当前默认评测入口已经去掉 `V2` 命名；旧 `*_v2_*` 文件仍保留为兼容入口，但不再推荐直接使用。

## 默认入口（推荐）

- Dispatch:
  - `TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py`
- 默认 runner:
  - `TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py`
- 默认配置（AIME24）:
  - `TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24.yaml`
- HF 对齐 per-head anchor 配置:
  - `TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor.yaml`
- 快速对齐脚本:
  - `TriAttention_vLLM/evaluation/scripts/run_hf_alignment_quick.sh`
- sample8 对齐脚本:
  - `TriAttention_vLLM/evaluation/scripts/run_hf_alignment_sample8.sh`

## 快速开始

```bash
# 默认 full pipeline（dispatch -> merge -> eval）
python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
  --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24.yaml

# dry-run（只打印命令）
python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
  --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24.yaml \
  --dry-run
```

```bash
# HF 对齐 quick（默认脚本入口）
TriAttention_vLLM/evaluation/scripts/run_hf_alignment_quick.sh

# sample8 对齐
TriAttention_vLLM/evaluation/scripts/run_hf_alignment_sample8.sh
```

## 单 shard 手工运行（当前默认 runner）

```bash
python TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py \
  --dataset-path R-KV/HuggingFace/data/aime24.jsonl \
  --output-dir TriAttention_vLLM/evaluation/outputs/test \
  --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B \
  --shard-id 0 \
  --num-shards 1 \
  --num-samples 1 \
  --kv-budget 2048 \
  --enable-experimental-kv-compaction true
```

## FullKV baseline（禁用压缩）

```bash
python TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py \
  --dataset-path R-KV/HuggingFace/data/aime24.jsonl \
  --output-dir TriAttention_vLLM/evaluation/outputs/fullkv \
  --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B \
  --shard-id 0 \
  --num-shards 1 \
  --num-samples 1 \
  --disable-compression true
```

## 配置与行为说明（最小版）

- 默认目标模式为 `per_head`（`per_layer` 需显式放行，否则报错）
- 历史 `per_layer` strict 配置文件仍可能存在于仓库中，仅用于回溯，不作为默认实验入口
- 输出字段与 HF/R-KV 兼容风格保持一致；详细字段定义请直接参考：
  - `TriAttention_vLLM/evaluation/runner/vllm_triattention_v2_runner.py`
  - `TriAttention_vLLM/evaluation/eval/eval_math_multi.py`

## 兼容说明

- `vllm_triattention_v2_runner.py` 与 `triattention_v2_*` 配置文件仍可运行（兼容）
- 当前实现内部兼容目录名仍为 `triattention_v2/`（为了降低重构风险），不代表旧版本逻辑

