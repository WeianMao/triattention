# TriAttention vLLM 使用说明（对外）

本文档给外部使用者说明：如何在本仓库运行带 TriAttention 压缩算法的 vLLM 版本，并复现我们当前常用的 Qwen3 / Qwen3-Coder 路径。

## 1. 代码位置与入口

- TriAttention 代码主目录：`TriAttention_vLLM/`
- 推荐调度入口：`TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py`
- 推荐 runner：`TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py`

## 2. 环境准备

1. 进入仓库根目录（当前文档所在目录）。
2. 使用 conda 环境：
   - 推理/调度：`trivllm`
   - 评测（如需）：`rkv1`

示例：

```bash
conda run -n trivllm python -V
```

## 3. Hugging Face 模型缓存位置（必须）

我们约定 HF 缓存根目录为：

- `HF_HOME=/data/rbg/users/weian/env/huggingface`

请在运行前设置：

```bash
export HF_HOME=/data/rbg/users/weian/env/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
```

注意：
- 不要手工把模型文件直接丢在 `HF_HOME` 根目录。
- 使用 Hugging Face 默认缓存结构（`hub/models--.../snapshots/...`）。

## 4. 模型下载

### 4.1 Qwen3-Coder（常用）

```bash
HF_HOME=/data/rbg/users/weian/env/huggingface \
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct
```

如需 FP8 版本：

```bash
HF_HOME=/data/rbg/users/weian/env/huggingface \
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
```

### 4.2 Qwen3（8B 路径）

当前仓库常用的是本地模型目录（示例）：

- `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B`

## 5. 一键启动（推荐）

### 5.1 Qwen3 对齐实验（多卡分 shard）

```bash
conda run -n trivllm python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
  --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor_qwen3.yaml
```

### 5.2 长 prefill 稳定性（开启 prefill chunk）

```bash
conda run -n trivllm python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
  --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor_qwen3_prefill_chunk.yaml
```

默认关键参数（在 config 内）：
- `kv_budget: 2048`
- `enable_experimental_kv_compaction: true`
- `prefill_auto_chunk: true`
- `prefill_chunk_threshold: 2048`
- `prefill_chunk_size: 2048`

## 6. Qwen3-Coder FP8 运行前置（重要）

TriAttention 压缩路径在 FP8 + Coder 场景需要有效 `sparse_stats_path`。如果缺少 stats，可能报错：
- `TRIATTN_FATAL_TRITON_SCORING_REQUIRED:stats_path_not_set`

可用已有 trace 做 calibration（示例）：

```bash
conda run -n trivllm python R-KV/weian_development/rkv_sparse_round_calibrate.py \
  --trace-root R-KV/outputs/aime_sampled8_qwen3/fullkv/aime24 \
  --model-path /data/rbg/users/weian/env/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/<snapshot_id> \
  --output-path R-KV/outputs/repository/sample8_fullkv_aime24_official_qwen3coder/stats/qwen3_coder_30b_a3b_fp8_plain_stats.pt \
  --num-traces 1 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --kv-budget 2048 \
  --device-map auto
```

然后在 runner/config 中把 `sparse_stats_path` 指向该 `.pt` 文件。

## 7. 如何判断“压缩已激活且运行正常”

建议看两类证据：

1. 启动日志中出现 TriAttention 启用参数，例如：
   - `experimental_compaction=True`
   - `prefill_chunk_size=2048`
   - `enable_chunked_prefill=True`
2. 输出 `run*.jsonl` 的完成状态与关键字段：
   - `status=complete`
   - `enable_experimental_kv_compaction=true`
   - 超长 prefill 样本可完成，不出现 OOM

## 8. 常见问题

1. 单卡跑 30B 初始化 OOM：
   - 优先用 FP8 模型，必要时提升 `tensor_parallel_size`，或降低 `gpu_memory_utilization`。
2. 开启 FP8 后 TriAttention 报 stats 相关错误：
   - 先完成第 6 节 calibration，并确认 `sparse_stats_path` 生效。
3. 需要只使用空闲 GPU：
   - 在 dispatch 命令增加 `--gpus <id列表>`，例如 `--gpus 0,1,2,3`。

## 9. 参考文档

- `TriAttention_vLLM/evaluation/README.md`
- `TriAttention_vLLM/docs/interface/NEXT_STAGE_EXECUTION_TODO.md`
- `TriAttention_vLLM/docs/interface/PREFILL_CHUNK_DEMO_RUNBOOK.md`
