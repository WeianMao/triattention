# Experiment Setup

## 1. 模型配置

### 基础模型

```yaml
model_path: DeepSeek-R1-Distill-Qwen-7B
attn_implementation: flash_attention_2
load_dtype: bfloat16
max_length: 32768
```

### 模型特点

- **架构**: Qwen2 based (28 layers)
- **RoPE**: 使用Yarn RoPE scaling
- **GQA**: Grouped Query Attention (num_attention_heads > num_key_value_heads)

---

## 2. 采样配置

### 共享配置

```yaml
eval_batch_size: 1
num_samples: 8        # 每题采样8次
temperature: 0.6
top_p: 0.95
seed: 888             # SpecKV/R-KV共享
```

### FullKV差异

```yaml
seed: 666             # FullKV使用不同seed
```

---

## 3. SpecKV特定配置

### 核心参数

```yaml
method: speckv
kv_budget: 2048
window_size: 128
sparse_round_window: 32
sparse_score_aggregation: mean
sparse_head_limit: -1           # 使用所有sampled heads
sparse_seed: 0
use_chat_template: false
```

### 统计文件

```yaml
# 使用AIME25的统计数据避免数据泄露
sparse_stats_path: R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt
sparse_offset_max_length: 65536
```

### 命令行额外参数

```bash
--sparse-normalize-scores     # z-score标准化
--include-prefill-in-budget   # prefill计入budget
--rkv-style-compression       # R-KV风格压缩
--rkv-style-slack-trigger     # 触发时机对齐
--divide-length 128           # 压缩间隔
--per-head-pruning            # per-head独立裁剪
```

---

## 4. R-KV特定配置

```yaml
method: rkv
kv_budget: 2048
# 无window_size/sparse相关参数
# 默认mix_lambda=0.1
```

---

## 5. FullKV配置

```yaml
method: fullkv
kv_budget: null  # 无压缩
```

---

## 6. 数据集

### AIME24

```yaml
dataset_path: R-KV/HuggingFace/data/aime24.jsonl
```

格式: JSONL，每行包含 `{"question": "...", "answer": "..."}`

---

## 7. 实验脚本

### 启动命令结构

```bash
python3 R-KV/weian_development/rkv_sharded_dispatch.py \
  --config <config.yaml> \
  --method-output-dir <output_dir> \
  --log-dir <log_dir> \
  --output-dir <shards_dir> \
  [--additional-flags]
```

### 分片执行

使用`rkv_sharded_dispatch.py`进行多GPU分片执行：
- 自动检测可用GPU
- 每个shard处理部分样本
- 结果合并到merged目录

### 评测

使用`R-KV/HuggingFace/evaluation/eval_math_multi.py`进行评测。

---

## 8. 输出结构

```
R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perhead/
├── shards/
│   ├── shard_0/
│   ├── shard_1/
│   └── ...
├── merged/
│   └── all_outputs.jsonl
└── stats/
    └── deepseek_r1_qwen7b_plain_stats.pt
```

---

## 9. 环境配置

### Conda环境

```bash
conda activate rkv
```

### Python路径

```bash
export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
```

### 进程命名

```bash
export VLLM_PROCESS_NAME_PREFIX="PD-L1_binder"
```
