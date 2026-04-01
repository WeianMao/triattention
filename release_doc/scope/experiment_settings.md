# 公布的实验 Setting 矩阵

## 决策状态：部分确认

## 已确认：全部公布（主实验 + 消融）

### Table 1：AIME24 + AIME25（默认 budget）

| 模型 | 数据集 | 方法 | Budget | num_samples | max_length |
|------|--------|------|--------|-------------|------------|
| Qwen3-8B | AIME24, AIME25 | Full/SnapKV/R-KV/TriAttention | 2048 | 8 | 32768 |
| DS-Qwen-7B | AIME24, AIME25 | Full/SnapKV/R-KV/TriAttention | 2048 | 8 | 32768 |
| DS-Llama-8B | AIME24, AIME25 | Full/SnapKV/R-KV/TriAttention | 512 | 8 | 32768 |
| GPT-OSS-20B | AIME24, AIME25 | Full/SnapKV/R-KV/TriAttention | 2048 | 8 | 32768 |

### Table 2：MATH-500

| 模型 | 数据集 | 方法 | Budget | num_samples | max_length |
|------|--------|------|--------|-------------|------------|
| Qwen3-8B | MATH-500 | Full/SnapKV/R-KV/TriAttention | 512 | 1 | 32768 |
| DS-Qwen-7B | MATH-500 | Full/SnapKV/R-KV/TriAttention | 512 | 1 | 32768 |
| DS-Llama-8B | MATH-500 | Full/SnapKV/R-KV/TriAttention | 512 | 1 | 32768 |
| GPT-OSS-20B | MATH-500 | Full/SnapKV/R-KV/TriAttention | 512 | 1 | 32768 |

### Figure 5 A-C：Budget Sweep（Qwen3-8B）

| 模型 | 数据集 | 方法 | Budget | num_samples |
|------|--------|------|--------|-------------|
| Qwen3-8B | AIME24 | Full/SnapKV/R-KV/TriAttention | 512, 1024, 2048, 3072, 4096 | 8 |
| Qwen3-8B | AIME25 | Full/SnapKV/R-KV/TriAttention | 512, 1024, 2048, 3072, 4096 | 8 |
| Qwen3-8B | MATH-500 | Full/SnapKV/R-KV/TriAttention | 512, 1024, 2048, 3072, 4096 | 1 |

### Figure 5 D：DFS Benchmark

| 模型 | 数据集 | 方法 | Budget | 参数 |
|------|--------|------|--------|------|
| Qwen3-8B | DFS/Recursive State Query | Full/R-KV/TriAttention | 2048 | step count: 6,8,10,12,14,16,18,20; 80 samples each |

### Table 3：消融实验

| ID | 消融内容 | 模型 | 数据集 | Budget |
|----|---------|------|--------|--------|
| 3A | w/o S_trig（去掉三角级数分量） | Qwen3-8B | AIME24, AIME25 | 2048 |
| 3B | w/o R weighting（去掉集中度加权） | Qwen3-8B | AIME24, AIME25 | 2048 |
| 3C | Cross-domain calibration（coding vs reasoning 校准数据） | Qwen3-8B | AIME24, AIME25 | 2048 |

### Table 4 + Table 5：吞吐量/对比

使用主实验相同 setting，额外测了吞吐量。不需要独立脚本，复用主实验脚本即可。

## TriAttention 方法配置（从论文和参考脚本提取）

### 核心 flag 组合

| 参数 | 值 | 说明 |
|------|---|------|
| `--per-head-pruning` | True | 每个 KV head 独立选择保留 token |
| `--sparse-normalize-scores` | True | z-score 归一化频域评分 |
| `--rkv-style-compression` | True | 在 attention layer 内触发压缩 |
| `--rkv-style-slack-trigger` | True | 允许 cache 增长到 budget+window 再触发 |
| `--include-prefill-in-budget` | True | prefill token 计入 budget |
| `--divide-length` | 128 | 每 128 个 token 触发一次压缩 |

### 其他参数

| 参数 | 值 |
|------|---|
| `window_size` | 128 |
| `sparse_round_window` | 32 |
| `sparse_offset_max_length` | 65536 |
| `sparse_head_limit` | -1（全部 head） |
| `sparse_seed` | 0 |
| `sparse_score_aggregation` | mean（per-head 模式下不生效） |
| `temperature` | 0.6 |
| `top_p` | 0.95 |
| `load_dtype` | bfloat16 |
| `attn_implementation` | flash_attention_2（GPT-OSS 用 FlashAttention-3 on H100） |
| `seed` | 888 |

### 校准数据

跨数据集校准（避免数据泄漏）：
- 评估 AIME24 时使用 AIME25 的校准 stats
- 评估 AIME25 时使用 AIME24 的校准 stats
- 每个模型有自己的 stats 文件

## 待确认事项

### 1. Figure 5 budget sweep 的 flag 差异

Agent 调查发现：Table 1 和 Figure 5 的 budget sweep 可能用了不同的压缩触发机制：
- Table 1（默认 budget）：`--rkv-style-compression` + `--rkv-style-slack-trigger`
- Figure 5（budget sweep）：可能用了 `--rkv-aligned-budget`（精确 budget 对齐）

这两个机制的 KV cache 峰值不同（~2176 vs ~2080，详见 [../execution/10_technical_notes.md](../execution/10_technical_notes.md)）。

**需要确认**：Figure 5 实际用的是哪套 flag？release 时是否统一为一种？

### 2. GPT-OSS-20B 的实验脚本

GPT-OSS 实验由协作者在 gptoss 分支上运行。本地 main 分支上没有对应的脚本。release 时需要从 gptoss 分支提取或重建。

**已确认**：模型是 GPT-OSS-20B（`openai/gpt-oss-20b`），不是 120B。HuggingFace 上可下载。

### 3. DFS Benchmark 代码来源

DFS 测试的数据生成/评估代码在 linxi-dev 分支的 `R-KV/linxi_development/AQA-Bench/dfs_state_query/` 下。
需要从该分支提取并整合到 release 中。

**需要在 release 前做一次代码审查**：确保 DFS 测试逻辑正确、学术合规。

### 4. 数据集

已确认 4 个数据集，全部自动下载：
- AIME24: `HuggingFaceH4/aime_2024`
- AIME25: `MathArena/aime_2025`
- MATH-500: `HuggingFaceH4/MATH-500`
- DFS/Recursive State Query: 需要公布生成脚本（来自 linxi-dev 分支）

### 5. 实验框架选择

两套脚本系统：
- `weian_script/` — 用户自己的脚本，产出了论文实际结果
- `speckv_experiments/` — 协作者 Xi Lin 的框架，更规范但覆盖不全

**待确认**：以哪套为基础整理 release 的实验脚本。建议以 speckv_experiments 为基础（更规范），补充缺失的方法和模型支持。
