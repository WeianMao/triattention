# SpecKV Paper Materials Overview

本文件夹包含SpecKV算法的论文撰写材料，基于R-KV项目的代码分析整理。

## 文件索引

| 文件 | 内容 |
|------|------|
| `01_speckv_algorithm.md` | SpecKV算法核心设计与实现 |
| `02_baseline_algorithms.md` | 对比算法(R-KV, FullKV)说明 |
| `03_key_observations.md` | 算法设计的关键观察发现与Motivation |
| `04_experiment_setup.md` | 实验配置与参数说明 |

## 算法定位

SpecKV是一种**基于频率域分析**的KV Cache压缩算法，用于减少长序列生成时的内存占用。

与R-KV（基于注意力分数+相似度）不同，SpecKV利用RoPE的数学性质，通过**预计算的频率统计量**来预测未来token对历史token的注意力权重，从而选择保留哪些KV对。

## 核心文件路径

- **主算法实现**: `R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py`
- **生成流程集成**: `R-KV/weian_development/speckv/rkv_speckv_generate.py`
- **R-KV风格实现**: `R-KV/weian_development/speckv/speckv_rkv_style.py`
- **工具函数**: `R-KV/weian_development/speckv/round_pruning_utils.py`

## 实验入口

- **SpecKV实验**: `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`
- **R-KV对比**: `R-KV/weian_script/aime_sampled8/rkv/aime24/run_rkv_aime24_qwen.sh`
- **FullKV基线**: `R-KV/weian_script/aime24_official_sampled8/run_fullkv_aime24_official_sampled8_qwen.sh`

## 模型与数据集

- **模型**: DeepSeek-R1-Distill-Qwen-7B (使用flash_attention_2, bfloat16)
- **数据集**: AIME24数学推理任务
- **采样配置**: 8次采样，temperature=0.6, top_p=0.95
