# 硬编码路径替换方案

## 决策状态：已确认

## 替换策略总表

| 路径类型 | 替换为 | 示例 |
|---------|--------|------|
| 本地 model 路径 | HuggingFace hub 名 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| 数据集路径 | 相对路径 | `./data/brumo_2025.jsonl` |
| 缓存路径 | 环境变量 + 默认值 | `${VLLM_CACHE_DIR:-.cache/vllm}` |
| .gitignore | 重写为相对路径 | `R-KV/outputs/` |

## 不需要处理的目录（已排除在 release 范围外）

| 目录 | 原因 |
|------|------|
| `weian_development/` | 不 release，个人开发工具 |
| `repository_archive/` | 不 release，历史归档 |
| `LazyEviction/data_processing/` | 不在第一阶段 release 范围 |
| `R-KV-backup-*` | 备份目录，不 release |
| `paper_visualizations/` | 不 release |

## 详细调查结果

以下是 agent 对代码库的完整扫描结果，列出所有需要替换的硬编码路径。

### 类型 A：Model 路径

需要替换为 HuggingFace hub 名称。

**Shell 脚本（scripts/）**：

| 文件 | 行号 | 当前路径 | 替换为 |
|------|------|---------|--------|
| `scripts/run_freq_magnitude_plots.sh` | 9 | `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` 或用 `${MODEL_PATH:-...}` |

**YAML 配置（scripts/configs/）**：

以下 4 个 YAML 文件都有相同的硬编码路径，需要统一替换：
- `deepseek_r1_qwen3_8b.yaml`
- `deepseek_r1_qwen3_8b_32trace.yaml`
- `deepseek_r1_qwen3_8b_32trace_debug.yaml`
- `deepseek_r1_qwen3_8b_64trace.yaml`

每个文件中的路径：
| 行号 | 当前路径 | 替换为 |
|------|---------|--------|
| 2 | `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` |
| 28 | `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` |

**Python 文件（将被 release 的）**：

| 文件 | 行号 | 当前路径 | 替换为 |
|------|------|---------|--------|
| `TriAttention_vLLM/benchmarks/reasoning/verify_compression.py` | 27 | `/data/rbg/users/weian/project/rl/datasets/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B` |
| `TriAttention_vLLM/benchmarks/reasoning/validate_setup.py` | 27 | `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| `TriAttention_vLLM/benchmarks/reasoning/test_initialization.py` | 27 | `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

注意：TriAttention_vLLM 暂不在第一阶段 release，但最终也需要清理。

### 类型 B：数据集路径

需要替换为相对路径。

**YAML 配置中的数据集引用**（同上 4 个 YAML 文件）：

| 行号 | 当前路径 | 替换为 |
|------|---------|--------|
| 19, 23 | `/data/rbg/users/weian/project/rl/deepconf/brumo_2025.jsonl` | `./data/brumo_2025.jsonl` |
| 31 | `/data/rbg/users/weian/project/rl/deepconf/aime25.jsonl` | `./data/aime25.jsonl` |

### 类型 C：缓存/工具路径

需要替换为环境变量 + 默认值。

**Shell 脚本中的 VLLM cache 路径**：

以下脚本都在第 8 行设置了 `VLLM_TORCH_COMPILE_CACHE`：

| 文件 | 当前值 | 替换为 |
|------|--------|--------|
| `scripts/yaml_runs/run_online_baseline_deepseek.sh` | `/data/rbg/users/weian/project/rl/cache/vllm` | `${VLLM_CACHE_DIR:-.cache/vllm}` |
| `scripts/yaml_runs/run_online.sh` | 同上 | 同上 |
| `scripts/yaml_runs/run_online_baseline.sh` | 同上 | 同上 |
| `scripts/yaml_runs/run_offline.sh` | 同上 | 同上 |
| `scripts/yaml_runs/run_offline_deepseek.sh` | 同上 | 同上 |
| `scripts/yaml_runs/run_online_deepseek.sh` | 同上 | 同上 |
| `scripts/yaml_runs_serialized/run_offline_deepseek_msgpack.sh` | 同上 | 同上 |
| `scripts/yaml_runs_serialized/run_offline_deepseek_msgpack_64.sh` | 同上 | 同上 |
| `scripts/yaml_runs_serialized/run_offline_deepseek_serialized.sh` | 同上 | 同上 |

### 类型 D：.gitignore

当前 `.gitignore` 引用了绝对路径（且指向旧 `/dc` 而非 `/dc1`）：

```
# 当前（错误的）
/data/rbg/users/weian/project/rl/dc/R-KV/outputs/
/data/rbg/users/weian/project/rl/dc/R-KV/logs/

# release 版本（重写）
R-KV/outputs/
R-KV/logs/
```

release 时 `.gitignore` 会整体重写。

### 类型 E：评估管线中的路径（需在 07_evaluation.md 中清理）

| 文件 | 问题 |
|------|------|
| `evaluation/model_utils.py` L448 | `../models/codellama_7b/v1-16k`（相对路径，低优先级） |
| `rm_maj_eval.py` `__main__` | 硬编码内部目录路径 |
| `run_math.py` | `import weian_development.*` 引用 |

### 类型 F：启动器中的路径

详见 [../components/08_launcher.md](../components/08_launcher.md)：
- `rkv_sharded_eval.py` 中 PYTHONPATH 添加了 `/R-KV` 和 `/R-KV/HuggingFace`
- `rkv_sharded_dispatch.py` 中可能有内部路径引用

## 执行要点

1. 在 clean-room 阶段统一处理所有路径替换
2. 处理完成后，用 `grep -r "/data/rbg" .` 和 `grep -r "weian" .` 做全局扫描
3. 完整的敏感关键词扫描清单见 [../scope/03_scope_exclude.md](../scope/03_scope_exclude.md)
