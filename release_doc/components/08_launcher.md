# 分布式启动器

## 决策状态：已确认

## 命名方案（已确认）

| 当前名字 | Release 名字 | 作用 | 说明 |
|---------|-------------|------|------|
| `rkv_sharded_dispatch.py` | `dispatch.py` | 主调度器：多GPU分配、断点恢复、自动评估 | 去掉 `rkv_sharded` 前缀 |
| `rkv_sharded_eval.py` | `worker.py` | 推理 worker：每GPU一个实例 | 更准确描述功能 |
| `rkv_sharded_runner.py` | **删除** | 原为 PD-L1_binder wrapper | 进程伪装删除后无存在意义，dispatch.py 直接调用 worker.py |
| `merge_rkv_shards.py` | `merge_shards.py` | 分片结果合并 | 去掉 `rkv` 前缀 |
| `process_utils.py` | `process_utils.py` | 进程工具 | 保留名字，**删除所有进程伪装功能**（mask_process_command、PD-L1_binder） |
| `rkv_cache_utils.py` | `cache_utils.py` | cache 管理 | 去掉 `rkv` 前缀 |

## 依赖关系

```
shell 脚本 → dispatch.py
               → worker.py（subprocess 调用）
               → merge_shards.py（合并结果）
               → eval_math_multi.py（评估）
```

注意：原来 dispatch.py → runner.py → worker.py 的调用链简化为 dispatch.py → worker.py。
dispatch.py 中调用 runner.py 的代码需要改为直接调用 worker.py。

## 功能

- **多GPU分配**：自动检测可用GPU，队列调度
- **断点恢复**：检测已完成的 shard，跳过重复计算
- **分片合并**：按 sample_idx + draw_idx 排序合并
- **自动评估**：合并后自动调用 eval_math_multi.py
- **错误处理**：fail-fast，任一 shard 失败终止全部

## 完整流程（Release 版本）

```
用户 shell 脚本 → dispatch.py
  → 分配任务到多个 GPU
  → 每个 GPU 运行 worker.py（推理）
  → merge_shards.py（合并分片）
  → eval_math_multi.py（评估）
```

## 清理要求

### 进程伪装代码 — 全部删除

| 文件 | 要删除的内容 |
|------|-------------|
| `rkv_sharded_runner.py` | **整个文件删除**（唯一功能是进程伪装） |
| `process_utils.py` | 删除 `mask_process_command()` 函数及所有 PD-L1_binder 相关代码 |
| `rkv_sharded_dispatch.py` | 删除对 `mask_process_command()` 的调用 |
| `rkv_sharded_eval.py` | 删除对 `mask_process_command()` 的调用 |

### 路径清理

- `rkv_sharded_eval.py` 中 PYTHONPATH 添加了内部路径，需改为相对路径
- `weian_development` 路径引用需要重构
- 详见 [../code_cleanup/06_path_cleanup.md](../code_cleanup/06_path_cleanup.md)

### 代码修改要点

1. `dispatch.py` 中 hardcoded 的 `rkv_sharded_eval.py` 路径（约 line 9）改为 `worker.py`
2. `dispatch.py` 中 hardcoded 的 `merge_rkv_shards.py` 路径（约 line 25）改为 `merge_shards.py`
3. `dispatch.py` 中删除对 runner.py 的调用，直接调用 worker.py
4. `worker.py` 中 `rkv_cache_utils` import 改为 `cache_utils`

## CLI 入口文件（`speckv_experiments_cli_v2.py` → `cli.py`）

**决策：已确认** — 移入 release repo 作为 `cli.py`，是实验框架的核心胶水层（660 行）。

### 需要清理的内容（按类别）

| 类别 | 行号 | 内容 |
|------|------|------|
| **docstring** | L2 | `"R-KV SpeckV experiments wrapper"` → 改为 TriAttention |
| **MODES 列表** | L39 | `["fullkv", "rkv", "snapkv", "speckv"]` — `speckv` 改为 `triattention` |
| **目录根路径** | L19-29 | `RKV_ROOT`, `speckv_experiments` 等内部目录名，需适配 release repo 布局 |
| **实验名前缀** | L266 | `f"speckv_{dataset}_{model}_{mode}_{tag}"` — 去掉 speckv 前缀 |
| **subprocess 路径** | L346, L579 | `weian_development/rkv_sharded_dispatch.py` 和 `rkv_sparse_round_calibrate.py` — 改为 release 路径 |
| **PD-L1_binder** | L343, L596 | `env.setdefault("VLLM_PROCESS_NAME_PREFIX", "PD-L1_binder")` — 删除 |
| **help text** | L17, L149, L401, L410-411, L694, L722 | 多处 `speckv`/`SpeckV` 出现在用户可见的日志和帮助文本中 |
| **错误消息中的内部脚本名** | L367, L411, L614 | `_v2.sh` 后缀暴露内部版本迭代 |
| **MODEL_SPECS** | L31-36 | `Qwen3-8B` 映射可能错误（`Qwen/Qwen3-8B` vs `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`），需确认 |
| **rkv_style_ 配置键** | L302-303 | `rkv_style_compression`, `rkv_style_slack_trigger` — 是否改名待定（属于 flag 清理范畴） |
| **控制流中的 speckv 字符串** | L393, L395, L770 | `if mode == "speckv":` 等 — 改为 `triattention` |

### 配置系统适配（非 trivial）

`runner_defaults.yaml` 硬编码 `runner_path: R-KV/weian_development/rkv_sharded_runner.py`。CLI 的配置加载逻辑（路径拼接、config 合并）绑定了内部目录结构，需要适配 release repo 布局。这不是简单 sed 替换，是配置系统的路径解析重构。
