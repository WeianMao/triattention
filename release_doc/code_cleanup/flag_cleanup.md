# Flag 清理方案

## 决策状态：已确认

## 清理原则

1. **保留**：用户理应能配置的基础接口（采样参数、模型路径、budget 等）
2. **保留**：RKV 官方的正常功能 flag（且确实有用的）
3. **删除**：我们开发过程中加的实验性 flag（消融专用、最终方法未采用）
4. **删除**：debug/模拟 flag
5. **删除**：死代码（定义了但从未使用）
6. **删除**：始终同一个值且不是用户有意义的选择（内部实现细节）
7. **删除**：RKV 官方的不规范残留 flag
8. **不搬到公布分支**：非 rkv-style 的实现（generate-wrapper 风格）及其相关 flag

## 删除清单

### Debug / 死代码

| Flag | 说明 | 删除原因 |
|------|------|---------|
| `--simulate-bug-phase-offset` | 模拟历史 bug（commit 896cbca6），偏移频域评分 phase | 纯 debug，1 个消融脚本 |
| `--simulate-attention-position-offset` | 同上，从 RoPE position_ids 角度模拟 | 纯 debug，0 个脚本 |
| `--output-name` | argparse 定义了但代码中从未读取 | 完全死代码 |

### 消融实验专用（最终方法未采用）

| Flag | 说明 | 删除原因 |
|------|------|---------|
| `--sparse-use-similarity` | 启用相似度去重（频域+余弦相似度结合） | 只在 ~18 个 lambda sweep 消融配置中，最终未采用 |
| `--sparse-similarity-mix-lambda` | 频域/相似度混合权重 | 依赖 `--sparse-use-similarity`，一起删 |
| `--use-rank-similarity-combination` | rank 聚合 + 相似度的组合模式 | 只在 ~9 个消融配置中，依赖上述 flag |
| `--use-rank-aggregation` | 评分聚合从 z-score+max 改为 rank-based min-pooling | 只在消融配置中，最终未采用 |
| `--disable-top-n-high-freq` | 禁用频域评分中 top N 高频分量 | 6 个消融脚本（high_freq_ablation/），默认 0=禁用 |

### 未使用的实现路径相关

| Flag | 说明 | 删除原因 |
|------|------|---------|
| `--rkv-aligned-budget` | 改变 budget 计算方式，用于 generate-wrapper 路径 | 0 个配置/脚本使用，generate-wrapper 实现不公布 |
| `--per-layer-pruning` | 同层所有 head 共享保留 token 集 | 0 个配置/脚本使用 |
| `--per-layer-aggregation` | `--per-layer-pruning` 的聚合方法 | 依赖上面的 flag，一起删 |

### 内部实现细节（始终同值，用户无需配置）

| Flag | 说明 | 始终值 | 删除原因 |
|------|------|--------|---------|
| `--sparse-head-limit` | 限制使用的 head 数量 | 始终 -1（全部） | 从未用过其他值 |
| `--reset-cache-each-batch` | 每个 sample 前清空 KV cache | 生产中始终 False | 早期 debug 产物 |

## 保留清单（确认保留，不需改动）

以下 flag 保留为用户可配置接口：

### 基础采样参数
- `--seed`、`--temperature`、`--top-p`、`--top-k`

### 核心压缩参数
- `--method`、`--kv-budget`、`--window-size`、`--divide-length`
- `--per-head-pruning`、`--per-layer-perhead-pruning`、`--layer-perhead-aggregation`
- `--sparse-normalize-scores`（z-score 归一化频域评分，生产中始终 True 但保留让用户可配置）
- `--sparse-seed`（pruner 独立随机种子，保留）
- `--sparse-score-aggregation`（评分聚合方式）
- `--sparse-offset-max-length`（频域评分窗口大小）
- `--sparse-round-window`（pruning round window）
- `--sparse-stats-path`（pruning stats 文件路径）

### RKV 官方参数
- `--fp32-topk`、`--divide-method`、`--retain-ratio`、`--retain-direction`
- `--first-tokens`、`--update-kv`、`--mix-lambda`
- `--protect-prefill`、`--allow-prefill-compression`、`--include-prefill-in-budget`
- `--compression-content`

### 模型/数据/输出
- `--model-path`、`--dataset-path`、`--output-dir`、`--log-dir`
- `--max-length`、`--eval-batch-size`、`--num-samples`
- `--load-dtype`、`--attn-implementation`
- `--use-chat-template`、`--chat-system-prompt`

### 分布式相关
- `--shard-id`、`--num-shards`、`--gpus`、`--config`
- `--skip-merge`、`--no-eval`、`--dry-run`、`--skip-existing`

## 需要改名的 flag（详见 04_naming.md）

保留的 flag 中含 `rkv`/`sparse`/`speckv` 等内部名称的，在代码清理阶段统一改名。

## 额外排查项

### KV cache 状态重置 bug

`--reset-cache-each-batch` 删除后，需要排查一个潜在 bug：

**问题**：多个问题在单进程中连续推理时，KV cache 或其他状态变量可能没有被正确重置，导致前一个问题的状态污染后一个问题。

**当前未触发的原因**：我们的分布式启动器为每个问题启动独立进程，进程退出时所有状态自然清零。

**风险**：外部用户如果在单进程中循环推理多个问题（不使用我们的启动器），可能会遇到这个 bug。

**处理**：在代码清理阶段需要 agent 排查以下文件中的状态管理：
- `R-KV/HuggingFace/rkv/modeling.py` — 检查 `self.length`、`self.config.compression` 等状态变量是否在每次推理间正确重置
- `R-KV/weian_development/speckv/speckv_rkv_style.py` — 检查 pruner 内部状态
- 如果确认 bug 存在，需要在代码中添加正确的重置逻辑（而非依赖 flag）

## 非 rkv-style 实现

项目中有两套压缩实现：
1. **rkv-style**（`--rkv-style-compression`）— 在 attention layer 内部触发，这是我们 release 的版本
2. **generate-wrapper**（非 rkv-style）— 通过 generate wrapper 触发

generate-wrapper 实现**不搬到公布分支**，其相关代码和 flag（如 `--rkv-aligned-budget`）直接不包含在 release 中。
