# 命名规范

## 核心原则

**公布的代码是全新整理过的，不是把现有代码原样搬过去。** 所有命名必须统一为 TriAttention 体系，内部开发名字不应出现在公布版本中。

## 命名映射表

| 类型 | 内部名字（不公布） | 公布名字 |
|------|-------------------|----------|
| 文件名 | `speckv_rkv_style.py` | `triattention.py` |
| 类名 | `SpeckVRKVStyle` | `TriAttention` |
| 函数名 | `apply_speckv_rkv_style_patch()` | `apply_triattention_patch()` |
| 配置key | `speckv_budget` | `triattention_budget` |
| 方法标识 | `method: speckv` | `method: triattention` |
| 脚本名 | `run_speckv_aime24_*.sh` | `run_triattention_*.sh` |
| 包名（baseline） | `rkv/compression/` | `kv_compress/` |
| 包名（我们的方法） | `rkv/compression/speckv.py` | `triattention/` |

## 重命名执行原则

1. **能小改就小改** — 能通过改变量名/函数名解决的，就小改，不要重构
2. **重构前必须报告** — 实在需要重构时，先说明：要改什么、为什么
3. **重构后必须验证** — 严格的 AB 对比测试 + 单元测试，确保行为一致

## `--method` 值改名（已确认）

| 当前值 | Release 值 |
|--------|-----------|
| `speckv` | `triattention` |
| `rkv` | `r1kv` |
| `fullkv` | `fullkv`（不变） |
| `snapkv` | `snapkv`（不变） |
| `h2o` | `h2o`（不变） |
| `streamingllm` | `streamingllm`（不变） |

## CLI Flag 改名（已确认）

**原则**：去掉 `sparse-` 前缀和 `rkv-style-` 前缀，用功能描述性名字。

### `rkv-style-*` flag

| 当前 flag | Release flag |
|----------|-------------|
| `--rkv-style-compression` | `--attention-layer-compression` |
| `--rkv-style-slack-trigger` | `--slack-budget-trigger` |

### `sparse-*` flag — TriAttention 方法特有（加 `triattention-` 前缀）

| 当前 flag | Release flag | 功能 |
|----------|-------------|------|
| `--sparse-normalize-scores` | `--triattention-normalize-scores` | z-score 归一化频域评分 |
| `--sparse-score-aggregation` | `--triattention-score-aggregation` | 多 head 频域评分聚合方式 |
| `--sparse-offset-max-length` | `--triattention-frequency-window` | 频域评分的几何偏移网格范围 |
| `--sparse-stats-path` | `--triattention-stats-file` | 预计算频率统计文件路径 |

### `sparse-*` flag — 通用（去掉前缀）

| 当前 flag | Release flag | 功能 |
|----------|-------------|------|
| `--sparse-seed` | `--pruning-seed` | 打分平局噪声的随机种子 |
| `--sparse-round-window` | `--round-window` | 压缩批次窗口大小 |

### 其他通用 flag

| 当前 flag | Release flag | 功能 |
|----------|-------------|------|
| `--rkv-style-compression` | `--attention-layer-compression` | 在 attention 层触发压缩 |
| `--rkv-style-slack-trigger` | `--slack-budget-trigger` | 允许 cache 增长到 budget+window 再触发 |
| `--include-prefill-in-budget` | `--count-prompt-tokens` | prefill token 计入 budget |

### 分类原则

- **TriAttention 特有**：和频域评分、频谱分析直接相关的参数 → 加 `triattention-` 前缀
- **通用**：任何 KV cache 压缩方法都可能用到的参数 → 去掉前缀，用功能描述名

## 文件名中的 "aime" 泛化

脚本名中的 "aime" 需要泛化处理，例如：
- `run_speckv_aime24_*.sh` → `run_triattention_*.sh`
- 校准结果文件名中不能出现 "aime"
