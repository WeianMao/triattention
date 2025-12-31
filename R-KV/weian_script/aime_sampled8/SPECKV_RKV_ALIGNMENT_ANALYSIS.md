# SpeckV 与 R-KV 基准测试对齐分析

本文档分析了 SpeckV（自定义算法）和 R-KV（官方基线）在 AIME 基准测试实验中的公平性问题，并记录解决方案。

## 问题汇总

| 问题 | 严重程度 | 对应 Arg | 状态 |
|------|----------|----------|------|
| 统计文件数据泄露 | **严重** | yaml 配置 (cross-dataset stats) | ✅ 已实现 |
| KV Budget 语义差异 | **严重** | `--include-prefill-in-budget` | ✅ 已实现 |
| **Budget 计算方式差异** | **严重** | `--rkv-aligned-budget` | ✅ 已实现 |
| 压缩频率差异 | **中等** | yaml 配置 (sparse_round_window=32) | ✅ 已实现 |
| 压缩实现方式差异 | **低**（等价但为严谨） | `--rkv-style-compression` | ✅ 已实现 |

---

## 解决方案原则

1. **每个问题对应一个独立的 arg**
2. **arg 默认关闭**：关闭状态下保持原有行为（有问题的默认做法）
3. **arg 激活后修复对应问题**
4. **代码隔离**：新实现不能修改原有代码逻辑，用 arg 隔离
5. **最终脚本**：4个新脚本，所有 arg 全部激活

---

## 问题 1：统计文件数据泄露

### 问题描述

SpeckV 的统计文件（`.pt`）来源于与测试集相同的数据集：
- 测试 AIME25 时使用 AIME25 的统计 ❌
- 测试 AIME24 时使用 AIME24 的统计 ❌

### 解决方案

| Arg | `--cross-dataset-stats` |
|-----|-------------------------|
| 默认值 | `False`（使用同数据集统计，有数据泄露） |
| 激活后 | `True`（交叉使用统计文件） |

**激活后的行为：**
- 测试 AIME25 → 使用 `sample8_fullkv_aime24_official_qwen` 的统计
- 测试 AIME24 → 使用 `sample8_fullkv_aime25_official_qwen` 的统计

### 实现方式

**使用方案B：新建 yaml 配置文件**

新建以下配置文件：
```
R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml
R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_rank_aligned.yaml
R-KV/weian_script/configs/aime_sampled8_speckv_aime25_qwen_norm_aligned.yaml
R-KV/weian_script/configs/aime_sampled8_speckv_aime25_qwen_rank_aligned.yaml
```

配置中切换 `sparse_stats_path`：
```yaml
# AIME25 测试的 aligned 配置
sparse_stats_path: R-KV/outputs/repository/sample8_fullkv_aime24_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt

# AIME24 测试的 aligned 配置
sparse_stats_path: R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt
```

---

## 问题 2：KV Budget 语义差异

### 问题描述

| 方法 | Budget 包含 Prefill？ | 实际 Decode 容量 |
|------|----------------------|------------------|
| R-KV | **是** | `budget - prefill_length` |
| SpeckV | **否** | `budget`（完整容量） |

SpeckV 会额外保留所有 prefill tokens，导致实际 KV cache 比 R-KV 多约 150 tokens（典型 AIME prefill）。

### 解决方案

| Arg | `--include-prefill-in-budget` |
|-----|-------------------------------|
| 默认值 | `False`（prefill 不计入 budget，SpeckV 原有行为） |
| 激活后 | `True`（prefill 计入 budget，对齐 R-KV） |

**激活后的行为：**
- SpeckV 的 `max_keys` 包含 prefill tokens
- 实际 decode 容量 = `budget - prefill_length`

### 参数传递路径

```
脚本 --include-prefill-in-budget
    ↓
rkv_sharded_eval.py (argparse 解析)
    ↓
apply_speckv_generate_patch(include_prefill_in_budget=args.include_prefill_in_budget)
    ↓
SparsePruningConfig(include_prefill_in_budget=...)  # dataclass 新增字段
    ↓
SparseRoundPruner(config)  # 通过 self.include_prefill_in_budget 访问
```

### 实现方式

需要在 4 个地方添加参数（标准传递，无风险）：

1. **`rkv_sharded_eval.py`**：添加 arg parser
```python
parser.add_argument("--include-prefill-in-budget", type=str2bool, default=False)
```

2. **`rkv_speckv_generate.py`**：函数签名添加参数
```python
def apply_speckv_generate_patch(..., include_prefill_in_budget: bool = False):
```

3. **`sparse_round_pruner_prefill_keep.py`**：SparsePruningConfig 添加字段
```python
@dataclass
class SparsePruningConfig:
    ...
    include_prefill_in_budget: bool = False  # 新增
```

4. **`sparse_round_pruner_prefill_keep.py`**：SparseRoundPruner 使用字段
```python
@property
def _dynamic_cache_size(self) -> int:
    if self.config.include_prefill_in_budget:
        return len(self.cache_positions)  # 含 prefill
    return max(0, len(self.cache_positions) - self.prefix_length)  # 原有逻辑
```

---

## 问题 2.5：Budget 计算方式差异

### 问题描述

即使启用 `--include-prefill-in-budget`，SpeckV 和 R-KV 在 budget 计算上仍存在两个差异：

**差异 A：压缩目标**

| 方法 | 压缩后保留数量 | 说明 |
|------|----------------|------|
| R-KV | `budget` | 精确压缩到 budget（选择 budget - window_size 个旧 token，加上 window_size 个新 token = budget） |
| SpeckV | `budget - round_window` | 压缩到 budget - round_window，然后生成新 token 直到达到 budget |

**差异 B：缓存波动方向**

| 方法 | 缓存大小波动范围 | 说明 |
|------|------------------|------|
| R-KV | `[budget, budget + window_size]` → 压缩回 `budget` | 先增加再压缩 |
| SpeckV | `[budget - round_window, budget]` | 先压缩再增加 |

### 解决方案

| Arg | `--rkv-aligned-budget` |
|-----|------------------------|
| 默认值 | `False`（SpeckV 原有行为：压缩到 budget - round_window） |
| 激活后 | `True`（对齐 R-KV：压缩到 budget 精确值） |

**激活后的行为：**
- `keep_capacity = budget`（而非 `budget - round_window`）
- 缓存大小波动范围变为 `[budget, budget]`（与 R-KV 一致）

### 参数传递路径

```
脚本 --rkv-aligned-budget
    ↓
rkv_sharded_eval.py (argparse 解析)
    ↓
apply_speckv_generate_patch(rkv_aligned_budget=args.rkv_aligned_budget)
    ↓
SparsePruningConfig(rkv_aligned_budget=...)  # dataclass 新增字段
    ↓
SparseRoundPruner(config)  # 通过 self.rkv_aligned_budget 访问
    ↓
ensure_capacity() / start_next_round()  # 条件逻辑
```

### 实现方式

在 `sparse_round_pruner_prefill_keep.py` 的两个方法中添加条件逻辑：

```python
def ensure_capacity(self, past_key_values):
    # 原有逻辑: keep_capacity = max(0, self.max_keys - self.round_window)
    # 新逻辑: 根据 rkv_aligned_budget 选择
    keep_capacity = self.max_keys if self.rkv_aligned_budget else max(0, self.max_keys - self.round_window)
    ...

def start_next_round(self, past_key_values):
    # 同上
    keep_capacity = self.max_keys if self.rkv_aligned_budget else max(0, self.max_keys - self.round_window)
    ...
```

### 测试用例

| 测试用例 | 配置 | 预期行为 |
|----------|------|----------|
| 默认行为 | `rkv_aligned_budget=False`, `include_prefill_in_budget=False` | 缓存大小 ∈ [budget - round_window, budget]，prefill 不计入 budget |
| 仅 Budget 对齐 | `rkv_aligned_budget=True`, `include_prefill_in_budget=False` | 缓存大小 = budget 精确值（部分对齐） |
| 完全对齐 | `rkv_aligned_budget=True`, `include_prefill_in_budget=True` | 缓存大小 = budget 精确值，prefill 计入 budget（完全对齐 R-KV） |

---

## 问题 3：压缩频率差异

### 问题描述

| 方法 | 压缩触发 | 压缩后大小 |
|------|----------|------------|
| R-KV | cache >= 2048 时每个 token 都压缩 | 2048 |
| SpeckV | 每 128 个 token 压缩一次 | 1920 |

SpeckV 的 cache 在 1920~2048 之间波动，平均比 R-KV 少约 3%。

### 关于 window_size 的澄清

**SpeckV 本身不使用 `window_size` 这个概念。**

yaml 配置中的 `window_size: 128` 只是 `sparse_round_window` 的 fallback：
```python
# rkv_sharded_eval.py:500
round_window = args.sparse_round_window if args.sparse_round_window > 0 else args.window_size
```

由于 `sparse_round_window: 128` 是显式设置的，`window_size` 实际上没有被使用。

**结论：只需要调整 `sparse_round_window`，不需要动 `window_size`。**

### 解决方案

| Arg | `--aligned-round-window` |
|-----|--------------------------|
| 默认值 | `False`（sparse_round_window=128） |
| 激活后 | `True`（sparse_round_window=32） |

**激活后的行为：**
- SpeckV 的 `sparse_round_window` 从 128 改为 32
- 压缩后大小 = 2048 - 32 = 2016
- 更接近 R-KV 的 2048（差距从 ~64 缩小到 ~32）

### 实现方式

在 yaml 配置中修改：
```yaml
# 激活前
sparse_round_window: 128

# 激活后
sparse_round_window: 32
```

---

## 问题 4：压缩实现方式差异

### 问题描述

虽然两者效果等价，但实现层面不同：
- R-KV：在 attention 层内触发压缩（每个 token forward 时检查并压缩）
- SpeckV：在 generate() 外层触发压缩（round-based wrapper）

### 解决方案

| Arg | `--rkv-style-compression` |
|-----|---------------------------|
| 默认值 | `False`（SpeckV 原有的 generate 外层实现） |
| 激活后 | `True`（切换到 R-KV 风格的 attention 层实现） |

**激活后的行为：**
- SpeckV 使用与 R-KV 相同的压缩触发方式
- 在 attention forward 中触发压缩，而非 generate wrapper

### 实现方式

**新建独立文件：`R-KV/weian_development/speckv/speckv_rkv_style.py`**

这个文件将：
1. 类似 `rkv/monkeypatch.py` 的方式，创建新的 attention forward 函数
2. 集成 SpeckV 的频率评分逻辑（复用 `SparseRoundPruner` 的评分方法）
3. 使用 R-KV 风格的触发方式（在 attention forward 中检查并压缩）

用 arg 控制使用哪个实现：
```python
if args.rkv_style_compression:
    from weian_development.speckv.speckv_rkv_style import apply_speckv_rkv_style_patch
    apply_speckv_rkv_style_patch(model, ...)
else:
    from weian_development.speckv.rkv_speckv_generate import apply_speckv_generate_patch
    apply_speckv_generate_patch(model, ...)
```

---

## 新脚本规划

### 原有脚本（不修改）

```
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm.sh
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_rank.sh
R-KV/weian_script/aime_sampled8/speckv/aime25/run_speckv_aime25_qwen_norm.sh
R-KV/weian_script/aime_sampled8/speckv/aime25/run_speckv_aime25_qwen_rank.sh
```

### 新脚本（全部 arg 激活）

```
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned.sh
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_rank_aligned.sh
R-KV/weian_script/aime_sampled8/speckv/aime25/run_speckv_aime25_qwen_norm_aligned.sh
R-KV/weian_script/aime_sampled8/speckv/aime25/run_speckv_aime25_qwen_rank_aligned.sh
```

### 新 yaml 配置文件

```
R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml
R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_rank_aligned.yaml
R-KV/weian_script/configs/aime_sampled8_speckv_aime25_qwen_norm_aligned.yaml
R-KV/weian_script/configs/aime_sampled8_speckv_aime25_qwen_rank_aligned.yaml
```

### 输出目录命名

```
# 原有
R-KV/outputs/aime_sampled8/speckv/aime25/norm/

# 新的（aligned 版本）
R-KV/outputs/aime_sampled8/speckv/aime25/norm_aligned/
```

---

## 需要修改/新建的文件

### 代码文件（需要添加 arg 支持）

| 文件 | 修改内容 |
|------|----------|
| `R-KV/weian_development/rkv_sharded_eval.py` | 添加 `--include-prefill-in-budget` 和 `--rkv-style-compression` args |
| `R-KV/weian_development/speckv/rkv_speckv_generate.py` | 添加 `include_prefill_in_budget` 参数传递 |
| `R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py` | SparsePruningConfig 添加字段，SparseRoundPruner 添加条件逻辑 |

### 需要新建的文件

| 文件 | 用途 |
|------|------|
| `R-KV/weian_development/speckv/speckv_rkv_style.py` | 问题4 的 R-KV 风格实现 |

### 配置文件（新建）

```
R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml
R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_rank_aligned.yaml
R-KV/weian_script/configs/aime_sampled8_speckv_aime25_qwen_norm_aligned.yaml
R-KV/weian_script/configs/aime_sampled8_speckv_aime25_qwen_rank_aligned.yaml
```

### 脚本文件（新建）

```
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned.sh
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_rank_aligned.sh
R-KV/weian_script/aime_sampled8/speckv/aime25/run_speckv_aime25_qwen_norm_aligned.sh
R-KV/weian_script/aime_sampled8/speckv/aime25/run_speckv_aime25_qwen_rank_aligned.sh
```
