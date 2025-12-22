# R-KV Fair Comparison Changes

## Background

在对比 R-KV 和 sparse_prefill_keep 时发现存在 KV budget 管理策略的不公平性：

| 方法 | 原始行为 | 平均 decode KV |
|------|----------|----------------|
| **R-KV** | 始终维持在 budget=1492 附近 | ~1492 |
| **sparse_prefill_keep** | 每 363 tokens 压缩到 1129，然后增长到 1492 | ~1310 |

R-KV 平均多使用约 14% 的 decode KV cache。

## Changes Made

### 1. `weian_development/rkv_lazy/compression/r1_kv.py`

**新增参数：**
```python
round_window: int = 0           # 0 = disabled; >0 = round size (e.g., 363)
round_base_budget: int = None   # target budget at round start (e.g., 1129)
```

**新增方法：**
- `_do_compression(key_states, query_states, value_states, target_budget)`: 压缩 KV cache 到指定大小

**修改 `update_kv` 逻辑：**
- 当 `round_window > 0` 时启用 round-based 压缩
- 每 `round_window` 个 tokens 压缩到 `round_base_budget`
- 保留原有逻辑作为 fallback（当 `round_window=0`）

**修改 `reset_compression_state`：**
- 添加 `self.tokens_in_round = 0` 重置 round 计数器

**修改 `update_kv` 新样本检测：**
- 当 `prefix_len == 0` 时（prefill 阶段），自动重置 `tokens_in_round = 0`
- 这确保了即使 `reset_cache_each_batch=false`，round 计数也会在新样本开始时重置

### 2. `weian_development/rkv_lazy_runner.py`

**新增命令行参数：**
```python
--round_window      # Round size (0=disabled, 363=match sparse_prefill_keep)
--round_base_budget # Target budget at round start (default: kv_budget - round_window)
```

**修改 `method_config`：**
- 将新参数传递给 R1KV 类

### 3. `LazyEviction/weian_script/configs/rkv_lazy_aime_fair.yaml` (新建)

公平对比配置文件，关键参数：
```yaml
kv_budget: 1492
round_window: 363              # 对齐 sparse_prefill_keep
round_base_budget: 1129        # = 1492 - 363
reset_cache_each_batch: true   # 确保每个样本重置状态
```

## Usage

### 运行公平对比实验

```bash
# 方法 1: 使用新配置文件
# 修改 run_rkv_sharded_eval.sh 中的 config 路径为:
# configs/rkv_lazy_aime_fair.yaml

# 方法 2: 直接传参
python weian_development/rkv_lazy_runner.py \
    --method rkv \
    --kv_budget 1492 \
    --round_window 363 \
    --round_base_budget 1129 \
    ... 其他参数
```

### 运行原始 R-KV（不对齐）

```bash
# 使用原配置文件 rkv_lazy_aime.yaml
# 或设置 round_window=0（默认值）
```

## Alignment Result

| 指标 | sparse_prefill_keep | R-KV (fair) |
|------|---------------------|-------------|
| 最大 decode KV | 1492 | 1492 |
| Round 开始时压缩到 | 1129 | 1129 |
| Round 大小 | 363 | 363 |
| 平均 decode KV | ~1310 | ~1310 |
| Prefill 处理 | pin 不计入 | pin 不计入 |
| Prompt template | 相同 | 相同 |

## Other Checked Items (No Changes Needed)

1. **Prompt template**: 两者都使用相同格式
   - System: `You are a helpful assistant.`
   - User prefix: `Please reason step by step, and put your final answer within \boxed{}.`

2. **Prefill handling**: 两者都 pin prefill tokens，不计入 budget

3. **Dataset**: 两者使用相同数据源（AIME test.jsonl）
