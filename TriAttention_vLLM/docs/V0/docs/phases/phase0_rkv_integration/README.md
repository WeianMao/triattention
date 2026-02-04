# Phase 0: R-KV/vLLM 框架内 SpeckV 集成

## 1. 目标

在 R-KV 的 vLLM fork 中实现 SpeckV 压缩，复用 R1KV 的接口和触发机制。

| 目标 | 验收标准 |
|-----|---------|
| 实现 `update_kv()` 接口 | 与 R1KV 接口兼容 |
| 准确率 | 与 HF 路径差异 < 1% |
| 隔离性 | 默认行为仍为 R1KV |

---

## 2. 架构

### 2.1 集成点

```
vLLM/vllm/v1/attention/backends/flash_attn.py
    ├── __init__: self.kvcompressor = SpeckVvLLM(...) or R1KV(...)
    └── forward: if seq_len >= budget + buffer: 调用 update_kv()
```

### 2.2 与 HF 路径的关键区别

| 方面 | HF 路径 | vLLM 路径 |
|-----|--------|----------|
| 触发条件 | `pos % divide_length == 0` | `seq_len >= budget + buffer` |
| 支持模式 | per_head / per_layer / per_layer_perhead | 三种模式都可支持 ✓ |
| RoPE 处理 | 反演后计算 | 优化版（直接用 K_rot）✓ |

---

## 3. 核心设计

### 3.1 SpeckVvLLM 类

```python
class SpeckVvLLM:
    def update_kv(self, key, query, value, position_indices, prefill_len) -> (K, V)
    def get_last_keep_indices() -> Tensor  # 供外部更新 position_indices
```

**关键点**：
- 接口与 R1KV 一致，只返回 `(K, V)`
- `position_indices` 由 `FlashAttentionImpl` 维护
- 三种模式（per_head / per_layer / per_layer_perhead）都可支持

### 3.2 打分算法（优化版）

**核心优化**（详见 `docs/design/optimization.md`）：
- 避免 RoPE 反演，直接用 K_rot
- 共享三角表 + 批量矩阵乘法

**与 HF 对齐**（需明确保留以下行为）：
- Union-based selection
- per-head z-score normalization
- tie-breaking noise（如 1e-6 级别）

### 3.3 Position Indices 追踪

| 事件 | 操作 |
|-----|------|
| Prefill | 写入 `[0, 1, ..., prefill_len-1]` |
| Decode | 写入新 token 位置 |
| 压缩后 | 保留 `kept_positions` |
| Request 结束 | **双重保险 reset**（见下文） |

**Reset 机制（双重保险）**：
1. **Slot 复用时 reset**：分配 slot 给新 request 时清零
2. **Scheduler hook**：request 结束时回调通知 attention 层

---

## 4. 环境配置

| 变量 | 默认值 | 说明 |
|-----|--------|------|
| `VLLM_COMPRESSION_ALGO` | `r1kv` | 算法选择 |
| `VLLM_SPECKV_STATS_PATH` | - | Stats 文件路径 |
| `VLLM_SPECKV_MODEL_PATH` | - | 模型路径（RoPE 配置） |

**Stats 文件**：`R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/`

---

## 5. 实现步骤

1. 创建 `rkv/compression/speckv_vllm.py`
2. 导出到 `rkv/modeling.py`
3. 修改 `flash_attn.py` 添加算法选择
4. 单元测试 + AIME24 准确率验证

---

## 6. 文件清单

| 类型 | 文件 |
|-----|------|
| 新建 | `rkv/compression/speckv_vllm.py` |
| 修改 | `flash_attn.py`, `rkv/compression/__init__.py`, `rkv/modeling.py`, `vllm/envs.py` |
| 参考 | `speckv_rkv_style.py`, `round_pruning_utils.py` |

---

## 7. 待验证事项

- [x] 模式选择 ✓（三种模式都可支持，R-KV 已证明 per-head 可行）
- [x] 优化版打分与 HF 原版的数学等价性 ✓
- [x] prefill_len 的获取准确性 ✓（通过 AttentionMetadata 传递）
- [x] position_indices 的 reset 机制 ✓（双重保险：slot 复用 + scheduler hook）

---

---

## 8. 等价性验证任务

### 8.1 验证目标

将原有的 HuggingFace 路径测试脚本的行为保持不变，但替换为 vLLM 后端，验证推理精度一致性。

**参考脚本**：
```
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh
```

**配置文件**：
```
R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml
```

**关键参数**：
| 参数 | 值 | 说明 |
|-----|-----|------|
| `kv_budget` | 2048 | KV 缓存预算 |
| `num_samples` | 8 | 每题采样次数 |
| `sparse_stats_path` | `sample8_fullkv_aime25_official_qwen/stats/` | 统计文件（AIME25） |
| `sparse_round_window` | 32 | 压缩窗口 |
| `--sparse-normalize-scores` | true | Z-score 归一化 |
| `--per-head-pruning` | true | 逐头剪枝模式 |

### 8.2 验证方法

1. **HF 基线**：运行原脚本获取 HF 路径结果
2. **vLLM 测试**：使用相同参数运行 vLLM 路径
3. **结果比对**：比较准确率、token 分布等指标

### 8.3 测试脚本位置

```
R-KV/vLLM/weian_script/tests/equivalence/
├── run_equivalence_test.sh    # 主测试脚本（执行全流程）
├── run_hf_baseline_quick.sh   # HF 基线快速测试（1题1采样）
├── run_vllm_speckv_quick.sh   # vLLM SpeckV 快速测试
├── compare_results.py         # 结果比对脚本
└── README.md                  # 测试说明
```

### 8.4 测试执行指南

**前置条件**：
1. 安装 vLLM fork: `cd R-KV/vLLM && pip install -e .`
2. 确保 stats 文件存在: `R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/`
3. 确保 quick 测试数据集存在: `R-KV/HuggingFace/data/quick/aime24.jsonl`

**执行完整测试**：
```bash
cd R-KV/vLLM/weian_script/tests/equivalence
GPU=0 bash run_equivalence_test.sh
```

**分步执行**：
```bash
# Step 1: 运行 HF 基线
GPU=0 bash run_hf_baseline_quick.sh

# Step 2: 运行 vLLM SpeckV
GPU=0 bash run_vllm_speckv_quick.sh

# Step 3: 比较结果
python compare_results.py \
    --hf-output outputs/hf_baseline/shard00/run000.jsonl \
    --vllm-output outputs/vllm_speckv/output.jsonl
```

**验收标准**：
- 预测答案匹配率 > 90%
- 准确率差异 < 1%

---

## 9. 开发规范

### 9.1 代码隔离原则

**核心原则**：只能修改 `R-KV/vLLM/` 目录下的代码，不能修改任何外部代码。

| 允许修改 | 禁止修改 |
|---------|---------|
| `R-KV/vLLM/vllm/` | `R-KV/rkv/` |
| `R-KV/vLLM/weian_script/` | `R-KV/weian_script/` |
| `TriAttention_vLLM/` (文档) | `R-KV/weian_development/` |
| | `R-KV/HuggingFace/` |

### 9.2 测试规范

1. **参考脚本不可修改**：只能调用，不能改动原有测试脚本
2. **新增测试脚本**：所有新测试代码放在 `R-KV/vLLM/weian_script/tests/`
3. **结果可追溯**：保存所有测试输出供后续对比

### 9.3 Bug 修复流程

1. 发现 bug 后，使用 agent 分析问题
2. 只在允许修改的目录内修复
3. 如需修改外部代码，需明确记录并获得授权

---

## 10. 架构改进建议：自定义 Attention Backend 注册机制

### 当前问题
- 压缩代码直接嵌入在 `flash_attn.py` 中（400+ 行修改）
- vLLM 升级时需要手动 merge 代码
- 侵入性高，可维护性差

### 推荐方案：通过环境变量指定自定义 Backend

**Step 1**: 在 `vllm/platforms/cuda.py` 的 `get_attn_backend_cls()` 方法开头添加：

```python
custom_backend = os.environ.get("VLLM_CUSTOM_ATTENTION_BACKEND")
if custom_backend:
    logger.info(f"Using custom attention backend: {custom_backend}")
    return custom_backend  # 直接返回完全限定类名
```

**Step 2**: 创建继承 FlashAttentionBackend 的子类：

```python
# vllm/v1/attention/backends/compressed_flash_attn.py
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl

class CompressedFlashAttentionImpl(FlashAttentionImpl):
    def forward(self, ...):
        # 压缩逻辑放在这里
        pass

class CompressedFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return CompressedFlashAttentionImpl
```

**Step 3**: 使用环境变量指定：

```bash
export VLLM_CUSTOM_ATTENTION_BACKEND="vllm.v1.attention.backends.compressed_flash_attn.CompressedFlashAttentionBackend"
```

### 优势
- 只需修改 cuda.py 5-10 行代码
- 压缩逻辑完全独立在子类文件中
- vLLM 升级时只需重新添加这几行
- 可以通过环境变量快速切换不同的压缩算法

### 状态
- 用户偏好此方案
- 待后续版本实现

---

*更新日期：2026-02-01*
