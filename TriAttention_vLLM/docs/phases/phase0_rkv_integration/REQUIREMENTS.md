# Phase 0 需求文档

## 1. 核心目标

在 R-KV 的 vLLM fork 中实现 SpeckV 压缩算法，复现 HF 参考实现的功能。

**参考实现**（3 个脚本）：
- `R-KV/weian_development/speckv/speckv_rkv_style.py` - 压缩器类
- `R-KV/weian_development/speckv/round_pruning_utils.py` - 打分工具
- `R-KV/weian_development/speckv/rkv_speckv_generate.py` - 生成入口

---

## 2. 功能要求

| 要求 | 说明 |
|-----|------|
| 接口兼容 | `update_kv()` 与 R1KV 一致 |
| 准确率 | 与 HF 路径差异 < 1% |
| 算法对齐 | 使用 HF 实现的配置（union-based, normalize_scores, seed） |

---

## 3. 效率要求

- 基本 PyTorch 优化（不要求 Triton）
- 避免明显浪费计算的写法
- 使用批量矩阵乘法、共享预计算等优化

---

## 4. 隔离要求

| 要求 | 说明 |
|-----|------|
| 默认行为不变 | 不设置环境变量时，使用 R1KV |
| 参数隔离 | SpeckV 通过 `VLLM_COMPRESSION_ALGO=speckv` 触发 |
| 无副作用 | 不修改 R1KV 的打分逻辑 |

---

## 5. 环境配置

**Conda 环境**：`rkv_vllm`

**环境变量**：
| 变量 | 默认值 | 说明 |
|-----|--------|------|
| `VLLM_COMPRESSION_ALGO` | `r1kv` | 算法选择 |
| `VLLM_SPECKV_STATS_PATH` | - | Stats 文件路径 |
| `VLLM_SPECKV_MODEL_PATH` | - | 模型路径 |
| `VLLM_SPECKV_NORMALIZE` | `1` | score normalization |
| `VLLM_SPECKV_SEED` | `0` | tie-breaking noise seed |

**Stats 文件**：
```
R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt
```

---

## 6. 验收标准

- [ ] `SpeckVvLLM` 类实现 `update_kv()` 接口
- [ ] 通过环境变量切换算法
- [ ] 默认配置下 R1KV 行为不变
- [ ] 单元测试通过
- [ ] AIME24 准确率与 HF 路径差异 < 1%

---

*创建日期：2025-01-31*
