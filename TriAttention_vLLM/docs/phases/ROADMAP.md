# SpeckV/TriAttention 实现路线图

## 阶段概览

```
已完成 (HF 路径)           Phase 0 (vLLM 集成)         Phase 1 (Triton 实现)
      ↓                          ↓                          ↓
speckv_rkv_style.py       SpeckVvLLM 类              高效独立版本
monkey patch HF           update_kv() 接口           batch>1, Triton
三种 pruning mode         per_layer_perhead          新接口设计
      ↓                          ↓                          ↓
  ┌─────────────────────────────────────────────────────────────┐
  │                  Phase 2 (高级功能)                           │
  │         vLLM PagedAttention 深度集成、CUDA Graph               │
  └─────────────────────────────────────────────────────────────┘
```

---

## Phase 0: vLLM 集成

**目标**：在 R-KV 的 vLLM fork 中快速验证 SpeckV

**详细文档**：[phase0_rkv_integration/README.md](./phase0_rkv_integration/README.md)

| 任务 | 文件 | 状态 |
|-----|------|------|
| 实现 SpeckVvLLM 类 | `rkv/compression/speckv_vllm.py` | □ |
| 集成到 flash_attn.py | `vLLM/.../flash_attn.py` | □ |
| 单元测试 | `tests/test_speckv_vllm.py` | □ |
| 准确率验证 | AIME24 | □ |

**完成标准**：
- `update_kv()` 接口与 R1KV 兼容
- 默认配置下 R1KV 行为不变
- AIME24 准确率与 HF 路径差异 < 1%

---

## Phase 1: Triton 实现

**目标**：高效独立实现，支持 batch > 1

**详细文档**：[phase1_triton_implementation/README.md](./phase1_triton_implementation/README.md)

| 任务 | 说明 | 状态 |
|-----|------|------|
| Triton 打分 kernel | 替代 PyTorch 打分 | □ |
| Triton TopK kernel | 高效 token 选择 | □ |
| batch > 1 支持 | 生产环境需求 | □ |
| 性能优化 | 目标 1.3-1.7x 加速 | □ |

**完成标准**：
- 正确性与 Phase 0 一致
- 端到端性能 >= 1.3x
- batch=8 正常运行

---

## Phase 2: 高级功能（规划中）

- vLLM PagedAttention 深度集成
- CUDA Graph 支持
- 动态 budget 调整

---

*更新日期：2025-01-31*
