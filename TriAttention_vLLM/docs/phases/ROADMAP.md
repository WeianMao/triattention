# SpeckV/TriAttention 实现路线图

## 阶段概览

```
已完成 (HF 路径)           Phase 0 (vLLM 集成)         Phase 1 (Triton 实现)
      ↓                          ↓                          ↓
speckv_rkv_style.py       SpeckVvLLM 类              高效独立版本
monkey patch HF           update_kv() 接口           batch>1, Triton
三种 pruning mode         PagedAttention 基础支持     PagedAttention 完整支持
      ↓                          ↓                          ↓
  ┌─────────────────────────────────────────────────────────────┐
  │                  Phase 2 (深度优化)                           │
  │           边界情况优化、CUDA Graph、动态 budget                  │
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

**目标**：高效独立实现，支持 batch > 1，完整支持 PagedAttention

**详细文档**：[phase1_triton_implementation/README.md](./phase1_triton_implementation/README.md)

| 任务 | 说明 | 状态 |
|-----|------|------|
| Triton 打分 kernel | 替代 PyTorch 打分 | □ |
| Triton TopK kernel | 高效 token 选择 | □ |
| batch > 1 支持 | 生产环境需求 | □ |
| PagedAttention 完整支持 | 多 request 并发场景 | □ |
| 性能优化 | 目标 1.3-1.7x 加速 | □ |

**完成标准**：
- 正确性与 Phase 0 一致
- 端到端性能 >= 1.3x
- batch=8 正常运行
- 多 request 并发下 PagedAttention 正确工作

---

## Phase 2: 深度优化（规划中）

- 边界情况优化（slot 复用、request 取消等）
- CUDA Graph 支持
- 动态 budget 调整
- 性能调优与稳定性

---

*更新日期：2026-01-31*
