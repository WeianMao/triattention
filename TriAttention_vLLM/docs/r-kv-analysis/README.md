# R-KV 对比分析

本目录包含 TriAttention 与 R-KV/vLLM 官方实现的对比分析文档。

> **关键决策汇总**：所有验证结论和关键决策已汇总到 [project/key_decisions.md](../project/key_decisions.md)

---

## 文档索引

| 文档 | 内容 | 核心价值 |
|-----|------|---------|
| [Q1_requirement_coverage.md](Q1_requirement_coverage.md) | 需求覆盖对比 | 阶段定义、需求清单、验证发现的问题 |
| [Q2_pros_cons_analysis.md](Q2_pros_cons_analysis.md) | 优缺点分析 | 架构对比、借鉴建议 |
| [Q3_reusable_code.md](Q3_reusable_code.md) | 代码复用分析 | 效率评估、复用策略 |

---

## 核心结论速查

### 验证确认 ✅

| 结论 | 详情 |
|-----|------|
| R-KV 无 Triton/CUDA kernel | 全部 native PyTorch |
| R-KV 无 noise injection | 之前分析有误，已修正 |
| R-KV batch > 1 静默失败 | 使用 `key_states[0]` 硬编码 |
| R-KV 相似度 O(n²) 内存 | seq_len=100K 需 ~80GB |

### 关键决策

| 决策 | 理由 |
|-----|------|
| 不用 noise injection | 不优雅，R-KV 也没实现 |
| 不用 Query cache | SpeckV 不依赖实时 Query |
| 阶段 1 必须 batch > 1 | R-KV 是静默失败 |
| 阶段 1 必须 Triton 重写 | R-KV 效率慢 1.8-2.8x |

### 三阶段策略

| 阶段 | 复用策略 | 效率目标 |
|-----|---------|---------|
| 阶段 0 | 直接用 R-KV 框架 | 不追求 |
| 阶段 1 | Triton 重写核心操作 | 2-3x 于 R-KV |
| 阶段 2 | 同阶段 1 | 同阶段 1 |

---

*创建日期：2025-01-31*
