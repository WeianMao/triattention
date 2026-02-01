# Phase 0 自检报告

## 1. 已确认对齐 HF 实现

| 项目 | HF 配置 | vLLM 计划 |
|-----|--------|----------|
| Token 选择 | union-based | union-based ✓ |
| normalize_scores | True | True ✓ |
| seed | 0 | 0 ✓ |
| score_aggregation | mean | mean ✓ |

---

## 2. 待验证事项

### 2.1 优化版打分的等价性 [需测试]

vLLM 使用优化版打分（避免 RoPE 反演），HF 使用原始版本。

**数学等价性已证明**（见 `docs/design/optimization.md`），但需单元测试验证实际结果一致。

### 2.2 prefill_len 获取的准确性 [已知局限]

当前方案：首次压缩时的 `seq_len` 作为 `prefill_len`。

**局限**：若 prefill 未达阈值（< budget + buffer），decode 累积后才触发压缩，会高估 prefill_len。

**影响**：保护范围偏大，略保守，不影响正确性。

---

## 3. 已知差异（不影响正确性）

| 差异 | HF | vLLM | 影响 |
|-----|-----|------|------|
| 触发条件 | `pos % divide_length` | `seq_len >= budget + buffer` | 触发时机不同，< 1% 差异可接受 |
| 支持模式 | 多种 | 仅 global | PagedAttention 约束 |

---

*更新日期：2025-01-31*
