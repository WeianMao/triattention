# Phase 0 自检报告

## 1. 已确认对齐 HF 实现

| 项目 | HF 配置 | vLLM 计划 |
|-----|--------|----------|
| Token 选择 | union-based | union-based ✓ |
| normalize_scores | True | True ✓ |
| seed | 0 | 0 ✓ |
| score_aggregation | mean | mean ✓ |

---

## 2. 已验证事项

### 2.1 优化版打分的等价性 [已验证 ✓]

vLLM 使用优化版打分（避免 RoPE 反演），HF 使用原始版本。

**数学等价性已证明**（见 `docs/design/optimization.md`），**单元测试已验证结果一致**。

---

## 3. prefill_len 获取方案 [已确定]

**问题**：原方案用"首次压缩时的 seq_len"作为 prefill_len，会高估。

**解决方案**：通过 AttentionMetadata 传递真实的 prefill_len

1. 扩展 `FlashAttentionMetadata`：添加 `prefill_lens: torch.Tensor` 字段
2. 在 builder 中填充：从 `req_states.prefill_len` 获取
3. 在压缩时使用：`kvcompressor.attach_prefill_length(prefill_len)`

**隔离要求**：
- R1KV 等现有算法如不使用 prefill_len，则传递不影响其行为
- SpeckV 通过 `attach_prefill_length()` 接收，仅在内部使用
- 默认不启用 prefill protection，需显式配置

---

## 4. 已知差异（不影响正确性）

| 差异 | HF | vLLM | 影响 |
|-----|-----|------|------|
| 触发条件 | `pos % divide_length` | `seq_len >= budget + buffer` | 触发时机不同，< 1% 差异可接受 |

> 注：支持模式已确认三种都可实现（R-KV 已证明 per-head 可行），不再是差异项。

---

*更新日期：2026-01-31*
