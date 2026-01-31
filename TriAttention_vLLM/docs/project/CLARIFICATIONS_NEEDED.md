# 设计问题澄清记录

本文档记录在实现 TriAttention 之前需要确认的设计细节及其结论。

---

## 问题状态总结

| 问题 | 状态 | 结论 |
|-----|------|-----|
| 1. 裁剪粒度与 Fill-in-Place | ✅ 已解决 | Per-head 可行，position_indices 需要 2D |
| 2. Prefill > Budget | ⏸️ 阶段 2 | 第一阶段不处理此边界情况 |
| 3. Budget 未满时合并 | ✅ 已解决 | vLLM 默认行为：直接追加，无空洞 |
| 4. 内存触发压缩 | ⏸️ 阶段 2 | 采用方案 A，第一阶段不实现 |
| 5. 多 Request 场景 | ✅ 已解决 | Per-request 独立，与 vLLM 一致 |
| 6. 打分聚合策略 | ✅ 已解决 | 严格对齐 R-KV 脚本 |
| 7. Position Indices 类型 | ✅ 已解决 | 使用 torch.long，对齐 R-KV |

---

## 问题 1：裁剪粒度与 Fill-in-Place ✅

**结论**：Per-head 模式在 vLLM 中可行

- vLLM KV cache 布局：`[2, num_blocks, block_size, num_kv_heads, head_size]`
- 同一 slot 的不同 heads 可以存储不同 token 的 KV
- Flash Attention 对每个 head 独立计算
- **position_indices 需要 2D**：`[num_slots, num_kv_heads]`
- Fill-in-Place kernel 需要 per-head 操作

---

## 问题 2：Prefill > Budget ⏸️

**结论**：第一阶段不处理，记录到阶段 2 待办

- R-KV 实现无参考价值（不面对此问题）
- 需要设计 vLLM 下的特殊处理策略
- 详见 todo.md 阶段 2 部分

---

## 问题 3：Budget 未满时合并 ✅

**结论**：vLLM 默认行为，无需特殊处理

- vLLM 的 `allocate_new_blocks` 直接追加到 `req_blocks` 末尾
- KV cache 始终紧凑，没有空洞
- Budget 未满时：正常追加（vLLM 默认）
- 只有裁剪后才产生空洞，需要 Fill-in-Place

---

## 问题 4：内存触发压缩 ⏸️

**结论**：采用方案 A，第一阶段不实现

- vLLM 默认：blocks 不足时 preempt 整个 request
- **方案 A**：在 preemption 之前介入，先尝试压缩
- 第一阶段只用 divide_length 触发
- 设计时考虑扩展点，不阻碍后续开发

---

## 问题 5：多 Request 场景 ✅

**结论**：Per-request 独立，与 vLLM 架构一致

- vLLM 已是 per-request 管理 blocks（`req_to_blocks`）
- TriAttention 在每个 request 内部：
  - 独立跟踪 budget 使用量
  - 独立触发裁剪
- 不需要修改 vLLM 的 block 分配机制
- 更灵活的策略（全局 budget 等）待阶段 2 决策

---

## 问题 6：打分聚合策略 ✅

**结论**：严格对齐 R-KV 参考脚本

- **聚合方式**：默认 `mean`
- **Offsets**：几何序列 `[1, 2, 4, 8, ..., 65536]`
- **divide_length**：与 offsets 独立，控制触发频率
- 参考脚本见 roadmap.md 0.2 节

---

## 问题 7：Position Indices 类型 ✅

**结论**：使用 `torch.long`（int64），对齐 R-KV

- R-KV 实际实现：`dtype=torch.long`
- bf16 优化是后续可选项，不是第一阶段要求
- 第一阶段严格对齐 R-KV 行为

---

*更新日期：2025-01-31*
