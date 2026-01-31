# 待办事项

待敲定的细节和待完成的任务。

---

## 1. 接口与配置

- [ ] 如何启用 TriAttention（vLLM 启动参数？配置文件？API？）
- [ ] 配置参数的具体接口设计
- [ ] Stats 文件的加载方式

---

## 2. 性能验证

- [ ] 打分开销：每 divide_length 步打分一次，耗时多少？
- [ ] Fill-in kernel 耗时 benchmark
- [ ] 对 TTFT / TPS 的影响
- [ ] 端到端性能测试

---

## 3. 实现细节

- [ ] Batch 中不同序列的状态管理方式
- [ ] 与 vLLM 其他特性的兼容性
  - Speculative decoding？
  - Chunked prefill？
  - Prefix caching？

---

## 4. 测试

- [ ] 正确性测试：输出与 R-KV 实现对比
- [ ] 打分验证：裁剪时打分结果是否相同
- [ ] 性能测试：吞吐量、解码延迟、打分延迟
- [ ] Benchmark：复用 R-KV 评估脚本（AIME24 等）

---

## 5. 数学正确性验证

- [ ] 验证：$\cos(t\omega + \phi) = \mathcal{A}\cos(t\omega) - \mathcal{B}\sin(t\omega)$
- [ ] 验证：$\text{score}(\mathbf{K}_{\text{rot}}, p) = \text{score}(\mathbf{K})$
- [ ] 集成测试：完整打分轮次匹配原始实现

**数值稳定性验收指标**：

| 精度 | 最大相对误差 |
|-----|------------|
| FP16 | < 1e-3 |
| BF16 | < 1e-2 |

---

## 6. 待确认问题

1. 对于 "half" 风格 RoPE，将前后两半视为复数的实部虚部，复数乘法是否正确？
2. 频率缩放因子 $s_f^2$ 在优化后的公式中是否需要调整？
3. 数值精度：$\phi_{\text{rot}} + p \cdot \omega$ 在 FP16/BF16 下是否稳定？
4. vLLM 的 block allocator 是否支持部分释放 block？
5. CUDA Graph 模式下，KV cache 布局变化是否会导致问题？

---

## 7. 阶段 2 待解决（第一阶段暂不处理）

- [ ] **Prefill > Budget 场景处理**：当 prefill 长度超过 budget 时，需要特殊处理。R-KV 没有参考价值（它不面对这个问题）。需要设计 vLLM 下的处理策略：
  - Prefill 阶段允许超额分配 pages？
  - 第一个 decode step 后立即压缩？
  - 还是限制 prefill 长度不能超过 budget？

- [ ] **内存触发压缩（方案 A）**：在 vLLM 触发 preemption 之前，先尝试压缩 KV cache
  - vLLM 默认行为：blocks 不足时 preempt 整个 request，丢失所有进度
  - 目标：在 preemption 之前介入，通过压缩释放 blocks，避免重算
  - 第一阶段要求：设计时考虑此扩展点，不采用会阻碍此功能开发的方案

- [ ] **更灵活的 Budget 策略**：第一阶段采用 per-request 独立 budget，后续可能需要：
  - 全局 budget 共享策略
  - 动态 budget 调整
  - 优先级驱动的 budget 分配
  - 具体方案待第二阶段决策

---

*最后更新：2025-01-31*
