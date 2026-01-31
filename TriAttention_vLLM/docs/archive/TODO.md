# TriAttention TODO

待敲定的细节和待完成的任务。

---

## 1. 接口与配置

- [ ] 如何启用 TriAttention（vLLM 启动参数？配置文件？API？）
- [ ] 配置参数的具体接口设计

---

## 2. 性能验证

- [ ] 打分开销：每 divide_length 步打分一次，耗时多少？
- [ ] Fill-in kernel 耗时 benchmark
- [ ] 对 TTFT / TPS 的影响
- [ ] 端到端性能测试

---

## 3. 实现细节

- [ ] Batch 中不同序列的状态管理方式
- [ ] 与 vLLM 其他特性的兼容性（Speculative decoding? Chunked prefill?）

---

## 4. 测试

- [ ] 正确性测试：输出与 R-KV 实现对比
- [ ] 打分验证：裁剪时打分结果是否相同
- [ ] 性能测试：吞吐量、解码延迟、打分延迟
- [ ] Benchmark：复用 R-KV 评估脚本（AIME24 等）

---

*最后更新：2025-01-30*
