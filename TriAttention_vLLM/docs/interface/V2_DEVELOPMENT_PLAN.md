# TriAttention_vLLM V2 开发计划

- 更新时间：2026-02-14
- 状态：Active
- 适用范围：vLLM 0.15.x（V1 Engine）

---

## 1. 本轮落地目标

1. 新建独立工程目录：`TriAttention_vLLM/triattention_v2/`。
2. 打通非侵入式接入骨架：`scheduler-cls + worker-cls + runner proxy`。
3. 落地 request 生命周期状态管理（`req_id` 唯一键）。
4. 完成 Phase 1 的触发链路基础版：
   - 长度触发（`kv_budget + divide_length`）
   - 可选 KV usage 触发（环境开关控制）
5. 保持 attention 计算路径不改动。

---

## 2. 分阶段执行计划

## Phase 1A（已开始）
- 范围：
  - 结构与接口先通，功能先可观测。
  - 不做真正 KV tensor gather/scatter。
- 交付件：
  - `triattention_v2/config.py`
  - `triattention_v2/scheduler.py`
  - `triattention_v2/worker.py`
  - `triattention_v2/runner.py`
  - `triattention_v2/planner.py`
  - `triattention_v2/state.py`
  - `tests_v2/` 最小单测
- 验收：
  - 可以通过 class path 加载自定义 scheduler/worker。
  - scheduler 输出可携带压缩触发信号，runner 可消费并记录状态。

## Phase 1B（进行中）
- 范围：
  - 在 runner 内接入真实压缩执行入口（保留失败降级）。
  - 单请求路径完成“触发 -> 压缩执行 -> 状态更新”闭环。
- 验收：
  - 单请求场景压缩动作稳定可复现。
  - 压缩失败不会中断推理主流程。

当前进度（2026-02-14）：

1. 已完成 runner 侧执行入口与失败降级闭环（executor + hook）。
2. 已完成 experimental KV compaction 原型（默认关闭，先 plan-only）。
3. 已接入 scheduler effective cache length tracker，触发语义从“总长度”向“有效缓存长度”对齐。
4. experimental compaction 已支持多 KV cache group 的基础映射（best-effort）。
5. 已补齐 `protect_prefill=false` 与 batch>1 状态隔离的基础测试。
6. 已新增 `tests_v2/run_smoke.py` 作为本地最小回归脚本。
7. 已新增 V2 专用评测入口（runner + quick dispatch config + 一键脚本），可执行小样本 HF 对照实验。
8. 待将 compaction 原型与 scheduler 容量语义进一步稳定化（覆盖更多场景）。

## Phase 2
- 范围：
  - batch>1。
  - `protect_prefill=false` 可裁剪模式。
  - KV usage 触发默认开启并引入阈值门禁。
- 验收：
  - 多请求生命周期无污染。
  - prefill 两种模式行为可配置、可测。

## Phase 3
- 范围：
  - 性能优化（再评估 Triton TopK/Gather）。
  - 回归与压力测试体系。
- 验收：
  - 性能收益可量化，且不破坏正确性。

---

## 3. 开发执行原则

1. 所有新开发放在 `triattention_v2/`，不在旧版目录上叠 patch。
2. 旧版仅作为参考与回归基线，不作为新功能落点。
3. 代码变更与文档变更同提交，不允许“代码先走、文档滞后”。
4. P0 问题优先：
   - 触发链路
   - 生命周期
   - 回归门禁

---

## 4. 当前 class path（V2）

```bash
--worker-cls triattention_v2.worker.TriAttentionWorker
--scheduler-cls triattention_v2.scheduler.TriAttentionScheduler
```
