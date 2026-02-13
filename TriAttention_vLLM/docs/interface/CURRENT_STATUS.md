# TriAttention_vLLM 当前状态

- 更新时间：2026-02-13
- 状态：Active
- 适用范围：vLLM 0.15.x

---

## 1. 执行摘要

项目已完成 V1 方案的大量算法与集成验证，但团队已确认进入 **V2 架构路线**：

1. 不再以“在 Attention 层后置压缩”作为主线。
2. 主线切换为“worker+runner+scheduler 非侵入式扩展”。
3. 文档系统已重构为 V2 体系，作为后续多人协作基线。

---

## 2. 已完成（可复用资产）

1. TriAttention 核心算法与配置体系（`triattention/config.py`, `triattention/compressor.py`）。
2. Triton 打分内核与相关验证资产。
3. V1 方案问题调查与根因沉淀（含 GQA 相关修复经验）。
4. 基于 vLLM 插件注册 custom attention backend 的实践路径。

---

## 3. 当前主线（V2）

1. 方案定义完成：`interface/V2_OVERVIEW.md`。
2. 技术规格主文档：`backend/ARCHITECTURE_REDESIGN.md`。
3. 决策日志已建立：`backend/DESIGN_DECISIONS.md`。
4. 开发规范与评审清单已重写：`backend/DEVELOPMENT_PRINCIPLES.md`, `backend/REVIEW_CHECKLIST.md`。

---

## 4. 当前阻塞

当前阻塞不在算法，而在工程落位：

1. V2 自定义 `worker_cls` / `scheduler_cls` / runner 数据流实现尚未落地。
2. prefill 模式默认值与显存触发阈值仍需最终拍板（见 `PENDING_DECISIONS.md`）。
3. Phase 1 的验收脚本与回归门禁尚未建立。

---

## 5. 下一里程碑

1. M1：提交 V2 骨架代码（可加载、可运行、可观测）。
2. M2：完成 Phase 1 基础功能（单请求 + prefill 保护）。
3. M3：完成 Phase 2（batch>1 + prefill 可裁剪 + KV usage 触发）。

---

## 6. 风险

1. vLLM 内部接口并非全部稳定公开接口，需版本锁定与适配层。
2. 若文档不按规范维护，极易再次出现“状态冲突与旧结论污染”。

