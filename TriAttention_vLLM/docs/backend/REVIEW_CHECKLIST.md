# TriAttention_vLLM 评审清单（Runtime）

- 更新时间：2026-02-13
- 状态：Active
- 适用范围：vLLM 0.15.x

---

## 1. 架构一致性（P0）

- [ ] 是否保持非侵入式（未修改 vLLM 源码）？
- [ ] 压缩主流程是否从 Attention 层移出？
- [ ] 是否遵循职责边界（scheduler 决策 / runner 执行）？
- [ ] request identity 是否仅使用 req_id？

---

## 2. 正确性（P0）

- [ ] 生命周期状态是否覆盖 new/running/finished/preempted/resumed？
- [ ] 压缩触发条件是否与配置一致？
- [ ] prefill 模式行为是否符合配置定义？
- [ ] 异常路径是否降级到“本步不压缩”而非中断？

---

## 3. 可测试性（P0）

- [ ] 是否有针对触发策略的独立测试？
- [ ] 是否有 request 状态隔离测试？
- [ ] 是否有基础端到端 smoke test？
- [ ] 新增逻辑是否可在日志中观测（触发/执行/回退）？

---

## 4. 回归风险（P1）

- [ ] 是否影响 batch>1 兼容路径？
- [ ] 是否引入新的全局状态污染点？
- [ ] 是否增加不可控的行为分支（隐式环境变量）？

---

## 5. 文档一致性（P0）

- [ ] `CURRENT_STATUS.md` 已更新为最新状态？
- [ ] `OPEN_ISSUES.md` 已新增/删除对应问题？
- [ ] 若有新决策，`DESIGN_DECISIONS.md` 已记录？
- [ ] 是否违反 `docs/DOCS_STANDARDS.md` 的 SSOT 规则？

---

## 6. 合并门槛

以下任一未满足，不应合并：

1. 架构边界被破坏。
2. request 状态管理无明确生命周期处理。
3. 文档未同步。
4. 无最小回归验证结果。

