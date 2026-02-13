# Guided Tour Plan - TriAttention_vLLM

## Meta Information

**Purpose**: 为项目负责人提供交互式导读，方便快速理解项目全貌
**Target Audience**: 项目负责人 (weian) 及后续接手人员
**Last Updated**: 2026-02-04
**Current Session**: Claude Opus 4.5 with weian

---

## User Requirements (用户需求)

1. **交互式导读**: 写一部分，用户看一部分，可随时插入问题
2. **断点续传**: 下一个接手人能从断点继续
3. **双层文档**:
   - `interface/` - 用户面向的导读文档
   - `backend/reference/` - 规划和详细内容（本文件所在位置）

---

## Conversation History (对话历史摘要)

### 2026-02-04 Session Summary

**完成的工作**:
1. TriAttention vLLM 集成实现 (V0 monkey patching)
2. AIME24 评估运行完成
   - 准确率: 41.7% (vs HF baseline 42.5%, gap 0.8%)
   - 正确 baseline: `R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perhead/...`
   - 30 questions × 8 samples = 240 predictions
   - 无超时、无空样本
3. 代码提交 (commit 66225ca5, 654af074)
4. GitIgnore 配置完成

**重要澄清** (2026-02-04):
- `protect_prefill` 是 **debug 功能**，不是最终目标
- 最终实现 **不保护题目部分 KV**（当前行为正确）
- 0.8% 差距在可接受范围

**参数审查结果** (2026-02-04):
- `sparse_round_window`: **待删除** - 未使用，只用 `divide_length` (slack mode)
- `--rkv-style-compression`: ✅ 正确实现 (MLR + Trig)
- `--rkv-style-slack-trigger`: ✅ 正确实现 (硬编码开启)

**🔴 重要决策** (2026-02-04 下午):
用户明确：当前 Monkey Patching 方案不是最终目标，需要切换到官方继承方案。

**用户原话**:
> "这不是 vLLM v1 引擎的官方配置方法把。v1 引擎设计的时候肯定会考虑可拓展性...现在这个侵入模式太不优雅了"
> "现在这个测试的整个 pipeline 作为备份放起来。但是我要的结果是把官方继承的方式的实现跑通并得到的结果。现在这个结果是不合格的"

**决策内容**:
1. **归档** Monkey Patching 方案的 pipeline 和结果（41.7%）
2. **切换到** 官方继承方案：`TriAttentionBackend` + Plugin 机制
3. **检查并调整** pipeline 组件（dispatch、runner、merge、eval）
4. **验收标准**：使用官方方案重新运行 AIME24，结果与 Monkey Patching 一致

详见：[DESIGN_DECISIONS.md - 决策 3.7](DESIGN_DECISIONS.md)

**当前进度**: Part 4 完成，下一步执行决策 3.7（切换到官方继承方案）

**2026-02-05 执行记录**:
- Agent 完成 Runner 和 Dispatch 组件修改
- Runner: 移除 `patch_vllm_attention()`，改用 `setup_triattention()` + 环境变量
- Dispatch: 添加 `VLLM_ATTENTION_BACKEND=TRIATTENTION` 环境变量
- Merge/Eval: 确认无需修改

**测试结果** (2026-02-05):
- 官方继承方案 AIME24 准确率：45.0%
- 与 baseline 差距过大（+3.3%），结果异常
- 需要调查原因

**待调查**：
1. 配置参数对比
2. 压缩行为对比
3. V0 API fallback 机制影响

---

## Guide Structure (导读结构规划)

### Part 1: 项目概览 (5 min)
- 项目目标: 什么是 TriAttention？解决什么问题？
- 核心价值: 为什么需要 KV cache 压缩？
- 当前状态: 已完成什么，还差什么？

### Part 2: 代码结构 (10 min)
- 目录结构总览
- 核心模块: `triattention/` 包
- 评估框架: `evaluation/` 目录

### Part 3: 核心算法 (15 min)
- SpeckV 压缩原理
- 频率统计 (stats) 的作用
- 分数计算和 TopK 选择

### Part 4: vLLM 集成 (10 min)
- Monkey Patching 方案
- V1 Backend 预留 (未来)
- 与 HuggingFace 实现的差异

### Part 5: 运行和评估 (10 min)
- 如何运行评估
- 配置文件说明
- 结果解读

### Part 6: 已知问题和下一步 (5 min)
- 1.6% accuracy gap 分析
- 待解决问题列表
- 后续开发计划

---

## Progress Tracking (进度追踪)

| Part | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| Part 1 | ✅ Complete | 2026-02-04 | 2026-02-04 | 含参数审查讨论 |
| Part 2 | 🟡 In Progress | 2026-02-04 | - | |
| Part 3 | ⚪ Pending | - | - | |
| Part 4 | ⚪ Pending | - | - | |
| Part 5 | ⚪ Pending | - | - | |
| Part 6 | ⚪ Pending | - | - | |

---

## Handoff Notes (交接备注)

### For Next Claude/Agent

1. **读取本文件** 了解导读规划和进度
2. **检查 `interface/GUIDED_TOUR.md`** 查看用户已读到哪里
3. **查看 Progress Tracking** 确定从哪个 Part 继续
4. **保持交互式风格**: 每次只写一小段，等待用户确认

### User Preferences

- 简洁为主，避免冗长
- 代码示例要实际可运行
- 遇到问题立即停下讨论，不要跳过
