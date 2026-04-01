# 当前工作状态（断点恢复文件）

> **所有 agent 必读**：开始工作前先读这个文件，了解当前进展。工作过程中及时更新本文件。

## 最后更新

- **时间**：2026-04-01
- **更新者**：Claude Opus agent（第 2 个接手的 agent）
- **对话状态**：进行中

## 当前所处阶段

**阶段 1: Open Items 确认** — 进行中

## 已完成的工作

- [x] 文档系统搭建（release_doc/ 目录结构、guidelines）
- [x] Open Items 确认（大部分已完成）：
  - R-KV 包重命名 → 双包策略（kv_compress + triattention）
  - 硬编码路径替换策略 → HF hub 名 / 相对路径 / 环境变量
  - 数据集 → 不放进 repo，代码中自动从 HuggingFace 下载
  - README 大纲 → 精致版（对标 MInference），含 demo 视频占位符
  - LICENSE → Apache 2.0
  - 启动器文件命名 → dispatch.py / worker.py 等，删除进程伪装
  - Flag 清理 → 14 个删除，保留的改名（方法特有加 triattention- 前缀）
  - 实验 setting 矩阵 → 论文全部主实验 + 消融全部公布
  - GPT-OSS 确认是 20B
  - Figure 5 flag 差异 → 不存在差异
  - DFS benchmark 代码审查 → 通过，有 5 个待修问题
- [x] Agent 模型策略变更：全程 Opus（不再用 Sonnet+Opus 混合）
- [x] 测试/发布流程拆分：单元测试 → 内部公布 → 头对头对比 → 正式公布

## 当前未完成的工作

### 优先级 1：阻塞后续阶段的待确认项

1. **实验框架选择** — 方向已定（speckv_experiments 为主，weian_script 为辅），但具体整合方案未确认
2. **第一阶段执行顺序** — 还没开始讨论

### 优先级 2：需要 agent 调查后让用户确认的 gap

3. **GPT-OSS-20B 处理方案** — 用户提出当作 1.5 阶段，需要 agent 调查 gptoss 分支代码差异和合并方案（对话中断前未执行）
4. **sys.path hack 清理** — 需要 agent 调查具体涉及哪些文件
5. **校准 stats 文件流程** — 需要文档化用户如何生成 stats（跑 fullkv → build-stats）

### 优先级 3：已有方案，可在执行阶段直接做

6. **DFS benchmark 5 个修复项** — 已列在 execution/15_checklist.md
7. **双 rkv 包统一** — 已有方案（重组为 kv_compress + triattention）
8. **KV cache 状态重置 bug 排查** — 已列在 flag_cleanup.md

## 下一步行动

接手的 agent 应该按以下顺序推进：

1. 先处理优先级 2 中的 gap 调查（启动 agent 调查，给用户方案选择）
2. 确认剩余的优先级 1 项（实验框架具体整合、执行顺序）
3. 所有 open items 确认后，进入阶段 2（代码清理）

## 关键文档指引

| 要了解什么 | 去看哪个文件 |
|-----------|------------|
| 项目全貌和发布策略 | scope/01_overview.md |
| 目标 repo 结构 | code_cleanup/05_repo_structure.md |
| 所有已确认/待确认决策 | tracking/14_open_items.md |
| 执行阶段和流程 | stages/README.md |
| 工作规范 | guidelines/ 目录下所有文件 |
| Release 前待办清单 | execution/15_checklist.md |
| 实验 setting 矩阵 | scope/experiment_settings.md |
