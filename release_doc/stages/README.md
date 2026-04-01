# Release 执行阶段

## 设计思路

整个 release 工作被拆分为多个独立阶段，每个阶段可由一个 agent 独立执行。阶段之间有明确的输入/输出和依赖关系。

## 阶段总览

```
阶段 0: 环境准备（创建 worktree）
  ↓
阶段 1: Open Items 确认（逐项确认所有待定决策）
  ↓
阶段 2: 代码清理 — 删除排除内容
  ↓
阶段 3: 代码清理 — 命名统一（speckv → triattention）
  ↓
阶段 4: 代码清理 — 路径替换（硬编码路径 → 通用路径）
  ↓
阶段 5: 代码清理 — 敏感信息扫描与删除
  ↓
阶段 6: 目录结构重组（按目标 repo 结构整理）
  ↓
阶段 6.5: GPT-OSS-20B 集成（从 gptoss 分支合并代码，独立 conda 环境，不阻塞前面阶段）
  ↓
阶段 7: 文档编写（README, LICENSE, 使用说明, 复现指南）
  ↓
阶段 8: 对比验证（AB 测试 + 单元测试）
  ↓
阶段 9: 敏感信息最终扫描
  ↓
阶段 10: 端到端测试（从零验证公布代码可用性）
  ↓
阶段 11: Clean-room 发布（创建干净 repo，push 到 GitHub）
  ↓
阶段 12: 清理（删除 worktree）
```

## 阶段文件

每个阶段一个文件，包含：
- **前置条件**：执行本阶段前需要什么
- **具体步骤**：一步步可执行的操作
- **验证标准**：如何确认本阶段完成
- **输出物**：本阶段产出什么
- **注意事项**：容易出错的地方

| 文件 | 阶段 | 状态 |
|------|------|------|
| [stage_00_setup.md](stage_00_setup.md) | 环境准备 | 待创建 |
| [stage_01_confirm.md](stage_01_confirm.md) | Open Items 确认 | 进行中 |
| [stage_02_remove.md](stage_02_remove.md) | 删除排除内容 | 待创建 |
| [stage_03_rename.md](stage_03_rename.md) | 命名统一 | 待创建 |
| [stage_04_paths.md](stage_04_paths.md) | 路径替换 | 待创建 |
| [stage_05_sensitive.md](stage_05_sensitive.md) | 敏感信息扫描 | 待创建 |
| [stage_06_restructure.md](stage_06_restructure.md) | 目录结构重组 | 待创建 |
| [stage_07_docs.md](stage_07_docs.md) | 文档编写 | 待创建 |
| [stage_08_verify.md](stage_08_verify.md) | 对比验证 | 待创建 |
| [stage_09_final_scan.md](stage_09_final_scan.md) | 最终扫描 | 待创建 |
| [stage_10_e2e_test.md](stage_10_e2e_test.md) | 端到端测试 | 已创建 |
| [stage_11_publish.md](stage_11_publish.md) | Clean-room 发布 | 待创建 |
| [stage_12_cleanup.md](stage_12_cleanup.md) | 清理 | 待创建 |

**注意**：每个阶段的具体步骤文件会在 Open Items 全部确认后，根据最终确认的方案来填充。现在先建立框架，后续逐步细化。
