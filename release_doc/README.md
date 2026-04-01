# TriAttention Release Documentation

## 文档标准

**本目录是 release 工作的唯一权威文档。** 所有决策、细节、调查结论都必须记录在此，确保：

1. **自包含** -- 任何接手的同事只看这些文档就能完整执行 release，不需要再问任何人
2. **消灭不确定性** -- 每个 open item 确认后，不光打勾，还要把决策理由、具体执行方案、边界条件都写清楚
3. **细节完备** -- 包括具体的替换示例、哪些目录不用管及原因、agent 调查出的建议（如环境变量写法、HuggingFace hub 名）等
4. **可追溯** -- 记录决策过程和理由，让后续同事理解"为什么这样决定"

## 文档索引

### scope/ -- 范围相关

| 文件 | 内容 |
|------|------|
| [scope/01_overview.md](scope/01_overview.md) | 发布目标、分阶段策略、时间线 |
| [scope/02_scope_include.md](scope/02_scope_include.md) | 第一阶段要公布的内容 |
| [scope/03_scope_exclude.md](scope/03_scope_exclude.md) | 排除清单、敏感信息识别 |
| [scope/datasets.md](scope/datasets.md) | 数据集：下载链接、字段映射、一致性验证 |
| [scope/experiment_settings.md](scope/experiment_settings.md) | 公布的实验 setting 矩阵 + TriAttention 方法配置 |

### code_cleanup/ -- 代码清理相关

| 文件 | 内容 |
|------|------|
| [code_cleanup/04_naming.md](code_cleanup/04_naming.md) | 命名规范（speckv -> triattention 映射表） |
| [code_cleanup/05_repo_structure.md](code_cleanup/05_repo_structure.md) | 目标 repo 目录结构 |
| [code_cleanup/06_path_cleanup.md](code_cleanup/06_path_cleanup.md) | 硬编码路径替换方案（含完整调查结果） |
| [code_cleanup/flag_cleanup.md](code_cleanup/flag_cleanup.md) | Flag 清理：删除/保留/改名清单 + KV cache bug 排查 |

### components/ -- 各组件详情

| 文件 | 内容 |
|------|------|
| [components/07_evaluation.md](components/07_evaluation.md) | 评估管线：公布/不公布文件清单、清理要求 |
| [components/08_launcher.md](components/08_launcher.md) | 分布式启动器：文件清单、功能、命名清理 |
| [components/09_reference_script.md](components/09_reference_script.md) | 起点脚本、关键 flag 组合、参数基准 |
| [components/readme_outline.md](components/readme_outline.md) | README 大纲（精致版，含占位符清单） |

### execution/ -- 执行相关

| 文件 | 内容 |
|------|------|
| [execution/10_technical_notes.md](execution/10_technical_notes.md) | KV cache 峰值对齐等技术发现 |
| [execution/11_implementation.md](execution/11_implementation.md) | Worktree + Clean-room 执行方案 |
| [execution/12_environment.md](execution/12_environment.md) | 开发环境背景（接手人须知） |
| [execution/15_checklist.md](execution/15_checklist.md) | Release 前待办清单 |

### tracking/ -- 追踪和计划

| 文件 | 内容 |
|------|------|
| [tracking/13_phase2.md](tracking/13_phase2.md) | 第二阶段 kvpress 详情 |
| [tracking/14_open_items.md](tracking/14_open_items.md) | 待确认事项 + 已确认决策记录 |

## 阅读顺序建议

- **快速了解**：先看 [scope/01_overview.md](scope/01_overview.md)，再看 [code_cleanup/05_repo_structure.md](code_cleanup/05_repo_structure.md)
- **执行 release**：按 [execution/11_implementation.md](execution/11_implementation.md) 步骤操作，配合 [execution/15_checklist.md](execution/15_checklist.md) 逐项确认
- **接手项目**：先看 [execution/12_environment.md](execution/12_environment.md) 了解环境，再看 [tracking/14_open_items.md](tracking/14_open_items.md) 了解待办

### stages/ -- 执行阶段

整个 release 工作被拆分为多个独立阶段，每个阶段可由一个 agent 独立执行。

| 文件 | 内容 |
|------|------|
| [stages/README.md](stages/README.md) | 阶段总览、依赖关系、状态追踪 |

阶段详情见 [stages/README.md](stages/README.md)，每个阶段的具体步骤文件会在 Open Items 全部确认后逐步填充。

### guidelines/ -- 工作规范

适用于所有参与 release 工作的 agent 和接手同事。

| 文件 | 内容 |
|------|------|
| [guidelines/agent_workflow.md](guidelines/agent_workflow.md) | Agent 执行任务的工作流程规范 |
| [guidelines/confirmation_protocol.md](guidelines/confirmation_protocol.md) | 与用户确认决策的标准流程 |
| [guidelines/documentation_standard.md](guidelines/documentation_standard.md) | 文档记录标准 |

## 历史归档

旧版单文件 release plan 保留在 [RELEASE_PLAN.md](RELEASE_PLAN.md)，仅供参考。
