# 开发准则（R-KV FullKV 对齐/消融）

> 沿用 `R-KV/docs/speckv_qwen` 的规范与记录方式，并内嵌 `R-KV/docs/rkv_fullkv_vs_lazy_ablation_plan.md` 的细节。新增约束：**不影响既有算法/实验，所有新开关默认保持原行为，消融只能通过 cfg 显式开启**。

## 1. 范围与目录
- 文档：本目录 `R-KV/docs/fullkv_ablation/`（记录差异、计划、留言、方案）。
- 代码：保持在 `R-KV/`，消融启动脚本集中于 `R-KV/weian_script/ablations_fullkv/`（命名清晰，避免与旧脚本混淆）。
- 继承规范：遵循 `R-KV/docs/speckv_qwen/` 的风格、记录习惯、停机告警条款。

## 2. 安全与隔离原则
- **不可影响现有算法/实验**：新增选项必须通过 cfg 开关控制，默认值 = 现有行为；不改旧脚本默认配置。
- **代码隔离优先**：必要时以子模块/新脚本方式实现消融，避免修改核心逻辑；修改核心文件需双重检查默认路径不变。
- **启动脚本集中管理**：所有消融脚本放到 `R-KV/weian_script/ablations_fullkv/`。

## 3. 开发与验证
- 先写 cfg，再落地代码：每个消融项对应 cfg 字段，默认关闭；代码读取 cfg，fallback 为旧逻辑。
- 冒烟优先：交付前做最小化 smoke（如 1 shard / 1 qid / num_samples=1），记录命令/结果。长跑由用户执行。
- 如果修改核心文件，确认默认路径/超参与现状一致；必要时使用隔离脚本。

## 4. 记录与告警
- 关键沟通/需求写入 `message.md`；进度与风险写入 `status_overview.md`；任务分解见 `project_todo.md`。
- 发现影响可比性的关键差异（prompt/数据/模型/kv_budget/位置处理等）必须先记录并告警，再行动。

## 5. 提交与收尾
- 对改动文件跑 `python -m compileall` 或等效静态检查；smoke 必记。
- 更新 `status_overview.md`/`project_todo.md` 后再提交；保持默认值不破坏旧实验。
- 流程要求：若用户要求执行 `project_todo.md` 的某一项，Agent 需先在该项下细化具体步骤（每步前加 `[ ]`），再按步骤执行并在完成后改为 `[x]`，同时记录命令/结果。
