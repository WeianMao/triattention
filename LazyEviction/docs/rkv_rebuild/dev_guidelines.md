# 开发准则（RKV 重构至 LazyEviction）

> 继承 `R-KV/docs/fullkv_ablation` 的记录与安全意识，适配 LazyEviction 代码规范。目标是隔离式实现，默认不扰动现有实验；所有新增行为须通过 cfg/脚本显式开启。

## 1. 目录与隔离
- 文档：`LazyEviction/docs/rkv_rebuild/`（本目录），记录消息、进展、计划、TODO。
- 代码：仅新增到 `LazyEviction/` 下；可放在 `weian_development/` 做可复用模块，调度/runner/配置放在 `LazyEviction/weian_script/`。
- 启动脚本：新增脚本与 cfg 使用独立命名，避免覆盖现有文件；默认不被其他方法调用。

## 2. 安全与公平
- **默认不变**：所有改动通过 cfg 控制，默认值复现 LazyEviction 当前行为；不可修改已有脚本的默认参数。
- **KV budget 对齐**：prefill 固定不压缩且不计入 KV budget；预算统计与 LazyEviction 现有方法一致（计量单位、层/头尺度、chunk 规则需对齐）。
- **公平性检查**：数据/提示词、解码策略（采样/贪心）、num_samples/聚合、dtype/attn 实现、长度截断、seed/reset、trust_remote_code 等都需可配置对齐；发现“RKV 占便宜”要记录。
- **隔离执行**：路径、日志、输出目录使用新前缀，避免覆盖旧结果；长跑命令保持 `PD-L1_binder` 前缀。

## 3. 流程与验证
- 先设计 cfg/接口，再动代码；必要时以子类/新 runner 避免修改核心逻辑。
- 静态检查：改动 Python 后至少跑 `python -m compileall` 覆盖变更文件。
- Smoke 优先：使用极小数据/分片验证新入口可运行；长跑留给用户。
- 不依赖仓库外文件（脚本、模块、权重路径除外的模型/数据可通过配置指定）。

## 4. 记录与告警
- 需求/指令追加到 `message.md`；阶段性进展与风险写入 `status_overview.md`；任务分解与勾选在 `project_todo.md`；具体差异/方案在 `rebuild_plan.md`。
- 发现潜在不公平（KV budget 计算、prefill 处理、采样策略等）先记录再改动。

## 5. 环境与习惯
- 默认环境：`lazy_evict` conda；遵守仓库 Python 3.9+ 规范，4 空格缩进，ASCII 优先。
- 长跑/多进程遵循 `VLLM_PROCESS_NAME_PREFIX=PD-L1_binder`。
