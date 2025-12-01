# FullKV 对齐/消融概览

> 面向不了解上下文的读者：说明目标、范围、规则。继承 `R-KV/docs/speckv_qwen` 规范，新增“默认不变、cfg 驱动”的安全要求。

## 1. 目标与交付
- 针对 R-KV 与 LazyEviction 全 KV Qwen 基准差异，设计可配置的消融项（见 `ablation_plan.md`），定位性能差距。
- 交付物：集中化的消融启动脚本（`R-KV/weian_script/ablations_fullkv/`），cfg 驱动开关（默认 = 旧逻辑），以及对应记录/计划。

## 2. 范围与安全
- 不修改现有实验默认行为；新增开关默认关闭。
- 核心文件若需改动，必须确保默认路径/超参与原先一致；必要时用隔离脚本。
- 长跑实验由用户执行；Agent 仅跑 smoke。

## 3. 继承的规范
- 文档/记录方式与 `R-KV/docs/speckv_qwen` 相同：告警优先、计划透明、结果留档。
- 编码风格、路径组织、停机条件同上。

## 4. 当前状态
- 文档体系已建立（dev_guidelines/status_overview/project_todo/message/ablation_plan）；尚未添加实际消融脚本或 cfg。
- 差异/优先级详见 `ablation_plan.md`（从 `R-KV/docs/rkv_fullkv_vs_lazy_ablation_plan.md` 汇总）。

## 5. 风险/关注
- 任何可能破坏可比性的改动（prompt/数据/模型/kv_budget/位置处理）需先记录并确认。
- cfg 默认值必须保持旧行为；如需修改核心文件，务必保证默认路径/超参与现状一致或采用隔离实现。
- 长跑由用户执行；Agent 仅跑 smoke，需记录命令与结果。
