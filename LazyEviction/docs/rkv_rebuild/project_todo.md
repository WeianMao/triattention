# RKV 重构 TODO（LazyEviction）

> 采用 `- [ ]` / `- [x]` 勾选。执行某项前补充可操作子步骤；发现关键风险先记录再修改。

## 1. 需求确认与素材整理
- [ ] 摘取 `message.md` 与用户要求，生成可执行规范（预算、prefill、不依赖外部、对齐脚本）。
- [ ] 清点 RKV 现有实现：`rkv_sharded_dispatch.py`、`rkv_sharded_runner.py`、相关 YAML（含默认超参与 budget 逻辑）。

## 2. 接口与配置设计（默认不变）
- [ ] 拟定 LazyEviction 版 RKV cfg schema（YAML + argparse），标注默认值与 LazyEviction 对齐点（budget 定义、prefill 处理、数据/提示词、解码、长度、attn/dtype）。
- [ ] 设计公平性开关：采样/贪心、num_samples/聚合、chat 模板切换、长度/截断模式、attn/dtype、seed/reset、trust_remote_code。

## 3. 调度/Runner 原型
- [ ] 定义新的 dispatch/runner 路径（放 `weian_development/` + `LazyEviction/weian_script/`），确保输出/日志目录隔离；路径/参数风格对齐 `run_sparse_prefill_keep_sharded_eval.sh`。
- [ ] 校准 KV budget 计数与 prefills 逻辑，必要时添加日志打印/断言，避免“RKV 占便宜”。

## 4. 验证与对比计划
- [ ] 设计 smoke 命令（单 shard、少量样本、最小预算），覆盖 budget 统计、prefill 处理、数据/模板切换、解码模式切换。
- [ ] 规划公平对比：与 LazyEviction 现有方法在相同设置下的小规模实验；评估口径记录（单样本 vs 多样本平均）。

## 5. 文档与脚本
- [ ] 更新 `status_overview.md`/`rebuild_plan.md`/`message.md` 与执行记录。
- [ ] 准备一键运行脚本与 YAML 占位（`LazyEviction/weian_script/`），确保默认不会覆盖旧实验；在脚本中标注依赖尚未完成的部分。

## 6. 收尾检查
- [ ] 运行 `python -m compileall` 覆盖改动文件；记录 smoke 结果或缺口。
- [ ] 列出剩余风险（预算定义、prompt/数据差异、评估口径）与后续步骤。
