# FullKV 消融 TODO（单 Agent）

> 使用 `- [ ]` / `- [x]` 记录；每次执行前补充可操作分解与预期输出。遵守 `speckv_qwen` 规范：发现关键差异先告警。

## 1. 差异梳理与假设确认
- [ ] 逐条提取 `ablation_plan.md` 差异 → 为每个差异标注：优先级、预期影响、需要的 cfg 字段名、默认值（旧逻辑）、落地位置（cfg+runner）。
  - 子步骤：  
    1) 列表化七大差异（数据/Prompt、解码、聚合、attn/dtype、长度、trust_remote_code、seed/reset）。  
    2) 对照现有脚本/runner，标记可插入 cfg 的具体键和值。  
    3) 在 `project_todo.md` 追加对应的执行项/子项。

## 2. 消融设计与实现（按优先级推进，默认旧逻辑）
- [ ] 数据/Prompt 对齐开关  
  - 设计：新增 cfg（如 `use_lazy_chat_prompt`, `lazy_dataset_path`），默认 False/旧路径。  
  - 实施：修改 runner 读取 cfg，默认 plain+AIME25；在 `ablations_fullkv/` 创建启动脚本引用新 cfg。  
  - 检查：默认路径不变；开启后使用 chat 模板和 LazyEviction 数据。
- [ ] 解码方式开关  
  - 设计：cfg `do_sample`，默认 True；若 False 切温度=0/top_p=1。  
  - 实施：runner 读取并传递；脚本示例写入新 cfg。  
  - 检查：默认保持采样，切换后日志打印生效。
- [ ] 多样本/聚合策略  
  - 设计：cfg `num_samples`、`eval_mode`（avg/best_of/majority），默认现有 avg。  
  - 实施：更新 eval 调用或在运行脚本中覆盖；默认不变。  
  - 检查：默认 8-draw 平均，切换后评估逻辑无副作用。
- [ ] Attn/DType 对齐开关  
  - 设计：cfg `attn_implementation`, `load_dtype`，默认 flash-attn2+bf16；可设 sdpa+fp16。  
  - 实施：runner 读取并应用；必要时隔离新启动脚本避免影响旧配置。  
  - 检查：默认值保持旧行为；切换后确认模型加载成功。
- [ ] 长度/截断控制  
  - 设计：cfg `max_length`（默认 32768）与可选 `max_new_tokens`；记录截断率。  
  - 实施：runner 支持两者，若同时配置则以新字段优先。  
  - 检查：默认逻辑不变；切换时打印截断监控。
- [ ] trust_remote_code / tokenizer 设置  
  - 设计：cfg `trust_remote_code`，默认旧值；一次性排查。  
  - 实施：runner 加载模型时读取；默认路径不变。  
  - 检查：默认行为一致；开启后无异常。
- [ ] Seed / 缓存复位  
  - 设计：cfg `seed`（默认 666/现值）、`reset_cache_each_batch`（默认 False）。  
  - 实施：runner 读取并应用；默认不变。  
  - 检查：默认一致；开关后日志显示。

## 3. 脚本与隔离
- [ ] 在 `R-KV/weian_script/ablations_fullkv/` 为每组消融生成启动脚本骨架（可共享 cfg），命名含差异点，避免覆盖旧脚本。
- [ ] 若修改核心 runner，验证默认 cfg 路径完全复现原行为；必要时提供隔离 runner 版本。

## 4. 验证流程
- [ ] 静态检查：`python -m compileall` 覆盖改动文件（脚本/runner）。  
- [ ] Smoke：对每个新增开关至少跑 1 shard / 1 样本 / 最小数据（如单 qid），记录命令、配置、GPU、结果/截断率。  
- [ ] 如 smoke 不可执行，记录原因与缺口（权限/资源/路径）。

## 5. 文档与收尾
- [ ] 在 `status_overview.md` 更新进展/风险；在 `message.md` 追加新指令或发现。  
- [ ] 汇总已做消融与剩余项；确认默认行为未变。  
- [ ] 提交前复核：默认 cfg 是否等价旧实验、脚本路径是否隔离、smoke 是否记录。
