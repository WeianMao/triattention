# Release 前待办清单

## 代码清理

- [ ] 统一命名：speckv -> TriAttention（详见 [../code_cleanup/04_naming.md](../code_cleanup/04_naming.md)）
- [ ] rkv 包轻度重构改名（行为不变）（详见 [../code_cleanup/05_repo_structure.md](../code_cleanup/05_repo_structure.md)）
- [ ] 去除进程伪装代码（详见 [../scope/03_scope_exclude.md](../scope/03_scope_exclude.md)）
- [ ] 去除校准脚本和校准语料
- [ ] 校准结果文件重命名（去掉 aime 字样）
- [ ] 识别并去除实验性 flag 及对应代码
- [ ] 文件夹结构重新组织
- [ ] 硬编码路径替换（详见 [../code_cleanup/06_path_cleanup.md](../code_cleanup/06_path_cleanup.md)）

## 功能开发

- [ ] 修复 TriAttention_vLLM 的 bug
- [ ] 开发 SGLang 版本

## 测试

- [ ] 单元测试：验证 RKV 和 TriAttention rkv-style 在相同 budget + divide_length 配置下峰值 KV cache 一致
- [ ] 确保清理后代码行为不变（AB 对比测试）
- [ ] 排查 KV cache 状态重置 bug：多问题单进程推理时状态变量是否正确重置（详见 ../code_cleanup/flag_cleanup.md）
- [ ] 全局扫描敏感信息（完整关键词清单见 [../scope/03_scope_exclude.md](../scope/03_scope_exclude.md)）

## DFS Benchmark 代码修复（审查已通过）

代码逻辑正确、学术合规。以下 5 项需在 release 前修复：

- [ ] 硬编码路径 `/home/linxi/...` 替换（3 个文件）
- [ ] 重复代码 `build_prompt` 合并去重
- [ ] 裸 `except:` 改为 `except Exception:`
- [ ] 中文文档翻译或删除
- [ ] 删除内部开发日志 `PROGRESS_SUMMARY.md`

## 跨分支代码合并

- [ ] 从 `gptoss` 分支提取 GPT-OSS-20B 模型支持代码（monkeypatch、modeling）— Phase 1.5
- [ ] 从 `linxi-dev` 分支提取 DFS benchmark 代码
- [ ] 统一双 rkv 包（`R-KV/rkv/` 和 `R-KV/HuggingFace/rkv/`）→ 重组为 `kv_compress/` + `triattention/`
- [ ] 清理 `sys.path.insert()` hack，改为正规包结构
- [ ] 生成每个模型×数据集的校准 stats .pt 文件（需先跑 fullkv 再 build-stats）

## 敏感信息泄露风险审查

> 需要启动 agent 专项调查，逐项扫描并修复。

- [ ] **AIME 交叉校准泄露**：speckv_experiments 的 stats 路径、config 中是否暴露了"用 AIME25 校准 AIME24"的交叉关系（文件名、yaml 字段、注释等）
- [ ] **内部路径泄露**：`/data/rbg/`、`/home/weian/`、`/home/linxi/` 等路径残留（shell 脚本、yaml config、Python 代码、.pt 文件 metadata）
- [ ] **进程伪装残留**：`PD-L1_binder`、`mask_process_command` 等
- [ ] **内部命名残留**：`speckv`、`weian`、`linxi` 等内部开发名出现在公布代码中
- [ ] **stats .pt 文件内部字段**：`trace_root`、`dataset` 等 metadata 字段是否暴露校准数据来源

## 文档

- [ ] README
- [ ] 使用说明
- [ ] 复现指南
