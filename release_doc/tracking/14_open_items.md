# 待确认事项 + 已确认决策记录

## 已确认项

- [x] **evaluation 评估管线**：已确认 -- 13个文件 + latex2sympy/ 全部公布。详见 [../components/07_evaluation.md](../components/07_evaluation.md)
- [x] **R-KV 包重命名**：已确认 -- 双包策略：baseline 通用方法用中性名 `kv_compress/`，我们的方法用 `triattention/`。与目录结构一致，在 clean-room 阶段随目录重组一起完成。详见 [../code_cleanup/05_repo_structure.md](../code_cleanup/05_repo_structure.md)
- [x] **paper_visualizations/**：已确认不公布
- [x] **experiments/**：第一阶段不公布，第二阶段视情况
- [x] **硬编码路径替换策略**：已确认 -- 本地 model 路径替换为 HuggingFace hub 名，数据集路径替换为相对路径，缓存路径替换为环境变量 + 默认值。详见 [../code_cleanup/06_path_cleanup.md](../code_cleanup/06_path_cleanup.md)

## 待确认项

- [x] **数据集**：已确认 -- 不 release 数据集文件，只在 README 提供 HuggingFace 下载链接。详见 [../scope/datasets.md](../scope/datasets.md)
- [x] **运行脚本硬编码路径**：已确认 — 替换策略见 [../code_cleanup/06_path_cleanup.md](../code_cleanup/06_path_cleanup.md)
- [x] **README 大纲**：已确认 -- 精致版（对标 MInference），含 demo 视频占位符。详见 [../components/readme_outline.md](../components/readme_outline.md)
- [x] **LICENSE**：已确认 — Apache 2.0
- [x] **公布的 setting 清单**：已确认 — 论文全部主实验（Table 1,2, Figure 5）+ 消融（Table 3）+ DFS benchmark 全部公布。详见 [../scope/experiment_settings.md](../scope/experiment_settings.md)
- [x] **GPT-OSS 模型**：已确认是 GPT-OSS-20B（`openai/gpt-oss-20b`），不是 120B。代码中 deepconf 示例文件的 120B default 是历史遗留，实际实验用的是 20B
- [x] **Figure 5 budget sweep flag 差异**：已确认 — 不存在差异。Table 1 和 Figure 5 都使用 `--rkv-style-compression` + `--rkv-style-slack-trigger`。之前发现的 `--rkv-aligned-budget` 脚本是 DS-Qwen-7B 的另一种实现路径，不是论文 Figure 5 用的。Qwen3-8B 的 budget sweep 通过 `speckv_experiments_cli_v2.py` 执行，CLI 默认即为 rkv_style
- [ ] **DFS benchmark 代码审查**：代码在 linxi-dev 分支，release 前需审查逻辑正确性和学术合规性
- [ ] **实验框架选择**：speckv_experiments vs weian_script，待确认
- [ ] **第一阶段执行顺序**：具体步骤排序
- [x] **启动器文件命名方案**：已确认 — 详见 [../components/08_launcher.md](../components/08_launcher.md)
- [x] **Flag 清理**：已确认 — 14 个 flag 删除，其余保留并改名。详见 [../code_cleanup/flag_cleanup.md](../code_cleanup/flag_cleanup.md)。额外排查项：KV cache 状态重置 bug
