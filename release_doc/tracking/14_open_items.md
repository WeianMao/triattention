# 待确认事项 + 已确认决策记录

## 已确认项

- [x] **evaluation 评估管线**：已确认 -- 13个文件 + latex2sympy/ 全部公布。详见 [../components/07_evaluation.md](../components/07_evaluation.md)
- [x] **R-KV 包重命名**：已确认 -- 双包策略：baseline 通用方法用中性名 `kv_compress/`，我们的方法用 `triattention/`。与目录结构一致，在 clean-room 阶段随目录重组一起完成。详见 [../code_cleanup/05_repo_structure.md](../code_cleanup/05_repo_structure.md)
- [x] **paper_visualizations/**：已确认不公布
- [x] **experiments/**：第一阶段不公布，第二阶段视情况
- [x] **硬编码路径替换策略**：已确认 -- 本地 model 路径替换为 HuggingFace hub 名，数据集路径替换为相对路径，缓存路径替换为环境变量 + 默认值。详见 [../code_cleanup/06_path_cleanup.md](../code_cleanup/06_path_cleanup.md)

## 待确认项

- [x] **数据集**：已确认 -- 不 release 数据集文件，只在 README 提供 HuggingFace 下载链接。详见 [../scope/datasets.md](../scope/datasets.md)
- [ ] **运行脚本硬编码路径**：`/data/rbg/users/weian/...`、本地 model 路径改成什么？HuggingFace hub 名称？（替换策略已确认，具体 hub 名称见 [../code_cleanup/06_path_cleanup.md](../code_cleanup/06_path_cleanup.md)）
- [x] **README 大纲**：已确认 -- 精致版（对标 MInference），含 demo 视频占位符。详见 [../components/readme_outline.md](../components/readme_outline.md)
- [x] **LICENSE**：已确认 — Apache 2.0
- [ ] **第一阶段执行顺序**：具体步骤排序
- [ ] **启动器文件命名方案**：`rkv_sharded` 等内部命名替换为正式名称的具体方案。详见 [../components/08_launcher.md](../components/08_launcher.md)
- [ ] **Flag 名清理**：起点脚本中 `rkv-style` 等 flag 名是否需要改名。详见 [../components/09_reference_script.md](../components/09_reference_script.md)
