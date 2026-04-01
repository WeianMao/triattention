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
- [ ] 全局扫描敏感信息（完整关键词清单见 [../scope/03_scope_exclude.md](../scope/03_scope_exclude.md)）

## 文档

- [ ] README
- [ ] 使用说明
- [ ] 复现指南
