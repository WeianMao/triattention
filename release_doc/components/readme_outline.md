# README 大纲

## 决策状态：已确认

## 方案：精致版（对标 MInference）

参考项目：MInference（微软）、R-KV（同一作者）

## 章节结构

### 1. 标题 + 徽章
- 项目名：TriAttention
- 徽章：Paper (arXiv)、License、Python version
- 一行 tagline

### 2. 导航链接
- Paper | Project Page | Demo Video
- Demo Video 先放占位符，后期补上

### 3. TL;DR / Overview
- 2-3 句话说明：做什么、核心方法、关键结果
- 例："TriAttention achieves X% accuracy at Y% KV cache budget..."

### 4. Demo 视频
- 嵌入 demo 视频（先占位符，后期替换）
- `<!-- TODO: 替换为实际 demo 视频链接 -->`

### 5. 方法概览图
- 方法流程图或核心思路图（1 张）
- 先占位符，后期补上

### 6. News
- Release 日期
- 论文投稿/接收状态（如适用）

### 7. 安装
```bash
git clone https://github.com/xxx/TriAttention.git
cd TriAttention
pip install -r requirements.txt
```
- 依赖说明（flash-attn、transformers 版本等）

### 8. Quick Start
- 最小可运行代码（~10 行）
- 展示如何用 TriAttention 做一次推理

### 9. 支持模型列表
- 表格形式列出支持的模型

| 模型 | 状态 |
|------|------|
| DeepSeek-R1-Distill-Qwen-7B | 已验证 |
| DeepSeek-R1-Distill-Llama-8B | 已验证 |
| Qwen2.5-* | 已验证 |
| ... | ... |

具体模型列表待确认（需要 agent 调查实验脚本中用了哪些模型）。

### 10. 结果表格
- 1-2 个关键结果表（accuracy vs cache budget）
- 对比 baseline（R-KV、SnapKV、H2O、StreamingLLM、FullKV）
- 数据来自论文

### 11. 复现指南
- 如何复现论文中的主要实验
- 指向 scripts/ 下的具体脚本
- 数据集自动下载说明（首次运行自动从 HuggingFace 下载）

### 12. TODO / Roadmap
- [ ] vLLM 后端支持
- [ ] SGLang 后端支持
- [ ] 更多模型支持
- 信号活跃开发

### 13. Citation
```bibtex
@article{...}
```

### 14. Acknowledgements
- 致谢（如适用）

### 15. License
- 具体 license 类型（待确认，见 [../tracking/14_open_items.md](../tracking/14_open_items.md)）

## 对比同类项目

| 章节 | TriAttention | MInference | R-KV | SnapKV |
|------|-------------|-----------|------|--------|
| 标题+徽章 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 导航链接 | :white_check_mark: | :white_check_mark: | :x: | :x: |
| TL;DR | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| Demo 视频 | :white_check_mark:（占位符） | :white_check_mark: | :x: | :x: |
| 方法图 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| News | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| 安装 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Quick Start | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 支持模型 | :white_check_mark: | :white_check_mark: | :x: | :x: |
| 结果表格 | :white_check_mark: | :x:（defer to paper） | :white_check_mark: | 图片 |
| 复现指南 | :white_check_mark: | :x: | :white_check_mark: | :x: |
| TODO | :white_check_mark: | :x: | :white_check_mark: | :white_check_mark: |
| Citation | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| License | :white_check_mark: | :white_check_mark: | :x: | :x: |

## 占位符清单

以下内容先放占位符，后期替换：
- [ ] Demo 视频链接
- [ ] 方法概览图
- [ ] GitHub repo URL
- [ ] arXiv 论文链接
- [ ] 支持模型完整列表
- [ ] 结果表格数据
- [ ] BibTeX citation
- [ ] License 类型
