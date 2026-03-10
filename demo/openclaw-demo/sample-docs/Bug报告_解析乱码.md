# Bug 报告：部分 PDF 解析后中文乱码

**报告人**: 刘洋 | **日期**: 2026-02-20 | **优先级**: P1 | **状态**: 临时修复已上线

---

## 一、问题描述

客户试用期间提供的 3 份合同 PDF（共 127 页），经 doc-parser 服务解析后，约 30% 的中文段落出现乱码，英文和数字内容正常。乱码表现为 Unicode 替换字符（U+FFFD）或错误的 CJK 字符映射。

## 二、影响范围

- **直接影响**：乱码段落无法正确分块和索引，导致这些内容在语义检索中完全丢失
- **间接影响**：用户对系统准确性的信任度下降
- **波及范围**：所有使用 WPS Office 导出且嵌入方正字体的 PDF 文件（根据客户反馈，约占其文档库的 15-20%）

## 三、复现条件

| 条件 | 详情 |
|------|------|
| 文件来源 | WPS Office 2023 导出的 PDF |
| 字体 | 嵌入方正小标宋简体、方正仿宋简体 |
| 编码 | 部分字体使用 CID 编码（非 Unicode ToUnicode 映射） |
| 解析器 | pdfplumber 0.11.x，使用 pdfminer 底层提取 |

**复现步骤**：
1. 使用 WPS Office 创建一份包含方正字体的文档
2. 导出为 PDF
3. 上传到 DocMind 系统
4. 查看解析结果，部分段落出现 □□□ 或乱码

## 四、根因分析

pdfplumber 底层使用 pdfminer 解析 PDF 内容流。pdfminer 依赖字体的 ToUnicode CMap 进行字符映射。问题在于：

1. WPS 导出的 PDF 中，部分方正字体使用了 **Adobe-GB1** CID 编码，但没有嵌入完整的 ToUnicode 映射表
2. pdfminer 在遇到缺失映射时，返回原始 CID 值而非正确的 Unicode 字符
3. 这导致约 30% 的中文字符无法正确解码

PyMuPDF (fitz) 使用了不同的字体解析策略（MuPDF 引擎内置了更完整的 CID 回退映射表），因此同样的文件用 PyMuPDF 可以正确提取大部分内容。

## 五、解决方案

### 5.1 临时方案（已实施）

在 doc-parser 中增加 fallback 逻辑：

```python
def extract_text(pdf_path: str) -> list[Section]:
    # 主路径：pdfplumber（表格提取更好）
    sections = pdfplumber_extract(pdf_path)

    # 检测乱码率
    garbled_ratio = count_garbled_chars(sections) / total_chars(sections)

    if garbled_ratio > 0.10:  # 乱码率超过 10%
        logger.warning(f"High garble ratio ({garbled_ratio:.1%}), falling back to PyMuPDF")
        fitz_sections = pymupdf_extract(pdf_path)
        sections = merge_extractions(sections, fitz_sections)

    return sections
```

合并策略：对于每个 page，如果 pdfplumber 提取的文本乱码率 > 10%，使用 PyMuPDF 的结果替换。表格数据仍优先使用 pdfplumber（表格检测更准确）。

**效果**：乱码率从 30% 降至 5%。剩余 5% 主要是罕见字体和手写体批注。

### 5.2 根本方案（待实施）

集成 OCR 兜底，当文本提取质量不达标时触发 OCR：

1. 在 fallback 后如果乱码率仍 > 5%，对该页面进行 OCR（PaddleOCR）
2. 将 OCR 结果与文本提取结果对比，选择置信度更高的版本
3. 在解析结果中标注每个段落的提取来源（text_extract / ocr）和置信度

**预估工作量**：3 人天
**性能影响**：OCR 路径增加 2-3 秒/页，但仅在检测到低质量提取时触发（约 15% 的 PDF）

### 5.3 长期方案

1. 向 pdfminer 社区提交 CID 回退映射的 PR（赵磊已在 GitHub 上开了 issue）
2. 建立字体兼容性测试集，覆盖主流中文字体（方正、华文、思源、微软雅黑等）
3. 考虑引入 Adobe PDF Services API 作为商业级 fallback（成本高，仅对 VIP 客户开放）

## 六、验证记录

| 测试文件 | 页数 | 修复前乱码率 | 修复后乱码率 | 提取方式 |
|----------|------|-------------|-------------|---------|
| 合同_甲乙_2026.pdf | 42 | 35% | 3% | pdfplumber + fitz fallback |
| 技术规范_V2.pdf | 85 | 28% | 6% | pdfplumber + fitz fallback |
| 采购协议_2025Q4.pdf | 12 | 31% | 4% | pdfplumber + fitz fallback |

## 七、当前状态

- ✅ 临时方案已合入 dev 分支（commit: a3f2e1d）
- ⚠️ 根本方案（OCR 兜底）待 PM 决策优先级
- ⚠️ pdfminer 社区 issue 已提交，等待响应
- 📌 王浩认为根本方案应在 M2 前完成，避免客户试用时遇到质量问题

---

**抄送**: 王浩、赵磊、张明
