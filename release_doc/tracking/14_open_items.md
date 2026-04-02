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
- [x] **DFS benchmark 代码审查**：已完成 — 代码逻辑正确，学术合规，可以公布。有 5 个需修复的问题：
  1. 硬编码路径 `/home/linxi/...`（3 个文件需替换）
  2. 重复代码 `build_prompt`（需合并去重）
  3. 裸 `except:` 需改为 `except Exception:`
  4. 中文文档需翻译或删除
  5. 内部开发日志 `PROGRESS_SUMMARY.md` 需删除
- [x] **实验框架选择**：已确认 — 以 `speckv_experiments/` 为 release 基础。覆盖论文全部主实验（Table 1/2, Figure 5 A-C）+ 消融（Table 3），含 math500 和 budget sweep。`weian_script/` 不公布，仅作内部参考。GPT-OSS（Phase 1.5）和 DFS（linxi-dev）两个缺口与框架选择无关，需从其他分支提取
- [ ] **第一阶段执行顺序**：具体步骤排序
- [x] **启动器文件命名方案**：已确认 — 详见 [../components/08_launcher.md](../components/08_launcher.md)
- [x] **Flag 清理**：已确认 — 14 个 flag 删除，其余保留并改名。详见 [../code_cleanup/flag_cleanup.md](../code_cleanup/flag_cleanup.md)。额外排查项：KV cache 状态重置 bug

## GPT-OSS-20B "Phase 1.5" 处理方案

**状态：已讨论，待启动调查**

GPT-OSS-20B 模型与其他模型有显著差异：
- 代码在 `gptoss` 分支，需要合并模型支持代码（monkeypatch、modeling）
- 使用不同的 conda 环境
- FlashAttention-3 on H100（其他模型用 flash_attention_2）

**决策**：当作 **1.5 阶段**处理 — 不阻塞其他模型的进度，但提前考虑方案。即：
- 第一阶段先完成 Qwen3-8B、DS-Qwen-7B、DS-Llama-8B 的代码整理
- GPT-OSS-20B 作为 1.5 阶段紧接着处理，不等到第二阶段
- 需要从 `gptoss` 分支提取相关代码并整合到 release 中

**待办**：需要启动 agent 调查 gptoss 分支的代码差异和合并方案（对话中断前尚未执行）。

## 已识别的 5 个 Gap

以下 gap 在对话中讨论确认，需要在执行阶段逐一解决：

| # | Gap | 说明 | 解决方案状态 |
|---|-----|------|-------------|
| 1 | GPT-OSS-20B 代码在 gptoss 分支 | 需合并模型支持代码（monkeypatch、modeling） | 待调查（见上方 1.5 阶段） |
| 2 | DFS benchmark 代码在 linxi-dev 分支 | 需合并数据生成/评估代码到 release | 代码审查已完成，有 5 个待修问题 |
| 3 | 校准 stats 文件 | 每个模型×数据集需要单独 .pt 文件；用户需先跑 fullkv 再 build-stats | **已确认**：混合方案（预生成 stats + 生成脚本），详见下方"校准 stats 处理方案" |
| 4 | 双 rkv 包混乱 | `R-KV/rkv/` 和 `R-KV/HuggingFace/rkv/` 需统一 | 已有方案：重组为 `kv_compress/` + `triattention/`（见 [../code_cleanup/05_repo_structure.md](../code_cleanup/05_repo_structure.md)） |
| 5 | sys.path hack | 多处 `sys.path.insert()` 需改成正规包结构 | **已确认**：完善 setup.py/pyproject.toml + 删除所有 hack，详见下方"sys.path 清理方案" |

## 实验框架 gap 分析（speckv_experiments 覆盖情况）

**已有分析结论**：

- **已覆盖模型**：Qwen3-8B、DS-Qwen-7B、DS-Llama-8B — speckv_experiments 中有对应脚本
- **缺失模型**：GPT-OSS-20B（在 gptoss 分支）、DFS benchmark（在 linxi-dev 分支）
- **Table 3 消融**：只有 Qwen3-8B — 论文只报告 Qwen3，这够了
- **H2O 和 StreamingLLM baseline**：不需要补充（用户明确说不管了）

## 待执行：用户最后的指令

对话中断前用户的最后问题："你说的这些 GAP 有什么东西是需要我来决定的吗？有没有已经有清晰的解决方案？如果没有的话能否让代理先调查方案让我来做决定"

**待办**：
1. 对上述 5 个 gap 逐一评估：哪些有清晰方案可直接执行，哪些需要先调查再让用户决策
2. Gap 4（双 rkv 包）：已有清晰方案（双包策略），可直接执行
3. Gap 1（GPT-OSS）：已完成调查，待用户确认合并方案
4. Gap 2（DFS）：审查已完成，5 个修复项可直接执行
5. Gap 3（校准 stats）：**已确认**，见下方方案
6. Gap 5（sys.path hack）：**已确认**，见下方方案

## 校准 Stats 处理方案

**状态：已确认**

### 决策：混合方案（方案 C）

1. **预生成主要模型的 stats 文件**放进 release — 用户开箱即用
2. **同时提供生成脚本** — 高级用户可自定义 budget、新模型等

### 关键约束：隐藏校准数据来源

**公布的代码和 stats 中绝对不能暴露"使用 AIME 数据做校准"这个信息。** 原因：AIME 同时是评估数据集，暴露会引发学术合规性质疑。

具体要求：
- **stats 文件名**：不能包含 aime 字样（如 `stats_budget_2048.pt`，不要 `aime25_stats_budget_2048.pt`）
- **stats 文件 metadata**：不能包含 `trace_root`、`dataset` 等暴露校准数据来源的字段
- **校准脚本的输入格式**：改为无模板纯文本输入（一段 raw text 直接送进模型），不带任何数据集特征（没有 AIME 的 question/answer 模板）

### 公布的校准流程

用户看到的流程：
1. 准备一段纯文本语料（任意文本，无特定格式要求）
2. 运行校准脚本，输入纯文本 → 输出 stats .pt 文件
3. 运行 TriAttention 时指定 `--triattention-stats-file` 加载 stats

### 内部转换脚本（不公布）

为了验证格式转换后校准结果仍然有效，需要：
1. 编写转换脚本：AIME 格式 → 无模板纯文本格式
2. 用转换后的纯文本重新生成 stats
3. 对比验证：新 stats 跑出的实验结果和原始 stats 一致
4. 此脚本仅用于内部测试，不公布

### 生成流程（3 步）

1. **跑 FullKV**（无压缩基准）→ 生成推理轨迹（jsonl）
2. **提取 Q/K 频域统计** → `rkv_sparse_round_calibrate.py` 从轨迹计算每个 attention head 的频域分布 → 保存为 .pt
3. **运行 TriAttention** 时指定 stats 路径

自动化：`speckv_experiments_cli_v2.py build-stats` 可一键完成第 2 步

### Stats 文件内容

PyTorch .pt 格式，包含：
- metadata（模型配置、RoPE 参数、计算精度 — **release 版需清理掉 trace_root 等敏感字段**）
- per-head 统计：每个 attention head 的 Q 分布特征（q_mean_real, q_mean_imag, q_abs_mean）

## sys.path 清理方案

**状态：已确认**

### 调查结果

全项目 363 处 sys.path 调用，release 范围内约 60 处需修。

### 分级处理

| 优先级 | 数量 | 问题 | 处理方法 |
|--------|------|------|---------|
| CRITICAL | 1 | `R-KV/SGLang/eval.py` 硬编码 `/tmp/kewan/...` | 删除或改为环境变量 |
| HIGH | 16 | R-KV/ 和 TriAttention_vLLM/ 中的项目根目录 hack | 完善 setup.py，用 `pip install -e .` 代替 |
| MEDIUM | 18 | latex2sympy 相对导入 hack（3 个副本各 7 处） | 添加 `__init__.py`，正规包导入 |
| LOW | 13+ | 文档构建/第三方代码 | 保留不动 |

### 执行方案

代码清理阶段统一处理：
1. 完善 `setup.py` / `pyproject.toml` 包声明（使用 `find_packages()`）
2. 删除所有 release 范围内的 sys.path hack
3. 用户通过 `pip install -e .` 安装
4. 预计工作量：~1 天
