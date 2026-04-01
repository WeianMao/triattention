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
- [ ] **实验框架选择**：speckv_experiments vs weian_script，待确认。已有 gap 分析结果，见下方"实验框架 gap 分析"
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
| 3 | 校准 stats 文件 | 每个模型×数据集需要单独 .pt 文件；用户需先跑 fullkv 再 build-stats | 待执行 |
| 4 | 双 rkv 包混乱 | `R-KV/rkv/` 和 `R-KV/HuggingFace/rkv/` 需统一 | 已有方案：重组为 `kv_compress/` + `triattention/`（见 [../code_cleanup/05_repo_structure.md](../code_cleanup/05_repo_structure.md)） |
| 5 | sys.path hack | 多处 `sys.path.insert()` 需改成正规包结构 | 待调查具体涉及文件并制定方案 |

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
3. Gap 1（GPT-OSS）：需要 agent 先调查 gptoss 分支代码差异
4. Gap 2（DFS）：审查已完成，5 个修复项可直接执行
5. Gap 3（校准 stats）：需要用户提供 fullkv 运行结果后才能生成
6. Gap 5（sys.path hack）：需要 agent 先调查涉及文件范围
