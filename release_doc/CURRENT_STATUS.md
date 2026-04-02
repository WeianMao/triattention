# 当前工作状态（断点恢复文件）

> **所有 agent 必读**：开始工作前先读这个文件，了解当前进展。工作过程中及时更新本文件。

## 最后更新

- **时间**：2026-04-02
- **更新者**：Claude Opus agent（第 4 个接手的 agent）
- **对话状态**：执行计划+治理规范已完成，待就绪检查和文档重构

## 当前所处阶段

**阶段 1: Open Items 确认** — 基本完成，GPT-OSS 搁置不阻塞，剩余 1 个未讨论项（执行顺序，待 agent 规划）

## 5 个 Gap 状态总览

| # | Gap | 状态 | 下一步 |
|---|-----|------|--------|
| 1 | GPT-OSS-20B 合并 | ⏸️ **搁置** — Phase 1 不含 GPT-OSS，不阻塞 release | 后续单独处理 |
| 2 | 校准 Stats 文件 | ✅ **已确认** — 方案 C + 隐藏 AIME 来源 | 执行阶段实施 |
| 3 | DFS benchmark | ✅ 代码审查通过，5 个修复项可直接执行 | 执行阶段实施 |
| 4 | 双 rkv 包统一 | ✅ 方案已定（kv_compress + triattention） | 执行阶段实施 |
| 5 | sys.path 清理 | ✅ **已确认** — 完善 setup.py + 删除 hack | 执行阶段实施 |

## ⚠️ GPT-OSS 阻塞项详情

### 已知信息（代码层面推断）

- gptoss 分支与 main 差异 586 文件，核心改动不多：
  - `modeling.py` — 新增 ~450 行：GPTOSSAttention
  - `monkeypatch.py` — 新增 ~20 行：replace_gptoss()
  - `speckv_experiments/scripts/gptoss/` — 实验脚本 ~1149 行
- 代码用 try-except 导入，不影响 Llama/Qwen
- 需要 transformers 4.57+（代码 `from transformers.models.gpt_oss` 要求）

### 关键风险（用户指出）

gptoss 分支把 `attn_implementation` 改成了 `eager`，但这**不是**说它在用低效 attention。实际情况是：
1. 协作者升级了 transformers 版本
2. 用上了 **FlashAttention-3**（不是 FA2）
3. H100 上 FA3 通过新版 transformers 内部 dispatch，所以配置虽写 eager 但实际是高效的
4. 这意味着环境依赖比纯代码 cherry-pick 复杂得多

### 看不到的信息（无法从代码推断）

- 协作者实际使用的 conda 环境配置
- flash-attn 精确版本（2.x 还是 3.x？hopper 专用版？）
- transformers 精确版本
- CUDA 版本和编译配置
- FA3 的启用方式（transformers 自动 dispatch？还是手动安装 flash-attn 3.x？）

### 需要向协作者确认的问题清单

1. **flash-attn 版本**：用的是 flash-attn 2.x 还是 3.x（hopper 版）？`pip show flash-attn` 输出是什么？
2. **transformers 版本**：精确版本号？`pip show transformers` 输出？
3. **CUDA 版本**：`nvcc --version` 和 `nvidia-smi` 输出？
4. **FA3 启用方式**：是 transformers 自动检测 H100 dispatch 的，还是需要额外安装/配置？
5. **attn_implementation=eager 的原因**：为什么配置写 eager？是 transformers 新版 GPT-OSS 模型类不支持 flash_attention_2 关键字？还是 FA3 走了不同的 code path？
6. **能否导出环境**：`conda list` 或 `pip freeze` 输出，方便我们复现环境？

**用户决策**：拿到以上信息后，才能评估 GPT-OSS 合并的真实复杂度和方案选择。

## 已完成的工作

- [x] 文档系统搭建（release_doc/ 目录结构、guidelines）
- [x] 所有 Open Items 确认（除 GPT-OSS 环境和下方 2 个未讨论项外）：
  - R-KV 包重命名 → 双包策略（kv_compress + triattention）
  - 硬编码路径替换策略 → HF hub 名 / 相对路径 / 环境变量
  - 数据集 → 不放进 repo，代码中自动从 HuggingFace 下载
  - README 大纲 → 精致版（对标 MInference），含 demo 视频占位符
  - LICENSE → Apache 2.0
  - 启动器文件命名 → dispatch.py / worker.py 等，删除进程伪装
  - Flag 清理 → 14 个删除，保留的改名（方法特有加 triattention- 前缀）
  - 实验 setting 矩阵 → 论文全部主实验 + 消融全部公布
  - GPT-OSS 确认是 20B
  - Figure 5 flag 差异 → 不存在差异
  - DFS benchmark 代码审查 → 通过，有 5 个待修问题
  - 校准 Stats → 方案 C（预生成 + 脚本），隐藏 AIME 来源，公布无模板纯文本输入
  - sys.path 清理 → 完善 setup.py + 删除 hack
- [x] Agent 模型策略变更：全程 Opus（不再用 Sonnet+Opus 混合）
- [x] 测试/发布流程拆分：单元测试 → 内部公布 → 头对头对比 → 正式公布
- [x] PD-L1 GPU 占座进程 → 已记录到 execution/12_environment.md

## 未讨论的项目

1. ~~**实验框架选择**~~：✅ 已确认 — 以 speckv_experiments 为 release 基础，weian_script 不公布
2. **第一阶段执行顺序**：具体步骤排序还没开始讨论

## 下一步行动

1. ~~**规划执行阶段拆分**~~ — ✅ 已完成。新计划在 `plan/execution_plan.md`（5 阶段 24 步 + 5 检查点）
2. **用户审阅计划** — 确认后开始执行 Phase 1
3. **Phase 1: Foundation** — 创建 worktree + triattention conda 环境

## 新增文档

| 文件 | 内容 |
|------|------|
| `plan/execution_plan.md` | 完整执行计划（替换旧 stages/README.md） |
| `plan/dev_standards.md` | 开发规范（角色职责、中断恢复、纠偏机制、命名、敏感词） |
| `plan/checkpoint_protocol.md` | 检查点协议（5 个检查点的详细检查项和失败处理） |
| `plan/execution_log.md` | 执行日志（空，等执行时填充） |

## 关键文档指引

| 要了解什么 | 去看哪个文件 |
|-----------|------------|
| 项目全貌和发布策略 | scope/01_overview.md |
| 目标 repo 结构 | code_cleanup/05_repo_structure.md |
| 所有已确认/待确认决策 | tracking/14_open_items.md |
| 校准 Stats 完整方案 | tracking/14_open_items.md §校准 Stats 处理方案 |
| sys.path 清理方案 | tracking/14_open_items.md §sys.path 清理方案 |
| 执行阶段和流程 | stages/README.md |
| 工作规范 | guidelines/ 目录下所有文件 |
| Release 前待办清单 | execution/15_checklist.md |
| 实验 setting 矩阵 | scope/experiment_settings.md |
| 环境信息和 PD-L1 | execution/12_environment.md |
