# Release 前待办清单

## 深度审查发现的问题（2026-04-02）

### 会导致 break 的问题（agent 执行时必须处理）

- [ ] **stats .pt metadata 含内部路径**：`rkv_sparse_round_calibrate.py` L337-341 的 `trace_root`, `model_path` 字段暴露 `/data/rbg/...`。预生成 stats 时必须 strip
- [ ] **`validate_stats_metadata()` 检查 prompt_template**：stats 中的 template 字符串必须和推理时 byte-identical，rename 时不能改动 template 内容
- [ ] **15+ 文件有 `weian_development.*` import**：不只是 CLI，整条链（worker、pruner、calibrate、rkv/compression）都有，需系统性重写 import
- [ ] **`rkv/compression/speckv.py` 不纳入 release**：generate-wrapper 路径，论文不用。源文件保留不动，release repo 不包含此文件。确认 release 版 `kv_compress/` 的 `__init__.py` 不引用它
- [ ] **校准脚本**：新写一个 raw text 输入的校准脚本用于 release。现有的 `rkv_sparse_round_calibrate.py` 和 `capture_qk_distributed.py` 均不公布
- [ ] **内部验证脚本（不公布）**：AIME 格式 → 纯文本 → 重新生成 stats → 对比确认和原始 stats 结果一致
- [ ] **`position_offset_patch.py` 条件 import**：flag 已在删除清单，但 worker 中 L664 的 import 也要一起删
- [ ] **`rkv_cache_utils.py` / `reset_model_cache`**：确认是否和被删 flag `reset_cache_each_batch` 绑定，如是则一起删

### 配置/数据不一致

- [ ] **DS-Llama-8B 默认 budget 不匹配**：`defaults.yaml` 默认 2048，但论文 Table 1 是 512。需在脚本中显式传 `--budget 512` 或修改 per-model 默认值
- [ ] **`budgets.yaml` 含 1536**：论文 Figure 5 没有此档（512, 1024, 2048, 3072, 4096），需删除

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

### 开发阶段（单元测试，轻量 GPU）

- [ ] **Level 1：纯评分函数等价性测试**（无需模型，~1秒）
  - 合成张量构造 K_unrot / K_rot
  - 原始 `score_keys_for_round()` vs release `compute_scores_pytorch()`
  - tolerance: `atol=1e-5`
  - 注意：原始对 K_unrot 算分，release 对 K_rot 算分，需正确对齐输入
- [ ] **Level 2：完整 pruner + 真实 stats 等价性测试**（不加载模型权重，~5秒，<100MB GPU）
  - 加载现有 stats .pt + 模型 config.json（仅读 RoPE 参数）
  - `SparseRoundPruner`（原始）vs `TriAttentionCompressor`（release）
  - 比较 per-head 分数 + keep/evict 索引
  - normalize_scores 和 per-head 采样范围两边必须对齐
- [ ] 单元测试：验证 RKV 和 TriAttention rkv-style 在相同 budget + divide_length 配置下峰值 KV cache 一致
- [ ] 排查 KV cache 状态重置 bug：多问题单进程推理时状态变量是否正确重置（详见 ../code_cleanup/flag_cleanup.md）
- [ ] 全局扫描敏感信息（完整关键词清单见 [../scope/03_scope_exclude.md](../scope/03_scope_exclude.md)）

### 内部发布后（GPU 推理测试）

- [ ] **Level 3：真实模型端到端对比**（~16GB GPU，~60秒）
  - 加载完整模型跑 decode，在压缩触发点对比 keep/evict 结果
  - 原始代码 vs release 代码必须数值一致
- [ ] 确保清理后代码行为不变（完整 AB 头对头对比测试，主要模型×主要 setting）

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

## 环境与依赖

- [ ] **新建 `triattention` conda 环境**（Python 3.10，cover DS-Qwen-7B / DS-Llama-8B / Qwen3-8B）
- [ ] 生成 `requirements.txt` 与 `triattention` 环境对齐
- [ ] `torch` 放入 requirements.txt（注释说明不同 CUDA 版本的安装方式）
- [ ] 核心依赖：`flash-attn>=2.5.8`, `transformers>=4.48.1`, `datasets`, `huggingface-hub`, `pyyaml`, `numpy`, `tqdm`
- [ ] 评估管线依赖：`pebble`, `latex2sympy2`, `word2number`, `antlr4-python3-runtime==4.7.2`, `sympy`
- [ ] 验证：`triattention` 环境下能跑通 Level 1+2 单元测试
- [ ] 版本号 `0.1.0` 不变

## 文档

- [ ] README
- [ ] 使用说明
- [ ] 复现指南
