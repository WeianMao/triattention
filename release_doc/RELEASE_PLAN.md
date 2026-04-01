# TriAttention Release Plan

## 1. Release Target
- GitHub public repo, name: **TriAttention**
- Release branch: `release` (branched from `main`)
- Local code stays unchanged, all cleanup only on release branch

## 2. Release 分阶段策略

- **第一阶段**：TriAttention 核心方法 + baselines（不含 kvpress）
- **第二阶段**：kvpress 相关代码和实验结果（检查、整理完后公布）

kvpress 相关代码目前只在 `dc1/rebuttal` 分支，需先转移到 main 上，第一阶段公布完后再检查并公布。

### 第二阶段涉及的代码（triattention_press/）

**转移到 main 的内容**：
- `triattention_press.py` — 主算法
- `triattention_press_v2.py` — V2 生产版本
- `scoring.py` — 频域评分引擎
- `__init__.py` — 导出接口
- `extra_weight.py`（improvements/ 中唯一有活跃依赖的）
- 测试文件（在主分支上通过后才 release）
- 校准结果 .pt 文件（改名去 aime）
- 最终版实验脚本：
  - `run_longbench_qwen3_final_v2.sh`（LongBench 最终结果）
  - `run_clnorm_only_full_12subtask.py`（LongBench 最新最优）
  - `run_ruler_norm_v2.py`（RULER 最终结果）

**不转移 / 不公布**：
- `improvements/` 其他文件（功能已合并到主类，中间产物）
- 废弃的 `score_norm.py`
- 消融/历史实验脚本（ablation A/B/D/F 等探索性实验）
- 校准脚本（`calibrate_qwen3.py` 等）
- `kvpress/` 外部库（作为依赖让用户自己安装，不包含在 repo 中）

**最终报告的配置**：
- LongBench 最优：`aligned_max_a1.5_w8`（α=1.5, max+max, window=8, 得分 42.35）
- RULER 最优：`aligned_max_a1.5_w8_norm`（加 normalize_scores=True, 得分 66.08）

### 第二阶段代码整理要求

与第一阶段相同的原则（参见 4.5 节）：**公布的代码是全新整理过的，不是内部代码的直接 copy。**

具体要求：
- **去掉实验性功能**：没有在最终版本使用的功能代码要删除（如废弃的 score_norm.py、kernel_pooling 如果最终未采用等）
- **命名规范化**：内部开发过程中的不规范命名要改成正式名字（如 `extra_weight` 改为正式术语）。不能暴露中间实验过程的命名
- **文件整理**：中间开发文件不直接公布，该合并的合并（如 extra_weight.py 的逻辑合并到主类）、该删的删
- **improvements/ 目录不公布**：功能已内化到主类，目录本身是开发过程产物
- **同样遵守重构原则**：能小改就小改，重构前报告，重构后 AB 测试 + 单元测试

## 3. What to Release（第一阶段）

### 3.1 TriAttention_vLLM/
- Main TriAttention implementation on vLLM
- Status: 有 bug 待修，还需要开发 SGLang 版本，暂不 release
- 待完成后再纳入 release

### 2.2 R-KV/ 中的内容
- **我们的方法（TriAttention/SpeckV）**：所有 speckv 相关代码和脚本
  - 统一使用 **TriAttention** 这个名字（不再叫 speckv）
- **官方 baseline 方法**：rkv, snapkv, h2o, streamingllm, fullkv
- **SparsePrefillKeep**：这是我们自己方法的一个变种，不是 RKV 的 baseline
- `rkv/` 包（核心压缩实现库）：需要 release，但要轻度重构改名，代码行为不能变

### 2.3 多个 Setting 都要公布
- 不同模型（Qwen, LLaMA 等）
- 不同数据集
- 不同配置变体（perhead 等）

## 3. What NOT to Release（排除清单）

### 3.1 校准相关
- ❌ 校准语料（calibration data）
- ❌ 校准脚本（calibration scripts）
- ❌ 任何涉及在 AIME 数据集上校准的信息
- ✅ 校准结果文件可以 release，但文件名/路径中**不能出现 "aime" 字样**（需要重命名或泛化）

### 3.2 敏感内容
- ❌ 数据敏感内容
- ❌ 进程伪装相关代码（`PD-L1_binder`、`mask_process_command`、`process_utils.py` 等）

### 3.3 历史遗留
- ❌ `deepconf/` — 历史遗留文件夹，与当前开发无关
- ❌ `TriAttention/` — 空目录

### 3.4 实验性 Flag 和代码
- ❌ 试验过但后来没有采用的 flag 及其对应代码实现
- 本地代码可以保留这些，但 release 版本中要去掉
- 需要后续步骤：对比起点脚本使用的 flag vs 代码中所有 flag，识别实验性的，列出确认后删除

### 3.5 内部开发命名（核心原则）

**公布的代码是全新整理过的，不是把现有代码原样搬过去。** 所有命名必须统一为 TriAttention 体系，内部开发名字不应出现在公布版本中。

命名清理示例：

| 类型 | 内部名字（不公布） | 公布名字 |
|------|-------------------|----------|
| 文件名 | `speckv_rkv_style.py` | `triattention.py` |
| 类名 | `SpeckVRKVStyle` | `TriAttention` |
| 函数名 | `apply_speckv_rkv_style_patch()` | `apply_triattention_patch()` |
| 配置key | `speckv_budget` | `triattention_budget` |
| 方法标识 | `method: speckv` | `method: triattention` |
| 脚本名 | `run_speckv_aime24_*.sh` | `run_triattention_*.sh` |

**原则：**
- 能通过改变量名/函数名解决的，就小改，不要重构
- 实在不行才重构。重构前必须报告：要改什么、为什么
- 重构后必须做严格的 AB 对比测试 + 单元测试，确保行为一致

## 4. 起点脚本（Reference Script）

**主实验脚本**：`R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`

关键 flag 组合：
- `--rkv-style-compression`（在 attention layer 内部触发压缩，和 RKV 一样的方式）
- `--rkv-style-slack-trigger`（slack trigger）
- `--per-head-pruning`（per-head 级别的剪枝）
- `--sparse-normalize-scores`（score 归一化）
- `--divide-length 128`

关键参数值：
- Model: DeepSeek-R1-Distill-Qwen-7B
- Budget: 2048
- Dataset: AIME24, sampled 8 draws, seed=888
- Attention: flash_attn2 + bfloat16

**重要说明**：
- 所有 setting 都要以这个脚本为准，不要搞错 flag 组合
- 其他 setting（不同模型、不同数据集）也要公布，但都以这个脚本的 flag 模式为基准
- 脚本名中的 "aime" 在 release 时需要泛化处理

## 5. 技术发现记录（重要）

### 5.1 KV Cache 峰值对齐

经过多轮代码审查确认：

**RKV 和 TriAttention（SpeckV rkv-style）的 KV cache 峰值是对齐的，都是 ~2176（budget + divide_length）。**

原因：
- RKV 官方实现在实验框架中也受 `divide_length` 门控（`CausalLM_forward` 中 `self.length % self.config.divide_length == 0`）
- 不是每个 token 都压缩，而是每 `divide_length`（默认 128）步才触发一次
- `r1_kv.py` 的 `update_kv` 虽然每次都检查，但上层 `modeling.py` 的 `CausalLM_forward` 通过 `config.compression` flag 控制是否真正执行压缩
- `compression = True` → 压缩；`compression = False` → 不压缩，cache 自由增长；`compression = None` → 初始状态

关键代码位置：
- 门控逻辑：`R-KV/HuggingFace/rkv/modeling.py` line 638: `is_newline = self.length % self.config.divide_length == 0`
- Flag 传播：`R-KV/HuggingFace/rkv/modeling.py` line 647: `layer.self_attn.config.compression = is_newline`
- Attention 层三分支：`R-KV/HuggingFace/rkv/modeling.py` lines 146-191

⚠️ 注意：之前的分析曾错误认为 RKV 峰值是 2049（每 token 压缩），这是因为只看了 `r1_kv.py` 底层实现，没追踪到 `CausalLM_forward` 的门控逻辑。

### 5.2 脚本对比

| 脚本 | 方式 | 峰值 | 和 RKV 对齐？ |
|------|------|------|--------------|
| `norm_aligned_perhead`（脚本 #1） | `--rkv-style-compression` + `--rkv-style-slack-trigger` | ~2176 | ✅ 对齐 |
| `norm_aligned_budget_perhead`（脚本 #2） | `--rkv-aligned-budget`（generate wrapper 风格） | ~2080 | ❌ 不对齐 |
| 官方 RKV | 原生 attention layer 压缩 + divide_length 门控 | ~2176 | — |

## 6. Release 前待办

### 6.1 代码清理
- [ ] 统一命名：speckv → TriAttention
- [ ] rkv 包轻度重构改名（行为不变）
- [ ] 去除进程伪装代码
- [ ] 去除校准脚本和校准语料
- [ ] 校准结果文件重命名（去掉 aime 字样）
- [ ] 识别并去除实验性 flag 及对应代码
- [ ] 文件夹结构重新组织

### 6.2 功能开发
- [ ] 修复 TriAttention_vLLM 的 bug
- [ ] 开发 SGLang 版本

### 6.3 测试
- [ ] 单元测试：验证 RKV 和 TriAttention rkv-style 在相同 budget + divide_length 配置下峰值 KV cache 一致
- [ ] 确保清理后代码行为不变

### 6.4 文档
- [ ] README
- [ ] 使用说明
- [ ] 复现指南

## 待确认事项（Open Items）

以下事项需要逐个和用户确认后才能执行：

- [x] **evaluation 评估管线**：已确认（见下方 §10）
- [ ] **数据集**：用户自己下载，我们提供链接。链接是什么？合规性确认？
- [x] **R-KV 包重命名**：已确认 — 双包策略：baseline 通用方法用中性名 `kv_compress/`，我们的方法用 `triattention/`。与 §8 目录结构一致，在 clean-room 阶段随目录重组一起完成
- [ ] **运行脚本硬编码路径**：`/data/rbg/users/weian/...`、本地 model 路径改成什么？HuggingFace hub 名称？
- [ ] **README 大纲**：内容规划
- [ ] **LICENSE**：暂定 Apache 2.0，需和导师确认
- [ ] **第一阶段执行顺序**：具体步骤排序
- [ ] **paper_visualizations/**：不公布（已确认）
- [ ] **experiments/**：第一阶段不公布，第二阶段视情况（已确认）

## 7. 实施方案：Worktree + Clean-room 两阶段

### 阶段 0：创建 worktree
```bash
git checkout main
git branch release/public
git worktree add ../dc1-release release/public
```
结果：
- `dc1/` — 主开发目录，不动，正常开发
- `dc1-release/` — release 分支，独立目录，在这里做所有清理工作

两个目录共享同一个 git，可以同时打开、同时跑测试对比。
`dc1/` 里的 gitignore 文件（如校准文件）完全不受影响。

### 阶段 1：在 dc1-release/ 中整理
- 删除不需要的文件和目录
- 重命名（speckv → triattention, aime → benchmark 等）
- 去除敏感信息（内部路径、进程伪装、校准脚本等）
- 去除实验性 flag 和对应代码
- 重新组织文件夹结构
- 添加 LICENSE, README, .gitignore
- 如需跑对比测试，从 dc1/ 手动复制或 symlink gitignore 的资源文件

### 阶段 2：对比验证
- 在 dc1/ 和 dc1-release/ 同时跑测试，验证代码行为一致
- 全局扫描敏感信息（aime, weian, /data/rbg, PD-L1 等）
- 单元测试：RKV 和 TriAttention 峰值 KV cache 一致

### 阶段 3：Clean-room 发布
```bash
mkdir ~/triattention-public
cp -r ../dc1-release/需要的文件 ~/triattention-public/
cd ~/triattention-public
git init && git add . && git commit -m "Initial release"
# push 到 GitHub public repo
```
最终 public repo 干净无历史。

### 阶段 4：清理
```bash
git worktree remove ../dc1-release  # 删除 worktree，不影响分支
```

## 8. Repo 结构

```
TriAttention/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── triattention/                   # 我们的方法（所有命名统一为 triattention）
│   ├── triattention.py             # 主实现（原 speckv_rkv_style.py）
│   ├── pruning_utils.py            # 核心工具（原 round_pruning_utils.py）
│   ├── stats_utils.py              # 统计验证
│   └── prompt_utils.py             # prompt 构建
│
├── kv_compress/                    # 通用 KV cache 压缩框架 + baseline 方法（原 baselines/）
│   ├── r1_kv.py
│   ├── snapkv.py
│   ├── h2o.py
│   └── streamingllm.py
│
├── integration/                    # HuggingFace 集成
│   ├── modeling.py
│   ├── monkeypatch.py
│   └── utils.py
│
├── evaluation/                     # 评估管线（13个文件 + latex2sympy/，详见 §10）
│
├── scripts/                        # 运行/复现脚本
│   ├── run_eval.py
│   ├── dispatch.py
│   ├── calibrate.py
│   └── merge_shards.py
│
├── configs/                        # 实验配置（待精简）
│
├── calibration/                    # 校准结果 .pt 文件（文件名不含 aime）
│
├── data/                           # 数据集（待确认）
│
├── tests/                          # 单元测试（公布）
│   ├── test_triattention.py
│   ├── test_baselines.py
│   └── test_peak_cache_alignment.py
│
└── backends/                       # 后端集成（待开发，先留空）
    ├── vllm/
    └── sglang/
```

**不公布的文件**：
- `sparse_round_pruner_prefill_keep.py` — 废弃旧实现
- `rkv_speckv_generate.py` — 废弃旧实现
- `analysiskv.py` — 内部分析工具
- 校准脚本和校准语料
- 进程伪装代码（PD-L1_binder, mask_process_command 等）
- `weian_development/` 中的个人开发工具

## 8. 开发环境背景（接手人须知）

### 8.1 Replica 信息
- 当前开发在 **dc1**（`/data/rbg/users/weian/project/rl/dc1`），是 dc 的 parallel copy
- 原始项目：`/data/rbg/users/weian/project/rl/dc`

### 8.2 共享 Symlink 目录（不应 release）
以下目录是 dc 和 dc1 之间的共享 symlink，包含运行时产物，不应进入 release：
- `R-KV/logs/`
- `R-KV/outputs/`
- `R-KV/vLLM/`
- `R-KV/SGLang/`

### 8.3 Conda 环境
- `dc1-env`：DeepConf 开发环境
- `trivllm1`：TriAttention_vLLM vLLM V1 后端开发
- `lazy_evict`：LazyEviction 子项目
- `rkv` / `rkv1`：R-KV 压缩实验（Qwen2.5 / Qwen3）

### 8.4 其他排除目录
以下目录为开发过程中间产物或个人工具，不应 release：
- `weian_development/` — 个人开发脚本和工具
- `scripts/gpu_occupier.py` 及测试脚本 — GPU 占领工具
- `.claude/`、`.workflow/` — 开发工具配置

以下目录**待确认**是否 release：
- `paper_visualizations/` — 论文可视化脚本
- `experiments/` — 实验性代码

## 10. Evaluation 评估管线（已确认）

### 10.1 公布的评估文件（13个 + latex2sympy/）

| 文件 | 来源 | 作用 |
|------|------|------|
| `evaluate.py` | 社区+自定义 | 核心评估函数 |
| `eval_math.py` | 自定义 | 单样本评估 CLI 入口 |
| `eval_math_multi.py` | 自定义 | 多样本 pass@k 评估（被启动器直接调用） |
| `grader.py` | 社区 | 数学等价性判断 |
| `parser.py` | 社区+自定义 | 答案提取（有 AIME 引用需清理） |
| `math_utils.py` | 社区+自定义 | sympy 数学工具 |
| `python_executor.py` | 自定义 | 安全代码执行 |
| `data_loader.py` | 自定义 | 数据集加载 |
| `utils.py` | 自定义 | 工具函数（有 AIME 引用需清理） |
| `trajectory.py` | 自定义 | chain-of-thought 解析 |
| `examples.py` | 社区 | few-shot 示例 |
| `model_utils.py` | 社区 | 模型加载/生成 |
| `rm_maj_eval.py` | 自定义 | majority voting（清理 __main__ 硬编码路径） |
| `latex2sympy/` | 第三方 | LaTeX 解析库 |

### 10.2 不公布的评估文件

| 文件 | 原因 |
|------|------|
| `length_eval.py` | 硬编码内部目录结构，外部用户无法使用 |
| `CHANGELOG_weian.md` | 内部开发日志，含 PD-L1_binder 等敏感信息 |
| `evaluation/aime24/` 等结果缓存 | 运行时产物 |
| `.DS_Store`、`__pycache__` | 系统文件 |

### 10.3 需要清理的内容

- `parser.py`、`utils.py` 中的 AIME 引用需泛化
- `run_math.py` 中 `import weian_development.*` 需重构
- `rm_maj_eval.py` 中 `__main__` 硬编码路径需清理

## 11. 分布式启动器（已确认，全部公布）

### 11.1 启动器核心文件

| 文件 | 作用 |
|------|------|
| `rkv_sharded_dispatch.py`（31KB） | 主调度器：多GPU分配、断点恢复、自动评估 |
| `rkv_sharded_eval.py`（33KB） | 推理 worker：每GPU一个实例 |
| `rkv_sharded_runner.py`（689B） | 轻量 wrapper（需去掉 PD-L1_binder） |
| `merge_rkv_shards.py`（2.8KB） | 分片结果合并 |
| `process_utils.py`（1.1KB） | 进程命名（需去掉 PD-L1_binder） |
| `rkv_cache_utils.py`（882B） | cache 管理 |

### 11.2 启动器功能

- **多GPU分配**：自动检测可用GPU，队列调度
- **断点恢复**：检测已完成的 shard，跳过重复计算
- **分片合并**：按 sample_idx + draw_idx 排序合并
- **自动评估**：合并后自动调用 eval_math_multi.py
- **错误处理**：fail-fast，任一 shard 失败终止全部

### 11.3 完整流程

```
用户 shell 脚本 → rkv_sharded_dispatch.py
  → 分配任务到多个 GPU
  → 每个 GPU 运行 rkv_sharded_eval.py（推理）
  → merge_rkv_shards.py（合并分片）
  → eval_math_multi.py（评估）
```

### 11.4 启动器命名清理

release 时启动器文件名中的 `rkv_sharded` 等内部命名需要替换为正式名称，
`weian_development` 路径引用需要重构。具体方案待确认。
