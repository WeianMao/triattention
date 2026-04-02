# 目标 Repo 结构

> **此文件已与 `plan/execution_plan.md` Step 2.1 对齐（2026-04-02）**

## 目录树

```
TriAttention/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py / pyproject.toml
├── .gitignore
│
├── triattention/                   # 我们的方法（所有命名统一为 triattention）
│   ├── __init__.py
│   ├── triattention.py             # 主实现（原 speckv_rkv_style.py）
│   ├── pruning_utils.py            # 核心工具（原 round_pruning_utils.py）
│   ├── stats_utils.py              # 统计验证
│   └── prompt_utils.py             # prompt 构建
│
├── kv_compress/                    # 通用 KV cache 压缩框架 + baseline 方法
│   ├── __init__.py
│   ├── r1_kv.py
│   ├── snapkv.py
│   ├── h2o.py
│   ├── streamingllm.py
│   └── utils.py                    # cal_similarity, compute_attention_scores（baseline 方法依赖）
│   # 注：FullKV 是无压缩模式，通过 --method fullkv 选择，不需要独立实现文件
│
├── integration/                    # HuggingFace 集成
│   ├── __init__.py
│   ├── modeling.py
│   └── monkeypatch.py
│
├── evaluation/                     # 评估管线（13个文件 + latex2sympy/，详见 07_evaluation.md）
│   ├── __init__.py
│   ├── evaluate.py
│   ├── eval_math.py
│   ├── eval_math_multi.py
│   ├── grader.py
│   ├── parser.py
│   ├── math_utils.py
│   ├── python_executor.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── trajectory.py
│   ├── examples.py
│   ├── model_utils.py
│   ├── rm_maj_eval.py
│   └── latex2sympy/                # 第三方 LaTeX 解析库（MIT，含子目录）
│
├── scripts/                        # 运行/复现脚本
│   ├── cli.py                      # 实验 CLI 入口（原 speckv_experiments_cli_v2.py）
│   ├── dispatch.py                 # 多 GPU 调度器
│   ├── worker.py                   # 推理 worker
│   ├── run_math.py                 # 推理入口脚本
│   ├── merge_shards.py             # 分片结果合并
│   ├── calibrate.py                # 校准脚本（新写，接受 raw text 输入）
│   ├── process_utils.py            # 进程工具（已去除 PD-L1 相关）
│   ├── cache_utils.py              # cache 管理
│   └── experiments/                # per-model 实验 shell 脚本（递归，含 budget 子目录）
│       ├── distill_llama8b/
│       ├── distill_qwen7b/
│       └── qwen3/
│
├── configs/                        # 实验 YAML 配置
│   ├── shared/                     # 默认配置
│   ├── extra_config/               # 方法变体覆盖
│   └── generated/                  # per-run 生成配置
│
├── calibration/                    # 校准结果 .pt 文件（文件名不含 aime 或 budget）
│   ├── qwen3_8b_stats.pt
│   ├── dsqwen_7b_stats.pt
│   └── dsllama_8b_stats.pt
│
├── benchmarks/
│   └── dfs/                        # DFS benchmark（从 linxi-dev 分支提取）
│
├── data/                           # 数据集（自动下载，不含数据文件）
│
└── tests/                          # 单元测试
    ├── test_triattention.py        # Level 1: 评分函数等价性
    ├── test_pruner_equivalence.py  # Level 2: pruner + 真实 stats
    └── test_peak_cache_alignment.py
```

## 包名决策说明

- `triattention/` — 我们的方法实现，对应内部 SpeckV rkv-style
- `kv_compress/` — 通用 KV cache 压缩框架 + 所有 baseline 方法（原 `baselines/`，改名原因：这个包不止包含 baseline 也包含通用压缩接口）

这两个包在 clean-room 阶段从原始 `R-KV/rkv/` 和 `R-KV/HuggingFace/rkv/` 重组而来，不是简单改名。

## 与旧版本的差异（2026-04-02 更新）

- 新增 `kv_compress/utils.py`（baseline 方法共享依赖）
- 新增 `scripts/cli.py`, `scripts/worker.py`, `scripts/process_utils.py`, `scripts/cache_utils.py`
- 新增 `scripts/experiments/`（per-model shell 脚本，用于复现论文实验）
- 新增 `benchmarks/dfs/`（DFS benchmark）
- 新增 `scripts/calibrate.py`（新写的 raw text 校准脚本）
- 删除 `integration/utils.py`（无来源文件）
- 删除 `scripts/run_eval.py`（评估入口在 `evaluation/eval_math*.py`）
- 删除 `backends/`（Phase 2 内容，Phase 1 不含）
- 明确 `evaluation/latex2sympy/` 子目录
