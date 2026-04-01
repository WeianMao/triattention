# 目标 Repo 结构

## 目录树

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
├── kv_compress/                    # 通用 KV cache 压缩框架 + baseline 方法
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
├── evaluation/                     # 评估管线（13个文件 + latex2sympy/，详见 07_evaluation.md）
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
├── data/                           # 数据集（待确认：用户自行下载 or 提供链接）
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

## 包名决策说明

- `triattention/` — 我们的方法实现，对应内部 SpeckV rkv-style
- `kv_compress/` — 通用 KV cache 压缩框架 + 所有 baseline 方法（原 `baselines/`，改名原因：这个包不止包含 baseline 也包含通用压缩接口）

这两个包在 clean-room 阶段从原始 `R-KV/rkv/` 和 `R-KV/HuggingFace/rkv/` 重组而来，不是简单改名。
