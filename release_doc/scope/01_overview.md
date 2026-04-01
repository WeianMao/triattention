# 发布概述

## Release Target

- GitHub public repo，名称：**TriAttention**
- Release branch: `release`（从 `main` 分出）
- 本地代码不动，所有清理只在 release 分支上进行

## 分阶段策略

### 第一阶段：TriAttention 核心方法 + baselines

公布内容：
- TriAttention 核心方法实现（原 SpeckV rkv-style）
- 官方 baseline 方法（RKV, SnapKV, H2O, StreamingLLM, FullKV）
- HuggingFace 集成层（modeling, monkeypatch）
- 评估管线（13 个文件 + latex2sympy/）
- 分布式启动器
- 多 setting 实验脚本（不同模型、数据集、配置变体）

不含：
- kvpress 相关代码（第二阶段）
- TriAttention_vLLM（有 bug 待修 + SGLang 版本待开发）
- LazyEviction（独立子项目，不在第一阶段范围内）

### 第二阶段：kvpress 相关代码

详见 [../tracking/13_phase2.md](../tracking/13_phase2.md)。

kvpress 相关代码目前只在 `dc1/rebuttal` 分支，需先转移到 main 上，第一阶段公布完后再检查并公布。
