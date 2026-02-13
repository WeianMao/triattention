# TriAttention_vLLM 终极目标定义

**文档创建日期**: 2026-02-02
**目标定义版本**: 1.0
**状态**: 📋 定义完成

> 说明（2026-02-13）：本文件只定义长期目标与验收口径，不作为当前实施方案文档。  
> 当前执行方案请以 `interface/V2_OVERVIEW.md` 与 `backend/ARCHITECTURE_REDESIGN.md` 为准。

---

## 1. 终极目标概述

### 1.1 目标声明

**实现与 HuggingFace SpeckV 版本完全等价的 vLLM 推理能力**，使用 Triton kernel 优化关键计算路径，在保持数值精度的同时大幅提升推理性能。

### 1.2 等价性定义

"完全等价" 指的是在相同输入条件下，TriAttention_vLLM 与 HuggingFace SpeckV 版本产生**数值上等价且准确率可比**的推理结果。

| 维度 | 等价性要求 | 验收标准 |
|-----|-----------|---------|
| **数学公式** | 打分公式、RoPE 处理完全一致 | 数值误差 < 1e-4 |
| **算法行为** | 裁剪逻辑、触发条件完全一致 | Token 选择匹配率 > 95% |
| **推理结果** | AIME24 准确率可比 | 准确率差异 < 1% |
| **配置参数** | 所有参数含义等价 | 参数映射完整且正确 |
| **输出格式** | JSONL 格式兼容 | 字段完全对齐 |

---

## 2. 参考基准：HuggingFace SpeckV 脚本

### 2.1 参考脚本路径

**主参考脚本**（per-head 模式，默认配置）:
```bash
R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh
```

**完整路径**:
```
/data/rbg/users/weian/project/rl/dc/R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh
```

**其他变种脚本**:
- per-layer-per-head: `run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh`
- per-layer: `run_speckv_aime24_qwen_norm_aligned_perlayer.sh`

### 2.2 配置文件路径

**YAML 配置**:
```bash
R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml
```

**完整路径**:
```
/data/rbg/users/weian/project/rl/dc/R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml
```

### 2.3 关键运行参数

#### 脚本参数
```bash
--sparse-normalize-scores          # 启用 Z-score 标准化
--include-prefill-in-budget        # prefill token 计入 budget
--rkv-style-compression            # R-KV 风格压缩（渐进式）
--rkv-style-slack-trigger          # slack 触发：budget + divide_length 时压缩
--divide-length 128                # 每 128 步检查一次是否需要压缩
--per-head-pruning                 # 每个 KV head 独立选择 token
```

#### 配置文件参数（YAML）
```yaml
# 模型配置
model_path: /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B
max_length: 32768
attn_implementation: flash_attention_2
load_dtype: bfloat16

# 压缩配置
method: speckv
kv_budget: 2048                    # KV cache 上限
window_size: 128                   # Recent window 大小
sparse_round_window: 32            # 压缩时使用的 offset 窗口
sparse_offset_max_length: 65536    # 最大 offset（位置相关打分）
sparse_score_aggregation: mean     # 多 offset 聚合策略：mean
sparse_head_limit: -1              # -1 表示不限制 head 数量
sparse_seed: 0                     # 随机种子（用于 tie-breaking）

# 推理配置
num_samples: 8                     # 每个问题采样 8 次
temperature: 0.6                   # 采样温度
top_p: 0.95                        # nucleus sampling
seed: 888                          # 全局随机种子
eval_batch_size: 1                 # 批处理大小（HF 限制）

# Stats 文件路径
sparse_stats_path: R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt

# 数据集
dataset_path: R-KV/HuggingFace/data/aime24.jsonl

# 输出
output_dir: R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perhead/shards
merged_dir_name: merged
```

### 2.4 核心实现文件

**主实现**:
```python
R-KV/weian_development/speckv/speckv_rkv_style.py    # SpeckV 主类
R-KV/weian_development/speckv/round_pruning_utils.py # 打分、RoPE 工具
```

**推理入口**:
```python
R-KV/weian_development/rkv_sharded_runner.py         # 单进程 runner
R-KV/weian_development/rkv_sharded_dispatch.py       # 多 GPU 调度器
```

### 2.5 Stats 文件

**预计算频率统计文件**:
```
R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt
```

**完整路径**:
```
/data/rbg/users/weian/project/rl/dc/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt
```

**格式**: PyTorch `.pt` 文件，包含 query 频率统计（用于打分）

### 2.6 数据集

**AIME24 数据集**:
```
R-KV/HuggingFace/data/aime24.jsonl
```

**完整路径**:
```
/data/rbg/users/weian/project/rl/dc/R-KV/HuggingFace/data/aime24.jsonl
```

**格式**: JSONL，每行包含 `{"question": "...", "answer": "..."}`

---

## 3. TriAttention_vLLM 等价实现要求

### 3.1 必须完全对齐的行为

#### 3.1.1 算法层面

| 项目 | HF 行为 | vLLM 要求 | 验证方法 |
|-----|---------|----------|---------|
| **打分公式** | SpeckV formula (base + extra) | 完全相同 | 数值误差 < 1e-4 |
| **RoPE 处理** | "half" 风格（前后半配对） | 完全相同 | 相位计算验证 |
| **聚合策略** | mean (多 offset) | 完全相同 | 配置参数对齐 |
| **裁剪粒度** | per-head（独立选择） | 完全相同 | 配置参数对齐 |
| **TopK 选择** | `torch.topk(k=budget)` | 行为等价 | Token 匹配率 > 95% |

#### 3.1.2 触发条件

| 条件 | HF 行为 | vLLM 要求 |
|-----|---------|----------|
| **压缩触发** | `current_length >= budget + divide_length` | 完全相同 |
| **Slack trigger** | 启用（buffer 机制） | 完全相同 |
| **Prefill 保护** | 启用（prefill 计入 budget） | 完全相同 |
| **Window 保护** | 最近 128 token 始终保留 | 完全相同 |

#### 3.1.3 配置参数

所有以下参数必须在 vLLM 版本中有对应映射：

| HF 参数 | HF 值 | vLLM 配置字段 | vLLM 值 | 状态 |
|--------|-------|--------------|--------|------|
| `kv_budget` | 2048 | `kv_budget` | 2048 | ✅ |
| `window_size` | 128 | `window_size` | 128 | ✅ |
| `sparse_round_window` | 32 | `round_window` | 32 | ⚠️ 需确认命名 |
| `sparse_offset_max_length` | 65536 | `offset_max_length` | 65536 | ✅ |
| `sparse_score_aggregation` | mean | `score_aggregation` | mean | ✅ |
| `--sparse-normalize-scores` | True | `normalize_scores` | True | ✅ |
| `--include-prefill-in-budget` | True | `include_prefill_in_budget` | True | ⚠️ |
| `--rkv-style-compression` | True | `rkv_style_compression` | True | ⚠️ |
| `--rkv-style-slack-trigger` | True | `use_slack_trigger` | True | ⚠️ |
| `--per-head-pruning` | True | `pruning_mode` | "per_head" | ✅ |

### 3.2 允许的差异（不影响等价性）

以下差异是允许的，因为不影响最终推理结果：

| 项目 | HF 实现 | vLLM 实现 | 理由 |
|-----|---------|----------|------|
| **打分实现** | PyTorch 原生 | Triton kernel | 数值等价即可，效率更高 |
| **TopK 实现** | `torch.topk` | `torch.topk` 或 Triton | 选择结果等价即可 |
| **Gather 实现** | `torch.gather` | `torch.gather` 或 Triton | 结果等价即可 |
| **Batch size** | 固定为 1 | 可支持 > 1（Phase 2） | vLLM 架构优势 |
| **状态管理** | 简单字典 | 请求级隔离 | vLLM 多请求需求 |

### 3.3 必须达到的性能指标

| 指标 | 目标 | 说明 |
|-----|------|------|
| **AIME24 准确率差异** | < 1% | 与 HF 版本对比 |
| **Token 匹配率** | > 95% | 确定性推理 (temp=0) |
| **打分加速** | > 1.5x | Triton vs PyTorch |
| **端到端延迟开销** | < 10% | vs 无压缩 baseline |
| **吞吐量提升** | > 1.5x | 长序列场景 (budget 2048) |
| **稳定性** | 100 问题零崩溃 | 完整 AIME24 数据集 |

---

## 4. 验收标准详细定义

### 4.1 正确性验收

#### 4.1.1 数学验证（已完成 ✅）

| 验证项 | 方法 | 标准 | 状态 |
|-------|------|------|------|
| RoPE 相位计算 | 单元测试 | 误差 < 1e-6 | ✅ 已修正 |
| MLR 公式 | 单元测试 | 误差 < 1e-6 | ✅ 已修正 |
| Triton-PyTorch 等价 | 对比测试 | FP32 误差 < 1e-4 | ✅ 通过 33/33 |

#### 4.1.2 算法验证（待完成 ❌）

**测试用例**: 使用固定随机种子在 AIME24 数据集上运行

| 测试项 | 方法 | 标准 |
|-------|------|------|
| Token 选择一致性 | 对比每个 step 的 topk 结果 | 匹配率 > 95% (确定性推理) |
| 压缩触发时机 | 记录触发位置 | 完全一致 |
| 最终输出 | 对比生成的 token 序列 | 匹配率 > 90% (随机采样) |

#### 4.1.3 端到端验证（待完成 ❌）

**测试集**: AIME24 完整数据集（30 道题）

| 指标 | HF 基线 | vLLM 目标 | 验收标准 |
|-----|---------|----------|---------|
| Pass@1 准确率 | 记录 HF 结果 | vLLM 结果 | 差异 < 1% |
| Pass@8 准确率 | 记录 HF 结果 | vLLM 结果 | 差异 < 1% |
| 平均生成长度 | 记录 HF 结果 | vLLM 结果 | 差异 < 5% |

### 4.2 性能验收

#### 4.2.1 打分性能（部分完成 ⚠️）

**测试场景**: budget=2048, seq_len=2048+128

| 指标 | PyTorch 基线 | Triton 目标 | 当前状态 |
|-----|-------------|------------|---------|
| 打分耗时 | 记录 PyTorch 时间 | < 50% PyTorch | ⚠️ 需实测 |
| GPU 利用率 | - | > 80% | ⚠️ 需 profiling |
| 数值精度 | FP32 参考 | FP32 < 1e-4 | ✅ 已验证 |

#### 4.2.2 端到端性能（待完成 ❌）

**测试场景**: AIME24, num_samples=8, budget=2048

| 指标 | vLLM 无压缩 | vLLM + TriAttention | 目标 |
|-----|------------|-------------------|------|
| TTFT (首 token 延迟) | 记录基线 | 期望值 | 增加 < 10% |
| TPS (吞吐量) | 记录基线 | 期望值 | > 1.5x (长序列) |
| GPU 内存占用 | 记录基线 | 期望值 | budget 相关 |

### 4.3 稳定性验收

| 测试项 | 方法 | 标准 |
|-------|------|------|
| 长时间运行 | 连续运行 AIME24 完整数据集 | 零崩溃 |
| 多样化输入 | 不同长度、不同模型 | 无异常退出 |
| 边界条件 | Prefill > budget, 空输入等 | 优雅处理或明确报错 |

### 4.4 兼容性验收

| 项目 | 要求 | 验证方法 |
|-----|------|---------|
| vLLM 版本 | 0.15.x | 实际运行测试 |
| GPU 型号 | Tesla T4 (sm_75+) | 实际运行测试 |
| 精度支持 | FP32 (必须), BF16 (可选) | 数值测试 |
| 模型架构 | Qwen, LLaMA, DeepSeek | 多模型测试 |

---

## 5. 实现路径

### 5.1 Phase 1: 基础等价性（当前阶段）

**目标**: 实现与 `run_speckv_aime24_qwen_norm_aligned_perhead.sh` 完全等价的推理

**里程碑**:
1. ✅ 核心库实现完成
2. ✅ Triton kernel 验证通过
3. ✅ 数学公式修正完成
4. ❌ 推理入口实现完成
5. ❌ 端到端验证通过

**当前进度**: 85% (4/5 完成)

**剩余工作**:
- [ ] 完成 `run_math_vllm.py` 初始化代码（预计 2-3 天）
- [ ] 运行端到端对比验证（预计 1-2 天）
- [ ] 修复发现的任何差异（预计 1 天）

**验收**: 通过所有 4.1 节的正确性验收标准

### 5.2 Phase 2: 性能优化与鲁棒性

**目标**: 提升性能，覆盖边界情况

**任务**:
- [ ] Triton TopK/Gather kernel（可选，取决于性能收益）
- [ ] 解除 batch_size=1 限制
- [ ] CUDA Graph 兼容性
- [ ] 内存触发压缩
- [ ] Prefill > budget 处理
- [ ] BF16 支持验证（需要 A100/H100）

**验收**: 通过所有 4.2-4.4 节的验收标准

### 5.3 Phase 3: 生产就绪（未来）

**目标**: 生产级稳定性和易用性

**任务**:
- [ ] 完善错误处理和日志
- [ ] 性能调优和 profiling
- [ ] 文档完善（用户手册、API 文档）
- [ ] CI/CD 集成
- [ ] 多 GPU (TP/PP) 支持（可选）

---

## 6. 运行对比流程

### 6.1 环境准备

#### HuggingFace 环境
```bash
conda activate rkv
cd /data/rbg/users/weian/project/rl/dc
export PYTHONPATH="${PWD}/R-KV:${PYTHONPATH:-}"
```

#### vLLM 环境
```bash
conda activate trivllm
cd /data/rbg/users/weian/project/rl/dc
export PYTHONPATH="${PWD}/TriAttention_vLLM:${PYTHONPATH:-}"
```

### 6.2 运行 HuggingFace 基线

```bash
conda activate rkv
bash R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh
```

**输出位置**:
```
R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perhead/
├── shards/           # 分片结果
│   ├── shard_0.jsonl
│   ├── shard_1.jsonl
│   └── ...
└── merged/           # 合并结果
    └── merged_results.jsonl
```

**预计运行时间**: 约 2-4 小时（8 GPU 并行，每题 8 个样本）

### 6.3 运行 vLLM 版本

```bash
conda activate trivllm
bash TriAttention_vLLM/benchmarks/reasoning/run_triattention_aime24_perhead.sh
```

**输出位置**:
```
TriAttention_vLLM/outputs/aime24/perhead/
└── results.jsonl
```

**关键参数确认**:
- `--kv-budget 2048`
- `--divide-length 128`
- `--window-size 128`
- `--sparse-round-window 32`
- `--pruning-mode per_head`
- `--sparse-stats-path R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt`
- `--sparse-normalize-scores`
- `--sparse-score-aggregation mean`

### 6.4 结果对比

```bash
python TriAttention_vLLM/benchmarks/reasoning/compare_results.py \
  --hf-output R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perhead/merged/merged_results.jsonl \
  --vllm-output TriAttention_vLLM/outputs/aime24/perhead/results.jsonl \
  --report-path comparison_report.json
```

**对比内容**:
1. Pass@1 和 Pass@8 准确率
2. 平均生成长度
3. Token 序列匹配率（如果有确定性推理）
4. 运行时间对比

### 6.5 验收判定

**成功标准**:
- ✅ Pass@1 准确率差异 < 1%
- ✅ Pass@8 准确率差异 < 1%
- ✅ 无 Python 异常或 CUDA 错误
- ✅ 输出 JSONL 格式正确

**如果不通过**:
1. 检查参数映射是否正确
2. 对比中间状态（打分结果、topk 选择）
3. 启用详细日志，定位差异来源
4. 参考 `test/debug_*.py` 工具进行调试

---

## 7. 参考材料

### 7.1 HuggingFace 实现文件

**核心实现**:
```
R-KV/weian_development/speckv/
├── speckv_rkv_style.py          # SpeckV 主类
├── round_pruning_utils.py       # 打分、RoPE 工具
└── rkv_speckv_generate.py       # 生成脚本
```

**推理框架**:
```
R-KV/weian_development/
├── rkv_sharded_runner.py        # 单进程 runner
└── rkv_sharded_dispatch.py      # 多 GPU 调度
```

**测试文件**:
```
R-KV/weian_development/tests/
├── test_speckv_scoring.py       # 打分测试
└── test_speckv_compression.py   # 压缩测试
```

### 7.2 TriAttention_vLLM 实现文件

**核心库**:
```
TriAttention_vLLM/triattention/
├── config.py                    # 配置类
├── state.py                     # 状态管理
├── compressor.py                # 压缩器
├── scoring.py                   # 打分逻辑
├── utils.py                     # 工具函数
├── vllm_integration.py          # vLLM 集成
└── kernels/triton_scoring.py    # Triton kernel
```

**Benchmark**:
```
TriAttention_vLLM/benchmarks/reasoning/
├── run_math_vllm.py             # vLLM 推理入口
├── run_triattention_aime24_*.sh # 启动脚本
└── compare_results.py           # 结果对比
```

**测试**:
```
TriAttention_vLLM/test/
├── test_scoring_kernel.py       # Kernel 测试
├── test_triton_pytorch_equivalence.py  # 等价性测试
└── test_bug_fixes.py            # Bug 修复测试
```

### 7.3 关键文档

**当前主文档（V2）**:
- `docs/interface/V2_OVERVIEW.md` - V2 方案总览（当前执行方案）
- `docs/interface/CURRENT_STATUS.md` - 当前进度
- `docs/interface/OPEN_ISSUES.md` - 当前问题
- `docs/interface/PENDING_DECISIONS.md` - 待决策项
- `docs/backend/ARCHITECTURE_REDESIGN.md` - 技术规格
- `docs/backend/DESIGN_DECISIONS.md` - 决策日志

**历史与参考**:
- `docs/backend/reference/` - 详细技术分析与历史修复记录
- `docs/archive/snapshots/2026-02-13/` - 文档重构前快照

---

## 8. 常见问题

### 8.1 为什么必须与 HF 版本等价？

**原因**:
1. **科学验证**: HF 版本已在论文中验证效果，需要保持一致性
2. **可信度**: 等价性是 vLLM 版本正确性的最强证明
3. **对比基准**: 便于评估 Triton 优化的性能收益

### 8.2 允许的差异有哪些？

**允许**:
- 实现技术不同（Triton vs PyTorch）
- 性能差异（Triton 应该更快）
- 批处理能力（vLLM 可支持 batch > 1）

**不允许**:
- 数值精度差异（超过容忍度）
- 算法行为差异（触发条件、选择逻辑）
- 准确率差异（> 1%）

### 8.3 如果验证不通过怎么办？

**步骤**:
1. 确认参数配置完全对齐
2. 启用详细日志，对比中间状态
3. 使用 `debug_*.py` 工具定位差异
4. 如果是数值误差，检查精度设置
5. 如果是算法差异，审查触发条件和选择逻辑

### 8.4 为什么使用 AIME25 stats 测试 AIME24？

**原因**: 避免数据泄漏（data leakage）

- Stats 文件从 AIME25 数据集收集（训练集）
- 在 AIME24 数据集上测试（测试集）
- 这是跨数据集泛化能力的验证

### 8.5 为什么 enforce_eager=True？

**原因**: CUDA Graph 不兼容

- vLLM 默认使用 CUDA Graph 优化
- TriAttention 的压缩操作是动态的（seq_len 变化）
- CUDA Graph 要求静态计算图
- Phase 1 使用 eager 模式，Phase 2 再考虑兼容

---

## 9. 成功标准总结

### 9.1 最低验收标准（Phase 1 完成）

| 项目 | 标准 |
|-----|------|
| ✅ 核心库实现 | 所有模块完成，测试通过 |
| ✅ Triton kernel 验证 | FP32 误差 < 1e-4 |
| ✅ 数学公式验证 | 所有公式修正并验证 |
| ❌ 推理入口实现 | `run_math_vllm.py` 可运行 |
| ❌ AIME24 准确率 | 与 HF 差异 < 1% |
| ❌ Token 匹配率 | 确定性推理 > 95% |
| ❌ 稳定性 | 零崩溃完成全数据集 |

### 9.2 理想目标（Phase 2 完成）

| 项目 | 标准 |
|-----|------|
| 打分性能 | Triton < 50% PyTorch 时间 |
| 端到端延迟 | 增加 < 10% |
| 吞吐量 | > 1.5x (长序列) |
| Batch size | 支持 > 1 |
| CUDA Graph | 兼容 |
| BF16 支持 | A100/H100 验证通过 |

---

## 10. 时间线与里程碑

### 10.1 短期目标（本周内）

- [ ] **Day 1-3**: 完成 `run_math_vllm.py` 实现
- [ ] **Day 4-5**: 运行 HF 和 vLLM 对比验证
- [ ] **Day 6-7**: 修复发现的问题，达到最低验收标准

### 10.2 中期目标（未来 2 周）

- [ ] 验证 per-layer-per-head 和 per-layer 模式
- [ ] 性能 benchmark 和优化
- [ ] 文档更新和清理

### 10.3 长期目标（Phase 2）

- [ ] 解除 batch_size=1 限制
- [ ] CUDA Graph 兼容性
- [ ] BF16 支持
- [ ] Triton TopK/Gather 优化

---

## 附录 A: 完整参数映射表

| HF 参数名 | HF 默认值 | TriAttention 字段 | vLLM 默认值 | 类型 | 说明 |
|----------|----------|------------------|------------|------|------|
| `kv_budget` | 2048 | `kv_budget` | 2048 | int | KV cache 上限 |
| `window_size` | 128 | `window_size` | 128 | int | Recent window |
| `sparse_round_window` | 32 | `round_window` | 32 | int | Offset 窗口 |
| `sparse_offset_max_length` | 65536 | `offset_max_length` | 65536 | int | 最大 offset |
| `sparse_score_aggregation` | mean | `score_aggregation` | mean | str | 聚合策略 |
| `sparse_head_limit` | -1 | - | - | int | -1 表示不限制 |
| `sparse_seed` | 0 | - | - | int | 随机种子 |
| `--sparse-normalize-scores` | flag | `normalize_scores` | True | bool | Z-score 标准化 |
| `--include-prefill-in-budget` | flag | `include_prefill_in_budget` | True | bool | Prefill 计入 |
| `--rkv-style-compression` | flag | `rkv_style_compression` | True | bool | 渐进式压缩 |
| `--rkv-style-slack-trigger` | flag | `use_slack_trigger` | True | bool | Slack 触发 |
| `--per-head-pruning` | flag | `pruning_mode` | "per_head" | str | 裁剪粒度 |
| `--divide-length` | 128 | `divide_length` | 128 | int | 检查间隔 |
| `num_samples` | 8 | - | - | int | 采样次数（推理参数） |
| `temperature` | 0.6 | - | - | float | 采样温度（推理参数） |
| `top_p` | 0.95 | - | - | float | Nucleus sampling |
| `seed` | 888 | - | - | int | 全局随机种子 |

---

*文档生成日期: 2026-02-02*
*维护者: TriAttention_vLLM 项目组*
*版本: 1.0*
