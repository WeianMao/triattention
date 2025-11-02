# 离线 Q/K 捕获工具说明

本目录提供两类脚本，用于从既有 DeepSeek-R1 离线推理 trace 中抽样，并在不修改 `text-generation-inference` 后端的前提下，重新回放 trace 以捕获 RoPE 之后的注意力 Q/K 张量。

## 目录结构

- `plan.md`：前期沟通确认的执行计划。
- `build_manifest.py`：从 JSON 离线结果中随机抽样题目与 trace。
- `trace_manifest.json`：最新一次抽样生成的任务清单（默认 8 条）。
- `capture_qk_distributed.py`：多 GPU 分发执行，捕获并落盘每条 trace 的 Q/K 与元数据。
- `visualize_attention_maps.py`：基于捕获的 Q/K 生成逐层逐头的高分辨率注意力热图。

## 运行环境

- Python 3.9（沿用仓库 `dc` conda 环境）。
- `transformers>=4.46`，已随环境安装。
- 依赖 `weian_development/process_utils.py` 提供的 `mask_process_command`，用于进程重命名。
- 模型路径默认 `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B`。

## 抽样 manifest

```bash
python weian_development/attention_qk_analysis/build_manifest.py \
    outputs/deepseek_r1_qwen3_8b/offline_reasoning_json \
    --output weian_development/attention_qk_analysis/trace_manifest.json \
    --count 8 --seed 20250107 --verbose
```

- `--count`：抽样问题数量；若实际可用题目不足，会自动降级并告警。
- 每个 `qid` 仅保留一条随机 `trace_index`。

## 捕获 Q/K

```bash
python weian_development/attention_qk_analysis/capture_qk_distributed.py \
    weian_development/attention_qk_analysis/trace_manifest.json \
    outputs/deepseek_r1_qwen3_8b/qk_bf16_traces \
    --gpus 0,1 --precision bfloat16 --verbose
```

参数说明：

- `manifest`：上一阶段生成的 JSON 任务清单。
- `output_dir`：每条 trace 的结果目录，默认写在 `outputs/` 下。
- `--gpus`：逗号分隔的物理 GPU 编号。脚本会为每张卡启动一个 `PD-L1_binder_*` worker。
- `--precision`：模型加载及输出 dtype，可选 `float16` / `bfloat16` / `float32`，默认 `float16`。建议使用 `bfloat16`，能将 64k token 样本的单条大小降到约 35 GB。
- `--dry-run`：仅打印调度信息，不实际执行。

运行时，脚本会：

1. 重建 DeepSeek-R1 的聊天模板，将选中 trace 的完整文本拼接到 prompt 后。
2. 在前向 pre-hook 中复制 Qwen3 注意力的投影、RMSNorm、RoPE 与 KV 扩充逻辑，捕获 post-RoPE 的 `query` / `key`。
3. 将各层 Q/K 累积到 CPU 缓冲区并以指定 dtype 存为 `qk.pt`，另写 `metadata.json` 记录 token 数、prompt 长度、层数等信息。
4. 每条 trace 运行完成即释放显存与内存，避免批次堆积。

## 输出结构

```
outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/
  ├── qid0013_trace55/
  │   ├── qk.pt            # {'q': Tensor[L,H,T,D], 'k': Tensor[L,H,T,D]}
  │   └── metadata.json    # token_count、prompt_tokens、dtype、源文件等
  ├── ...
  └── capture_summary.json # 采样摘要（自动生成，可选）
```

> 仓库的 `.gitignore` 已忽略 `outputs/`，请勿尝试将上述产物纳入版本控制；`AGENTS.md` 已记录该约束。

## 注意力热图（可选）

若需要可视化注意力权重，可在捕获完成后运行：

```bash
scripts/run_attention_maps.sh
```

- 默认在每个 `qidXXXX_traceYY` 目录下创建 `attention_maps/` 子目录，按 `layer_##_head_##.png` 生成 4K 级 PNG。脚本默认参数为 `patch_size=32`、`head_batch=8`、`q_tile=4096`、`dtype=float32`，如需覆盖可在命令后追加参数。
- 计算过程中先对 `Q·K^T` 分块并完成 softmax，然后对 key / query 维度分别做最大池化，最后对概率值做线性 min-max 归一化；若序列过长，可通过 `--patch-size` 直接指定窗口大小，或使用 `--target-size` 自动推断。绘制时保持 Query 轴向下、Key 轴向左（反转 X 方向），便于观察远端 token 的注意力分布。
- `attention_maps/README.md` 会记录池化倍数、图片尺寸以及生成脚本参数。

## 性能与资源提示

- 64k token 的 trace 在 `bfloat16` 下单条约 35 GB；`float32` 会翻倍，请谨慎选择。
- 捕获时需保证 GPU/CPU 内存有足够余量，可通过 `--gpus` 控制并发数量。
- 若只想测试流程，可加 `--max-workers 1` 并限制 manifest 中的条目数量。

## 常见问题

- **两个进程显示在同一 GPU？**
  - 从 `2025-11-01` 的修订起，脚本改为直接使用 `cuda:<gpu_id>`，每个 worker 绑定对应物理卡；如仍出现混用，检查命令行 `--gpus` 参数是否设置正确。
- **无法删除输出目录**
  - NFS 可能残留 `.nfs*` 文件，请确保相关进程已结束后再删除或等待系统回收。

如需扩展功能（例如捕获 V、注意力权重或按层切片写盘），建议在本目录新增脚本并复用现有的 hook/缓冲封装。

## 频段幅值诊断（可选）

若需分析 RoPE 频段的幅值特征，可运行：

```bash
python weian_development/attention_qk_analysis/freq_magnitude_plots.py \
    outputs/deepseek_r1_qwen3_8b/qk_bf16_traces \
    --output-root outputs/deepseek_r1_qwen3_8b/freq_magnitude_plots \
    --device cuda:0 --dtype float32 --max-distance 10000 --verbose
```

- 每个 trace 会在输出目录生成 `layer_##_head_##_freq.png`，包含四幅子图：`|K|` 平均、`|Q|` 平均、自回归遮罩下的 `|Q||K|` 平均，以及依据这些幅值重构的 Σ_f |Q||K| cos(ω_f Δ) 曲线，用于观察随距离的理论衰减趋势（Δ 在 logspace 中采样，最大值可通过 `--max-distance` 控制，默认 10k token）。
- 横轴为 RoPE 频段索引，纵轴为聚合后的幅值。
- 附带 `README.md` 说明统计口径，便于后续复现。
