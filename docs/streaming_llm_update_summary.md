# StreamingLLM 集成实现说明

## 综述
本轮改动围绕“使用 HuggingFace + StreamingLLM 重现离线 DeepThink 流程”展开，核心目标是：

- 复制旧 vLLM 离线脚本的行为与输出协议（msgpack 结构、评估逻辑）。
- 在完全隔离的脚本/配置目录中实现新功能，保证旧脚本零改动仍可运行。
- 为 Qwen3 模型补齐 StreamingLLM 支持，并解决 HuggingFace `DynamicCache` 与缓存裁剪的兼容问题。
- 提供镜像旧流程的调度、分析脚本，同时加入进度条与进程别名方便监控。

以下分模块列出关键文件与逻辑。

## HuggingFace 推理脚本
- **development/example_offline_streaming_hf.py**：仿照 `example_offline_serialized.py` 实现单 QID 离线推理。主要功能：
  - `SamplingConfig` 和 prompt 构造沿用旧逻辑。
  - `generate_streaming_trace()` 使用 HuggingFace 推理循环，支持采样、logprob 记录、StreamingLLM KV 裁剪。
  - 通过 `DeepThinkOutput` 组织结果，保持 msgpack 字段兼容。
- 引入 `tqdm.trange` 输出 trace 级进度条；根据环境变量设置进程名称为 `PD_L1_affinity`。
- 支持通过 `--attention_backend` 强制切换 Transformers 注意力后端（`flash_attn2` / `flash_attn3` / `sdpa` / `eager`），并在运行前检测所需依赖是否可用。
- 在 `dc` 环境中安装 `flash-attn==2.6.3`（需 `CUDA_HOME=/usr/local/cuda-12.4`）以启用 `flash_attn2` 后端，若需使用其他版本，请同步更新依赖。

## StreamingLLM 底层补丁
- **streaming-llm/streaming_llm/pos_shift/modify_qwen3.py**：新增 Qwen3 Attention 的位置偏移实现，确保起始窗口 + 最近窗口策略与 Qwen3 的 RoPE 对齐。
- **streaming-llm/streaming_llm/enable_streaming_llm.py**：增加 Qwen3 分支，加载上述补丁，并设置 `k_seq_dim` / `v_seq_dim`。
- **streaming-llm/streaming_llm/kv_cache.py**：
  - 为 HuggingFace `DynamicCache` 增加兼容层（`to_legacy_cache` / `from_legacy_cache`）。
  - 忽略空层 / None 值，避免出现 `AttributeError`。
  - `StartRecentKVCache` 仍保留起始窗口 + 最近窗口裁剪逻辑。

## 调度与分析脚本
- **scripts/yaml_runs_streaming/run_dispatch_streaming.py**：
  - 复刻旧调度器的“按 GPU 列表并行执行”方式：`parse_gpu_argument`、`split_qids`、`ThreadPoolExecutor`。
  - 为每个子进程设置环境变量 `PD_L1_AFFINITY_ALIAS=1`，由推理脚本统一改进程名。
- **scripts/yaml_runs_streaming/run_offline_deepseek_streaming_hf.sh**：与旧脚本风格一致，固定配置路径、默认 GPU 列表并允许额外参数覆盖。
- **scripts/analyze/run_offline_msgpack_analysis_streaming.sh**：沿用旧分析器，对 streaming 输出目录做覆盖。
- **scripts/configs/streaming/deepseek_r1_qwen3_8b_streaming_64.yaml**：Streaming 流程的默认配置（模型路径、窗口大小、采样参数）。

## 额外辅助文件
- **development/tmp_streaming_smoke.py**：最小化冒烟测试脚本，验证 HuggingFace + StreamingLLM 能加载模型并生成结果。
- **docs/streaming_llm_integration_plan.md**：详细的中文开发计划，记录阶段目标与测试点。

## 使用方式
1. **单 QID 调试**（进入 `dc` 环境）：
   ```bash
   conda run -n dc python development/example_offline_streaming_hf.py \
     --model /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B \
     --dataset aime25.jsonl --qid 0 --budget 1 --max_tokens 64000 \
     --start_size 4 --recent_size 2048 --output_dir outputs/streaming_hf_test
   ```
2. **批量运行**：
   ```bash
   bash scripts/yaml_runs_streaming/run_offline_deepseek_streaming_hf.sh \
     --rid myrun --gpus 0,1,2,3 --qids 0-15
   ```
3. **结果分析**：
   ```bash
   bash scripts/analyze/run_offline_msgpack_analysis_streaming.sh \
     --output_dir outputs/deepseek_r1_qwen3_8b/streaming_hf_64trace --max_qid 29 --rids myrun
   ```

## 注意事项
- Streaming 版本仍按 trace 顺序生成（batch=1），若需更高 GPU 利用率，可在调度层或 HuggingFace 推理脚本中继续扩展并行。
- `start_size`、`recent_size` 默认为 `4` 与 `2048`（YAML 可覆盖），请结合显存预算调整。
- 进程名通过 `prctl` 设置为 `PD_L1_affinity`，便于在 htop 中识别。
- 旧 vLLM 流程仍可通过原脚本运行，未做任何改动。
