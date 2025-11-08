# HF Offline Runner (Sparse KV Variant)

## 需求回顾
- **隔离开发**：所有 HuggingFace 推理相关改动都放在 `weian_development/hf_offline_runner_sparse/`，原目录保持不动。
- **稀疏注意力**：真实删除 KV（不仅仅是 mask），并遵循文档中的轮次维护逻辑：
  - 每轮解码 `round_window` 个 token，预留空间给新 token，旧 KV 通过频域打分保留 `max_keys - round_window`。
  - 打分重用 `docs/online_k_pruning_round_based.md` 的方案，包含 `mean/max` 聚合、几何 offset、cross-trace 统计。
  - 频段需要乘上实际 RoPE scaling（YaRN `attention_scaling` + per-frequency 放大），否则与模型前向不一致。
- **统计来源**：使用 `weian_development/hf_offline_runner_sparse/export_round_pruning_stats.py` 预先从指定 trace（例如 `qid0008_trace46`）导出 `|E[q]|`, `E[|q|]`。
- **KV 上限**：实验关注 `max_keys = 2048`，确保在真实推理中也只保留 2048 个 token 的 KV。
- **日志/命名**：长任务进程名需为 `PD-L1_binder`，以便在 htop 中统一识别。

## 目录结构
- `example_offline_hf_serialized.py`：复制原 HF runner 并加入：
  - CLI 开关 `--enable_sparse_pruning` 与各项稀疏参数。
  - 手写采样/生成循环 `run_sparse_generation`，在每个 round 触发 `SparseRoundPruner` 切 KV，并按 head 取并集后裁剪。
  - **仅支持单卡**：`tensor_parallel_size` 必须为 1，避免多卡下的 KV 复制导致额外显存。
- `example_offline_hf_serialized_streaming.py`：与上面脚本共享逻辑，但会把实时新增的文本打印到终端，并可选通过 `--stream-log-path` 同步写入文件，便于长回答过程中观察模型状态。
- `round_pruning_utils.py`：频域共用工具（采样 head、构造 RoPE 表、读取/写入 stats、计算频段 scaling）。
- `export_round_pruning_stats.py`：从 `qk.pt` + `metadata.json` 生成 `.pt` 统计，输出路径示例：`weian_development/hf_offline_runner_sparse/stats/qid0008_trace46_stats.pt`。
- `sparse_round_pruner.py`：核心裁剪器；对每个 head 独立选出 top-K，取并集后切实删除 KV，确保 multi-query head 需求满足。
- `run_dispatch_hf_serialized.py`：批量调度器，复用 `scripts/configs/*.yaml`，但 offline mode 会调用本目录的稀疏 runner 并把输出存到独立目录。

## 实验记录
- **短测试（GPU1）**：
  ```bash
  CUDA_VISIBLE_DEVICES=1 conda run -n dc python weian_development/hf_offline_runner_sparse/example_offline_hf_serialized.py \
    --model /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B \
    --dataset aime25.jsonl --qid 1 --rid sparse_verify --model_type deepseek \
    --max_tokens 64 --temperature 0 --top_p 1 --top_k 0 \
    --enable_sparse_pruning --sparse-stats-path weian_development/hf_offline_runner_sparse/stats/qid0008_trace46_stats.pt \
    --sparse-max-keys 512 --sparse-round-window 64 --sparse-offset-max-length 65536 --sparse-head-limit 8 \
    --output_dir tmp_eval
  ```
  - 输出：`tmp_eval/deepthink_offline_qid1_ridsparse_verify_20251106_232334.msgpack`
  - 文本：`tmp_eval/qid1_sparse_response.txt`

- **无上限测试（GPU1, 2048->4096)**：`--max_tokens 2048` / `-1`，用于验证长上下文 + 稀疏裁剪时的行为，输出位于 `tmp_eval/deepthink_offline_qid7_ridsparse_full_*`、`..._sparse_unbounded_*`。

- **TP=2 + 2048 KV（GPU2,3）**：当前正在运行：
  ```bash
  CUDA_VISIBLE_DEVICES=2,3 conda run -n dc python ... --qid 7 --rid sparse_maxctx_2048_tp2 \
    --max_tokens -1 --tensor_parallel_size 2 --enable_sparse_pruning \
    --sparse-max-keys 2048 --sparse-round-window 64 ...
  ```
  - PID: 3332266 (`PD-L1_binder`)，占用 GPU2 ~34GB、GPU3 ~9GB。
  - 结束后结果将落在 `tmp_eval/deepthink_offline_qid7_ridsparse_maxctx_2048_tp2_*.msgpack`，并会同步生成 `tmp_eval/qid7_sparse_maxctx_2048_tp2_response.txt`。

## 使用说明
1. **导出统计**（仅需一次）：
   ```bash
   conda run -n dc python -m weian_development.hf_offline_runner_sparse.export_round_pruning_stats \
     outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0008_trace46 \
     --output-path weian_development/hf_offline_runner_sparse/stats/qid0008_trace46_stats.pt \
     --device cuda:1 --dtype float32
   ```
2. **运行推理**：将以上 `stats.pt` 路径传给 `--sparse-stats-path`。当前实现仅支持单卡，请通过 `CUDA_VISIBLE_DEVICES` 选择目标 GPU。
3. **查看结果**：
   - MsgPack：`tmp_eval/deepthink_offline_qid*_rid*.msgpack*`
   - 解码文本：`tmp_eval/qid*_sparse_*_response.txt`
   - 实时日志（可选）：`run_streaming_sparse_example.sh` / `*_default.sh` 会把 stdout 同步写入 `--stream-log-path` 指定文件，可随时 `tail -f`。

## 批量运行脚本
- `run_offline_deepseek_hf_msgpack.sh`：一键跑完整个 AIME（或 YAML 中配置的离线数据集），内部调用 `run_dispatch_hf_serialized.py`，会将稀疏 runner 分发到多张 GPU 上，每题只生成一个回答，输出目录为 `outputs/deepseek_r1_qwen3_8b/offline_hf_sparse`。运行结束后会自动调用 `weian_development/hf_offline_runner/offline_accuracy_report.py`，把正确率写入 `outputs/deepseek_r1_qwen3_8b/offline_hf_sparse/accuracy_rid*.json`。
- `run_offline_deepseek_hf_msgpack_smoke.sh`：快速冒烟脚本，只跑 qid 0 和 1，其余参数与完整实验一致，方便在小样本上验证稀疏实现。完成后同样生成准确率日志。
- `run_streaming_sparse_example.sh`：单题调试脚本，支持通过 `STREAM_LOG_PATH=/path/to/log` 把实时输出写入文件；`run_streaming_sparse_example_default.sh` 则提供零参数默认值（已对齐 0.6/0.95 的采样策略）。

## 注意事项
- `--max_tokens -1` 表示“不限制新 token 数”，真实生成长度上限由模型上下文决定。
- 每轮裁剪阈值为 `max_keys - round_window`；如果未达阈值，KV 不会删减，显存看起来会持续上升，这是预期行为。
- 多 GPU 加载时，`SparseRoundPruner` 会在需要的地方把索引/张量搬到各自的设备，不需要额外配置。
- 若看到 `CUDA out of memory`，优先检查 GPU 是否被其它 `PD-L1_binder` 占用，必要时用 TP 或低精度加载。
