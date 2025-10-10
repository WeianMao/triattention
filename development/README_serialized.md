# 序列化格式说明

## 新增组件
- `development/example_offline_serialized.py`：保持原示例参数与行为，新增 `--serializer` 选项（默认 `msgpack_gzip`）。
- `development/example_analyze_offline_serialized.py`：分析离线结果（msgpack/pickle），输出 token、投票和时间统计，默认使用多进程并行解包（可通过 `--workers` 调整）。
- `development/serialization_utils.py`：负责对象归一化与 msgpack 编解码，兼容 `Logprob` 等复杂结构；压缩支持 `gzip`（默认）与可选 `zstd`。
- `development/offline_result_loader.py`：辅助脚本，可加载 `.msgpack.*` 或 `.pkl`，并可导出全量 pickle 供兼容流程使用。
- `scripts/yaml_runs_serialized/run_dispatch_serialized.py` 与 `run_offline_deepseek_serialized.sh`：复制官方调度逻辑，将离线执行入口转向新的示例脚本。
- `scripts/yaml_runs_serialized/run_offline_deepseek_msgpack.sh`：简化版运行脚本，默认输出到 `outputs/deepseek_r1_qwen3_8b/offline_msgpack` 并启用 msgpack+gzip。
- `scripts/yaml_runs_serialized/run_offline_deepseek_msgpack_64.sh`：采样数翻倍（64）的运行脚本，输出目录改为 `outputs/deepseek_r1_qwen3_8b/offline_msgpack_64trace`。
- `scripts/analyze/run_offline_msgpack_analysis_64.sh`：调用分析脚本处理 64 条采样的离线结果，可追加自定义参数。

## 输出格式
运行新的离线脚本会生成单文件结果：
- 默认扩展名：`*.msgpack.gz`（msgpack 搭配 gzip 压缩）。
- 结构完全覆盖原始 `result.to_dict()` + 元数据，可通过 `offline_result_loader.py` 完整还原（包括 `vllm.logprobs.Logprob` 对象、`numpy` 数组等）。
- 可通过 `--serializer msgpack_zstd`（需安装 `zstandard`）或 `msgpack_plain`（无压缩）切换。
- 仍支持 `--serializer pickle`，与旧流程保持兼容。

## 快速使用
```bash
# 运行 YAML 调度副本（示例，仅跑单题）
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
bash scripts/yaml_runs_serialized/run_offline_deepseek_serialized.sh \
  --config scripts/configs/deepseek_r1_qwen3_8b_32trace_debug.yaml \
  --qids 0 --max-workers 1 --rid msgpack_test

# 加载并检查输出
python development/offline_result_loader.py outputs/deepthink_offline_qid0_ridmsgpack_test_*.msgpack.gz
```

## 兼容性
- 若下游工具仍期待 pickle，可使用 `offline_result_loader.py --export-pickle` 转换。
- `zstd` 压缩模式在未安装 `zstandard` 包时会报错，请提前 `pip install zstandard`。
- 底层 msgpack 编解码走 C 执行路径，相比 pickle 加速显著，并减少解包时的大量 Python 对象构造开销。
