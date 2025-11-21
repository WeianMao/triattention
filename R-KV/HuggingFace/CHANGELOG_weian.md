## 2025-03-xx (weian agent)

- 增强 HuggingFace 跑法：`run_math.py` 与 `weian_development/rkv_sharded_eval.py` 支持 `num_samples`/`temperature`/`top_p`（默认 64/0.6/0.95），AIME24/25 默认 `max_length=32768`，强制 `eval_batch_size=1`，输出带 `draw_idx`。
- 分片改为“每卡跑全集题目、均分采样次数”，按 `shard_id` 偏移随机种子；合并按 `(sample_idx, draw_idx)` 排序，调度默认跑多样本评测。
- 新增多样本评估脚本 `evaluation/eval_math_multi.py`，`rkv_sharded_dispatch` 调用它生成 pass@1 统计。
- 新增采样版脚本/YAML（64/8 draws）覆盖 rkv/fullkv/snapkv/streamingllm/h2o，输出与日志目录使用全新命名；`weian_script/README.md` 记录入口。

## 2025-02-17 (weian agent)

### 变更概览
- 重建 `rkv` conda 环境（Python 3.10），核心依赖：torch 2.3.1+cu121，transformers 4.48.1，accelerate 0.33.0，flash-attn 2.5.8（CUDA_HOME=/usr/local/cuda-12.4，PIP_CACHE_DIR=/data/rbg/users/weian/.cache/pip），numpy 2.2.2，tqdm 4.67.1，matplotlib 3.10.0；安装 `evaluation/latex2sympy2`、`evaluation/requirements.txt`，及 `rkv` editable。
- `run_math.py` 添加进程名遮罩 `mask_process_command("PD-L1_binder")` 并补充仓库根目录到 `sys.path`。
- `rkv/monkeypatch.py`、`rkv/modeling.py` 对 qwen3 相关 import 做可选处理，以兼容 transformers 4.48.1；不影响 llama 路径和已有逻辑。
- 新增 gitignore 规则，排除 HuggingFace 目录下运行产物（outputs、evaluation 结果、egg-info、all_results.csv 等）。
- 从 `R-KV-backup-20250219/weian_script` 搬运了所有自定义脚本与配置到 `R-KV/weian_script/`（不同方法的 AIME24 单卡/分片调度脚本和 YAML 配置），仅更改位置，未改动内容。
- 将 `R-KV-backup-20250219/weian_development` 下的 rkv 分片调度/合并/评测工具拷贝到 `R-KV/weian_development/` 以支撑 weian_script；不改算法逻辑。
- weian_script 默认 kv_budget 更新为 2048（rkv、h2o、snapkv、streamingllm 分片配置），与当前实验设定一致。
- 更新 `R-KV/rkv_task_log.md` 记录环境与实验。

### 实验记录
- AIME24，kv_budget=128（脚本 `scripts/run.sh`）：输出 `outputs/output.jsonl`，评测 `evaluation/aime24/default-aime24_math_eval_cot_metrics.json`，acc=3.3%，empty_samples=1。
- AIME24，kv_budget=2048（命令直跑 `run_math.py`）：输出 `outputs/output_kv2048.jsonl`，评测 `evaluation/evaluation_kv2048/aime24/default-aime24_math_eval_cot_metrics.json`，acc=43.3%，empty_samples=0。
- AIME24，kv_budget=2048，8 卡分片（`weian_script/run_rkv_aime24_sharded.sh`）：输出 `R-KV/outputs/rkv_aime24_sharded/merged/merged.jsonl`，评测 `R-KV/outputs/rkv_aime24_sharded/eval/.../default-aime24_math_eval_cot_metrics.json`，acc=33.3%，empty_samples=0。

### 兼容性说明
- 算法逻辑未改动；仅对 transformers 4.48.1 缺少 qwen3 模块时的导入做保护，qwen3 依赖满足时行为不变。进程名遮罩不影响模型推理结果。
