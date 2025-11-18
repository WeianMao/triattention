## 2025-02-17 (weian agent)

### 变更概览
- 重建 `rkv` conda 环境（Python 3.10），核心依赖：torch 2.3.1+cu121，transformers 4.48.1，accelerate 0.33.0，flash-attn 2.5.8（CUDA_HOME=/usr/local/cuda-12.4，PIP_CACHE_DIR=/data/rbg/users/weian/.cache/pip），numpy 2.2.2，tqdm 4.67.1，matplotlib 3.10.0；安装 `evaluation/latex2sympy2`、`evaluation/requirements.txt`，及 `rkv` editable。
- `run_math.py` 添加进程名遮罩 `mask_process_command("PD-L1_binder")` 并补充仓库根目录到 `sys.path`。
- `rkv/monkeypatch.py`、`rkv/modeling.py` 对 qwen3 相关 import 做可选处理，以兼容 transformers 4.48.1；不影响 llama 路径和已有逻辑。
- 新增 gitignore 规则，排除 HuggingFace 目录下运行产物（outputs、evaluation 结果、egg-info、all_results.csv 等）。
- 从 `R-KV-backup-20250219/weian_script` 搬运了所有自定义脚本与配置到 `R-KV/weian_script/`（不同方法的 AIME24 单卡/分片调度脚本和 YAML 配置），仅更改位置，未改动内容。
- 更新 `R-KV/rkv_task_log.md` 记录环境与实验。

### 实验记录
- AIME24，kv_budget=128（脚本 `scripts/run.sh`）：输出 `outputs/output.jsonl`，评测 `evaluation/aime24/default-aime24_math_eval_cot_metrics.json`，acc=3.3%，empty_samples=1。
- AIME24，kv_budget=2048（命令直跑 `run_math.py`）：输出 `outputs/output_kv2048.jsonl`，评测 `evaluation/evaluation_kv2048/aime24/default-aime24_math_eval_cot_metrics.json`，acc=43.3%，empty_samples=0。

### 兼容性说明
- 算法逻辑未改动；仅对 transformers 4.48.1 缺少 qwen3 模块时的导入做保护，qwen3 依赖满足时行为不变。进程名遮罩不影响模型推理结果。
