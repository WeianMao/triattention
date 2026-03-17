# RKV 重构 TODO（LazyEviction）

> 采用 `- [ ]` / `- [x]` 勾选。执行某项前补充可操作子步骤；发现关键风险先记录再修改。

## 1. 需求确认与素材整理
- [x] 摘取 `message.md` 与用户要求，生成可执行规范（预算、prefill、不依赖外部、对齐脚本）。
  - 需求：实现全部放在 `LazyEviction/` 内，默认遵循 LazyEviction 规则（prefill 不压缩且不计入 KV budget、贪心单样本），提供 `run_sparse_prefill_keep_sharded_eval.sh` 风格一键脚本，PD-L1 进程前缀，避免改动现有默认值/路径。
- [x] 清点 RKV 现有实现：`rkv_sharded_dispatch.py`、`rkv_sharded_runner.py`、相关 YAML（含默认超参与 budget 逻辑）。
  - YAML（`sample8_rkv_aime25_official_qwen.yaml`）：kv_budget=2048，max_length=32768（总长），num_samples=8，temp=0.6/top_p=0.95，flash_attn2 + bf16，reset_cache_each_batch=false，fp32_topk=false，eval_batch_size=1，seed=666。
  - Runner（`rkv_sharded_eval.py`）：强制 do_sample=True（无贪心路径），max_length 作用于总长，prefill 计入 max_length；默认 plain prompt（忽略 argparse 中的 use_chat_template），build_prompt 复用 SpeckV 模板；kv_budget 仅透传至 monkeypatch，未看到显式排除 prefill/padding；attn 实现/精度/kv 更新等均来自 args。

## 2. 接口与配置设计（默认不变）
- [x] 拟定 LazyEviction 版 RKV cfg schema（YAML + argparse），标注默认值与 LazyEviction 对齐点（budget 定义、prefill 处理、数据/提示词、解码、长度、attn/dtype）。
  - YAML 方案：沿用 `experiment` + `runner_args`，默认 conda_env=lazy_evict，runner_path 放 `weian_development/rkv_lazy_runner.py`，log/method_output_dir 使用新前缀（如 `outputs/rkv_lazy/`）；runner_args 含 dataset_path/model_path/tokenizer_path、kv_budget、max_length_mode（legacy_total|max_new_tokens，默认 max_new_tokens=16384）、eval_batch_size、seed、method=rkv、attn_implementation、load_dtype、reset_cache_each_batch、fp32_topk、do_sample（默认 False）、temperature/top_p/num_samples/aggregation、prompt_style（plain|chat）、system_prompt、trust_remote_code。
- [x] 设计公平性开关：采样/贪心、num_samples/聚合、chat 模板切换、长度/截断模式、attn/dtype、seed/reset、trust_remote_code。
  - 公平性开关：prompt_style plain/chat + system_prompt；decode 模式（do_sample、temperature/top_p、num_samples、aggregation=pass_at_1|majority）；kv_budget_mode（count_prefill=False 默认、count_padding=False 默认、per_layer/head 均分与否）；length_mode（max_new_tokens vs total_max_length=32768 兼容）；attn/dtype（flash2/sdpa/eager，bfloat16/float16），reset_cache_each_batch/fp32_topk，trust_remote_code，seed + per-run stride。

## 3. 调度/Runner 原型
- [x] 定义新的 dispatch/runner 路径（放 `weian_development/` + `LazyEviction/weian_script/`），确保输出/日志目录隔离；路径/参数风格对齐 `run_sparse_prefill_keep_sharded_eval.sh`。
  - 方案：新增 `LazyEviction/weian_script/run_rkv_lazy_sharded_eval.sh`（调用 `weian_development/rkv_lazy_dispatch.py` + 默认 YAML `LazyEviction/weian_script/configs/rkv_lazy_aime.yaml`），调度逻辑参考 `lazy_eviction_sparse_prefill_keep_dispatch.py`，保留 skip-existing、auto GPU、PD-L1 前缀、merge+eval 入口。
- [x] 校准 KV budget 计数与 prefills 逻辑，必要时添加日志打印/断言，避免“RKV 占便宜”。
  - 计划：在 runner 内显式计算 prefill_tokens/output_tokens，并打印/记录可用 kv_budget（按 LazyEviction 定义：prefill 不计 budget；padding 不计；预算按 token 计而非 head 倍数）。未对齐时通过 cfg 切换 legacy 模式，并在 meta 中写入 budget 口径。

## 4. 验证与对比计划
- [x] 设计 smoke 命令（单 shard、少量样本、最小预算），覆盖 budget 统计、prefill 处理、数据/模板切换、解码模式切换。
  - 示例：`python weian_development/rkv_lazy_dispatch.py --config LazyEviction/weian_script/configs/rkv_lazy_aime.yaml --num-shards 1 --gpus 0 --dry-run`；实际 smoke 用 `--max-examples 2 --num-samples 1 --kv_budget 256 --max_new_tokens 512 --do-sample False --prompt_style chat` 验证 chat/greedy 口径。
- [x] 规划公平对比：与 LazyEviction 现有方法在相同设置下的小规模实验；评估口径记录（单样本 vs 多样本平均）。
  - 对照：固定 chat 模板 + 贪心单样本，与 `sparse_prefill_keep`/fullkv 用相同 max_new_tokens/kv_budget；另备采样 8 次、pass@1 聚合配置复现 RKV 原版；评估脚本沿用 LazyEviction 现有 eval，多样本输出需支持合并后求 pass@1。

## 5. 文档与脚本
- [x] 更新 `status_overview.md`/`rebuild_plan.md`/`message.md` 与执行记录。
- [x] 准备一键运行脚本与 YAML 占位（`LazyEviction/weian_script/`），确保默认不会覆盖旧实验；在脚本中标注依赖尚未完成的部分。

## 6. 收尾检查
- [x] 运行全量 AIME：8 卡，输出 `outputs/rkv_lazy_aime/merged/merged.jsonl`，评测 `accuracy=0.5333 (16/30)`（LazyEviction 判分口径），指标写入 `outputs/rkv_lazy_aime/merged/metrics.json`。
- [ ] 运行 `python -m compileall` 覆盖改动文件；记录 smoke 结果或缺口。
- [ ] 列出剩余风险（预算定义、prompt/数据差异、评估口径）与后续步骤。
