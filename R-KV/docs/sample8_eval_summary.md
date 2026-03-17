# sample8 多抽样评测（pass@1 均值）

- 评测代码已修正为“同一题多次采样先取均值，再对题目取均值”（与论文一致），路径：`R-KV/HuggingFace/evaluation/evaluate.py`。
- 调用方式：`python R-KV/HuggingFace/evaluation/eval_math_multi.py --base_dir <merged_or_shards> --output_dir <exp>/eval --exp_name <run_name> --dataset aime24`。
- 下表基于 2025-03 修复后的评测重新计算；acc 单位为 %。

| run_name | num_questions | acc (pass@1 mean) | eval file | notes |
| --- | --- | --- | --- | --- |
| sample8_fullkv_aime24_official | 30 | 50.4 | R-KV/outputs/sample8_fullkv_aime24_official/eval/sample8_fullkv_aime24_official/aime24/default-default_math_multi_eval.jsonl | 8 draws/题，完整 |
| sample8_rkv_aime24_official | 30 | 52.9 | R-KV/outputs/sample8_rkv_aime24_official/eval/sample8_rkv_aime24_official/aime24/default-default_math_multi_eval.jsonl | 8 draws/题，完整 |
| sample8_snapkv_aime24_official | 30 | 38.8 | R-KV/outputs/sample8_snapkv_aime24_official/eval/sample8_snapkv_aime24_official/aime24/default-default_math_multi_eval.jsonl | 8 draws/题，完整 |
| sample8_speckv_aime24_quick_clean | 6 | 4.2 | R-KV/outputs/sample8_speckv_aime24_quick_clean/eval/sample8_speckv_aime24_quick_clean/aime24/default-default_math_multi_eval.jsonl | 6 题 × 8 draws |
| sample8_speckv_aime24_quick_v2 | 2 | 37.5 | R-KV/outputs/sample8_speckv_aime24_quick_v2/eval/sample8_speckv_aime24_quick_v2/aime24/default-default_math_multi_eval.jsonl | 仅 2 题，draw 数不齐（1–8） |

缺失：`sample8_h2o_aime24_official` 目录为空，未评测。
