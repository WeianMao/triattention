# SpeckV RoPE 反旋偏差修复记录

## 根因/现状
- 之前将 Llama/YaRN 误判为 interleaved（偶/奇维度配对），`invert_rope` 走错分支，导致 `apply_rotary_pos_emb` → `invert_rope` 最大误差 ~3。HF Llama（含 llama3 rope_scaling）实际使用 front/back 半维配对。
- RoPE 同一频段的 x/y 分量仍共享同一 cos/sin 频率；「even/odd 频率不同」的修复方向并非问题根源。

## 已修复
- `determine_rope_style` 现在对 Llama 返回 half 风格，保持 Qwen 行为不变；interleaved 逆解分支保留并在测试覆盖。
- 新增验证：`R-KV/weian_development/speckv/tests/test_rope_inversion.py`  
  - DeepSeek-R1 llama3 配置下，`apply_rotary_pos_emb` → `invert_rope` 误差 < 1e-6，并断言风格为 half。  
  - 合成的 interleaved（偶/奇 cos/sin 不对称）案例也能正确反旋。
- 校验命令：`PYTHONPATH=R-KV pytest R-KV/weian_development/speckv/tests/test_rope_inversion.py`

## 下一步
1) **验证与校准**  
   ```
   conda run -n rkv python R-KV/weian_development/rkv_sparse_round_calibrate.py \
     --trace-root R-KV/outputs/sample8_fullkv_aime24_official \
     --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B \
     --output-path R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt \
     --head-sample-file R-KV/weian_development/speckv/stats/deepseek_r1_llama8b_heads.json \
     --kv-budget 2048 \
     --attn-implementation flash_attention_2
   ```

2) **性能复测**  
   - 使用修正后的统计文件重跑 SpeckV 实验：`bash R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`（保持 use_chat_template=false，kv_budget=2048 等）。

## 参考信息
- 相关文件：  
  - `R-KV/weian_development/speckv/round_pruning_utils.py`（RoPE 反旋/配对）  
  - `R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py`（打分使用的反旋）  
  - `R-KV/weian_development/rkv_sparse_round_calibrate.py`（统计生成）
- 模型：`/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B`
- 当前统计路径（需覆盖）：`R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt`
