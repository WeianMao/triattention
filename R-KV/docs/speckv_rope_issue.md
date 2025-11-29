# SpeckV RoPE 反旋偏差待修复事项

## 现状
- 经过多轮修复，SpeckV 的前向裁剪、模板对齐已落地，但 Llama/YaRN 场景下的 RoPE 反旋仍有偏差。
- 当前 `invert_rope` 在 interleaved（Llama）模式下使用了矢量化解法，已考虑 cos_even/sin_even/cos_odd/sin_odd 不相等的情况，但实测用模型的 `apply_rotary_pos_emb` 先旋转、再调用 `invert_rope` 还原，最大误差仍约 2.7，说明反旋公式仍未与真实前向完全一致。
- 该偏差会导致频域统计/打分与模型实际 RoPE 不一致，可能是 SpeckV 性能未恢复的核心原因。

## 待办目标
1) **精确反旋 interleaved RoPE**  
   - 在 `R-KV/weian_development/speckv/round_pruning_utils.py` 的 `invert_rope` 中，针对 Llama/YaRN 的 interleaved 模式改为逐对 2×2 矩阵显式求逆（单独处理每一对 even/odd 维度的 cos/sin，不假设对称），确保 `apply_rotary_pos_emb` → `invert_rope` 误差 ~ 0。  
   - 保持 Qwen half 模式行为不变。

2) **验证与校准**  
   - 用单元测试或脚本验证：随机向量 `x`，`apply_rotary_pos_emb` 旋转 → `invert_rope` 还原，max error ≈ 0（CPU 模式即可）。  
   - 重新运行校准：  
     ```
     conda run -n rkv python R-KV/weian_development/rkv_sparse_round_calibrate.py \
       --trace-root R-KV/outputs/sample8_fullkv_aime24_official \
       --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B \
       --output-path R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt \
       --head-sample-file R-KV/weian_development/speckv/stats/deepseek_r1_llama8b_heads.json \
       --kv-budget 2048 \
       --attn-implementation flash_attention_2
     ```

3) **性能复测**  
   - 使用修正后的统计文件重跑 SpeckV 实验：`bash R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`（保持 use_chat_template=false，kv_budget=2048 等）。

## 参考信息
- 相关文件：  
  - `R-KV/weian_development/speckv/round_pruning_utils.py`（RoPE 反旋/配对）  
  - `R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py`（打分使用的反旋）  
  - `R-KV/weian_development/rkv_sparse_round_calibrate.py`（统计生成）
- 模型：`/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B`
- 当前统计路径（需覆盖）：`R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt`
