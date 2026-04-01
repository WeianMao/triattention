# 起点脚本（Reference Script）

## 主实验脚本

`R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`

## 关键 flag 组合

```bash
--rkv-style-compression      # 在 attention layer 内部触发压缩，和 RKV 一样的方式
--rkv-style-slack-trigger     # slack trigger
--per-head-pruning            # per-head 级别的剪枝
--sparse-normalize-scores     # score 归一化
--divide-length 128
```

## 关键参数值

- Model: DeepSeek-R1-Distill-Qwen-7B
- Budget: 2048
- Dataset: AIME24, sampled 8 draws, seed=888
- Attention: flash_attn2 + bfloat16

## 重要说明

1. **所有 setting 都要以这个脚本为准**，不要搞错 flag 组合
2. 其他 setting（不同模型、不同数据集）也要公布，但都以这个脚本的 flag 模式为基准
3. 脚本名中的 "aime" 在 release 时需要泛化处理（详见 [../code_cleanup/04_naming.md](../code_cleanup/04_naming.md)）
4. Flag 名中的 `rkv-style` 在 release 时需要考虑是否改名（待确认）
