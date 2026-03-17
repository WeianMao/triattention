# R-KV 分片脚本 vs 官方单卡脚本问题记录

> 记录时间：2025-11-19  
> 记录人：Codex Agent  
> 相关脚本：`run_rkv_aime24_single.sh`、`run_rkv_aime24_sharded.sh`

## 背景与现象

- 官方单卡脚本 `R-KV/weian_script/run_rkv_aime24_single.sh`（kv_budget=2048，flash-attn2）在 `tmp_eval/single_outputs_eval_results/all_results.csv` 得到 43.3 分。
- 分片脚本 `R-KV/weian_script/run_rkv_aime24_sharded.sh` 在 `R-KV/outputs/rkv_aime24_sharded/eval/all_results.csv` 只有 33.3 分，生成阶段就出现大幅偏差。
- 其他配置（如 `sdpa`、`fp16`、`reset_cache_each_batch`）能与单卡对齐，但默认设置始终得分偏低。

## 核心发现（已确认）

1. **输出完全确定，但压缩触发点错位**  
   - `do_sample=False`，同一脚本多次运行字节级一致，说明差异不是随机性。  
   - 对比单卡与分片的 JSONL，默认配置有 25/30 条输出不同；`sdpa_fp16_reset` 也有 29/30 条不同（只是分数巧合接近）。

2. **根因：跨样例的 `self.length` 没有重置**  
   - `CausalLM_forward` 用 `self.length % divide_length == 0` 决定何时开启 KV 压缩（`divide_length=128`）。  
   - 单卡脚本串行 30 题，`self.length` 贯穿所有题目，第二题起始就带着第一题的 token 数，压缩时间点被整体平移。  
   - 分片脚本每个 shard 从 0 开始计数，压缩点落在 0,128,256…，与单卡全局累积的轨迹完全不同。统计显示 29/30 个样例的“起始 mod128”在单卡与分片间不同。

3. **受影响的脚本范围**  
   - 所有基于 HuggingFace 路径的 R-KV 脚本（`R-KV/weian_script/*` 单卡、分片，以及 sdpa/fp16/reset 变体）都会经历同一 `CausalLM_forward`，因此默认配置下都受此状态污染。

## 采取的修复

- 在 `R-KV/HuggingFace/rkv/modeling.py` 将 `self.length` 在检测到新样例（`past_key_values` 为空）时重置为 0，再累加本轮输入长度。这样每个题目都从 0 计数，压缩调度仅由当前题目决定，单卡与分片的调度对齐。

## 接下来建议验证

1. 重新运行 `run_rkv_aime24_single.sh` 与 `run_rkv_aime24_sharded.sh`，比较 `merged/merged.jsonl` 与分数，确认已对齐（本修复应同步覆盖所有 weian_script 变体）。  
2. 如需继续测试 `sdpa`/`fp16`/`reset_cache_each_batch`，可在修复后再跑一遍，记录新的得分。  
3. 若未来确实有“跨轮长会话”需要全局计数，可在模型里显式做成开关；当前 AIME24 场景按题重置是正确且兼容分片的策略。
