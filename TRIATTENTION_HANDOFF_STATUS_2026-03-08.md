# TriAttention vLLM 交接状态（2026-03-08）

## 1. 结论先看

当前 `master` 分支上，TriAttention demo 路径已经切到 **runtime V2 接口**，不再依赖已退役的 V1 `CUSTOM backend` 接线。  
已完成的关键目标：

1. V2 接口切换与兼容性修复完成（可在 `trivllm` 环境下激活插件）。
2. 旧版 streaming 卡顿根因已定位（V1 token-loop gather/scatter 热点）。
3. 单机 smoke 结果显示输出可用；在已跑完的 SpeckV 压缩实验日志中可确认压缩触发。

## 2. 本阶段已解决的问题

### 2.1 Demo 接口用错版本（V1 vs V2）

已修复为默认走 runtime V2：

- `TriAttention_vLLM/triattention/plugin.py`
- `TriAttention_vLLM/linxi_dev/run_vllm_serve.sh`
- `demo/DEMO_STARTUP.md`
- `TRIATTENTION_V2_INTERFACE_SWITCH.md`

效果：

1. 默认不再要求 `--attention-backend CUSTOM`。
2. 通过 `VLLM_PLUGINS=triattention` + `TRIATTN_RUNTIME_*` 配置走 V2。
3. 保留 legacy 开关：`TRIATTENTION_INTERFACE=legacy_custom`（仅兼容，不推荐）。

### 2.2 插件在 editable 环境导入不稳

已修复 runtime 导入 fallback 的路径鲁棒性（避免硬编码绝对路径），并处理临时 `sys.path` 清理，降低跨机器路径变化导致的失败概率。

### 2.3 启动器环境依赖问题（无 pixi 时）

已修复 `run_vllm_serve.sh` 启动策略：

1. 优先 `pixi run`
2. 其次直接 `vllm`
3. 再次 `conda run -n trivllm`

减少“机器没有 pixi 导致无法启动”的问题。

### 2.4 Streaming 卡顿根因定位

已通过旁路基准定位：旧 V1 路径在压缩触发时存在 Python token-loop gather/scatter 热点，容易在大模型上形成秒级顿挫。  
证据文档/产物：

- `weian_development/demo_debug/STUTTER_ROOT_CAUSE.md`
- `weian_development/demo_debug/artifacts/stutter_root_cause_bench_cpu.json`
- `weian_development/demo_debug/artifacts/stutter_root_cause_bench_cpu_l36.json`

## 3. 压缩触发与输出质量现状

### 3.1 是否触发压缩

在已完成的 SpeckV 压缩实验日志中可以确认触发：

- `R-KV/logs/aime_sampled8/speckv/aime25/norm_aligned_perhead/rkv_aime24_shard00.log:11446`
- `R-KV/logs/aime_sampled8/speckv/aime25/norm_aligned_perhead/rkv_aime24_shard01.log:8467`

可见大量：

- `Effective size: 2176, Should compress: True`

### 3.2 触发压缩后输出是否“胡言乱语”

抽样检查输出文件：

- `R-KV/outputs/aime_sampled8/speckv/aime25/norm_aligned_perhead/shards/shard00/run000.jsonl`

观察结果：

1. 大部分样本前中段推理文本可读、非随机乱码。
2. 存在长输出尾部重复（超长生成场景下出现循环句式）。
3. 粗筛结果：30 条中约 4 条有明显重复尾巴（约 13.3%）。

结论：不是全面胡言乱语，但超长生成仍有退化风险。

## 4. 未解决 / 未完成验证

### 4.1 本机 `vllm serve` 启动异常（环境侧阻断）

在当前这台机器上，近期出现了一个环境级问题：  
`vllm serve` 进程能起来，但端口长时间不监听（包括 `--no-triattention` 对照也出现），因此阻断了“当场重跑并实时抓压缩日志”的验证链路。

这意味着：

1. TriAttention 逻辑不一定有新回归；
2. 但当前机器的 serve 可用性不稳定，建议换机器或先修环境再做最终演示录制。

### 4.2 本轮未完成的演示级对比

以下仍建议补齐：

1. 同机同 prompt 的 `fullkv vs triattention` 端到端 streaming 延迟对比（含压缩触发点）。
2. “baseline OOM、triattention 不 OOM”的现场复现实验（用户已允许后续再做）。

## 5. 运行与测试说明（交接可直接用）

## 5.1 环境准备

1. 进入仓库：
   - `cd /data/rbg/users/weian/project/rl/dc`
2. 使用 conda 环境：
   - `conda activate trivllm`
3. 确认 TriAttention editable 安装：
   - `python -m pip install -e TriAttention_vLLM`

## 5.2 启动 vLLM（runtime V2）

示例（单卡）：

```bash
CUDA_VISIBLE_DEVICES=5 \
MODEL=/data/rbg/users/weian/project/rl/datasets/Qwen2.5-0.5B \
STATS_PATH=/data/rbg/users/weian/project/rl/dc/weian_development/demo_debug/artifacts/qwen25_05b_stats.pt \
HOST=127.0.0.1 PORT=8031 \
KV_BUDGET=512 DIVIDE_LENGTH=128 WINDOW_SIZE=128 \
TRIATTENTION_INTERFACE=runtime \
TriAttention_vLLM/linxi_dev/run_vllm_serve.sh
```

健康检查：

```bash
curl -s http://127.0.0.1:8031/v1/models
```

## 5.3 快速功能探针（stream）

```bash
conda run -n trivllm python weian_development/demo_debug/stream_stutter_probe.py \
  --base-url http://127.0.0.1:8031 \
  --prompt-file weian_development/demo_debug/artifacts/long_prefill_prompt.txt \
  --prompt-index 0 \
  --max-tokens 256 \
  --temperature 0.0 \
  --top-p 1.0 \
  --stutter-threshold-ms 800 \
  --output-json weian_development/demo_debug/artifacts/runtime_longprefill_stutter_probe.json
```

查看结果：

- `weian_development/demo_debug/artifacts/runtime_longprefill_stutter_probe.json`

## 5.4 检查“压缩是否触发”

对已完成日志做检查：

```bash
rg -n "Should compress: True" R-KV/logs/aime_sampled8/speckv/aime25/norm_aligned_perhead/rkv_aime24_shard00.log
```

## 5.5 检查输出是否退化

重点看输出尾部是否大量重复：

```bash
python - <<'PY'
import json
path='R-KV/outputs/aime_sampled8/speckv/aime25/norm_aligned_perhead/shards/shard00/run000.jsonl'
with open(path,'r',encoding='utf-8') as f:
    for i,line in zip(range(5),f):
        o=json.loads(line)
        out=str(o.get('output',''))
        print(i, o.get('index'), o.get('output_tokens'), out[-200:].replace('\n',' '))
PY
```

## 6. 相关提交（本阶段关键）

- `f945b71d` feat: switch demo serve interface from legacy CUSTOM backend to runtime v2
- `f8d9813a` fix: make runtime plugin import resilient in editable envs
- `989abf11` refactor: make runtime import fallback path-temporary and relocation-safe
- `08d03d96` fix: add no-pixi fallback for vllm serve launcher

## 7. 交接建议

1. 优先在一台“serve 启动稳定”的机器复跑 5.2~5.5（避免被当前机器环境问题干扰）。
2. Demo 演示先控制输出长度，避免把“超长尾部重复”放大为算法失效误判。
3. 若要做最终 release 验收，补齐 4.2 中两项对比测试并固化为脚本。
