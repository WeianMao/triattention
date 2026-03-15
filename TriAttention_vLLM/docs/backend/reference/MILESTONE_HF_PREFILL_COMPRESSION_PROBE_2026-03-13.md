# HF Prefill Compression Probe Milestone

Updated: 2026-03-13
Status: Milestone reached
Owner: Codex

## 1. Why we ran this experiment

当前已经有比较强的现象证据表明：

1. `vLLM + TriAttention` 在长 prefill 触发压缩后，可能出现明显输出退化。
2. 但单看 baseline，并不足以解释这种退化，因为 baseline 更多是长尾质量下降，不是明显胡言乱语。

因此，这一步的目标不是继续纠结 baseline，而是回答一个更关键的问题：

`prefill 压缩本身是否就会把模型搞坏？`

如果答案是“会”，问题方向更偏算法。
如果答案是“不会”，问题方向更偏 vLLM 实现链路。

## 2. Experiment design

这次实验刻意不走 vLLM，而是在 HF backend 上做一个隔离判别实验。

设计原则：

1. 不复用 vLLM runtime 代码，避免把 vLLM 的实现问题带到对照侧。
2. 不追求完整 runtime 复刻，只追求“是否有判别力”。
3. 只要能回答“压缩一发生，模型会不会立刻坏”，这个实验就有意义。

本次采用的模拟方式是：

1. 先完成整段 prefill。
2. 在 HF 侧手动执行一次 prefill 压缩。
3. 再继续正常 decode。

这不是完整 runtime 等价实验，但足够回答当前阶段最核心的问题。

## 3. Code and branch isolation

本实验在隔离分支上完成：

- branch: `codex/hf-prefill-trigger-sim-20260313`

实验脚本：

- `weian_development/demo_debug/hf_prefill_manual_compression_probe.py`

该脚本是隔离 helper，不修改已有 HF baseline 主路径，也不依赖 vLLM runtime 逻辑。

## 4. Model / setting used

模型：

- `JunHowie/Qwen3-32B-GPTQ-Int4`

输入样本：

- openclaw-like 长样本

关键 setting：

1. `top_k = 20`
2. chat-style 输入
3. `prompt_tokens = 7817`
4. 压缩目标：`compressed_prompt_tokens = 7000`
5. `max_new_tokens = 256`

本次只验证 `per-head` 方向，不把 `per-layer-per-head` 混进来。

## 5. Environment notes

这次 HF 32B GPTQ 的判别实验最终在 `trivllm` 环境中完成，而不是 `rkv` 环境。

原因不是算法问题，而是工具链兼容性：

1. `rkv` 环境里一开始缺 `safetensors`，后续虽补上，但 GPTQ 路径仍不顺。
2. `optimum.gptq` 路径出现 `QuantizeConfig` 相关报错。
3. `auto_gptq` 对这类 Qwen3 GPTQ 加载路径支持不完整。
4. `trivllm` 环境已有 `gptqmodel`，因此 HF 原生 GPTQ 加载可以走通。

这件事只影响“在哪个环境里完成了实验”，不影响本次实验结论本身。

## 6. Result summary

输出目录：

- `/tmp/hf_prefill_probe_32b_7000_perhead_v6`

关键结果文件：

1. baseline:
   - `/tmp/hf_prefill_probe_32b_7000_perhead_v6/baseline.jsonl`
2. compress once:
   - `/tmp/hf_prefill_probe_32b_7000_perhead_v6/compress_once.jsonl`

便于人工检查的拷贝：

1. `debug/hf_qwen3_32b_fullkv_prefill7817_out256_topk20.jsonl`
2. `debug/hf_qwen3_32b_prefillcompress7000_prefill7817_out256_topk20.jsonl`

定量摘要：

### baseline

1. `prompt_tokens = 7817`
2. `compressed_prompt_tokens = 7817`
3. `output_tokens = 256`
4. `max_same_ws_run = 1`
5. `max_same_char_run = 2`

### compress_once

1. `prompt_tokens = 7817`
2. `compressed_prompt_tokens = 7000`
3. `output_tokens = 256`
4. `max_same_ws_run = 1`
5. `max_same_char_run = 2`
6. 手动压缩日志显示：
   - `before = 7817`
   - `after = 7000`
   - `reclaimed = 817`

## 7. Human inspection result

人工检查结论：

1. `HF fullkv baseline` 输出正常，能讲人话。
2. `HF prefill compress_once` 输出也正常，能讲人话。
3. 没有复现 vLLM 那边那种明显胡言乱语、乱码、疯狂重复。
4. 两份输出都只是因为 `max_new_tokens = 256` 被截断，不是质量崩坏。

## 8. What this milestone supports

这次实验给出的核心方向性结论是：

`已有证据明显更偏向 vLLM 实现问题，而不是算法本身一做 prefill 压缩就会坏。`

更具体一点：

1. 在同一个 32B INT4 模型上，HF baseline 正常。
2. 在同一个模型、同一个长 prompt、同一个 budget 级别下，HF 手动执行一次 prefill 压缩后也正常。
3. 说明“prefill 压缩一发生，算法就必然把输出搞坏”这个说法，目前不成立。

因此，当前更合理的工作假设是：

`问题主要在 vLLM 的 runtime / 选择 / 压缩 / 长度覆盖 / 回收 等实现链路。`

## 9. What this milestone does NOT prove

这次实验很重要，但边界也要写清楚。

它没有证明以下事情：

1. 没有证明“算法绝对没有任何问题”。
2. 没有证明“HF 模拟一定与 vLLM runtime 完全等价”。
3. 没有证明“只要切回 vLLM 就一定是某一个单点 bug”。
4. 没有覆盖 `per-layer-per-head`。
5. 没有覆盖完整在线 runtime 多次重压缩链路。

所以这次实验的定位应该是：

- 它是一个强方向证据。
- 它足够支撑“下一阶段优先查 vLLM 实现”。
- 它还不是最终形式化证明。

## 10. Confidence assessment

我认为这次实验的可信度已经足够支持当前方向判断，理由如下：

1. 压缩是真实触发了，不是假实验。
2. 压缩后确实继续 decode 了，不是只压缩不生成。
3. 样本长度已经是长 prompt 场景，不是玩具输入。
4. 模型就是当前重点排查的 `Qwen3-32B-GPTQ-Int4`，不是换成别的更容易模型。

局限主要在于：

1. decode 长度只有 `256`，不是超长 decode。
2. 只做了“一次 prefill 压缩再继续生成”，不是完整 runtime 连续触发。
3. 某些 chat/template 细节没有追求和线上 demo 完全一致。

综合判断：

- 用来支持“当前优先查 vLLM 实现”这个决策：可信度高。
- 用来宣称“算法已经被彻底排除”：还不够。

## 11. Immediate implication for next phase

下一阶段不该继续在“算法是不是天然不 work”上消耗主要精力，而应该：

1. 全面审视 vLLM 的实现链路。
2. 先列出所有真正有可能导致当前坏输出的嫌疑点。
3. 按嫌疑大小排序。
4. 再进入逐项验证阶段。

对应嫌疑审计文档：

- `TriAttention_vLLM/docs/backend/reference/VLLM_PREFILL_COMPRESSION_SUSPECT_AUDIT_2026-03-13.md`
