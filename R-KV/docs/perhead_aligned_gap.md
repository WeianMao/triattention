# SpeckV per-head 在 aligned 重构路径未生效的问题说明

## TL;DR
- **问题**：`aligned` 代码路径启用 `--rkv-style-compression`/`--rkv-style-slack-trigger`，进入 `speckv_rkv_style.py`，该实现没有 per-head 支持，导致 `--per-head-pruning` 在 aligned 版被忽略。
- **现象**：`run_speckv_aime24_qwen_norm_aligned_perhead.sh` 实际行为 = aligned 基线，无 per-head；无法与非 aligned per-head 做公平对比。
- **解决选项**：
  1) 在 `speckv_rkv_style.py` 中补齐 per-head 逻辑（推荐，保持 rkv-style 路径）。
  2) 临时移除 rkv-style 参数，让 aligned-perhead 走原始 wrapper（会偏离 aligned 基线的压缩路径，不推荐长期使用）。

## 背景
- 目标：对比 “aligned 基线” 与 “aligned + per-head”。
- 基线脚本：`R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned.sh`
- 对比脚本：`R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`
- 配置文件：`aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml`（aligned 与 aligned_perhead 共享）

## 代码路径对比
- **非 aligned / 原始 SpeckV**
  - 调用 `apply_speckv_generate_patch` (`weian_development/speckv/rkv_speckv_generate.py`)
  - per-head 逻辑位于 `SparseRoundPruner`：`weian_development/speckv/sparse_round_pruner_prefill_keep.py:432-470`
  - `--per-head-pruning` 生效。

- **aligned / rkv-style**
  - 触发条件：`--rkv-style-compression`（脚本行 22）/ `--rkv-style-slack-trigger`
  - 代码路径：`weian_development/rkv_sharded_eval.py:602-660` -> `apply_speckv_rkv_style_patch`
  - 实现文件：`weian_development/speckv/speckv_rkv_style.py`
  - **问题点**：没有 per-head 开关/参数，未实现 per-head 独立 top-k；`--per-head-pruning` 被忽略。

## 影响
- 当前的 aligned_perhead 脚本与 aligned 基线输出一致，per-head 实验未实际运行。
- 对比实验失效：无法评估 per-head 在 aligned 重构上的效果。

## 复现步骤（快速验证）
1. 运行 aligned_perhead（任意小样本，`--qids 0 --max-workers 1`）：
   ```bash
   R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh --qids 0 --max-workers 1 --max-examples 1
   ```
2. 检查日志/元数据：`per_head_pruning` 不会出现在 runner args；压缩路径仍为 rkv-style。
3. 对比非 aligned perhead（原始路径）可看到 `per_head_pruning=True` 进入 `SparseRoundPruner`。

## 修复选项
### 选项 1（推荐）：在 rkv-style 实现中补齐 per-head
- 位置：`weian_development/speckv/speckv_rkv_style.py`
- 动作：
  - 为 `SpeckVRKVStyleConfig` / `apply_speckv_rkv_style_patch` 增加 `per_head_pruning` 参数。
  - 在 `compute_keep_indices` 中增加 per-head 独立 top-k 逻辑（可参考 `sparse_round_pruner_prefill_keep.py` 的 per-head 分支），保持与原始 per-head 聚合方式一致。
  - 确认与 `divide_length`/`use_slack_trigger` 触发节奏兼容。
- 验证：
  - 单元/烟测：对比 per-head on/off，在同一 seed 下日志中 `per_head_pruning=True`，且 cache 截断位置、保留 token 数有差异。
  - 最小样本命令同上（可加 `--dry-run` 验证 args 传递）。

### 选项 2（临时绕过，不推荐长期）
- 从 aligned_perhead 脚本移除 `--rkv-style-compression` / `--rkv-style-slack-trigger`，让其走原始 wrapper，使 `--per-head-pruning` 生效。
- 代价：算法路径不再与 aligned 基线一致（压缩触发/实现不同），对比意义下降。

## 建议执行步骤
1. 选定修复方案（推荐选项 1）。
2. 实现 per-head 支持于 `speckv_rkv_style.py`，复用原始 per-head 聚合逻辑。
3. 添加最小化回归测试（或烟测脚本）：
   - 同一输入，比较 per-head on/off 的保留索引差异。
   - 确认 rkv-style 仍按预期节奏压缩（divide_length/slack）。
4. 更新脚本/文档：
   - 如果有新参数，确保 `rkv_sharded_dispatch.py` 传递到 rkv-style 分支。
   - 在脚本注释中标注“aligned per-head 现已生效”。

## 参考文件
- `weian_development/rkv_sharded_eval.py:602-660`
- `weian_development/speckv/speckv_rkv_style.py`
- `weian_development/speckv/sparse_round_pruner_prefill_keep.py:432-470`
- `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`
- `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_perhead.sh`
