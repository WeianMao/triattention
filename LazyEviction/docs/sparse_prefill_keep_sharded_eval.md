# SparseRound Prefill-Keep：执行与算法（方法部分风格）

本文像论文“方法”一节，聚焦两件事：1）从启动到产出指标的执行链；2）SparseRound Prefill-Keep KV 压缩的原理与校准（统计提取）流程。读完即可把握决策点和默认超参。

## 1. 执行链（从命令到结果）
**入口 → 分片 → 单 shard 推理 → 合并**
1) 启动命令（bash 封装）设置环境并调用“分片调度器”。  
2) 调度器读取 YAML：确定 GPU 列表、总 shard 数；为每个 shard 启独立 Python 进程，传递 `--shard_id`/`--num_shards`，日志各写独立文件。  
3) 每个 shard 内：加载模型/分词器 → 按 shard 切分样本 → 运行 SparseRound Prefill-Keep 推理 → 写预测与指标（按 KV 容量分目录）。  
4) 所有 shard 结束后（默认）合并：同一 KV 容量下的预测/指标合并为一份。

## 2. 默认配置（AIME 任务，SparseRound Prefill-Keep）
- GPU：自动挑显存占用 ≤200 MiB 的卡，失败回退 0-7；`num_shards=8`。  
- 模型：DeepSeek-R1-Distill-Qwen-7B；同路径 tokenizer。  
- 评测：`benchmark=aime`，`data_type=test`，`eval_batch_size=1`，`max_new_tokens=16384`，`temperature=0.0`。  
- 方法与关键超参：  
  - `method=sparse_round_prefill`  
  - `max_kv_capacity=1492`（动态 KV 上限，前缀固定保留）  
  - `decoding_recent_size=363`；`sparse_round_window=363`（<=0 时回退到 `decoding_recent_size`）  
  - 统计文件：预采样头部频域统计（同模型）  
  - 其他：`sparse_offset_max_length=65536`，`sparse_score_aggregation=mean`，`sparse_head_limit=-1`，`sparse_seed=0`，`attn_implementation=sdpa`，`use_cache=true`，`alpha=1e-4`。

## 3. 数据与分片
- 样本：从 AIME 配置生成统一的 `test.jsonl`。  
- 分片规则：全局索引取模 `idx % num_shards == shard_id`；每 shard 拿到子集。  
- 提示词：Qwen/Qwen3 聊天模板，要求答案写在 `\boxed{}`。  
- 输出：按 KV 容量号（此处 1492）分目录记录预测与指标；合并时汇总同容量结果。

## 4. 算法原理：SparseRound Prefill-Keep
目标：**保留完整前缀**，仅对解码阶段新增 KV 进行轮次化稀疏化，维持在 `max_kv_capacity` 下尽量保留“高贡献”键值。

### 4.1 状态与配置
- 容量与轮长：`max_kv_capacity` 必须 ≥ `sparse_round_window`。  
- 统计：仅对“被采样的头”有频域统计（均值相位/幅值）；可用 `sparse_head_limit` 再截断。未采样头在打分中等价于“无评分、不参与排序”。  
- 几何距离：`offsets`=1,2,4,... 直到 `sparse_offset_max_length`。  
- RoPE 相关：`omega`、`freq_scale_sq`、`attention_scaling` 从模型配置推导。  
- 随机性：`sparse_seed` 控制微扰，避免分数并列。

### 4.2 生命周期（单次推理）
1) **Prefill**：完整前向一次，KV 全入缓存；记录 `prefix_length`，后续剪枝仅作用于前缀之后。  
2) **解码循环**（贪心：temperature=0）：每生成一 token 手动推进 `position_ids`。  
   - 追加位置并累积 `tokens_in_round`。  
   - 若达到 `sparse_round_window`，执行一轮剪枝：动态区保留到 `max_kv_capacity - sparse_round_window`，为下一轮留空间。  
   - 若任何时刻动态 KV 超过 `max_kv_capacity`，立即强制剪到上限。  
3) **评分与选留**（仅动态区）：  
   - 对候选位置生成 `key_positions`（全局步）。  
   - 对每个采样头：取对应 KV，必要时做 KV-head 映射；反转 RoPE 得到未旋转 K，与统计的 Q 频域均值组合；在几何距离 `offsets` 上计算相位/幅值，聚合成得分（`mean` 或 `max`）。  
   - 先在每头取 top-k 并集，不足则用全局最大得分补齐到目标数；前缀索引固定保留、排在最前。  
4) **缓存切片**：按保留索引裁剪各层 KV 的序列维，更新位置映射；兼容 Transformers 动态缓存。

### 4.3 关键公式与决策（更细）
- 反转 RoPE：`k_unrot = (k_rot / scale) * cos - rotate_half(k_rot / scale) * sin`。把已旋转的 K 恢复到未旋转坐标，以便和 Q 统计对齐。  
- 频域统计含义：校准阶段为每个采样头存两组统计  
  - `q_mean_complex`：未旋转 Q 在复平面上的逐频率平均（实部/虚部）；  
  - `q_abs_mean`：未旋转 Q 的模长均值（逐频率）。  
  在线步骤 `compute_frequency_statistics_from_means`：  
  - 把 `k_unrot` 拆成复数对 `k_complex`；  
  - `q_mean_abs = |q_mean_complex|`；  
  - `relative = q_mean_complex * conj(k_complex)`；  
  - `phi = atan2(Im(relative), Re(relative))`（相位差）；  
  - `amp = q_mean_abs * |k_complex|`（主幅值项）；  
  - `extra = (q_abs_mean - q_mean_abs) * |k_complex|`（补偿项，刻画幅值偏差）。  
- 距离打分（含补偿项）：  
  - 基础距离：`base_delta = round_start - key_idx`；几何距离网格：`delta_grid = base_delta + offsets`。  
  - 相位：`phase = delta_grid * omega + phi`；`freq_scale_sq` 来自 RoPE 缩放。  
  - 每频率得分：`amp * freq_scale_sq * cos(phase)`，再加补偿项 `(extra * freq_scale_sq)`；其中 **extra** = `(q_abs_mean - |q_mean_complex|) * |k_complex|`，用于补偿 “Q 的模长均值” 与 “Q 的相位均值的模长” 之间的差，刻画头在幅值波动上的额外贡献。  
  - 聚合：对距离和频率做 `mean`（默认）或 `max` 得到该头对每个候选键的分数。  
- 选留策略：每头取 top-k 并集 → 若不足目标保留数，用全局 `combined = max(per_head_scores)` 补齐 → 排序 → 前缀永远保留且在前。
  - 未被采样的头：在评分矩阵中不存在，等价于对选留没有正负贡献，只依赖被采样头的分数；如果统计文件缺失所需头则直接报错（保证使用时统计完备）。
  - 某 KV 头未被采样：它的键不会得到专属分数，仍会受到“已采样头”驱动的全局裁剪（即用其他头的得分排序后截断）。这会对与该头强相关的信息更为苛刻，属于“静默压缩”；若希望覆盖所有 KV 头，应在校准时增加采样覆盖或放宽 `head_limit`。
  - 同一 token、同一层的所有 KV head 同步保留/删除：`keep_indices` 是序列位置集合，对该层的 K/V 在所有 head 上统一 `index_select`；不会出现某个 head 留下某个位置而另一个 head 丢弃的情况，前缀位置同样整体保留。

## 5. 校准（统计提取）流程
目的：为特定模型生成“头部频域统计”文件，供 SparseRound 评分使用。

1) **收集 Q/K 轨迹**：离线运行一次模型，保存 `qk.pt`（含每层每头的 Q/K 序列）与 `metadata.json`（含序列长度、模型信息）。输入样本需代表目标分布（如同一数据集的部分题目）。  
2) **头部采样**：读取 Q Tensor 形状 `(layer, head, seq, head_dim)`，按设定的 `sample_count`、`sample_seed` 采样若干 (layer, head)。若已有采样文件则复用，保证不同统计一致性。  
3) **反转 RoPE 与频域统计**：  
   - 依据模型配置重建 RoPE，计算 `cos/sin` 表与 `attention_scaling`。  
   - 对采样头：反转 RoPE 得到未旋转 Q；转换为复数对（实/虚）；求 `q_mean_complex` 与 `q_abs_mean`。  
4) **序列化**：把采样头列表、统计张量、元数据（序列长度、head_dim、模型路径、trace 名称、dtype）写入统计文件（torch.save）。  
5) **使用**：推理时加载统计文件，若指定 `sparse_head_limit` 则截断；评分只用“统计文件里出现的头”。文件缺失某个已列出的头会直接报错；没被采样的头在评分中被视为“缺席”，不影响保留顺序（依赖已有头的分数）。

实质上，校准把“典型分布下的 Q 频域均值”提取出来，推理时用它与在线生成的 K 做相似性/相位对齐，估计键在未来几何距离处的贡献，从而决定保留顺序。

## 6. 压缩相关的优先调参
- `max_kv_capacity`：压缩率/精度主旋钮。  
- `sparse_round_window`（与 `decoding_recent_size`）：轮长；大轮=少剪但每次剪得多。  
- `sparse_score_aggregation`：`mean` 更稳健，`max` 更激进。  
- `sparse_stats_path` / `sparse_head_limit`：换统计文件适配新模型；裁头可加速。  
- `sparse_offset_max_length`：控制关注的最远距离。  
- 采样/贪心：默认贪心；若改为采样需评估一致性。

## 7. 运行与验证要点
- 启动：直接运行封装脚本即可，默认自动 GPU、多 shard、自动合并。  
- 日志：每 shard 独立；出现非零退出需要检查对应日志。  
- 成品：按 KV 容量号的目录下应有 `predictions.jsonl` 与 `metrics.json`；合并结果在 `shard_merged/`。  
- 校准重跑：若更换模型/域，需重新收集 Q/K 轨迹并导出统计文件再替换。
