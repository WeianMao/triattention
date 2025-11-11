# 可以直接照搬的 LazyEviction 实现细节

以下条目都来自 LazyEviction 的公开实现，属于容易忽视但确实能迁移到你现有裁剪器中的工程技巧；每一条后面附了源码位置，方便对应。

## 1. 生成前重置全局统计，避免跨样本污染
- **做法**：在每个 `model.generate` 批次之前，无条件调用 `TempCache.reset()`（`LazyEviction/eval/utils.py:60`）。
- **效果**：上一条样本中积累的 `layer_last_recurrent` / `layer_max_period` 不会影响下一条样本，避免在长序列跑完后出现“缓存状态已经偏向旧样本”的情况。
- **迁移建议**：你的 round pruner 也可在每次推理入口清空与 token 选择有关的运行期统计（比如频域打分的缓存、自适应阈值等），保证不同样本之间没有隐形耦合。

## 2. 注意力权重一律先转 fp32 再回落
- **做法**：q·k 结果永远以 `torch.float32` 做 softmax，然后再 `to(query_states.dtype)`（`LazyEviction/model/kv_utils.py:59-60`、`model/qwen_model.py:70-75`）。
- **效果**：`TempCache` 里的阈值比较和 sigmoid 计算使用的是数值稳定的权重，尤其在 bf16/fp16 下不会因为舍入噪声导致“误判是否超过 alpha”。
- **迁移建议**：即便你的方法最终不依赖注意力，也可以在需要读 softmax 时参考该流程，确保统计信号来自高精度结果。

## 3. 扩容元数据时用 `current_step-1` 作为默认值
- **做法**：当序列拉长，需要在 `layer_last_recurrent`、`layer_max_period` 末尾拼接一列占位符时，LazyEviction 用 `current_decoding_step-1` 和 `0` 作为新 token 的上次命中时间 / 周期（`LazyEviction/model/temp_cacheobs.py:30-35`）。
- **效果**：新 token 初始“上次命中”几乎等于当前步，后续第一次被真正关注时周期计算才会生效，避免刚拼接就被视作“很久未出现”从而触发过度衰减。
- **迁移建议**：若你的方法也需要给新 token 初始化统计量（例如频域估计的先验时间戳），可以采用类似的占位策略，避免初始值导致的虚假长周期。

## 4. 只在观测计数命中倍数时才触发裁剪
- **做法**：`obs_count % obs_size != 0` 时直接返回，不动 KV（`LazyEviction/model/kv_utils.py:68-69`）。
- **效果**：让裁剪发生在规律的“观测窗口”边界，期间你可以积累更多统计证据，并减少频繁的 gather/cat 带来的显存碎片。
- **迁移建议**：在 round-based 框架里也可以引入“延迟触发”的概念，例如只有当 round 内 token 数达到某个倍数时才重新评分与裁剪，稳住缓存抖动。

## 5. 将“不可裁剪尾部”与“可裁剪前段”分别处理并同步元数据
- **做法**：把最后 `obs_size` 个 token 视为必保留区，只在前段执行 top-k；裁完后用相同的 gather/cat 重建 `layer_last_recurrent`、`layer_max_period`（`LazyEviction/model/kv_utils.py:75-99`）。
- **效果**：
  1. 最近 token 永远留存，避免刚生成的 token 还没机会证明价值就被删；
  2. 元数据和 KV 始终同步，后续周期/时间计算不会错位。
- **迁移建议**：引入“尾部保留”后，你也能保证最新一段上下文稳定存在；若要附带其他统计（频谱幅度、相位等），在裁剪 KV 时同步 gather/cat 就不会额外出错。

## 6. 结合 sigmoid 平滑和零除保护，保证重要性分数稳定
- **做法**：
  - 利用 `torch.where` 在 `layer_max_period` 为 0 时直接置 0，其他情况下走 `2 / (1 + exp(...))` 的 Sigmoid（`LazyEviction/model/kv_utils.py:72-74`）。
  - 分母加 `1e-5`，同时把历史周期信息拆成一个独立的“惩罚项” `s`。
- **效果**：重要性分数落在可控区间，不会因为周期为零或极小而爆炸，也不会因为长时间未命中就突然掉到 -inf。
- **迁移建议**：如果你的得分函数里也涉及类似的比值或周期，可以直接复用这一套“Sigmoid + epsilon + where”的写法，提升数值稳定性。

以上技巧都不改变算法的大方向，但能让任何 KV 裁剪策略在工程层面更稳。如果需要把其中某条嵌入到 `SparseRoundPruner` 代码里，我可以继续协助细化。
