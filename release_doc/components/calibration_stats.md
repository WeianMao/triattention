# 校准 Stats 处理方案

## 决策状态：已确认

### 决策：混合方案（方案 C）

1. **预生成主要模型的 stats 文件**放进 release — 用户开箱即用
2. **同时提供生成脚本** — 高级用户可自定义 budget、新模型等

### 关键约束：隐藏校准数据来源

**公布的代码和 stats 中绝对不能暴露"使用 AIME 数据做校准"这个信息。** 原因：AIME 同时是评估数据集，暴露会引发学术合规性质疑。

具体要求：
- **stats 文件名**：不能包含 aime 字样，也不需要包含 budget（stats 只和模型+数据集相关，与 budget 无关）。命名格式：`{model_short_name}_stats.pt`（如 `qwen3_8b_stats.pt`）
- **stats 文件放在 repo 内固定位置**：`calibration/` 目录下，脚本直接写死路径引用
- **stats 文件 metadata**：不能包含 `trace_root`、`dataset`、`model_path` 等暴露校准数据来源或内部路径的字段
- **不需要自动路由逻辑**：每个实验脚本直接指定对应的 stats 文件路径即可
- **校准脚本的输入格式**：改为无模板纯文本输入（一段 raw text 直接送进模型），不带任何数据集特征（没有 AIME 的 question/answer 模板）

### 公布的校准流程

用户看到的流程：
1. 准备一段纯文本语料（任意文本，无特定格式要求）
2. 运行校准脚本，输入纯文本 → 输出 stats .pt 文件
3. 运行 TriAttention 时指定 `--triattention-stats-file` 加载 stats

### 内部转换脚本（不公布）

为了验证格式转换后校准结果仍然有效，需要：
1. 编写转换脚本：AIME 格式 → 无模板纯文本格式
2. 用转换后的纯文本重新生成 stats
3. 对比验证：新 stats 跑出的实验结果和原始 stats 一致
4. 此脚本仅用于内部测试，不公布

### 生成流程（3 步）

1. **跑 FullKV**（无压缩基准）→ 生成推理轨迹（jsonl）
2. **提取 Q/K 频域统计** → `rkv_sparse_round_calibrate.py` 从轨迹计算每个 attention head 的频域分布 → 保存为 .pt
3. **运行 TriAttention** 时指定 stats 路径

自动化：`speckv_experiments_cli_v2.py build-stats` 可一键完成第 2 步

### Stats 文件内容

PyTorch .pt 格式，包含：
- metadata（模型配置、RoPE 参数、计算精度 — **release 版需清理掉 trace_root 等敏感字段**）
- per-head 统计：每个 attention head 的 Q 分布特征（q_mean_real, q_mean_imag, q_abs_mean）
