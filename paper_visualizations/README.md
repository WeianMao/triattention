# Paper Visualizations

论文可视化实验脚本集合。本目录包含用于生成论文图表的所有可视化脚本。

## 目录结构

```
paper_visualizations/
├── README.md              # 本文档
├── scripts/               # 可视化脚本
│   ├── visualize_attention_maps.py
│   ├── freq_magnitude_plots.py
│   └── freq_magnitude_single_plot_meanvec_scatter.py
└── outputs/               # 可视化输出
```

## 数据来源

所有脚本依赖捕获的 Q/K 张量数据，位于：
- `outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qidXXXX_traceYY/`
  - `qk.pt`: 包含 Q 和 K 矩阵的张量文件，形状为 `[layers, heads, seq_len, head_dim]`
  - `metadata.json`: 包含 `sequence_length` 等元信息

---

## 脚本详细说明

### 1. visualize_attention_maps.py

**功能**: 生成注意力权重热图，展示 Query-Key 注意力分布模式。

**原输出位置**: `outputs/deepseek_r1_qwen3_8b/attention_maps_full2/`

#### 可视化方法

1. **注意力计算**:
   - 计算 `scores = Q @ K^T * scale`，其中 `scale = 1/sqrt(head_dim)`
   - 应用因果掩码（causal mask），屏蔽未来位置
   - 对每行进行 softmax 归一化得到注意力权重

2. **池化处理**（处理长序列）:
   - 将序列按 `patch_size`（默认32）分组
   - 对 Key 维度进行最大池化：取每个 key group 内的最大注意力值
   - 对 Query 维度进行最大池化：取每个 query group 内的最大值
   - 最终得到 `[num_q_groups, num_k_groups]` 的热图

3. **归一化**:
   - 对每行进行 min-max 归一化到 [0, 1]

#### 运行示例

```bash
python paper_visualizations/scripts/visualize_attention_maps.py \
    outputs/deepseek_r1_qwen3_8b/qk_bf16_traces \
    --output-root paper_visualizations/outputs/attention_maps \
    --patch-size 32 \
    --max-layers 5 \
    --verbose
```

#### 输出文件

- `layer_XX_head_YY.png`: 第 XX 层第 YY 头的注意力热图
- 热图横轴为 Key group index，纵轴为 Query group index
- 颜色越亮表示注意力权重越高

---

### 2. freq_magnitude_plots.py

**功能**: 生成频率幅值诊断图，分析 RoPE 各频率分量的幅值和相位特性。

**原输出位置**: `outputs/deepseek_r1_qwen3_8b/freq_magnitude_plots/`

#### 可视化方法

1. **RoPE 逆变换**:
   - 从 RoPE 编码后的 Q/K 恢复原始向量：`x = y/scale * cos - rotate_half(y/scale) * sin`
   - 其中 `scale` 是 YaRN 的 attention_scaling 因子

2. **频率分量提取**:
   - 将 head_dim 维向量视为 `head_dim/2` 个复数对 `(real, imag)`
   - 每个复数对对应一个 RoPE 频率分量
   - 计算每个频率的幅值：`|z| = sqrt(real^2 + imag^2)`

3. **统计量计算**:
   - **Avg |K|**: 所有 key token 在各频率的平均幅值
   - **Avg |Q|**: 所有 query token 在各频率的平均幅值
   - **Avg |Q||K|**: 因果遮罩下的 `|Q_i| * |K_j|` 平均值（i >= j）
   - **Mean angle φ_f**: 各频率的平均相位差
   - **Angle variance σ²_f**: 相位差的圆形方差

4. **重建曲线**:
   - **Plain**: `Σ_f |Q||K| cos(ω_f Δ)` - 不考虑相位偏移
   - **Phased**: `Σ_f |Q||K| cos(ω_f Δ + φ_f)` - 考虑平均相位偏移
   - **Ground-truth**: 实际的 Q·K 点积按距离 Δ 的平均值

#### 运行示例

```bash
python paper_visualizations/scripts/freq_magnitude_plots.py \
    outputs/deepseek_r1_qwen3_8b/qk_bf16_traces \
    --output-root paper_visualizations/outputs/freq_magnitude \
    --max-layers 3 \
    --max-heads 8 \
    --verbose
```

#### 输出文件

- `layer_XX_head_YY_freq.png`: 包含 8 个子图的频率诊断图
  - 图1-5: 各频率统计量（横轴为 RoPE 周期，log scale）
  - 图6-7: 重建曲线（横轴为 token 距离 Δ，log scale）
  - 图8: Ground-truth Q·K 平均值

---

### 3. freq_magnitude_single_plot_meanvec_scatter.py

**功能**: 生成 Q/K 向量在高贡献频率上的散点图，展示向量分布特性。

**原输出位置**: `outputs/deepseek_r1_qwen3_8b/vis/layer_*_freq_meanvec_scatter*.png`

#### 可视化方法

1. **频率排序**:
   - 计算 `|E[q_f]| * |E[k_f]|` 作为每个频率的贡献度
   - 选取贡献度最高的 top-k 个频率（默认 k=6）

2. **复平面散点**:
   - 对每个选中频率 f，提取所有 token 的复数值
   - **Centered 模式**（默认）: 绘制 `z - E[z]`，即减去均值后的偏移
   - **Raw 模式**: 绘制原始复数值 z

3. **图表布局**:
   - 每行对应一个频率
   - 左列: Query 向量散点
   - 右列: Key 向量散点
   - 标题显示频率索引 f 和 `|E[q]E[k]|` 值

#### 运行示例

```bash
# Centered scatter (default)
python paper_visualizations/scripts/freq_magnitude_single_plot_meanvec_scatter.py \
    outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34 \
    --layer 1 --head 21 --top-k 3 \
    --output-path paper_visualizations/outputs/scatter_l1_h21_centered.png

# Raw scatter
python paper_visualizations/scripts/freq_magnitude_single_plot_meanvec_scatter.py \
    outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34 \
    --layer 1 --head 21 --top-k 3 --no-center \
    --output-path paper_visualizations/outputs/scatter_l1_h21_raw.png
```

#### 输出文件

- 单个 PNG 文件，包含 `top_k * 2` 个子图
- 每个子图为复平面上的散点图
- 散点分布形状反映该频率分量的向量聚集特性

---

## 依赖环境

```bash
conda activate rkv  # 或 rkv1 for Qwen3
```

依赖包:
- torch
- matplotlib
- transformers (Qwen3RotaryEmbedding)
- numpy

## 批量生成示例

```bash
# 生成所有 trace 的 attention maps
for trace in outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid*; do
    python paper_visualizations/scripts/visualize_attention_maps.py \
        "$trace" \
        --output-root paper_visualizations/outputs/attention_maps/$(basename $trace) \
        --patch-size 32
done
```
