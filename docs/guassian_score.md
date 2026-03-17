Attention RoPE 频段级 Top-p Quantile 打分算法说明（Method 1 Only）

本文档描述一个用于分析 Transformer Attention 中 RoPE（Rotary Position Embedding）频域结构的新打分算法。
目标是：给定一个 key token（位置为 τ），预测该 key 在未来相对位置 Δ 上的注意力强度，无需真实 Q，只依赖历史 Q 的统计分布。

该算法基于以下思想：
	•	一个频段 f 的 query 向量 q_f = (x, y) 可近似为二维高斯分布；
	•	在 RoPE 下，key 向量被旋转，因此 dot-product d_f(\Delta;\tau) 也是高斯分布；
	•	使用该高斯分布右尾概率质量为 p 的最小值（top-p quantile）作为频段打分；
	•	最终 head 的打分是所有频段打分的和。

本文档将精确说明需要计算的统计量、公式以及代码应如何实现。

⸻

1. 输入数据结构

对某个 attention head：
	•	对每个 RoPE 频段 f，我们有大量 query 样本：
q_f = (x, y)^\top \in \mathbb R^2
	•	对于要打分的某个 key token τ，我们能获得其在频段 f 上的向量：
k_{\tau,f} = (k_x, k_y)^\top
	•	RoPE 在相对位置 Δ 上提供一个旋转：
R_f(\Delta) =
\begin{pmatrix}
\cos(\omega_f \Delta) & -\sin(\omega_f \Delta) \\
\sin(\omega_f \Delta) & \cos(\omega_f \Delta)
\end{pmatrix}
	•	超参数：
p \in (0,1)
\quad\text{例如 } p = 0.1

⸻

2. 需要从 query 样本估计的统计量

对每个频段 f，用所有历史 query 统计：

2.1 均值

\mu_f = \mathbb E[q_f] =
\begin{pmatrix}
\mu_{x,f} \\
\mu_{y,f}
\end{pmatrix}

2.2 协方差矩阵（2×2）

\Sigma_f = \text{Cov}(q_f) =
\begin{pmatrix}
\sigma_{xx,f} & \sigma_{xy,f}\\
\sigma_{xy,f} & \sigma_{yy,f}
\end{pmatrix}

这三项都由真实 q 样本计算（简单均值与协方差即可）。

⸻

3. 旋转 key 得到位置 Δ 的向量

k_{\tau,f}(\Delta) = R_f(\Delta)\,k_{\tau,f}

记旋转后的结果为：

k_{\tau,f}(\Delta) =
\begin{pmatrix}
u_{f} \\
v_{f}
\end{pmatrix}

⸻

4. 定义 dot-product 随机变量的均值与方差

本算法基于假设：

d_f(\Delta;\tau) := q_f^\top k_{\tau,f}(\Delta)

由于 q_f\sim\mathcal N(\mu_f, \Sigma_f)，
可知 d_f 是一维高斯：

d_f(\Delta;\tau) \sim \mathcal N\big(m_f(\Delta;\tau), v_f(\Delta;\tau)\big)

其中：

4.1 均值

m_f(\Delta;\tau) = \mu_f^\top k_{\tau,f}(\Delta)
= \mu_{x,f} u_f + \mu_{y,f} v_f

4.2 方差

v_f(\Delta;\tau)
= k_{\tau,f}(\Delta)^\top \Sigma_f\,k_{\tau,f}(\Delta)
展开后为：

v_f(\Delta;\tau)
= \sigma_{xx,f} u_f^2 + 2\sigma_{xy,f} u_f v_f + \sigma_{yy,f} v_f^2

⸻

5. Top-p Quantile 打分（方法 1）

定义右尾概率为 p 的 quantile（阈值）：

\alpha_p = \Phi^{-1}(1 - p)

其中 \Phi^{-1} 是标准正态分布的反 CDF。

对于频段 f，
top-p quantile 是：

t_{f,p}(\Delta;\tau)
= m_f(\Delta;\tau)
	•	\sqrt{v_f(\Delta;\tau)}\;\alpha_p

这是本频段的最终 score。

⸻

6. Head 级总打分（频段逐个算，再求和）

本算法选择频段独立计算 + 求和（而非跨频段合并）。

最终 head 级别的打分为：

\boxed{
\text{HeadScore}(\Delta;\tau)
= \sum_f
\Big[
m_f(\Delta;\tau)
	•	\sqrt{v_f(\Delta;\tau)}\;\alpha_p
\Big]
}

其中：
	•	m_f(\Delta;\tau) 来自 query 均值；
	•	\sqrt{v_f(\Delta;\tau)} 来自 query 协方差；
	•	\alpha_p = \Phi^{-1}(1-p) 是常数。

⸻

7. 实现步骤总结（代码该做的事）

Step 1：预处理（一次性）

对每个频段 f：
	•	从所有 q 样本计算
	•	\mu_f（均值，shape = [2]）
	•	\Sigma_f（协方差，shape = [2,2]）

Step 2：对每个 key token τ、每个相对位置 Δ

循环所有频段 f：
	1.	旋转 key：
k_{\tau,f}(\Delta) = R_f(\Delta)k_{\tau,f}
	2.	计算均值项：
m_f = \mu_f^\top k_{\tau,f}(\Delta)
	3.	计算方差：
v_f = k_{\tau,f}(\Delta)^\top \Sigma_f k_{\tau,f}(\Delta)
	4.	计算频段 score：
t_{f,p} = m_f + \sqrt{v_f}\,\alpha_p

Step 3：累加得到 head score

\text{HeadScore}(\Delta;\tau) = \sum_f t_{f,p}

9. 这个算法解决的问题

该方法自然融合两种情况：
	•	Query 在该频段有明显相位方向 ⇒ 均值项 m_f 主导
	•	Query 分布近似绕原点、方向混乱 ⇒ 方差项 \sqrt{v_f} 主导
	•	不需要手工 gate，不需要 heuristic 加权，两者通过几何结构自动平衡

⸻

附：实验记录与复现方法

环境与数据
- trace：primary `outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34`，统计 trace `outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0008_trace46`
- 设备：GPU `cuda:0`
- 默认 round 参数：`--max-keys 2048 --round-window 64 --score-aggregation mean --offset-max-length 65536`
- 采样头：默认 `weian_development/online_k_pruning_viz/hybrid_sample_heads.json`（自动生成 100 头）；额外低命中筛选文件 `weian_development/online_k_pruning_viz/hybrid_sample_heads_lowret_top10.json`

主要脚本
- 原始混合评分：`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`
- 高斯评分：`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_gaussian.py`（新增 `--quantile-p`，默认 0.1）

基本命令示例（GPU）
```bash
# 原始算法，默认 100 头采样
python weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py \
  outputs/deepseek_r1_qwen3_8b/qk_bf16_traces \
  --trace qid0003_trace34 \
  --stats-trace outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0008_trace46 \
  --device cuda:0 \
  --output-root outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_xtrace_default_heads_gpu

# 高斯算法，示例 p=0.90
python weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_gaussian.py \
  outputs/deepseek_r1_qwen3_8b/qk_bf16_traces \
  --trace qid0003_trace34 \
  --stats-trace outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0008_trace46 \
  --device cuda:0 \
  --quantile-p 0.90 \
  --output-root outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_gaussian_default_heads_p0.90
```

量化扫描结果（默认 100 头采样，GPU）
- 原算法整体命中率：0.9902（`attention_pruning_case_studies_xtrace_default_heads_gpu/.../retention_metrics.json`）
- 高斯算法整体命中率（按 `p`）：
  - 0.85 → 0.9117
  - 0.90 → 0.9283
  - 更高 `p`（0.95/0.99）出现回落（≈0.99→0.987），最佳区间约 0.85–0.90
- 低命中头集中在文件 `weian_development/online_k_pruning_viz/hybrid_sample_heads_lowret_top10.json`，来自 `p=0.90` 结果中最差的 10 个 (layer, head)，可用来做针对性调参。

对比（小样本 8 头，GPU，同一采样文件 `hybrid_sample_heads_gaussian_compare.json`）
- 原算法整体命中率：0.9989
- 高斯算法：`p=0.85` 0.9942，`p=0.90` 0.9933，`p≤0.10` 时显著下降（<0.70）

输出位置说明
- 原算法：`outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_xtrace_*`
- 高斯算法：`outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_gaussian_*`
- 每个 run 下的 `retention_metrics.json` 记录 overall/per-head 命中率，对应热图、argmax 可视化同目录保存。
