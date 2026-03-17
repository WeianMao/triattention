Attention RoPE 频段合并式 Top-p Quantile 打分算法说明（Method 5.2）

本文档描述一种用于分析 Transformer Attention 中 RoPE（Rotary Position Embedding）频域结构的预测打分方法。
本方法与 5.1 的区别是：

先对所有频段的 dot-product（在高斯近似下）累加成一个整体高斯分布，再一次性对这个整体高斯做 top-p quantile。

本文件给出的公式、步骤足够精确，使另一个 agent 能直接据此实现代码。

⸻

1. 输入结构

对某个 attention head：
	•	F 个 RoPE 频段，每段的 query 向量为
q_f = (x, y)^\top \in \mathbb R^2
	•	对 key token τ，在频段 f 的 key 向量为
k_{\tau,f} = (k_x, k_y)^\top
	•	RoPE 旋转矩阵（角频率为 \omega_f）：
R_f(\Delta) =
\begin{pmatrix}
\cos(\omega_f \Delta) & -\sin(\omega_f \Delta) \\
\sin(\omega_f \Delta) & \cos(\omega_f \Delta)
\end{pmatrix}
	•	超参数：右尾概率质量
p \in (0,1)

⸻

2. 预处理：统计所有频段的 Query 分布

对每个频段 f：

2.1 均值

\mu_f = \mathbb E[q_f] =
\begin{pmatrix}
\mu_{x,f} \\
\mu_{y,f}
\end{pmatrix}

2.2 协方差

\Sigma_f = \text{Cov}(q_f) =
\begin{pmatrix}
\sigma_{xx,f} & \sigma_{xy,f} \\
\sigma_{xy,f} & \sigma_{yy,f}
\end{pmatrix}

⸻

3. 构造 dot-product 随机变量（频段级）

对于 key token τ、相对位置 Δ：

3.1 RoPE 旋转 key

k_{\tau,f}(\Delta) = R_f(\Delta)\,k_{\tau,f}

记旋转后的向量
k_{\tau,f}(\Delta) = (u_f, v_f)^\top

3.2 频段级 dot-product

d_f(\Delta;\tau) := q_f^\top k_{\tau,f}(\Delta)

由于 q_f\sim \mathcal N(\mu_f,\Sigma_f)，有：

d_f(\Delta;\tau) \sim \mathcal N\big(m_f(\Delta;\tau),\; v_f(\Delta;\tau)\big)

其中：

均值：

m_f(\Delta;\tau) = \mu_f^\top k_{\tau,f}(\Delta)
= \mu_{x,f} u_f + \mu_{y,f} v_f

方差：

v_f(\Delta;\tau)
= k_{\tau,f}(\Delta)^\top \Sigma_f\, k_{\tau,f}(\Delta)
展开：
v_f(\Delta;\tau)
= \sigma_{xx,f} u_f^2 + 2\sigma_{xy,f} u_f v_f + \sigma_{yy,f} v_f^2

⸻

4. 关键步骤：合并所有频段成为一个总的 dot-product

我们假设各频段之间统计独立（用于高斯加和近似）。

定义总 dot-product 随机变量：

D(\Delta;\tau) := \sum_{f=1}^F d_f(\Delta;\tau)

由于高斯相加仍然是高斯，

D(\Delta;\tau) \sim \mathcal N\big(
M(\Delta;\tau),\; V(\Delta;\tau)
\big)

其中

4.1 组合均值

\boxed{
M(\Delta;\tau)
= \sum_{f=1}^F m_f(\Delta;\tau)
}

4.2 组合方差

\boxed{
V(\Delta;\tau)
= \sum_{f=1}^F v_f(\Delta;\tau)
}

这两个值完全决定后续的 top-p quantile。

⸻

5. 对总分布做一次 Top-p Quantile

5.1 定义 right-tail quantile 系数

\alpha_p = \Phi^{-1}(1-p)
其中 \Phi^{-1} 是标准正态反 CDF。

5.2 总分布的 top-p quantile（方法仅此一种）

对总随机变量 D(\Delta;\tau)\sim\mathcal N(M,V)，
右尾概率质量为 p 的最小阈值为：

\boxed{
T_{p}(\Delta;\tau)
= M(\Delta;\tau)
	•	\sqrt{V(\Delta;\tau)}\;\alpha_p
}

该值即为该 key 在相对位置 Δ 的最终预测 score。

⸻

6. 最终 Head 打分公式

\boxed{
\text{HeadScore}(\Delta;\tau)
= T_{p}(\Delta;\tau)
= M(\Delta;\tau) + \sqrt{V(\Delta;\tau)}\;\alpha_p
}

其中：
	•	M(\Delta;\tau) = \sum_f m_f(\Delta;\tau)
	•	V(\Delta;\tau) = \sum_f v_f(\Delta;\tau)
	•	\alpha_p = \Phi^{-1}(1-p)

8. 方法 5.2 与方法 5.1 的区别（便于 agent 理解）
	•	方法 5.1：
	•	每个频段先取 top-p quantile
	•	然后 把所有频段的 quantile 相加
	•	方法 5.2（本文件）：
	•	所有频段的分布先合成一个高斯
	•	再对这个合成高斯 一次性取 top-p quantile

方法 5.2 通常会出现：
	•	方差较大的频段主导整体行为；
	•	各频段的“峰值方向”在合并过程中可能部分抵消；
	•	但整体 score 更稳定、更“全局化”。

⸻

实验记录（Method 5.2 vs. 5.1/xtrace，低命中头 Top-10）

- 脚本：`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_gaussian_combined.py`（Method 5.2，总分布一次取 quantile）
- 对比：`attention_pruning_case_study_hybrid_rounds_gaussian.py`（Method 5.1，频段逐个 quantile）与 `attention_pruning_case_study_hybrid_rounds_xtrace.py`
- 数据：`outputs/deepseek_r1_qwen3_8b/qk_bf16_traces`，主 trace `qid0003_trace34`，统计 trace `qid0008_trace46`
- 采样头：`weian_development/online_k_pruning_viz/hybrid_sample_heads_lowret_top10.json`
- 主要参数：`--max-keys 2048 --round-window 64 --score-aggregation mean --offset-max-length 65536 --quantile-p 0.1 --device cuda:0`

运行命令示例（GPU）
```bash
conda run -n dc python weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_gaussian_combined.py \
  outputs/deepseek_r1_qwen3_8b/qk_bf16_traces \
  --trace qid0003_trace34 \
  --stats-trace outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0008_trace46 \
  --head-sample-file weian_development/online_k_pruning_viz/hybrid_sample_heads_lowret_top10.json \
  --device cuda:0 \
  --output-root outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_gaussian_combined_lowret \
  --quantile-p 0.1
```

结果汇总（overall retention，10 heads）
- Method 5.2（combined 一次取 quantile）：0.4285（`outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_gaussian_combined_lowret/qid0003_trace34/agg_mean_max2048_w64/retention_metrics.json`）
	- Method 5.1（频段逐个 quantile）：0.4202（`outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_gaussian_lowret/qid0003_trace34/agg_mean_max2048_w64/retention_metrics.json`）
	- xtrace baseline：0.9647（`outputs/deepseek_r1_qwen3_8b/attention_pruning_case_studies_xtrace_lowret/qid0003_trace34/agg_mean_max2048_w64/retention_metrics.json`）

	补充观察
	- 合成高斯方案相对 5.1 略升（+0.0083），但仍显著低于 xtrace（差距 ~0.54），说明一次性 quantile 仍不足以弥补频段信号缺失。
	- 最低头主要出现在层 9、33、34，Method 5.2 在部分缓存均值主导的头（如 layer 24/11、24/14、17/25）有小幅提升，波动性仍大。

Quantile 扫描（低命中头 Top-10，GPU）
- 说明：`--quantile-p` 定义为右尾概率质量（p 越大，alpha 趋向负值，阈值越低，普遍更保守）；本文记录 Method 5.2 与 Method 5.1 在相同采样头上的表现。
- 运行参数同上，仅调整 `--quantile-p`，输出根目录按 `..._pXXX` 区分。
- Method 5.2（combined quantile）overall retention：
  - p=0.05 → 0.4314（`.../attention_pruning_case_studies_gaussian_combined_lowret_p005/.../retention_metrics.json`）
  - p=0.10 → 0.4285（`..._gaussian_combined_lowret/.../retention_metrics.json`）
  - p=0.20 → 0.4274（`..._gaussian_combined_lowret_p02/.../retention_metrics.json`）
  - p=0.50 → 0.4243（`..._gaussian_combined_lowret_p05/.../retention_metrics.json`）
  - p=0.85 → 0.4209（`..._gaussian_combined_lowret_p085/.../retention_metrics.json`）
  - p=0.90 → 0.4205（`..._gaussian_combined_lowret_p09/.../retention_metrics.json`）
  - p=0.99 → 0.4176（`..._gaussian_combined_lowret_p099/.../retention_metrics.json`）
  - 最佳值出现在 p≈0.05，其余 p 单调下降。
- Method 5.1（per-frequency quantile）overall retention：
  - p=0.05 → 0.4147（`..._gaussian_lowret_p005/.../retention_metrics.json`）
  - p=0.10 → 0.4202（`..._gaussian_lowret/.../retention_metrics.json`）
  - p=0.85 → 0.4717（`..._gaussian_lowret_p085/.../retention_metrics.json`）
  - p=0.90 → 0.4963（`..._gaussian_lowret_p09/.../retention_metrics.json`）
  - p=0.99 → 0.9166（`..._gaussian_lowret_p099/.../retention_metrics.json`）
  - 与 Method 5.2 不同，Method 5.1 随着 p 增大会逐步逼近 xtrace（p=0.99 接近全保留，命中率≈0.9166；仍低于 xtrace 的 0.9647）。
