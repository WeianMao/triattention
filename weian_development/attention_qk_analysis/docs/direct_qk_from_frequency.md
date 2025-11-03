# 从频域统计理解 `direct q·k`

> 目标：用尽量少、直观的统计量来描述一个注意力头的行为，最好这些统计量和 `direct q·k` 之间有明确的公式关系。幅值部分（`|Q| * |K|`）已经比较清楚；剩下的难点在于余弦项 `cos(θ_q - θ_k)`，也就是“方向”或“相位”这一块。下面把问题系统整理，并给出更严谨、易计算的统计方案。

---

## 1. 单频段点积到底长什么样？
逆 RoPE 后，每个频段是一对二维向量 `(q_x, q_y)`、`(k_x, k_y)`。写成复数：

- $\tilde{q} = q_x + i\,q_y = |Q|\,e^{i\theta_q}$
- $\tilde{k} = k_x + i\,k_y = |K|\,e^{i\theta_k}$

于是该频段的点积就是：

$$
q \cdot k = \operatorname{Re}\big(\tilde{q}\,\overline{\tilde{k}}\big) = |Q|\,|K|\,\cos(\theta_q - \theta_k).
$$

也就是说，只要知道每个配对 $(i,j)$ 的幅值 $|Q|\,|K|$ 和相位差 $\delta = \theta_q - \theta_k$，就完全确定单频段的贡献；真正的 `direct q·k` 只是把这些贡献在所有频段上求和。

---

## 2. 对所有频段求和就是 `direct q·k`
实际的注意力点积是：

$$
\begin{aligned}
Q_i \cdot K_j
 &= \sum_f q_{i,f} \cdot k_{j,f} \\
 &= \sum_f \operatorname{Re}\big(\tilde{q}_{i,f}\,\overline{\tilde{k}_{j,f}}\big) \\
 &= \sum_f |Q_{i,f}|\,|K_{j,f}|\,\cos(\theta_{q,i,f} - \theta_{k,j,f}).
\end{aligned}
$$

脚本里直接对捕获的 Q/K（RoPE 之后）做点积，得到的就是这条公式。

---

## 3. 为什么“平均相位”会失真？
常见做法是先算

$$
\phi_f = \operatorname{atan2}\big(\mathbb{E}[\sin \delta],\, \mathbb{E}[\cos \delta]\big),
$$

再用 $\mathbb{E}[|Q|\,|K|]\cos(\omega_f\Delta + \phi_f)$ 重构。问题在于这隐含 **幅值与相位差独立** 的假设。现实中：

- 近邻配对幅值大、相位集中；
- 远距离配对幅值小、相位像噪声；
- 简单平均 $\cos \delta$ 会被大量远距配对拉偏，使 $\phi_f$ 接近 $\pm\pi/2$，从而“考虑相位反而更糟”。

因此需要一个同时记录“幅值 × 相位”耦合信息的统计量。

---

## 4. 更严谨的基础量：复相关
最自然的做法是直接统计复相关系数：

$$
C_f = \mathbb{E}\big[\tilde{q}_{i,f}\,\overline{\tilde{k}_{j,f}}\big] = A_f + i B_f,
$$

其中：

- $A_f = \mathbb{E}\big[|Q|\,|K|\,\cos \delta\big]$
- $B_f = \mathbb{E}\big[|Q|\,|K|\,\sin \delta\big]$

好处：

1. 没有额外假设；
2. $\operatorname{Re}(C_f)$ 就是频段对 `direct q·k` 的平均贡献，$\operatorname{Im}(C_f)$ 表明相位滞后，$|C_f|$ 是贡献强度，$\arg C_f$ 是平均相位差；
3. 可直接用于重构注意力曲线：

$$
\text{Recon}(\Delta) = \sum_f \Big( \operatorname{Re}(C_f)\cos(\omega_f \Delta) - \operatorname{Im}(C_f)\sin(\omega_f \Delta) \Big).
$$

这一公式与真实的平均点积严格等价。

---

## 5. 如何刻画“近邻 vs 远距”？
RoPE 把距离 $\Delta = i-j$ 编码成 $\cos(\omega_f \Delta)$、$\sin(\omega_f \Delta)$；要区分近邻/远距，可按 Δ 分桶：

$$
C_f(\Delta) = \mathbb{E}\big[\tilde{q}_{i,f}\,\overline{\tilde{k}_{j,f}} \mid i-j = \Delta\big].
$$

每个桶就是“在这个 Δ 范围内，该频段的平均贡献”。如果想得到单一指标，再按权重 $p(\Delta)$ 求平均：

$$
C_f = \sum_{\Delta} p(\Delta)\, C_f(\Delta).
$$

权重可以取配对数量、真实注意力权重 $\mathbb{E}[\text{softmax}(QK)]$，或用户关注的距离窗口（如 $\Delta \le 32$）。这样可以避免近邻和远距的信号互相抵消。

---

## 6. 推荐记录的统计量
1. **幅值**：`|Q|*|K|` 的分布或平均值，反映最大可能贡献。
2. **复相关**：`C_f`（必要时 `C_f(\Delta)`），是真正对应 `direct q·k` 的量。
3. **一致性**：`|C_f| / \mathbb{E}[|Q|\,|K|]` 粗略衡量相位集中程度。
4. **距离行为**：`C_f(\Delta)` 随 Δ 的曲线，展示近邻/远距的差异。

这些量都能直接映射到 `direct q·k` 的组成部分。

---

## 7. 与 `direct q·k` 的核心公式
\[
\begin{aligned}
Q_i \cdot K_j &= \sum_f \operatorname{Re}(\tilde{q}_{i,f}\,\overline{\tilde{k}_{j,f}}), \\
\mathbb{E}[Q \cdot K] &= \sum_f \operatorname{Re}(C_f), \\
\text{Recon}(\Delta) &= \sum_f \Big( \operatorname{Re}(C_f)\cos(\omega_f \Delta) - \operatorname{Im}(C_f)\sin(\omega_f \Delta) \Big).
\end{aligned}
\]

因此，只要手里掌握 `C_f`（或 `C_f(\Delta)`），就能描述平均注意力、重构距离曲线，并构建简明的统计图表。

---

## 8. 实际操作流程建议
1. **逆 RoPE**：把捕获的 Q/K 逆旋得到 $\tilde{q}, \tilde{k}$。对于 Yarn RoPE，需要先除以 `attention_scaling` 再执行逆旋，以免幅值被放大。
2. **选择权重**：原始配对、softmax 注意力、只看近邻等。
3. **计算复相关**：
   - 全局：`C_f = average(tilde_q * conjugate(tilde_k))`
   - 分桶：`C_f(Δ) = average(...)` 在各个 Δ 桶里
4. **展示统计图**：幅值、`|C_f|`、`arg(C_f)`、`C_f(Δ)` 等。
5. **重构验证**：用 `Recon(Δ)` 与真实曲线对比，确认统计量描述得是否准确。

---

## 9. 小结
- 单频段点积 = 幅值 × 余弦 ⇒ 把两个因素一起统计最自然的方式就是用复相关 `C_f`。
- `C_f` 同时记录幅值和相位差，且能直接用于重构注意力曲线，是最“简单但严谨”的频域统计量。
- 若想强调不同距离的行为，可统计 `C_f(Δ)`；这样可以避免近邻与远距信号互相抵消。
- 将 `|Q|*|K|` 与 `C_f`（必要时带 Δ 桶）封装成图表，就能用少量指标清晰描述一个注意力头的行为。
