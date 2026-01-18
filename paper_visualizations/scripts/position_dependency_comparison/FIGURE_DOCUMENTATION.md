# Figure Documentation: Position Dependency Comparison

This document provides comprehensive details for the position dependency comparison figure, which visualizes the relationship between attention patterns and Q/K distribution concentration. This is intended for paper writing.

---

## 1. Overview

**Research Question**: How do attention patterns relate to the concentration of Q/K vectors in frequency space?

**Key Finding**: Attention heads with Relative-Position-Dependent patterns (Local Attention, Attention Sink) exhibit highly concentrated Q/K distributions (R > 0.99), while Relative-Position-Independent heads (Vertical Stripes, Retrieval Head) show dispersed distributions (R ~ 0.7-0.98).

**Visual Message**: The figure demonstrates a clear correspondence:
- **Top axis**: Relative-Position-Dependent → Relative-Position-Independent
- **Bottom axis**: Concentrated → Dispersed

---

## 2. Experimental Setup

### 2.1 Model and Data

| Item | Value |
|------|-------|
| Model | DeepSeek-R1-0528-Qwen3-8B |
| Architecture | 36 layers, 32 attention heads per layer |
| Head dimension | 128 |
| RoPE type | YaRN (with attention scaling) |
| Trace source | Single inference trace (qid0003_trace34) |
| Sequence length | ~10,938 tokens |

### 2.2 Head Selection

Four representative heads were selected to demonstrate the spectrum:

| Category | Layer | Head | Label | R_Q | R_K |
|----------|-------|------|-------|-----|-----|
| Position-Dependent | 6 | 9 | Local Attention | 0.9996 | 0.9988 |
| Position-Dependent | 9 | 20 | Attention Sink | 0.9983 | 0.9975 |
| Position-Independent | 3 | 11 | Vertical Stripes | 0.9795 | 0.9554 |
| Position-Independent | 17 | 25 | Retrieval Head | 0.8207 | 0.7128 |

---

## 3. Core Concepts

### 3.1 Mean Resultant Length (R)

The Mean Resultant Length measures how concentrated complex vectors are around a mean direction:

$$R = \frac{||\mathbb{E}[z]||}{\mathbb{E}[||z||]}$$

Where:
- $z$ are complex-valued vectors (Q or K at a specific frequency)
- $R = 1.0$ means all vectors point in the same direction (perfectly concentrated)
- $R = 0.0$ means vectors are uniformly distributed (completely dispersed)

### 3.2 Dominant Frequency Selection

For each head, we select the dominant frequency using amplitude-based criterion:

$$f_{dom} = \arg\max_f ||\mathbb{E}[q_f]|| \cdot ||\mathbb{E}[k_f]||$$

This selects the frequency where both Q and K have the strongest coherent signal.

### 3.3 Pre-RoPE Vector Recovery

Since we only have access to post-RoPE Q/K vectors, we invert the RoPE transformation:

$$q_{orig} = \frac{q_{rotated}}{scale} \cdot \cos(\theta) - \text{rotate\_half}\left(\frac{q_{rotated}}{scale}\right) \cdot \sin(\theta)$$

### 3.4 Complex Representation

The head dimension (128) is split into 64 frequency pairs:
- Real part: dimensions $[0, d/2)$
- Imaginary part: dimensions $[d/2, d)$

---

## 4. Figure Layout

### 4.1 Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│  Relative-Position-Dependent    ───────→    Relative-Position-Independent  │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│  Local Attention │  Attention Sink │ Vertical Stripes│  Retrieval Head │
│   (Attn Map)    │   (Attn Map)    │   (Attn Map)    │   (Attn Map)    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│   Q/K Scatter   │   Q/K Scatter   │   Q/K Scatter   │   Q/K Scatter   │
│   R_Q=0.9996    │   R_Q=0.9983    │   R_Q=0.9795    │   R_Q=0.8207    │
│   R_K=0.9988    │   R_K=0.9975    │   R_K=0.9554    │   R_K=0.7128    │
├─────────────────┴─────────────────┴─────────────────┴─────────────────┤
│  Concentrated           ───────→              Dispersed               │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Row 0: Attention Maps

**Content**: Pooled attention score heatmaps (max-pooling within patches)

**Parameters**:
- Patch size: 80 tokens
- Colormap: Custom blue gradient (matches Q point color)
- Background: Light gray-purple `(231/250, 231/250, 240/250)`

**Interpretation**:
- **Local Attention**: Strong diagonal pattern (attends to nearby tokens)
- **Attention Sink**: Strong first-column pattern (attends to initial tokens)
- **Vertical Stripes**: Vertical stripe patterns (position-independent)
- **Retrieval Head**: No fixed pattern (content-dependent attention)

### 4.3 Row 1: Q/K Scatter at Dominant Frequency

**Content**: Complex-plane scatter plots of Q and K vectors at the dominant frequency

**Visual Elements**:
- Blue-purple points: Q vectors
- Gray points: K vectors
- Background: Light gray-purple (same as attention map background)
- R values displayed in top-right corner

**Interpretation**:
- **Concentrated (high R)**: Points cluster tightly around a direction
- **Dispersed (low R)**: Points spread across the complex plane

---

## 5. Color Scheme

| Element | Color (RGB normalized) | Description |
|---------|------------------------|-------------|
| Q points | `(85/250, 104/250, 154/250)` | Blue-purple |
| K points | Gray | Standard gray |
| Background | `(231/250, 231/250, 240/250)` | Light gray-purple |
| Attention map peak | `(60/250, 75/250, 120/250)` | Deep blue-purple |
| Attention map low | Same as background | Seamless integration |

---

## 6. Key Numerical Results

| Head | Dominant Freq | R_Q | R_K | Pattern Type |
|------|---------------|-----|-----|--------------|
| L6H9 (Local Attention) | 49 | 0.9996 | 0.9988 | Position-Dependent |
| L9H20 (Attention Sink) | 55 | 0.9983 | 0.9975 | Position-Dependent |
| L3H11 (Vertical Stripes) | 60 | 0.9795 | 0.9554 | Position-Independent |
| L17H25 (Retrieval Head) | 62 | 0.8207 | 0.7128 | Position-Independent |

---

## 7. Suggested Figure Caption

> **Figure X: Relationship between attention patterns and Q/K distribution concentration.**
> Top row shows attention maps for four representative heads, progressing from Relative-Position-Dependent patterns (Local Attention, Attention Sink) to Relative-Position-Independent patterns (Vertical Stripes, Retrieval Head).
> Bottom row shows Q (blue) and K (gray) vector distributions in the complex plane at each head's dominant frequency.
> Position-dependent heads exhibit highly concentrated Q/K distributions (Mean Resultant Length R > 0.99), while position-independent heads show dispersed distributions (R ~ 0.7-0.98).
> This correspondence suggests that Q/K concentration in frequency space is a key indicator of relative-position dependency in attention patterns.

---

## 8. Scripts Reference

| Script | Purpose |
|--------|---------|
| `generate_comparison_figure.py` | **Main script** - Generate the 2x4 comparison figure |
| `visualize_lowret_heads.py` | Visualize multiple low-R heads for candidate selection |
| `compare_sink_heads.py` | Compare attention sink head candidates |

### Running the Analysis

```bash
# Generate the figure with default settings
cd /data/rbg/users/weian/project/rl/dc
python paper_visualizations/scripts/position_dependency_comparison/generate_comparison_figure.py

# With custom colormap (requires pre-registration)
python -c "
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

face_color = (231/250, 231/250, 240/250)
peak_color = (85/250, 104/250, 154/250)
r, g, b = peak_color
fr, fg, fb = face_color

colors = [face_color, ((fr+r)/2, (fg+g)/2, (fb+b)/2), (r, g, b), (r*0.7, g*0.7, b*0.7)]
cmap = LinearSegmentedColormap.from_list('custom_blue', colors)
matplotlib.colormaps.register(cmap=cmap, force=True)

from paper_visualizations.scripts.position_dependency_comparison.generate_comparison_figure import generate_figure
from pathlib import Path
import torch

generate_figure(
    Path('outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34'),
    Path('/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B'),
    torch.device('cuda:0'), 80, 512, 200,
    Path('paper_visualizations/outputs/position_dependency_comparison/fig_position_dependency_comparison.png'),
    attn_cmap='custom_blue')
"
```

---

## 9. Implications for Paper

### What This Figure Demonstrates

1. **Visual Taxonomy**: Clear categorization of attention heads into Position-Dependent vs Independent
2. **Quantitative Metric**: Mean Resultant Length (R) as a measurable indicator of position dependency
3. **Mechanistic Insight**: High Q/K concentration enables position-dependent attention patterns through coherent frequency-domain interactions

### Connections to Method

This analysis supports the claim that:
- Attention heads can be categorized by their reliance on relative position
- Q/K distribution concentration (R) is predictive of attention pattern type
- Position-independent heads (low R) may benefit from different sparse attention strategies than position-dependent heads (high R)

---

## 10. Output Files

| File | Description |
|------|-------------|
| `fig_position_dependency_comparison.png` | Main figure (2 rows × 4 columns) |
| `lowret_heads_top10.png` | Visualization of top-10 low-R heads |
| `sink_head_candidates.png` | Comparison of attention sink candidates |
