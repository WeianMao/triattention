# Position Dependency Comparison Visualization Requirements

## Task Overview

Create a comparative visualization showing the difference between Relative-Position-Dependent and Relative-Position-Independent attention heads across two dimensions: attention maps and dominant frequency Q/K scatter plots.

## Dimensions

### Dimension 1: Attention Head Types

| Type | Term | Head IDs | Characteristics |
|------|------|----------|-----------------|
| Relative-Position-Dependent | Relative-Position-Dependent Heads | layer_06_head_09, layer_13_head_12 | local attention, attention sink |
| Relative-Position-Independent | Relative-Position-Independent Heads | layer_05_head_10, layer_15_head_11 | vertical stripes, no fixed pattern |

### Dimension 2: Visualization Types

1. **Attention Map**: Pooled attention heatmap
   - Requirement: Increase pooling size to reduce resolution (current patch_size=32 may be too fine)

2. **Dominant Frequency Q/K Scatter**: 2D scatter plot of Q/K vectors at the dominant frequency
   - Requirement: NO centering (raw values, not mean-subtracted)
   - Requirement: Only show top-1 dominant frequency
   - Dominant frequency selection: Based on **variance** of actual attention scores, NOT simulation/amplitude

## Key Technical Requirements

### 1. Dominant Frequency Selection (Variance-based)

Instead of using amplitude product `|E[q]| * |E[k]|` to select dominant frequency, use **variance of actual attention scores** at each frequency:

```
For each frequency f:
    Compute actual attention contribution at f across all (i,j) pairs
    dominant_freq = argmax(variance of attention scores across f)
```

This measures which frequency has the **largest variation** in attention pattern, not just the largest absolute value.

### 2. Attention Map Resolution

Current pooling (patch_size=32) is too fine-grained. Increase pooling size to make patterns more visible:
- Suggestion: patch_size=64 or patch_size=128

### 3. Scatter Plot Style

- Use `--no-center` mode (raw values, not delta from mean)
- Only show top-1 frequency (not top-3 or top-6)
- Reference style: `outputs/deepseek_r1_qwen3_8b/vis/layer_01_head_21_freq_meanvec_scatter_raw_top3.png`

## Figure Layout

Expected output: 2x4 or 4x2 grid comparing:

```
                    | Attention Map | Dominant Freq Scatter |
--------------------|---------------|----------------------|
layer_06_head_09    |     (A1)      |        (B1)          | (Dependent - local)
layer_13_head_12    |     (A2)      |        (B2)          | (Dependent - sink)
layer_05_head_10    |     (A3)      |        (B3)          | (Independent - stripes)
layer_15_head_11    |     (A4)      |        (B4)          | (Independent - no pattern)
```

## Reference Scripts

1. **Attention Map**: `paper_visualizations/scripts/visualize_attention_maps.py`
   - Key function: `compute_attention_heatmap_block()` with `patch_size` parameter

2. **Scatter Plot**: `paper_visualizations/scripts/freq_magnitude_single_plot_meanvec_scatter.py`
   - Key flag: `--no-center` for raw (non-centered) visualization
   - Need to modify top-k selection to use variance-based method

## Styling Requirements (from previous session)

```python
# Colors
color_gt = (187 / 250, 130 / 250, 90 / 250)      # warm brown/orange
color_recon = (85 / 250, 104 / 250, 154 / 250)   # blue
color_bar = (187 / 250, 130 / 250, 90 / 250)
face_color = (231 / 250, 231 / 250, 240 / 250)   # light gray-purple background

# Unified font size
FONT_SIZE = 14

# Style function
def style_ax(ax):
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
    ax.set_axisbelow(True)  # Grid behind data
    ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)
```

## Data Source

- Trace directory: `outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34/`
- Contains: `qk.pt` (Q/K tensors) and `metadata.json`
- Model: DeepSeek-R1-0528-Qwen3-8B (36 layers, 32 heads)
