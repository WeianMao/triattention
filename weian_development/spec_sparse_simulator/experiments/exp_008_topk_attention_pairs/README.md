# Experiment 007: Probe Activation Loss Ablation Study

## Experiment Overview

This experiment implements and validates **Phase 1** of the anti-collapse loss design from [exp_007_anti_collapse_losses.md](../exp_007_anti_collapse_losses.md):

- **Loss 1: Probe Activation Loss** - Activates "dead" probes that receive near-zero query routing probability
- **Loss 2: Load Balancing Loss** - *Placeholder only* (interface reserved for Phase 2)

### Objectives

1. Validate the effectiveness of Probe Activation Loss in addressing bin collapse
2. Quantify impact on TopK Hit Rate through ablation study (sweeping `lambda_activation`)
3. Establish baseline for Phase 2 (combined activation + load balancing losses)

## Motivation

Analysis from [exp_006 ANALYSIS_REPORT](../exp_006_module2_reverse_cross_trace_validation/ANALYSIS_REPORT.md) revealed severe **bin collapse** in the Query Network:

| Metric | Key Network | Query Network |
|--------|-------------|---------------|
| Used bins | 125/128 (97.7%) | **3/128 (2.3%)** |
| Effective bins (exp(entropy)) | 106.66 | **1.00** |
| Gini coefficient | 0.32 | **0.99** |

**Critical finding**: 99.96% of queries route to a single bin (Bin 37), while the remaining 125 bins receive no queries. This extreme collapse prevents the model from learning true sparse routing.

The Probe Activation Loss addresses this by forcing "dead" probes to learn from active keys in each training batch.

## Loss Function Specification

### Dead Probe Detection

For N probes (bins), define the death threshold:

$$\tau = \frac{\alpha}{N}$$

where $\alpha$ is the allowed imbalance ratio (default: 0.05).

Given query batch probabilities $\mathbf{P} \in \mathbb{R}^{B \times N}$:

1. Compute batch-averaged probe usage: $\bar{\mathbf{p}} = \frac{1}{B} \sum_{i=1}^{B} \mathbf{P}_{i,:}$
2. Generate dead probe mask: $\mathbf{m}_{\text{dead}} = \mathbb{1}[\bar{\mathbf{p}} < \tau]$

### Activation Loss Computation

For dead probe set $\mathcal{D}$ and positive key set $\mathcal{K}^+$ (argmax keys from current batch):

$$\mathcal{L}_{\text{activation}} = -\frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \frac{1}{|\mathcal{K}^+|} \sum_{k \in \mathcal{K}^+} \log \sigma_{d,k}$$

where $\sigma_{d,k}$ is the Key Network's softmax score for key $k$ in probe $d$.

**Implementation**: Uses `F.log_softmax` for numerical stability instead of `softmax` + `log`.

### Hyperparameters

| Parameter | Config Key | Default | Range | Description |
|-----------|------------|---------|-------|-------------|
| Death threshold ratio | `alpha_dead_threshold` | 0.05 | 0.01-0.2 | Controls death threshold $\tau = \alpha/N$ |
| Activation loss weight | `lambda_activation` | 0.0 | 0.0-1.0 | Weight for activation loss (0.0 = disabled) |
| Balance loss weight | `lambda_balance` | 0.0 | - | *Phase 2 placeholder* |

## Configuration

The `config.yaml` includes anti-collapse hyperparameters:

```yaml
training:
  epochs: 100
  round_window: 128
  query_batch_size: 32
  learning_rate: 0.001

  # Anti-collapse hyperparameters (Phase 1)
  alpha_dead_threshold: 0.05    # Death threshold ratio
  lambda_activation: 0.0        # Activation loss weight (ablation variable)
  lambda_balance: 0.0           # Phase 2 placeholder
```

## Running Experiments

### Train Single Configuration

```bash
# Train with default config (lambda_activation=0.0, baseline)
python run.py --mode train

# Train then evaluate
python run.py --mode all
```

### Run Ablation Study

```bash
# Run ablation with default lambda values [0.0, 0.01, 0.05, 0.1, 0.5]
python run.py --mode ablation

# Run ablation with custom lambda values
python run.py --mode ablation --lambdas 0.0,0.001,0.01,0.05,0.1,0.2,0.5
```

### Generate Visualizations

```bash
# After ablation completes, generate plots and summary
python visualize_ablation.py --results-dir output/ablation
```

### Expected Outputs

| Output | Path | Description |
|--------|------|-------------|
| Checkpoints | `output/ablation/lambda_*/checkpoints/` | Model checkpoints per lambda |
| Results JSON | `output/ablation/ablation_results.json` | Aggregated metrics |
| Loss curves | `output/ablation/loss_curves.png` | Training loss over epochs |
| Metric comparison | `output/ablation/metric_comparison.png` | Hit rate bar charts (K=50/500/1000) |
| Bin utilization | `output/ablation/bin_utilization.png` | Heatmap of bin usage |
| Summary report | `output/ablation/summary_report.txt` | Best config and recommendations |

## Results Analysis

### Key Metrics

- **TopK Hit Rate**: Primary metric; percentage of queries whose argmax key is in TopK
- **Bin Utilization**: Number of bins actively used (receiving queries)
- **Effective Bins**: `exp(entropy)` of query routing distribution

### Interpretation Guide

| Observation | Interpretation |
|-------------|----------------|
| Higher hit rate with λ > 0 | Activation loss improves key ranking in dead probes |
| Increased effective bins | Activation loss reduces collapse |
| Optimal λ in mid-range | Balance between activation and main loss |
| λ = 0 is best | Activation loss may not be beneficial for this task |

### Comparison Criteria

1. **Primary**: K=1000 Hit Rate (most lenient, shows ceiling)
2. **Secondary**: K=50 Hit Rate (practical constraint)
3. **Auxiliary**: Effective bins count, loss convergence speed

## Phase 2 Roadmap

Phase 2 will add **Load Balancing Loss** (see [exp_007_anti_collapse_losses.md](../exp_007_anti_collapse_losses.md) lines 64-93):

$$\mathcal{L}_{\text{balance}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

Current status:
- `lambda_balance` config key exists (default 0.0)
- `compute_load_balancing_loss()` function exists (returns zero tensor)
- Integration point in `train_epoch()` ready for Phase 2

## References

- [exp_007_anti_collapse_losses.md](../exp_007_anti_collapse_losses.md) - Complete loss design specification
- [exp_006 ANALYSIS_REPORT.md](../exp_006_module2_reverse_cross_trace_validation/ANALYSIS_REPORT.md) - Bin collapse evidence and analysis
- [exp_006a README.md](../exp_006a_top2_bin_inference/README.md) - Top-2 bin inference variant
