# Phase Shift Estimation Without Inverting RoPE

## Context
- Existing workflow (`weian_development/attention_qk_analysis/freq_magnitude_plots.py`) reconstructs phase offsets after undoing RoPE. Suspected numerical drift in the inverse motivates a new approach.
- Target run traces (`qid0003_trace34`) come from the DeepSeek-R1 Qwen3-8B model launched via `scripts/run_perplexity_tensor_parallel.sh`, which dispatches `weian_development/run_perplexity_distributed.py` → `compute_reasoning_perplexity.py`. That loader relies on `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` for Qwen3, so the runtime RotaryEmbedding matches `transformers`’ `Qwen3RotaryEmbedding`.

## Key Observations
- The model applies YaRN-style RoPE scaling (config advertises `rope_scaling` with `rope_type="yarn"` and `attn_factor`). When instantiating `Qwen3RotaryEmbedding`, the class exposes the effective inverse frequencies through `rotary.inv_freq`; this is the same tensor used during forward passes, so we can treat `ω_f = 2π / period_f` derived from `inv_freq` as authoritative despite the YaRN warning.
- Directly inverting RoPE is unnecessary if we work in the complex plane using the already rotated Q/K: the geometric phase `ω_f·Δ` is known analytically, and we just need to estimate the residual phase offset that best aligns reconstruction with ground-truth average dot products.

## Proposed Estimator
1. Interpret each frequency pair as a complex number:
   - `q_f[t] = q_x + i q_y`, `k_f[s] = k_x + i k_y` from the captured tensors (post-RoPE).
2. For every causal pair `(t, s)` where `t ≥ s`, accumulate
   `z_f[t, s] = q_f[t] · conj(k_f[s]) · exp(-i · ω_f · (t - s))`.
   - This removes the deterministic RoPE rotation so `arg(z_f)` approximates the true pre-RoPE phase gap.
   - Weighting comes “for free” because `|z_f| = |Q||K|`, matching the magnitude term used in `Σ_f |Q||K| cos(ω_f Δ + φ_f)`.
3. Efficient accumulation via prefix sums per frequency: maintain running sums of `k_f[s] · exp(i · ω_f · s)` and combine with `q_f[t] · exp(-i · ω_f · t)` to avoid explicit `O(T²)` loops.
4. Define `φ_f = angle( Σ_{t} q_f[t] e^{-i ω_f t} · (Σ_{s ≤ t} conj(k_f[s]) e^{i ω_f s}) )`. This is the phase we will plug into the reconstructed curve.
5. Keep the existing plots for `Σ_f |Q||K| cos(ω_f Δ)` and the FFT-derived ground-truth correlation unchanged for comparability.

## Implementation Milestones

### Single-token sanity checks
- Script `weian_development/attention_qk_analysis/freq_single_token_debug.py` now reconstructs one query’s historical scores using only the RoPE frequencies.
- Workflow: harvest the raw causal dot products, subtract the per-query mean, project onto the RoPE complex basis (`ω_f = inv_freq`), divide by the window length to keep coefficients bounded, and then add the mean back after the inverse projection.
- With the normalization fix, the reconstructed curve stays on the same scale as the raw attention scores; the figure at `outputs/deepseek_r1_qwen3_8b/layer_00_head_00_token_debug.png` verifies this alignment.

### Multi-query aggregation
- Script `weian_development/attention_qk_analysis/freq_magnitude_single_series.py` generalises the same RoPE-frequency projection to all queries.
- For each query we:
  1. Collect its full-length causal score series (default window = full history, configurable via `--fft-window`).
  2. Subtract the query’s mean and compute RoPE-frequency coefficients with the same `1/window` scaling.
  3. Normalise each query’s spectrum by its total energy so every query contributes equally while preserving per-frequency ratios.
- Aggregation simply averages the normalised spectra, rescales by the mean energy, restores the global mean, and produces the reconstructed curve `Σ_f Re(C_f e^{iω_f Δ})`.
- The plot now shows (i) the RoPE-frequency reconstruction, (ii) the true per-distance average obtained directly from scores, and (iii) the FFT-based ground truth from the original pipeline, all drawn on a log-spaced Δ axis so it lines up with `freq_magnitude_plots.py`.

### Usage reminders
- `freq_single_token_debug.py` defaults to CPU execution to avoid GPU memory spikes; adjust `--fft-window` as needed.
- `freq_magnitude_single_series.py` accepts the same `--fft-window`, but leaving it empty uses the full sequence. Both scripts require a valid Qwen3 model path to fetch `inv_freq`.
- Outputs land under `outputs/deepseek_r1_qwen3_8b/` by default to mirror the existing visualisation hierarchy.

These changes let us study phase behaviour directly from RoPE frequencies without ever inverting the rotation, while keeping the visual comparisons against the legacy plots.
