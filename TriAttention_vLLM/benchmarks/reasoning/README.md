# TriAttention vLLM Reasoning Benchmark

**Status**: ✅ Ready for Testing | 📅 Last Updated: 2026-02-01

**Quick Start**: See [QUICK_START.md](./QUICK_START.md)

This directory contains benchmark scripts for evaluating TriAttention KV compression on reasoning tasks using vLLM as the backend. The framework is designed for fair comparison with the HuggingFace SpeckV implementation.

## Overview

**Purpose**: Compare accuracy and performance between HuggingFace SpeckV and vLLM TriAttention implementations on mathematical reasoning tasks (AIME dataset).

**Key Features**:
- Three pruning modes: `per_head`, `per_layer_per_head`, `per_layer`
- Identical hyperparameters to HuggingFace baseline for fair comparison
- Token-level comparison tools for equivalence verification
- Support for R-KV style compression and statistics

## Documentation Index

### For New Users
- **[QUICK_START.md](./QUICK_START.md)** - TL;DR commands and quick reference
- **[READY_TO_RUN.md](./READY_TO_RUN.md)** - Detailed setup and usage guide

### For Developers
- **[NEXT_STEPS.md](./NEXT_STEPS.md)** - Action items and troubleshooting
- **[VERIFICATION_STATUS.md](./VERIFICATION_STATUS.md)** - Current verification status
- **[PARAMETER_ALIGNMENT.md](./PARAMETER_ALIGNMENT.md)** - HF vs vLLM parameter comparison

### Development History
- **[AGENT3_SUMMARY.md](./AGENT3_SUMMARY.md)** - Latest verification work

## Directory Structure

```
benchmarks/reasoning/
├── README.md                                # This file
├── QUICK_START.md                           # Quick reference guide
├── READY_TO_RUN.md                          # Detailed usage guide
├── NEXT_STEPS.md                            # Developer action items
├── VERIFICATION_STATUS.md                   # Current verification status
├── PARAMETER_ALIGNMENT.md                   # HF vs vLLM comparison
├── AGENT3_SUMMARY.md                        # Agent #3 work summary
├── run_math_vllm.py                         # Main vLLM benchmark script
├── run_aime24_vllm.sh                       # Production benchmark script
├── test_minimal.sh                          # Quick test script
├── verify_compression.py                    # Compression verification
├── check_output_format.py                   # Format validation
├── compare_results.py                       # HuggingFace vs vLLM comparison tool
├── run_verification_suite.sh                # Complete verification suite
├── run_triattention_aime24_perhead.sh       # Per-head pruning benchmark
├── run_triattention_aime24_layer_perhead.sh # Per-layer-per-head pruning benchmark
└── run_triattention_aime24_perlayer.sh      # Per-layer pruning benchmark
```

## Prerequisites

### Environment Setup

```bash
# Activate the trivllm conda environment
conda activate trivllm

# Ensure TriAttention module is in PYTHONPATH (automatically set by shell scripts)
export PYTHONPATH="/data/rbg/users/weian/project/rl/dc/TriAttention_vLLM:$PYTHONPATH"
```

### Required Files

1. **Model**: DeepSeek-R1-Distill-Qwen-7B
   - Default: `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B`
   - Override with: `MODEL_PATH` environment variable

2. **Dataset**: AIME24 (American Invitational Mathematics Examination)
   - Default: `R-KV/HuggingFace/data/aime24.jsonl`
   - Format: `{"id": int, "question": str, "answer": str, ...}`

3. **Statistics File**: Precomputed attention statistics
   - Default: `R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt`
   - Used for R-KV style compression alignment

## Usage

### Running Benchmarks

**Per-Head Pruning** (each head selects its own tokens independently):
```bash
cd TriAttention_vLLM/benchmarks/reasoning
./run_triattention_aime24_perhead.sh
```

**Per-Layer-Per-Head Pruning** (each (layer, KV head) pair selects tokens independently):
```bash
./run_triattention_aime24_layer_perhead.sh
```

**Per-Layer Pruning** (all heads in a layer share the same token selection):
```bash
./run_triattention_aime24_perlayer.sh
```

**Output Locations**:
- Results: `TriAttention_vLLM/outputs/aime24/{perhead,layer_perhead,perlayer}/results.jsonl`
- Logs: `TriAttention_vLLM/logs/aime24/{perhead,layer_perhead,perlayer}/run_*.log`

### Comparing Results

After running both HuggingFace and vLLM benchmarks:

```bash
python compare_results.py \
  --hf-results /path/to/hf_results.jsonl \
  --vllm-results TriAttention_vLLM/outputs/aime24/perhead/results.jsonl \
  --output-report comparison_report.txt \
  --detailed
```

The comparison tool will:
- Calculate accuracy for both implementations
- Perform token-level comparison for identical inputs
- Generate statistical summary of differences
- Identify any discrepancies in output quality

### Custom Configuration

Override default parameters using environment variables or command-line arguments:

```bash
# Override model path
MODEL_PATH=/custom/model/path ./run_triattention_aime24_perhead.sh

# Override KV budget and other parameters
./run_triattention_aime24_perhead.sh --kv-budget 4096 --num-samples 16
```

## Key Parameters

### Compression Parameters
- `--kv-budget`: Maximum KV cache tokens to retain (default: 2048)
- `--divide-length`: Compression trigger interval (default: 128)
- `--sparse-round-window`: Sparse round window size (default: 32)
- `--pruning-mode`: Token selection strategy (`per_head`, `per_layer`, `per_layer_per_head`)

### Generation Parameters
- `--num-samples`: Number of samples per question (default: 8)
- `--temperature`: Sampling temperature (default: 0.6)
- `--top-p`: Nucleus sampling parameter (default: 0.95)
- `--seed`: Random seed (default: 888)

### Alignment Flags
- `--sparse-normalize-scores`: Enable score normalization (HuggingFace alignment)
- `--include-prefill-in-budget`: Include prefill tokens in budget
- `--rkv-style-compression`: Use R-KV style compression mode
- `--rkv-style-slack-trigger`: Use R-KV style slack trigger

## Reference: HuggingFace Baseline Scripts

The vLLM benchmarks mirror the following HuggingFace scripts for fair comparison:

1. **Per-Head**: `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`
2. **Per-Layer-Per-Head**: `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh`
3. **Per-Layer**: `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perlayer.sh`

## Implementation Status

### ✅ Completed (2026-02-01)
- [x] Benchmark framework structure
- [x] Shell scripts for 3 pruning modes
- [x] Result comparison tool (supports both HF and vLLM formats)
- [x] vLLM integration complete
- [x] TriAttention wrapper functional
- [x] Parameter alignment with HF baseline
- [x] Output format compatibility verified
- [x] Verification tools created
- [x] Comprehensive documentation

### ⏳ Needs Testing
- [ ] Run compression verification (`verify_compression.py`)
- [ ] Run full AIME24 benchmark (`run_aime24_vllm.sh`)
- [ ] Compare accuracy with HF baseline
- [ ] Performance benchmarking

**Current Status**: Infrastructure 100% complete, ready for testing

## Process Naming Convention

All scripts use `PD-L1_binder` process name prefix for cluster compatibility (set via `VLLM_PROCESS_NAME_PREFIX` environment variable).

## Expected Results Format

**JSONL Output Structure**:
```json
{
  "id": 60,
  "question": "Every morning Aya goes for a ...",
  "ground_truth": "204",
  "generated_answers": [
    "The answer is \\boxed{204}",
    "... (7 more samples)"
  ],
  "num_samples": 8,
  "temperature": 0.6,
  "kv_budget": 2048,
  "pruning_mode": "per_head"
}
```

## Troubleshooting

### Common Issues

1. **Conda environment not activated**:
   ```bash
   conda activate trivllm
   ```

2. **Module import errors**:
   - Ensure PYTHONPATH includes project root
   - Shell scripts automatically set this

3. **GPU memory issues**:
   - Reduce `--kv-budget` or `--num-samples`
   - Adjust `--gpu-memory-utilization` (default: 0.9)

4. **Missing statistics file**:
   - Verify `STATS_PATH` exists or generate new statistics
   - Can run without stats by omitting `--sparse-stats-path`

## Contributing

When modifying benchmark scripts:
- Maintain parameter parity with HuggingFace baseline
- Update both shell scripts and Python argument parsers
- Add new parameters to this README
- Ensure changes don't break comparison logic

## See Also

- **TriAttention Implementation**: `TriAttention_vLLM/triattention/`
- **HuggingFace Reference**: `R-KV/weian_development/rkv_sharded_runner.py`
- **Dataset Docs**: `R-KV/HuggingFace/data/README.md` (if available)
