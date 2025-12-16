# Data Directory

This directory should contain the trace data file `qk.pt` required for training and evaluation.

## Setup Instructions

### Option 1: Symlink to existing trace data

If you have trace data generated from the reference implementation:

```bash
# From this directory, create symlink to trace data
ln -s /path/to/trace_directory/qk.pt qk.pt

# Example (adjust path as needed):
# ln -s ../../../../outputs/qk_bf16_traces/qid0003_trace34/qk.pt qk.pt
```

### Option 2: Copy trace data

```bash
# Copy trace file to this directory
cp /path/to/trace_directory/qk.pt .
```

## Expected Data Format

The `qk.pt` file should contain a dictionary with the following structure:

- `Q`: Query tensor
- `K`: Key tensor
- Additional metadata (if any)

Refer to the reference implementation for trace generation:
`../../attention_pruning_case_study_hybrid_rounds_xtrace.py`

## Verification

After setup, verify the file exists:

```bash
ls -lh qk.pt
```

You should see the file (or symlink) listed.
