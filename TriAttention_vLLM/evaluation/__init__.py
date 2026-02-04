"""TriAttention vLLM Evaluation Framework.

This package provides a complete evaluation pipeline for TriAttention KV compression,
reusing the R-KV sharded dispatch framework with vLLM as the inference backend.

Directory Structure:
    dispatch/       - Sharded task dispatcher
    runner/         - vLLM inference runner with TriAttention
    merge/          - Shard output merger
    eval/           - Math answer evaluation scripts
"""
