#!/usr/bin/env python3
"""Test script to validate run_math_vllm.py structure without loading models."""

import json
import sys
import tempfile
from pathlib import Path

# Test 1: Import test
print("Test 1: Importing modules...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from run_math_vllm import (
        parse_args,
        load_dataset,
        setup_triattention_config,
        PROMPT_TEMPLATE
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Prompt template validation
print("\nTest 2: Validating prompt template...")
try:
    test_prompt = PROMPT_TEMPLATE.format(question="What is 2+2?")
    assert "Problem:" in test_prompt
    assert "chain-of-thought" in test_prompt
    assert "\\boxed" in test_prompt
    print(f"✓ Prompt template valid ({len(PROMPT_TEMPLATE)} chars)")
    print(f"  Sample: {test_prompt[:100]}...")
except Exception as e:
    print(f"✗ Prompt template invalid: {e}")
    sys.exit(1)

# Test 3: Dataset loading
print("\nTest 3: Testing dataset loading...")
try:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps({"question": "Test Q1", "answer": "A1"}) + "\n")
        f.write(json.dumps({"question": "Test Q2", "answer": "A2"}) + "\n")
        temp_file = f.name

    dataset = load_dataset(temp_file)
    assert len(dataset) == 2
    assert dataset[0]["question"] == "Test Q1"
    assert dataset[0]["index"] == 0
    assert dataset[1]["index"] == 1
    print(f"✓ Dataset loading works (loaded {len(dataset)} items)")
    Path(temp_file).unlink()
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    sys.exit(1)

# Test 4: Argument parsing
print("\nTest 4: Testing argument parsing...")
try:
    sys.argv = [
        'test',
        '--model-path', 'test_model',
        '--dataset', 'test.jsonl',
        '--output-dir', '/tmp/test',
        '--kv-budget', '256',
        '--divide-length', '64',
        '--window-size', '128',
        '--pruning-mode', 'per_head',
    ]
    args = parse_args()
    assert args.model_path == 'test_model'
    assert args.dataset == 'test.jsonl'
    assert args.output_dir == '/tmp/test'
    assert args.kv_budget == 256
    assert args.divide_length == 64
    assert args.window_size == 128
    assert args.pruning_mode == 'per_head'
    print("✓ Argument parsing works")
    print(f"  Parsed {len(vars(args))} arguments")
except Exception as e:
    print(f"✗ Argument parsing failed: {e}")
    sys.exit(1)

# Test 5: TriAttention config creation
print("\nTest 5: Testing TriAttention config creation...")
try:
    config = setup_triattention_config(args)
    assert config.kv_budget == 256
    assert config.divide_length == 64
    assert config.window_size == 128
    assert config.pruning_mode == 'per_head'
    print("✓ TriAttention config creation works")
    print(f"  Budget: {config.kv_budget}, Divide: {config.divide_length}")
except Exception as e:
    print(f"✗ Config creation failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All structural tests passed! ✓")
print("Script is ready for full inference testing.")
print("="*60)
