"""
Test script for Module1KeyPruningNetwork.

Verifies:
1. Class exists and can be instantiated
2. 3 components are initialized
3. Forward pass works and returns drop probabilities in [0,1]
4. Parameter count is approximately 82,180
"""

import torch
from model import Module1KeyPruningNetwork

def test_module1_network():
    print("Testing Module1KeyPruningNetwork...")

    # Create model
    model = Module1KeyPruningNetwork()
    print("✓ Model instantiated successfully")

    # Verify components
    assert hasattr(model, 'kernel_layer'), "Missing kernel_layer"
    assert hasattr(model, 'mlp'), "Missing mlp"
    assert hasattr(model, 'position_scaling'), "Missing position_scaling"
    print("✓ All 3 components initialized")

    # Test forward pass with batch
    num_keys = 10
    head_dim = 128
    num_freqs = 64

    # Create dummy inputs
    K = torch.randn(num_keys, head_dim)
    key_positions = torch.arange(1000, 1000 + num_keys)
    reference_angles = torch.randn(num_freqs)

    # Forward pass
    drop_probs = model(K, key_positions, reference_angles)

    # Verify output shape
    assert drop_probs.shape == (num_keys,), f"Expected shape {(num_keys,)}, got {drop_probs.shape}"
    print(f"✓ Forward pass successful, output shape: {drop_probs.shape}")

    # Verify output range [0, 1]
    assert torch.all((drop_probs >= 0) & (drop_probs <= 1)), "Drop probabilities not in [0, 1]"
    print(f"✓ Drop probabilities in [0, 1], range: [{drop_probs.min():.4f}, {drop_probs.max():.4f}]")

    # Test with single key
    K_single = torch.randn(head_dim)
    pos_single = 1000
    drop_prob_single = model(K_single, pos_single, reference_angles)
    assert drop_prob_single.shape == (), f"Expected scalar, got shape {drop_prob_single.shape}"
    print(f"✓ Single key forward pass successful, output: {drop_prob_single.item():.4f}")

    # Check parameter count
    param_count = model.get_param_count()
    print("\nParameter count breakdown:")
    print(f"  - Kernel Layer: {param_count['kernel_layer']:,}")
    print(f"  - MLP: {param_count['mlp']:,}")
    print(f"  - Position Scaling: {param_count['position_scaling']:,}")
    print(f"  - Total: {param_count['total']:,}")

    # Verify total is approximately 82,180
    expected = 82180
    tolerance = 100
    assert abs(param_count['total'] - expected) <= tolerance, \
        f"Parameter count {param_count['total']} differs from expected {expected} by more than {tolerance}"
    print(f"✓ Parameter count matches expected: {param_count['total']:,} ≈ {expected:,} (±{tolerance})")

    # Verify individual component counts
    assert param_count['kernel_layer'] == 73856, \
        f"Kernel layer params {param_count['kernel_layer']} != 73856"
    print(f"✓ Kernel layer parameter count correct: {param_count['kernel_layer']:,}")

    # MLP params: (128*64 + 64) + (64*1 + 1) = 8256 + 64 + 64 + 1 = 8385
    expected_mlp = 8385
    assert param_count['mlp'] == expected_mlp, \
        f"MLP params {param_count['mlp']} != {expected_mlp}"
    print(f"✓ MLP parameter count correct: {param_count['mlp']:,}")

    assert param_count['position_scaling'] == 3, \
        f"Position scaling params {param_count['position_scaling']} != 3"
    print(f"✓ Position scaling parameter count correct: {param_count['position_scaling']:,}")

    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_module1_network()
