"""
Unit test for PositionScalingLayer interpolation.

Verifies:
1. Parameter count is 3
2. Softplus initialization produces weights ≈ 1.0
3. Interpolation produces correct scaling at known positions
"""

import torch
import sys
sys.path.insert(0, '.')
from model import PositionScalingLayer


def test_position_scaling_layer():
    """Test PositionScalingLayer implementation."""
    print("Testing PositionScalingLayer...")

    # Create layer
    layer = PositionScalingLayer(anchors=[1000, 10000, 100000])

    # Test 1: Parameter count
    param_count = sum(p.numel() for p in layer.parameters())
    print(f"\n1. Parameter count: {param_count}")
    assert param_count == 3, f"Expected 3 parameters, got {param_count}"

    # Test 2: Initial weights ≈ 1.0
    initial_weights = layer.anchor_weights.detach()
    print(f"2. Initial anchor weights: {initial_weights.tolist()}")
    for i, w in enumerate(initial_weights):
        assert 0.95 <= w <= 1.05, f"Anchor {i} weight {w:.4f} not close to 1.0"
    print("   All weights ≈ 1.0 ✓")

    # Test 3: Interpolation at known positions
    test_positions = torch.tensor([500, 1000, 3162, 10000, 31623, 100000, 200000], dtype=torch.float32)
    interpolated = layer._interpolate_weights(test_positions)

    print(f"\n3. Interpolation results:")
    print(f"   Position | log10(pos) | Interpolated Weight | Expected")
    print(f"   ---------|------------|---------------------|----------")

    w0, w1, w2 = initial_weights[0].item(), initial_weights[1].item(), initial_weights[2].item()

    expected = [
        (500, 2.7, w0, "w0 (below 1k)"),
        (1000, 3.0, w0, "w0"),
        (3162, 3.5, 0.5 * w0 + 0.5 * w1, "0.5*w0 + 0.5*w1"),
        (10000, 4.0, w1, "w1"),
        (31623, 4.5, 0.5 * w1 + 0.5 * w2, "0.5*w1 + 0.5*w2"),
        (100000, 5.0, w2, "w2"),
        (200000, 5.3, w2, "w2 (above 100k)")
    ]

    for i, (pos, log_pos, expected_w, desc) in enumerate(expected):
        actual_w = interpolated[i].item()
        print(f"   {pos:7.0f} | {log_pos:10.1f} | {actual_w:19.4f} | {desc}")

        # Check within tolerance (0.01)
        assert abs(actual_w - expected_w) < 0.01, \
            f"Position {pos}: expected {expected_w:.4f}, got {actual_w:.4f}"

    print("\n   All interpolations correct ✓")

    # Test 4: Forward pass
    logits = torch.tensor([2.0, 1.5, 1.0], dtype=torch.float32)
    positions = torch.tensor([1000, 10000, 100000], dtype=torch.float32)
    scaled = layer.forward(logits, positions)

    print(f"\n4. Forward pass test:")
    print(f"   Input logits: {logits.tolist()}")
    print(f"   Positions: {positions.tolist()}")
    print(f"   Scaled logits: {scaled.tolist()}")
    print(f"   Expected (logits * weights ≈ logits * 1.0): {logits.tolist()}")

    # Since weights ≈ 1.0, scaled should ≈ logits
    assert torch.allclose(scaled, logits, atol=0.1), \
        f"Scaled logits {scaled} not close to input logits {logits}"
    print("   Scaling applied correctly ✓")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_position_scaling_layer()
