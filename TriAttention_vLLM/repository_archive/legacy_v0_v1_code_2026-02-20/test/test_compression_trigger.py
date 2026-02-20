"""Test compression trigger logic matches R-KV slack mode.

This test verifies that the compression trigger follows R-KV's slack mode:
- Trigger at budget + divide_length
- Compress to budget
- Cache fluctuates in range [budget, budget + divide_length]
"""
from triattention.config import TriAttentionConfig
from triattention.state import CompressionState


def test_slack_mode_trigger():
    """Test that compression triggers at budget + divide_length (R-KV slack mode)."""
    config = TriAttentionConfig(
        kv_budget=2048,
        divide_length=128,
        protect_prefill=False,
    )
    state = CompressionState(config)

    # Cache below budget - no compression
    assert not state.should_compress(1000), "Should not compress when cache < budget"
    assert not state.should_compress(2048), "Should not compress exactly at budget"

    # Cache in slack range (budget, budget + divide_length) - no compression yet
    assert not state.should_compress(2049), "Should not compress just above budget"
    assert not state.should_compress(2100), "Should not compress in slack range"
    assert not state.should_compress(2175), "Should not compress near trigger"

    # Cache at trigger threshold - compress!
    assert state.should_compress(2176), "MUST compress at budget + divide_length (2048 + 128)"
    assert state.should_compress(2200), "MUST compress above trigger threshold"
    assert state.should_compress(3000), "MUST compress far above trigger"


def test_slack_mode_with_prefill_protection():
    """Test slack mode with prefill protection enabled."""
    config = TriAttentionConfig(
        kv_budget=2048,
        divide_length=128,
        protect_prefill=True,
    )
    state = CompressionState(config)

    # Simulate prefill of 512 tokens
    state.prefill_length = 512

    # Total cache = 2000 (512 prefill + 1488 decode)
    # Effective size = 1488 (only decode tokens count)
    # Trigger = 2048 + 128 = 2176
    # Should NOT compress
    assert not state.should_compress(2000), "Should not compress when decode tokens < trigger"

    # Total cache = 2688 (512 prefill + 2176 decode)
    # Effective size = 2176 (only decode tokens count)
    # Trigger = 2048 + 128 = 2176
    # Should compress!
    assert state.should_compress(2688), "MUST compress when decode tokens >= budget + divide_length"


def test_compression_state_update():
    """Test that state updates correctly after compression."""
    config = TriAttentionConfig(
        kv_budget=2048,
        divide_length=128,
        protect_prefill=False,
    )
    state = CompressionState(config)

    # Simulate growing cache
    state.initialize(seq_len=100)
    assert state.current_cache_len == 100
    assert state.absolute_position == 100

    # Add tokens until trigger
    for _ in range(2076):  # Total = 2176 = budget + divide_length
        state.append_tokens(1)

    assert state.current_cache_len == 2176
    assert state.should_compress(2176)

    # Simulate compression to budget
    state.update_after_compression(new_cache_len=2048)

    assert state.current_cache_len == 2048
    assert state.compression_count == 1
    assert state.last_prune_step == 2176  # Absolute position when compressed

    # After compression, should NOT compress until hitting trigger again
    assert not state.should_compress(2048), "Should not compress right after compression"
    assert not state.should_compress(2100), "Should not compress in slack range"

    # Add more tokens to reach trigger again
    for _ in range(128):
        state.append_tokens(1)

    assert state.current_cache_len == 2176
    assert state.should_compress(2176), "Should trigger again at budget + divide_length"


def test_different_budgets_and_intervals():
    """Test compression trigger with various budget and divide_length settings."""
    test_cases = [
        (1024, 64, 1088),   # Small budget, small interval
        (2048, 128, 2176),  # Default settings
        (4096, 256, 4352),  # Large budget, large interval
        (512, 32, 544),     # Tiny budget, tiny interval
    ]

    for budget, divide_length, expected_trigger in test_cases:
        config = TriAttentionConfig(
            kv_budget=budget,
            divide_length=divide_length,
            protect_prefill=False,
        )
        state = CompressionState(config)

        # Should not compress below trigger
        assert not state.should_compress(expected_trigger - 1), \
            f"Should not compress at {expected_trigger - 1} (trigger={expected_trigger})"

        # Should compress at trigger
        assert state.should_compress(expected_trigger), \
            f"MUST compress at {expected_trigger} (budget={budget}, divide_length={divide_length})"


if __name__ == "__main__":
    # Run tests
    test_slack_mode_trigger()
    print("✅ test_slack_mode_trigger passed")

    test_slack_mode_with_prefill_protection()
    print("✅ test_slack_mode_with_prefill_protection passed")

    test_compression_state_update()
    print("✅ test_compression_state_update passed")

    test_different_budgets_and_intervals()
    print("✅ test_different_budgets_and_intervals passed")

    print("\n🎉 All compression trigger tests passed!")
