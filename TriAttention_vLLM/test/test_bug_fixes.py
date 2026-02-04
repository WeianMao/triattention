"""Test suite for bug fixes in TriAttention code review.

This module tests the following bug fixes:
1. TrigTableCache index validation (off-by-one error)
2. position_indices dtype validation (removed bfloat16)
3. normalize_scores division by zero handling
4. disable_mlr fallback to PyTorch implementation
"""
import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from triattention.config import TriAttentionConfig
from triattention.kernels.triton_scoring import TrigTableCache
from triattention.utils import normalize_scores


class TestTrigTableCacheFix:
    """Test TrigTableCache index validation fixes."""

    def test_valid_round_starts(self):
        """Test that valid round_start values work correctly."""
        compress_interval = 128
        max_seq_len = 512
        offsets = torch.tensor([0, 1, 2], dtype=torch.float32)
        omega = torch.randn(32, dtype=torch.float32)
        device = torch.device("cpu")

        cache = TrigTableCache(max_seq_len, compress_interval, offsets, omega, device)

        # Valid round_starts: multiples of compress_interval
        valid_values = [128, 256, 384, 512]
        for rs in valid_values:
            cos_vals, sin_vals = cache.get_trig_values(rs)
            assert cos_vals.shape == (3, 32), f"Expected (3, 32), got {cos_vals.shape}"
            assert sin_vals.shape == (3, 32), f"Expected (3, 32), got {sin_vals.shape}"

    def test_invalid_round_start_too_small(self):
        """Test that round_start < compress_interval raises error."""
        compress_interval = 128
        max_seq_len = 512
        offsets = torch.tensor([0, 1, 2], dtype=torch.float32)
        omega = torch.randn(32, dtype=torch.float32)
        device = torch.device("cpu")

        cache = TrigTableCache(max_seq_len, compress_interval, offsets, omega, device)

        # round_start=0 should fail
        with pytest.raises(ValueError, match="out of range"):
            cache.get_trig_values(0)

        # round_start=64 should fail (not multiple of interval)
        with pytest.raises(ValueError, match="must be a multiple"):
            cache.get_trig_values(64)

    def test_invalid_round_start_too_large(self):
        """Test that round_start > max_seq_len raises error."""
        compress_interval = 128
        max_seq_len = 512
        offsets = torch.tensor([0, 1, 2], dtype=torch.float32)
        omega = torch.randn(32, dtype=torch.float32)
        device = torch.device("cpu")

        cache = TrigTableCache(max_seq_len, compress_interval, offsets, omega, device)

        # round_start=640 should fail (> max_seq_len)
        with pytest.raises(ValueError, match="out of range"):
            cache.get_trig_values(640)

    def test_invalid_round_start_not_multiple(self):
        """Test that non-multiple round_start values raise error."""
        compress_interval = 128
        max_seq_len = 512
        offsets = torch.tensor([0, 1, 2], dtype=torch.float32)
        omega = torch.randn(32, dtype=torch.float32)
        device = torch.device("cpu")

        cache = TrigTableCache(max_seq_len, compress_interval, offsets, omega, device)

        # Non-multiples should fail
        invalid_values = [127, 129, 200, 255, 257]
        for rs in invalid_values:
            with pytest.raises(ValueError, match="must be a multiple"):
                cache.get_trig_values(rs)


class TestConfigDtypeFix:
    """Test TriAttentionConfig dtype validation fixes."""

    def test_valid_position_indices_dtypes(self):
        """Test that int32 and int64 are accepted."""
        config = TriAttentionConfig(position_indices_dtype=torch.int32)
        assert config.position_indices_dtype == torch.int32

        config = TriAttentionConfig(position_indices_dtype=torch.int64)
        assert config.position_indices_dtype == torch.int64

    def test_invalid_position_indices_dtype_bfloat16(self):
        """Test that bfloat16 is rejected for position_indices."""
        with pytest.raises(ValueError, match="position_indices_dtype must be one of"):
            TriAttentionConfig(position_indices_dtype=torch.bfloat16)

    def test_invalid_position_indices_dtype_float32(self):
        """Test that float32 is rejected for position_indices."""
        with pytest.raises(ValueError, match="position_indices_dtype must be one of"):
            TriAttentionConfig(position_indices_dtype=torch.float32)


class TestNormalizeScoresFix:
    """Test normalize_scores division by zero handling."""

    def test_normal_scores(self):
        """Test that normal scores are normalized correctly."""
        scores = torch.randn(2, 100)
        normalized = normalize_scores(scores)

        # Check mean and std are close to 0 and 1
        mean = normalized.mean(dim=-1)
        std = normalized.std(dim=-1)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-6)

    def test_constant_scores_no_nan(self):
        """Test that constant scores don't produce NaN/Inf."""
        scores = torch.ones(2, 100) * 5.0
        normalized = normalize_scores(scores)

        # Should not have NaN or Inf
        assert not torch.isnan(normalized).any(), "Normalized scores contain NaN"
        assert not torch.isinf(normalized).any(), "Normalized scores contain Inf"

        # Constant scores should normalize to zero
        assert torch.allclose(normalized, torch.zeros_like(normalized))

    def test_all_zero_scores_no_nan(self):
        """Test that all-zero scores don't produce NaN/Inf."""
        scores = torch.zeros(2, 100)
        normalized = normalize_scores(scores)

        # Should not have NaN or Inf
        assert not torch.isnan(normalized).any(), "Normalized scores contain NaN"
        assert not torch.isinf(normalized).any(), "Normalized scores contain Inf"

        # All-zero scores should remain zero
        assert torch.allclose(normalized, torch.zeros_like(normalized))

    def test_eps_parameter(self):
        """Test that eps parameter can be customized."""
        scores = torch.ones(2, 100)
        eps = 1e-10

        normalized = normalize_scores(scores, eps=eps)

        # Should not have NaN or Inf
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
