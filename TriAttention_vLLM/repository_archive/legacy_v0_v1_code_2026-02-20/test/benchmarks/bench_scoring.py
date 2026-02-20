"""
Performance benchmarks for TriAttention scoring.

Measures throughput and latency of scoring operations across
different configurations and dtypes.
"""

import time
import torch
import argparse
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_scoring_correctness import TriAttentionScorer, apply_rope_rotation


class ScoringBenchmark:
    """
    Benchmark harness for TriAttention scoring operations.
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = []

    def benchmark_single_position(
        self,
        num_keys: int,
        freq_count: int,
        num_trials: int = 100,
        dtype: torch.dtype = torch.float32,
    ) -> Dict:
        """
        Benchmark single position scoring.

        Args:
            num_keys: Number of keys to score
            freq_count: Number of frequency components
            num_trials: Number of timing trials
            dtype: Data type to use

        Returns:
            dict: Timing statistics
        """
        # Generate test data
        Q_mean_real = torch.randn(1, 1, freq_count, dtype=dtype, device=self.device)
        Q_mean_imag = torch.randn(1, 1, freq_count, dtype=dtype, device=self.device)
        freq_scale_sq = torch.rand(1, 1, freq_count, dtype=dtype, device=self.device) * 2.0
        extra_coef = torch.randn(1, 1, freq_count, dtype=dtype, device=self.device)
        omega = torch.rand(freq_count, dtype=dtype, device=self.device) * 0.1

        K = torch.randn(num_keys, freq_count * 2, dtype=dtype, device=self.device)
        position_indices = torch.arange(num_keys, dtype=torch.long, device=self.device)

        K_rot_real, K_rot_imag = apply_rope_rotation(K, position_indices, omega)

        scorer = TriAttentionScorer(Q_mean_real, Q_mean_imag, freq_scale_sq, extra_coef, omega)

        # Warmup
        for _ in range(10):
            _ = scorer.score_single_position(K_rot_real, K_rot_imag, position_indices, 100, 0, 0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        timings = []
        for _ in range(num_trials):
            start = time.perf_counter()
            _ = scorer.score_single_position(K_rot_real, K_rot_imag, position_indices, 100, 0, 0)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

        timings = torch.tensor(timings)

        return {
            "mean_ms": timings.mean().item() * 1000,
            "std_ms": timings.std().item() * 1000,
            "min_ms": timings.min().item() * 1000,
            "max_ms": timings.max().item() * 1000,
            "median_ms": timings.median().item() * 1000,
            "throughput_keys_per_sec": num_keys / timings.mean().item(),
        }

    def benchmark_multi_position(
        self,
        num_keys: int,
        freq_count: int,
        num_positions: int,
        num_trials: int = 100,
        dtype: torch.dtype = torch.float32,
        aggregation: str = "mean",
    ) -> Dict:
        """
        Benchmark multi-position scoring.

        Args:
            num_keys: Number of keys
            freq_count: Frequency components
            num_positions: Number of future positions
            num_trials: Timing trials
            dtype: Data type
            aggregation: "mean" or "max"

        Returns:
            dict: Timing statistics
        """
        Q_mean_real = torch.randn(1, 1, freq_count, dtype=dtype, device=self.device)
        Q_mean_imag = torch.randn(1, 1, freq_count, dtype=dtype, device=self.device)
        freq_scale_sq = torch.rand(1, 1, freq_count, dtype=dtype, device=self.device) * 2.0
        extra_coef = torch.randn(1, 1, freq_count, dtype=dtype, device=self.device)
        omega = torch.rand(freq_count, dtype=dtype, device=self.device) * 0.1

        K = torch.randn(num_keys, freq_count * 2, dtype=dtype, device=self.device)
        position_indices = torch.arange(num_keys, dtype=torch.long, device=self.device)

        K_rot_real, K_rot_imag = apply_rope_rotation(K, position_indices, omega)

        scorer = TriAttentionScorer(Q_mean_real, Q_mean_imag, freq_scale_sq, extra_coef, omega)

        target_positions = torch.arange(100, 100 + num_positions, device=self.device)

        # Warmup
        for _ in range(10):
            _ = scorer.score_multi_position(
                K_rot_real, K_rot_imag, position_indices, target_positions, 0, 0, aggregation
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        timings = []
        for _ in range(num_trials):
            start = time.perf_counter()
            _ = scorer.score_multi_position(
                K_rot_real, K_rot_imag, position_indices, target_positions, 0, 0, aggregation
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

        timings = torch.tensor(timings)

        return {
            "mean_ms": timings.mean().item() * 1000,
            "std_ms": timings.std().item() * 1000,
            "throughput_keys_per_sec": num_keys / timings.mean().item(),
        }

    def benchmark_rope_rotation(
        self, num_keys: int, head_dim: int, num_trials: int = 100, dtype: torch.dtype = torch.float32
    ) -> Dict:
        """
        Benchmark RoPE rotation performance.

        Args:
            num_keys: Number of keys
            head_dim: Head dimension
            num_trials: Timing trials
            dtype: Data type

        Returns:
            dict: Timing statistics
        """
        freq_count = head_dim // 2
        K = torch.randn(num_keys, head_dim, dtype=dtype, device=self.device)
        position_indices = torch.arange(num_keys, dtype=torch.long, device=self.device)
        omega = torch.rand(freq_count, dtype=dtype, device=self.device) * 0.1

        # Warmup
        for _ in range(10):
            _ = apply_rope_rotation(K, position_indices, omega)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        timings = []
        for _ in range(num_trials):
            start = time.perf_counter()
            _ = apply_rope_rotation(K, position_indices, omega)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

        timings = torch.tensor(timings)

        return {
            "mean_ms": timings.mean().item() * 1000,
            "std_ms": timings.std().item() * 1000,
        }

    def run_suite(self, config: str = "default"):
        """
        Run comprehensive benchmark suite.

        Args:
            config: "default", "large", or "small"
        """
        if config == "small":
            num_keys_list = [64, 128]
            freq_counts = [32]
            num_positions_list = [8]
            dtypes = [torch.float32]
        elif config == "large":
            num_keys_list = [512, 1024, 2048, 4096]
            freq_counts = [64, 128]
            num_positions_list = [16, 32]
            dtypes = [torch.float32, torch.bfloat16]
        else:  # default
            num_keys_list = [128, 256, 512]
            freq_counts = [32, 64]
            num_positions_list = [16]
            dtypes = [torch.float32, torch.bfloat16]

        print(f"\n{'=' * 80}")
        print(f"TriAttention Scoring Benchmark Suite ({config} config)")
        print(f"Device: {self.device}")
        print(f"{'=' * 80}\n")

        # Single position benchmarks
        print("Single Position Scoring:")
        print(f"{'Keys':<8} {'Freqs':<8} {'Dtype':<12} {'Mean (ms)':<12} {'Throughput (keys/s)':<20}")
        print("-" * 80)

        for num_keys in num_keys_list:
            for freq_count in freq_counts:
                for dtype in dtypes:
                    result = self.benchmark_single_position(num_keys, freq_count, dtype=dtype)
                    print(
                        f"{num_keys:<8} {freq_count:<8} {str(dtype).split('.')[-1]:<12} "
                        f"{result['mean_ms']:<12.3f} {result['throughput_keys_per_sec']:<20.1f}"
                    )

        # Multi-position benchmarks
        print(f"\n{'=' * 80}")
        print("Multi-Position Scoring (mean aggregation):")
        print(
            f"{'Keys':<8} {'Freqs':<8} {'Positions':<12} {'Dtype':<12} {'Mean (ms)':<12} {'Throughput':<20}"
        )
        print("-" * 80)

        for num_keys in num_keys_list[:2]:  # Fewer configs for multi-position
            for freq_count in freq_counts[:1]:
                for num_positions in num_positions_list:
                    for dtype in dtypes:
                        result = self.benchmark_multi_position(
                            num_keys, freq_count, num_positions, dtype=dtype, aggregation="mean"
                        )
                        print(
                            f"{num_keys:<8} {freq_count:<8} {num_positions:<12} "
                            f"{str(dtype).split('.')[-1]:<12} {result['mean_ms']:<12.3f} "
                            f"{result['throughput_keys_per_sec']:<20.1f}"
                        )

        # RoPE rotation benchmarks
        print(f"\n{'=' * 80}")
        print("RoPE Rotation:")
        print(f"{'Keys':<8} {'Head Dim':<12} {'Dtype':<12} {'Mean (ms)':<12}")
        print("-" * 80)

        for num_keys in num_keys_list[:3]:
            for head_dim in [64, 128]:
                for dtype in dtypes:
                    result = self.benchmark_rope_rotation(num_keys, head_dim, dtype=dtype)
                    print(
                        f"{num_keys:<8} {head_dim:<12} {str(dtype).split('.')[-1]:<12} "
                        f"{result['mean_ms']:<12.3f}"
                    )

        print(f"\n{'=' * 80}")
        print("Benchmark suite completed")
        print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="TriAttention scoring benchmarks")
    parser.add_argument(
        "--config", type=str, default="default", choices=["small", "default", "large"]
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-trials", type=int, default=100)

    args = parser.parse_args()

    benchmark = ScoringBenchmark(device=args.device)
    benchmark.run_suite(config=args.config)


if __name__ == "__main__":
    main()
