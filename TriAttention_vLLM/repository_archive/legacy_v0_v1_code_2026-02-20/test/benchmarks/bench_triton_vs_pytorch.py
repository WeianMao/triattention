"""
Comprehensive throughput benchmark comparing Triton kernel vs PyTorch reference.

Measures:
- Triton kernel latency
- PyTorch reference latency
- Speedup ratio
- Memory bandwidth utilization estimates

Configuration space:
- batch_size: 1, 8, 32
- seq_len: 1024, 2048, 4096, 8192
- num_heads: 4, 8 (typical Qwen KV heads)
- head_dim: 128
- num_offsets: 16
"""

import time
import torch
import argparse
from typing import Dict, List, Tuple
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from triattention.kernels.triton_scoring import speckv_scoring
from test.test_scoring_kernel import pytorch_reference_scoring, build_omega


class ThroughputBenchmark:
    """
    Benchmark harness comparing Triton vs PyTorch implementations.
    """

    def __init__(self, device="cuda", warmup_trials=10, num_trials=100):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.warmup_trials = warmup_trials
        self.num_trials = num_trials

        if self.device.type != "cuda":
            raise RuntimeError("Benchmarks require CUDA")

    def create_test_data(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_offsets: int,
        dtype: torch.dtype = torch.float32,
    ) -> Dict:
        """Generate test data for benchmarking."""
        freq_count = head_dim // 2

        torch.manual_seed(42)  # Reproducibility

        K_rot = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            device=self.device, dtype=dtype
        ).contiguous()

        q_mean_real = torch.randn(num_heads, freq_count, device=self.device, dtype=dtype)
        q_mean_imag = torch.randn(num_heads, freq_count, device=self.device, dtype=dtype)
        q_mean_complex = torch.complex(q_mean_real, q_mean_imag)

        q_abs_mean = torch.abs(torch.randn(num_heads, freq_count, device=self.device, dtype=dtype))
        freq_scale_sq = torch.ones(num_heads, freq_count, device=self.device, dtype=dtype)

        position_indices = torch.arange(seq_len, device=self.device, dtype=torch.long)
        omega = build_omega(head_dim, self.device)
        offsets = torch.tensor([float(2**i) for i in range(num_offsets)], device=self.device)

        round_start = 1000

        return {
            "K_rot": K_rot,
            "position_indices": position_indices,
            "q_mean_real": q_mean_real,
            "q_mean_imag": q_mean_imag,
            "q_mean_complex": q_mean_complex,
            "q_abs_mean": q_abs_mean,
            "freq_scale_sq": freq_scale_sq,
            "omega": omega,
            "offsets": offsets,
            "round_start": round_start,
        }

    def benchmark_triton(
        self,
        data: Dict,
        aggregation: str = "max",
    ) -> Dict:
        """Benchmark Triton kernel."""
        # Warmup
        for _ in range(self.warmup_trials):
            _ = speckv_scoring(
                data["K_rot"],
                data["position_indices"],
                data["q_mean_real"],
                data["q_mean_imag"],
                data["q_abs_mean"],
                data["freq_scale_sq"],
                data["omega"],
                data["offsets"],
                data["round_start"],
                aggregation,
            )

        torch.cuda.synchronize()

        # Benchmark
        timings = []
        for _ in range(self.num_trials):
            start = time.perf_counter()
            _ = speckv_scoring(
                data["K_rot"],
                data["position_indices"],
                data["q_mean_real"],
                data["q_mean_imag"],
                data["q_abs_mean"],
                data["freq_scale_sq"],
                data["omega"],
                data["offsets"],
                data["round_start"],
                aggregation,
            )
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

        timings = torch.tensor(timings)

        return {
            "mean_ms": timings.mean().item() * 1000,
            "std_ms": timings.std().item() * 1000,
            "min_ms": timings.min().item() * 1000,
            "max_ms": timings.max().item() * 1000,
        }

    def benchmark_pytorch(
        self,
        data: Dict,
        aggregation: str = "max",
    ) -> Dict:
        """Benchmark PyTorch reference."""
        # Warmup
        for _ in range(self.warmup_trials):
            _ = pytorch_reference_scoring(
                data["K_rot"],
                data["position_indices"],
                data["q_mean_complex"],
                data["q_abs_mean"],
                data["freq_scale_sq"],
                data["omega"],
                data["offsets"],
                data["round_start"],
                aggregation,
            )

        torch.cuda.synchronize()

        # Benchmark
        timings = []
        for _ in range(self.num_trials):
            start = time.perf_counter()
            _ = pytorch_reference_scoring(
                data["K_rot"],
                data["position_indices"],
                data["q_mean_complex"],
                data["q_abs_mean"],
                data["freq_scale_sq"],
                data["omega"],
                data["offsets"],
                data["round_start"],
                aggregation,
            )
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

        timings = torch.tensor(timings)

        return {
            "mean_ms": timings.mean().item() * 1000,
            "std_ms": timings.std().item() * 1000,
            "min_ms": timings.min().item() * 1000,
            "max_ms": timings.max().item() * 1000,
        }

    def estimate_memory_bandwidth(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_offsets: int,
        latency_ms: float,
        dtype: torch.dtype = torch.float32,
    ) -> float:
        """
        Estimate memory bandwidth utilization.

        Returns GB/s
        """
        bytes_per_element = 4 if dtype == torch.float32 else 2
        freq_count = head_dim // 2

        # Input reads:
        # - K_rot: batch * num_heads * seq_len * head_dim
        # - position_indices: seq_len (or batch * seq_len)
        # - q_mean_real/imag: num_heads * freq_count * 2
        # - q_abs_mean: num_heads * freq_count
        # - freq_scale_sq: num_heads * freq_count
        # - omega: freq_count
        # - offsets: num_offsets

        k_rot_bytes = batch_size * num_heads * seq_len * head_dim * bytes_per_element
        pos_bytes = seq_len * 8  # int64
        q_stats_bytes = num_heads * freq_count * bytes_per_element * 4  # real, imag, abs, scale
        omega_bytes = freq_count * bytes_per_element
        offsets_bytes = num_offsets * bytes_per_element

        # Output writes:
        # - scores: batch * num_heads * seq_len * 4 (float32 output)
        output_bytes = batch_size * num_heads * seq_len * 4

        total_bytes = k_rot_bytes + pos_bytes + q_stats_bytes + omega_bytes + offsets_bytes + output_bytes
        total_gb = total_bytes / (1024**3)

        latency_s = latency_ms / 1000
        bandwidth_gb_s = total_gb / latency_s

        return bandwidth_gb_s

    def run_comprehensive_suite(
        self,
        batch_sizes: List[int] = [1, 8, 32],
        seq_lens: List[int] = [1024, 2048, 4096, 8192],
        num_heads_list: List[int] = [4, 8],
        head_dim: int = 128,
        num_offsets: int = 16,
        aggregation: str = "max",
        dtype: torch.dtype = torch.float32,
    ):
        """Run comprehensive benchmark suite."""

        print(f"\n{'=' * 120}")
        print(f"TriAttention Scoring: Triton vs PyTorch Throughput Benchmark")
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Aggregation: {aggregation}")
        print(f"Dtype: {dtype}")
        print(f"Trials: {self.num_trials}")
        print(f"{'=' * 120}\n")

        # Print header
        header = (
            f"{'Batch':<8} {'Seq Len':<10} {'Heads':<8} "
            f"{'Triton (ms)':<15} {'PyTorch (ms)':<15} "
            f"{'Speedup':<10} {'Bandwidth (GB/s)':<20}"
        )
        print(header)
        print("-" * 120)

        results = []

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for num_heads in num_heads_list:
                    # Create test data
                    data = self.create_test_data(
                        batch_size, num_heads, seq_len, head_dim, num_offsets, dtype
                    )

                    # Benchmark both implementations
                    triton_result = self.benchmark_triton(data, aggregation)
                    pytorch_result = self.benchmark_pytorch(data, aggregation)

                    triton_ms = triton_result["mean_ms"]
                    pytorch_ms = pytorch_result["mean_ms"]
                    speedup = pytorch_ms / triton_ms

                    bandwidth = self.estimate_memory_bandwidth(
                        batch_size, num_heads, seq_len, head_dim, num_offsets, triton_ms, dtype
                    )

                    # Print row
                    row = (
                        f"{batch_size:<8} {seq_len:<10} {num_heads:<8} "
                        f"{triton_ms:<15.3f} {pytorch_ms:<15.3f} "
                        f"{speedup:<10.2f}x {bandwidth:<20.1f}"
                    )
                    print(row)

                    results.append({
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "num_heads": num_heads,
                        "triton_ms": triton_ms,
                        "pytorch_ms": pytorch_ms,
                        "speedup": speedup,
                        "bandwidth_gb_s": bandwidth,
                    })

        print(f"\n{'=' * 120}")

        # Summary statistics
        speedups = [r["speedup"] for r in results]
        bandwidths = [r["bandwidth_gb_s"] for r in results]

        print(f"\nSummary Statistics:")
        print(f"  Average Speedup: {sum(speedups) / len(speedups):.2f}x")
        print(f"  Min Speedup: {min(speedups):.2f}x")
        print(f"  Max Speedup: {max(speedups):.2f}x")
        print(f"  Average Bandwidth: {sum(bandwidths) / len(bandwidths):.1f} GB/s")
        print(f"  Peak Bandwidth: {max(bandwidths):.1f} GB/s")
        print(f"{'=' * 120}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description="Triton vs PyTorch throughput benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--num-heads", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-offsets", type=int, default=16)
    parser.add_argument("--aggregation", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--warmup-trials", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=100)

    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    benchmark = ThroughputBenchmark(
        device="cuda",
        warmup_trials=args.warmup_trials,
        num_trials=args.num_trials,
    )

    benchmark.run_comprehensive_suite(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        num_heads_list=args.num_heads,
        head_dim=args.head_dim,
        num_offsets=args.num_offsets,
        aggregation=args.aggregation,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
