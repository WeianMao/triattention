"""
Pytest fixtures for TriAttention testing.

Provides reusable fixtures for model configurations, random data generation,
and frequency statistics.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Tuple


@pytest.fixture
def qwen_7b_config() -> Dict:
    """
    Qwen-7B model configuration.

    Returns:
        dict: Model hyperparameters
    """
    return {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "hidden_size": 4096,
        "rope_theta": 10000.0,
        "max_position_embeddings": 8192,
    }


@pytest.fixture
def qwen_14b_config() -> Dict:
    """
    Qwen-14B model configuration.

    Returns:
        dict: Model hyperparameters
    """
    return {
        "num_layers": 40,
        "num_heads": 40,
        "num_kv_heads": 40,
        "head_dim": 128,
        "hidden_size": 5120,
        "rope_theta": 10000.0,
        "max_position_embeddings": 8192,
    }


@pytest.fixture
def small_test_config() -> Dict:
    """
    Small configuration for fast testing.

    Returns:
        dict: Reduced model hyperparameters
    """
    return {
        "num_layers": 4,
        "num_heads": 8,
        "num_kv_heads": 8,
        "head_dim": 64,
        "hidden_size": 512,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
    }


@pytest.fixture
def rope_frequencies(small_test_config: Dict) -> torch.Tensor:
    """
    Generate RoPE frequency values.

    Args:
        small_test_config: Model configuration

    Returns:
        torch.Tensor: RoPE frequencies [freq_count]
    """
    head_dim = small_test_config["head_dim"]
    theta = small_test_config["rope_theta"]
    freq_count = head_dim // 2

    freqs = 1.0 / (theta ** (torch.arange(0, freq_count, dtype=torch.float32) / freq_count))
    return freqs


@pytest.fixture
def random_kv_cache(small_test_config: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random KV cache data for testing.

    Args:
        small_test_config: Model configuration

    Returns:
        tuple: (key_states, value_states, position_indices)
            - key_states: [batch, num_kv_heads, seq_len, head_dim]
            - value_states: [batch, num_kv_heads, seq_len, head_dim]
            - position_indices: [seq_len]
    """
    batch_size = 2
    num_kv_heads = small_test_config["num_kv_heads"]
    head_dim = small_test_config["head_dim"]
    seq_len = 128

    key_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    position_indices = torch.arange(seq_len, dtype=torch.long)

    return key_states, value_states, position_indices


@pytest.fixture
def random_query_stats(small_test_config: Dict) -> Dict[str, torch.Tensor]:
    """
    Generate random query statistics (simulating stats file).

    Args:
        small_test_config: Model configuration

    Returns:
        dict: Query statistics
            - Q_mean_real: [num_layers, num_heads, freq_count]
            - Q_mean_imag: [num_layers, num_heads, freq_count]
            - freq_scale_sq: [num_layers, num_heads, freq_count]
            - extra_coef: [num_layers, num_heads, freq_count]
    """
    num_layers = small_test_config["num_layers"]
    num_heads = small_test_config["num_heads"]
    head_dim = small_test_config["head_dim"]
    freq_count = head_dim // 2

    return {
        "Q_mean_real": torch.randn(num_layers, num_heads, freq_count),
        "Q_mean_imag": torch.randn(num_layers, num_heads, freq_count),
        "freq_scale_sq": torch.rand(num_layers, num_heads, freq_count) * 2.0 + 0.5,
        "extra_coef": torch.randn(num_layers, num_heads, freq_count),
    }


@pytest.fixture
def deterministic_seed():
    """
    Set deterministic random seed for reproducible tests.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield
    # Reset after test
    torch.manual_seed(torch.initial_seed())


@pytest.fixture(params=[torch.float32, torch.float16, torch.bfloat16])
def test_dtype(request):
    """
    Parametrize tests across multiple dtypes.

    Returns:
        torch.dtype: One of float32, float16, bfloat16
    """
    return request.param


@pytest.fixture
def tolerance_for_dtype(test_dtype):
    """
    Get appropriate numerical tolerance for a given dtype.

    Args:
        test_dtype: torch.dtype

    Returns:
        float: Absolute tolerance value
    """
    if test_dtype == torch.float32:
        return 1e-5
    elif test_dtype == torch.float16:
        # FP16 has ~3.5 decimal digits precision, complex scoring formulas
        # can accumulate errors up to ~0.3 on some hardware (Tesla T4)
        return 3e-1
    elif test_dtype == torch.bfloat16:
        return 1e-1
    else:
        return 1e-5


@pytest.fixture
def pruning_budget_config() -> Dict:
    """
    Configuration for pruning budget and overflow parameters.

    Returns:
        dict: Pruning configuration
    """
    return {
        "budget_slots": 128,
        "overflow_slots": 32,
        "divide_length": 32,
        "prefill_len": 64,
        "protect_prefill": False,
    }


@pytest.fixture
def multi_position_offsets() -> torch.Tensor:
    """
    Position offsets for multi-position scoring.

    Returns:
        torch.Tensor: Offsets [0, 1, 2, ..., 15]
    """
    return torch.arange(16, dtype=torch.long)


@pytest.fixture(params=["mean", "max"])
def aggregation_strategy(request):
    """
    Parametrize multi-position score aggregation strategies.

    Returns:
        str: "mean" or "max"
    """
    return request.param


@pytest.fixture(params=["per_head", "per_layer", "per_layer_per_head"])
def pruning_mode(request):
    """
    Parametrize pruning granularity modes.

    Returns:
        str: Pruning mode
    """
    return request.param


# Device fixtures
@pytest.fixture
def device():
    """
    Get test device (CUDA if available, else CPU).

    Returns:
        torch.device: Test device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cuda_only(device):
    """
    Skip test if CUDA is not available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device


# GPU capability detection fixtures
@pytest.fixture
def gpu_capability():
    """
    Get current GPU compute capability.

    Returns:
        tuple: (major, minor) version numbers, or None if no CUDA device
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(torch.cuda.current_device())
    return None


@pytest.fixture
def requires_sm80(gpu_capability):
    """
    Skip test if GPU compute capability < 8.0 (required for bf16).

    This fixture checks if the current GPU supports compute capability SM80 or higher,
    which is required for native bfloat16 support.
    """
    if gpu_capability is None:
        pytest.skip("CUDA not available")
    if gpu_capability[0] < 8:
        pytest.skip(f"Test requires SM80+, got SM{gpu_capability[0]}{gpu_capability[1]}")
    return gpu_capability


@pytest.fixture
def requires_sm70(gpu_capability):
    """
    Skip test if GPU compute capability < 7.0 (required for some FP16 operations).

    This fixture checks if the current GPU supports compute capability SM70 or higher.
    """
    if gpu_capability is None:
        pytest.skip("CUDA not available")
    if gpu_capability[0] < 7:
        pytest.skip(f"Test requires SM70+, got SM{gpu_capability[0]}{gpu_capability[1]}")
    return gpu_capability
