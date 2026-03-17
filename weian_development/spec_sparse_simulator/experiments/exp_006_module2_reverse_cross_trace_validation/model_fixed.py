"""
Module 2 Multi-Bin Sparse Attention Model

Network Architecture:
- KernelEncodingLayer: von Mises kernel encoding (shared architecture, separate instances)
- Module2KeyNetwork: KernelEncoding -> logits (external softmax dim=0)
- Module2QueryNetwork: KernelEncoding -> logits (external softmax dim=-1)

NO MLP, NO PositionScaling (unlike Module 1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelEncodingLayer(nn.Module):
    """
    Kernel Encoding Layer using von Mises kernels.

    Implements frequency-based encoding with 64 frequency bands and 3 von Mises kernels per band.
    Total parameters: 73,856 (128 bins * 64 freqs * 3 kernels * 3 params + 128 bias)

    Vector layout: [real_0...real_63, imag_0...imag_63] (front/back, NOT interleaved)
    This matches transformers/Qwen3RotaryEmbedding.
    """

    def __init__(self, num_bins=128, num_freqs=64, num_kernels=3, inv_freq=None):
        """
        Args:
            num_bins: Number of output bins (default: 128)
            num_freqs: Number of frequency bands (default: 64, from head_dim=128)
            num_kernels: Number of von Mises kernels per frequency band (default: 3)
            inv_freq: Optional inverse frequency tensor from model's RoPE.
                      If provided, uses this instead of computing from base=10000.
        """
        super().__init__()
        self.num_bins = num_bins
        self.num_freqs = num_freqs
        self.num_kernels = num_kernels

        # Store inv_freq as a buffer (not a parameter)
        if inv_freq is not None:
            self.register_buffer('inv_freq', inv_freq.clone())
        else:
            self.inv_freq = None

        # Initialize mu: 3 kernels centered at -pi, 0, pi with small noise
        mu_init = torch.zeros(num_bins, num_freqs, num_kernels)
        for m in range(num_kernels):
            center = 2 * math.pi * m / num_kernels - math.pi
            mu_init[:, :, m] = torch.randn(num_bins, num_freqs) * 0.1 + center
        self.mu = nn.Parameter(mu_init)

        # Initialize kappa: small value (~1.0) for wide kernel coverage
        self.kappa = nn.Parameter(torch.ones(num_bins, num_freqs, num_kernels) * 1.0)

        # Initialize weight: small random values
        self.weight = nn.Parameter(torch.randn(num_bins, num_freqs, num_kernels) * 0.1)

        # Initialize bias: zeros
        self.bias = nn.Parameter(torch.zeros(num_bins))

    def _compute_reference_angles(self, round_start, round_window=128, base=10000):
        """
        Compute reference angles from round position using RoPE formula.

        The reference angle represents the rotation of a zero-angle vector (1, 0)
        at the round midpoint position.

        Args:
            round_start: Starting position of current round
            round_window: Window size (default: 128)
            base: RoPE base (default: 10000), only used if inv_freq is None

        Returns:
            reference_angles: Tensor of shape (num_freqs,)
        """
        # Reference position: middle of the round
        ref_position = round_start + round_window // 2

        # Use provided inv_freq or compute from base
        if self.inv_freq is not None:
            omega = self.inv_freq.to(device=self.mu.device, dtype=self.mu.dtype)
        else:
            # Compute RoPE angular frequencies omega from base
            dim_indices = torch.arange(0, self.num_freqs, device=self.mu.device, dtype=self.mu.dtype)
            omega = 1.0 / (base ** (2 * dim_indices / (self.num_freqs * 2)))

        # Reference angles: zero-angle vector (1,0) rotated by RoPE
        # angle_after = angle_before + pos * omega_j
        # For zero-angle vector: angle_before = 0
        # Therefore: reference_angle_j = ref_position * omega_j
        reference_angles = ref_position * omega

        return reference_angles

    def _prepare_round(self, reference_angles):
        """
        Prepare effective mu for a round by adding reference angles.

        This optimization computes mu_effective once per round instead of
        adjusting angles for every key.

        Args:
            reference_angles: Tensor of shape (num_freqs,)

        Returns:
            mu_effective: Tensor of shape (num_bins, num_freqs, num_kernels)
        """
        # Add reference_angles to mu: (num_bins, num_freqs, num_kernels)
        # Broadcasting: (1, num_freqs, 1)
        mu_effective = self.mu + reference_angles.view(1, -1, 1)
        return mu_effective

    def forward(self, K, reference_angles):
        """
        Forward pass of kernel encoding.

        Vector layout: [real_0...real_63, imag_0...imag_63] (front/back, NOT interleaved)
        This matches transformers/Qwen3RotaryEmbedding.

        Args:
            K: Key vectors of shape (num_keys, head_dim) or (head_dim,)
            reference_angles: Reference angles of shape (num_freqs,)
                Computed via _compute_reference_angles

        Returns:
            logits: Encoded features of shape (num_keys, num_bins) or (num_bins,)
        """
        # Handle single vector case
        is_single = K.dim() == 1
        if is_single:
            K = K.unsqueeze(0)  # (1, head_dim)

        num_keys = K.shape[0]

        # Split K into real and imag parts (front/back layout)
        # Vector layout: [real_0...real_63, imag_0...imag_63]
        K_real = K[:, :self.num_freqs]  # (num_keys, num_freqs)
        K_imag = K[:, self.num_freqs:]  # (num_keys, num_freqs)

        # Compute magnitude: sqrt(real^2 + imag^2)
        magnitude = torch.sqrt(K_real ** 2 + K_imag ** 2 + 1e-8)  # (num_keys, num_freqs)

        # Compute angle: atan2(imag, real)
        angle = torch.atan2(K_imag, K_real)  # (num_keys, num_freqs)

        # Prepare effective mu for this round
        mu_effective = self._prepare_round(reference_angles)

        # Compute normalized von Mises kernel
        # angle: (num_keys, num_freqs) -> (num_keys, 1, num_freqs, 1)
        # mu_effective: (num_bins, num_freqs, num_kernels)
        angle_expanded = angle.view(num_keys, 1, self.num_freqs, 1)
        mu_expanded = mu_effective.unsqueeze(0)  # (1, num_bins, num_freqs, num_kernels)

        # Normalized von Mises: exp(kappa * (cos(angle - mu) - 1))
        # Shape: (num_keys, num_bins, num_freqs, num_kernels)
        kernel = torch.exp(
            self.kappa.unsqueeze(0) * (torch.cos(angle_expanded - mu_expanded) - 1)
        )

        # Weighted sum over kernels: (num_keys, num_bins, num_freqs)
        weighted_kernels = (kernel * self.weight.unsqueeze(0)).sum(dim=3)

        # Multiply by magnitude and sum over frequencies
        # magnitude: (num_keys, num_freqs) -> (num_keys, 1, num_freqs)
        magnitude_expanded = magnitude.unsqueeze(1)
        weighted = weighted_kernels * magnitude_expanded  # (num_keys, num_bins, num_freqs)
        logits = weighted.sum(dim=2) + self.bias  # (num_keys, num_bins)

        # Return to original shape if input was single vector
        if is_single:
            logits = logits.squeeze(0)  # (num_bins,)

        return logits


class Module2KeyNetwork(nn.Module):
    """
    Module 2 Key Network for bin assignment.

    Architecture: KernelEncodingLayer only (NO MLP, NO PositionScaling)

    Input: K (post-RoPE key vectors) of shape (num_keys, head_dim)
    Output: logits of shape (num_keys, num_bins)

    Softmax is applied EXTERNALLY over keys (dim=0):
        key_probs = F.softmax(logits, dim=0)  # Each column sums to 1

    Execution timing: Once per round (round_start)
    """

    def __init__(self, num_bins=128, num_freqs=64, num_kernels=3, inv_freq=None):
        """
        Args:
            num_bins: Number of bins (default: 128)
            num_freqs: Number of frequency bands (default: 64)
            num_kernels: Number of von Mises kernels (default: 3)
            inv_freq: Optional inverse frequency tensor from model's RoPE.
        """
        super().__init__()
        self.kernel_layer = KernelEncodingLayer(num_bins, num_freqs, num_kernels, inv_freq=inv_freq)

    def forward(self, K, reference_angles):
        """
        Forward pass: K -> KernelEncoding -> logits

        Args:
            K: Key vectors of shape (num_keys, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)

        Returns:
            logits: Raw logits of shape (num_keys, num_bins)
                Apply softmax(dim=0) externally to get key_probs
        """
        # KernelEncoding -> logits
        logits = self.kernel_layer(K, reference_angles)
        return logits

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.kernel_layer._compute_reference_angles(round_start, round_window)


class Module2QueryNetwork(nn.Module):
    """
    Module 2 Query Network for bin routing.

    Architecture: KernelEncodingLayer only (NO MLP, NO PositionScaling)
    IDENTICAL architecture to KeyNetwork but SEPARATE parameters.

    Input: Q (post-RoPE query vector) of shape (head_dim,) or (num_queries, head_dim)
    Output: logits of shape (num_bins,) or (num_queries, num_bins)

    Softmax is applied EXTERNALLY over bins (dim=-1):
        bin_probs = F.softmax(logits, dim=-1)  # Each row sums to 1

    Execution timing: Every decoding step
    """

    def __init__(self, num_bins=128, num_freqs=64, num_kernels=3, inv_freq=None):
        """
        Args:
            num_bins: Number of bins (default: 128)
            num_freqs: Number of frequency bands (default: 64)
            num_kernels: Number of von Mises kernels (default: 3)
            inv_freq: Optional inverse frequency tensor from model's RoPE.
        """
        super().__init__()
        self.kernel_layer = KernelEncodingLayer(num_bins, num_freqs, num_kernels, inv_freq=inv_freq)

    def forward(self, Q, reference_angles):
        """
        Forward pass: Q -> KernelEncoding -> logits

        Args:
            Q: Query vector(s) of shape (head_dim,) or (num_queries, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)

        Returns:
            logits: Raw logits of shape (num_bins,) or (num_queries, num_bins)
                Apply softmax(dim=-1) externally to get bin_probs
        """
        # KernelEncoding -> logits
        logits = self.kernel_layer(Q, reference_angles)
        return logits

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.kernel_layer._compute_reference_angles(round_start, round_window)


class Module2Network(nn.Module):
    """
    Complete Module 2 Network combining Key and Query networks.

    This is a convenience wrapper that holds both networks and provides
    unified interface for training and inference.
    """

    def __init__(self, num_bins=128, num_freqs=64, num_kernels=3, inv_freq=None):
        """
        Args:
            num_bins: Number of bins (default: 128)
            num_freqs: Number of frequency bands (default: 64)
            num_kernels: Number of von Mises kernels (default: 3)
            inv_freq: Optional inverse frequency tensor from model's RoPE.
        """
        super().__init__()
        self.num_bins = num_bins

        # Separate Key and Query networks with independent parameters
        self.key_network = Module2KeyNetwork(num_bins, num_freqs, num_kernels, inv_freq=inv_freq)
        self.query_network = Module2QueryNetwork(num_bins, num_freqs, num_kernels, inv_freq=inv_freq)

    def forward_keys(self, K, reference_angles):
        """
        Process keys: K -> logits -> key_probs (softmax over keys)

        Args:
            K: Key vectors of shape (num_keys, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)

        Returns:
            key_probs: Probability distribution of shape (num_keys, num_bins)
                Each column sums to 1 (softmax over keys)
        """
        logits = self.key_network(K, reference_angles)
        # Softmax over keys (dim=0): each column represents a bin's distribution over keys
        key_probs = F.softmax(logits, dim=0)
        return key_probs

    def forward_queries(self, Q, reference_angles, empty_bin_mask=None):
        """
        Process queries: Q -> logits -> bin_probs (softmax over bins)

        Args:
            Q: Query vector(s) of shape (head_dim,) or (num_queries, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)
            empty_bin_mask: Optional bool tensor of shape (num_bins,)
                True for empty bins (will be masked with -inf)

        Returns:
            bin_probs: Probability distribution of shape (num_bins,) or (num_queries, num_bins)
                Each row sums to 1 (softmax over bins)
        """
        logits = self.query_network(Q, reference_angles)

        # Mask empty bins with -inf before softmax
        if empty_bin_mask is not None:
            if logits.dim() == 1:
                logits = logits.masked_fill(empty_bin_mask, float('-inf'))
            else:
                logits = logits.masked_fill(empty_bin_mask.unsqueeze(0), float('-inf'))

        # Softmax over bins (dim=-1): each query selects a bin
        bin_probs = F.softmax(logits, dim=-1)
        return bin_probs

    def compute_reference_angles(self, round_start, round_window=128):
        """Compute reference angles for a round."""
        return self.key_network._compute_reference_angles(round_start, round_window)

    def get_param_count(self):
        """Get parameter count breakdown."""
        key_params = sum(p.numel() for p in self.key_network.parameters())
        query_params = sum(p.numel() for p in self.query_network.parameters())
        return {
            'key_network': key_params,
            'query_network': query_params,
            'total': key_params + query_params
        }


def load_model_inv_freq(model_path, logger=None):
    """
    Load inverse frequency tensor from model's RoPE embedding.

    Args:
        model_path: Path to the model
        logger: Optional logger instance

    Returns:
        inv_freq: Tensor of shape (num_freqs,) or None if loading fails
    """
    try:
        from transformers import AutoConfig
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        rope_scaling = dict(config.rope_scaling or {})
        if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
            rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
        rope_scaling.pop("attn_factor", None)
        config.rope_scaling = rope_scaling

        rotary = Qwen3RotaryEmbedding(config=config, device='cpu')
        inv_freq = rotary.inv_freq.to(dtype=torch.float32)
        if logger:
            logger.info(f"Loaded model inv_freq from {model_path}, shape: {inv_freq.shape}")
        return inv_freq
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load model inv_freq: {e}, using default omega")
        return None


def create_model(config, inv_freq=None):
    """
    Factory function to create Module 2 network.

    Args:
        config: Configuration dict with model parameters
        inv_freq: Optional inverse frequency tensor from model's RoPE.
                  If None, will try to load from model_path in config.

    Returns:
        Module2Network instance
    """
    model_cfg = config.get('model', {})

    # Load inv_freq if not provided
    if inv_freq is None:
        model_path = model_cfg.get('model_path',
            "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B")
        inv_freq = load_model_inv_freq(model_path)

    model = Module2Network(
        num_bins=model_cfg.get('num_bins', 128),
        num_freqs=model_cfg.get('num_freqs', 64),
        num_kernels=model_cfg.get('num_kernels', 3),
        inv_freq=inv_freq
    )
    return model
