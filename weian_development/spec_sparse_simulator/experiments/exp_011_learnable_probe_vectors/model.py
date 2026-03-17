"""
Module 2 Multi-Bin Sparse Attention Model - Learnable Probe Vectors

Network Architecture:
- LearnableProbeLayer: Learnable probe vectors with RoPE rotation (replaces KernelEncodingLayer)
- Module2KeyNetwork: LearnableProbe -> logits (external softmax dim=0)
- Module2QueryNetwork: LearnableProbe -> logits (external softmax dim=-1)

Key changes from exp_006:
- Simplified architecture: direct dot product instead of von Mises kernels
- RoPE rotation applied to probe vectors before dot product
- Parameters reduced from ~74K to ~16.5K per network
- Initialization: randn / sqrt(head_dim) for proper variance scaling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope_rotation(vectors, position, base=10000):
    """
    Apply RoPE rotation to vectors.

    Args:
        vectors: Tensor of shape (num_vectors, head_dim) or (head_dim,)
        position: Target position for rotation (scalar)
        base: RoPE base (default: 10000)

    Returns:
        rotated_vectors: Same shape as input
    """
    is_single = vectors.dim() == 1
    if is_single:
        vectors = vectors.unsqueeze(0)

    num_vectors, head_dim = vectors.shape
    num_freqs = head_dim // 2  # 64 for head_dim=128

    # Compute angular frequencies: omega_j = 1 / (base^(2j/d))
    dim_indices = torch.arange(num_freqs, device=vectors.device, dtype=vectors.dtype)
    omega = 1.0 / (base ** (2 * dim_indices / head_dim))

    # Compute rotation angles: theta_j = pos * omega_j
    theta = position * omega  # shape: (num_freqs,)

    # Compute cos and sin
    cos_theta = torch.cos(theta)  # shape: (num_freqs,)
    sin_theta = torch.sin(theta)  # shape: (num_freqs,)

    # Reshape vectors to complex pairs: (num_vectors, num_freqs, 2)
    vectors_complex = vectors.view(num_vectors, num_freqs, 2)

    # Apply 2D rotation
    # v'_0 = v_0 * cos - v_1 * sin
    # v'_1 = v_0 * sin + v_1 * cos
    rotated = torch.stack([
        vectors_complex[..., 0] * cos_theta - vectors_complex[..., 1] * sin_theta,
        vectors_complex[..., 0] * sin_theta + vectors_complex[..., 1] * cos_theta
    ], dim=-1)

    # Restore original shape
    rotated = rotated.view(num_vectors, head_dim)

    if is_single:
        rotated = rotated.squeeze(0)

    return rotated


class LearnableProbeLayer(nn.Module):
    """
    Learnable Probe Layer with RoPE rotation.

    Each probe is a learnable vector that gets rotated by RoPE
    to the reference position before computing dot product with keys/queries.

    Total parameters: 16,512 (128 bins * 128 head_dim + 128 bias)
    """

    def __init__(self, num_bins=128, head_dim=128, base=10000):
        """
        Args:
            num_bins: Number of output bins (default: 128)
            head_dim: Dimension of key/query vectors (default: 128)
            base: RoPE base (default: 10000)
        """
        super().__init__()
        self.num_bins = num_bins
        self.head_dim = head_dim
        self.base = base
        self.num_freqs = head_dim // 2  # For compatibility with reference_angles interface

        # Probe vectors: each row is a probe vector (base vector, unrotated)
        # Shape: (num_bins, head_dim)
        # Initialization: same as Transformer QK linear layers
        # std = 1 / sqrt(head_dim) for proper dot product variance
        self.probes = nn.Parameter(
            torch.randn(num_bins, head_dim) / math.sqrt(head_dim)
        )

        # Bias for each probe
        # Shape: (num_bins,)
        self.bias = nn.Parameter(torch.zeros(num_bins))

    def _compute_reference_angles(self, round_start, round_window=128, base=10000):
        """
        Compute reference angles from round position using RoPE formula.

        This method maintains compatibility with the original interface.

        Args:
            round_start: Starting position of current round
            round_window: Window size (default: 128)
            base: RoPE base (default: 10000)

        Returns:
            reference_angles: Tensor of shape (num_freqs,)
        """
        # Reference position: middle of the round
        ref_position = round_start + round_window // 2

        # Compute RoPE angular frequencies omega
        dim_indices = torch.arange(0, self.num_freqs, device=self.probes.device)
        omega = 1.0 / (base ** (2 * dim_indices / (self.num_freqs * 2)))

        # Reference angles: ref_position * omega
        reference_angles = ref_position * omega

        return reference_angles

    def forward(self, x, reference_angles):
        """
        Forward pass: apply RoPE rotation to probes, then compute dot product.

        Args:
            x: Input vectors (post-RoPE) of shape (num_vectors, head_dim) or (head_dim,)
            reference_angles: Reference angles of shape (num_freqs,)
                Note: We extract ref_position from reference_angles[0] since omega[0] = 1.0

        Returns:
            logits: Shape (num_vectors, num_bins) or (num_bins,)
        """
        is_single = x.dim() == 1
        if is_single:
            x = x.unsqueeze(0)

        # Extract ref_position from reference_angles
        # Since omega[0] = 1.0, reference_angles[0] = ref_position * 1.0 = ref_position
        ref_pos = reference_angles[0].item()

        # Apply RoPE rotation to all probe vectors
        # probes: (num_bins, head_dim) -> rotated_probes: (num_bins, head_dim)
        rotated_probes = apply_rope_rotation(self.probes, ref_pos, self.base)

        # Ensure dtype compatibility (input may be BFloat16, model is Float32)
        rotated_probes = rotated_probes.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype)

        # Dot product + bias
        # x: (num_vectors, head_dim)
        # rotated_probes.T: (head_dim, num_bins)
        # logits: (num_vectors, num_bins)
        logits = torch.matmul(x, rotated_probes.t()) + bias

        if is_single:
            logits = logits.squeeze(0)

        return logits


class Module2KeyNetwork(nn.Module):
    """
    Module 2 Key Network for bin assignment.

    Architecture: LearnableProbeLayer (replaces KernelEncodingLayer)

    Input: K (post-RoPE key vectors) of shape (num_keys, head_dim)
    Output: logits of shape (num_keys, num_bins)

    Softmax is applied EXTERNALLY over keys (dim=0):
        key_probs = F.softmax(logits, dim=0)  # Each column sums to 1

    Execution timing: Once per round (round_start)
    """

    def __init__(self, num_bins=128, head_dim=128):
        """
        Args:
            num_bins: Number of bins (default: 128)
            head_dim: Dimension of key vectors (default: 128)
        """
        super().__init__()
        self.probe_layer = LearnableProbeLayer(num_bins, head_dim)

    def forward(self, K, reference_angles):
        """
        Forward pass: K -> LearnableProbe -> logits

        Args:
            K: Key vectors of shape (num_keys, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)

        Returns:
            logits: Raw logits of shape (num_keys, num_bins)
                Apply softmax(dim=0) externally to get key_probs
        """
        logits = self.probe_layer(K, reference_angles)
        return logits

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.probe_layer._compute_reference_angles(round_start, round_window)


class Module2QueryNetwork(nn.Module):
    """
    Module 2 Query Network for bin routing.

    Architecture: LearnableProbeLayer (replaces KernelEncodingLayer)
    IDENTICAL architecture to KeyNetwork but SEPARATE parameters.

    Input: Q (post-RoPE query vector) of shape (head_dim,) or (num_queries, head_dim)
    Output: logits of shape (num_bins,) or (num_queries, num_bins)

    Softmax is applied EXTERNALLY over bins (dim=-1):
        bin_probs = F.softmax(logits, dim=-1)  # Each row sums to 1

    Execution timing: Every decoding step
    """

    def __init__(self, num_bins=128, head_dim=128):
        """
        Args:
            num_bins: Number of bins (default: 128)
            head_dim: Dimension of query vectors (default: 128)
        """
        super().__init__()
        self.probe_layer = LearnableProbeLayer(num_bins, head_dim)

    def forward(self, Q, reference_angles):
        """
        Forward pass: Q -> LearnableProbe -> logits

        Args:
            Q: Query vector(s) of shape (head_dim,) or (num_queries, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)

        Returns:
            logits: Raw logits of shape (num_bins,) or (num_queries, num_bins)
                Apply softmax(dim=-1) externally to get bin_probs
        """
        logits = self.probe_layer(Q, reference_angles)
        return logits

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.probe_layer._compute_reference_angles(round_start, round_window)


class Module2Network(nn.Module):
    """
    Complete Module 2 Network combining Key and Query networks.

    This is a convenience wrapper that holds both networks and provides
    unified interface for training and inference.
    """

    def __init__(self, num_bins=128, head_dim=128):
        """
        Args:
            num_bins: Number of bins (default: 128)
            head_dim: Dimension of key/query vectors (default: 128)
        """
        super().__init__()
        self.num_bins = num_bins

        # Separate Key and Query networks with independent parameters
        self.key_network = Module2KeyNetwork(num_bins, head_dim)
        self.query_network = Module2QueryNetwork(num_bins, head_dim)

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


def create_model(config):
    """
    Factory function to create Module 2 network.

    Args:
        config: Configuration dict with model parameters
            - num_bins: Number of bins (default: 128)
            - num_freqs: Ignored (kept for backward compatibility)
            - num_kernels: Ignored (kept for backward compatibility)

    Returns:
        Module2Network instance
    """
    model_cfg = config.get('model', {})

    # Get num_bins from config
    num_bins = model_cfg.get('num_bins', 128)

    # Derive head_dim from num_freqs if available, otherwise use default
    # head_dim = num_freqs * 2 (since num_freqs = head_dim // 2)
    num_freqs = model_cfg.get('num_freqs', 64)
    head_dim = num_freqs * 2  # 128 by default

    model = Module2Network(
        num_bins=num_bins,
        head_dim=head_dim
    )
    return model
