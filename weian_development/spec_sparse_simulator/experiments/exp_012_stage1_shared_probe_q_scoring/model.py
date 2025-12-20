"""
Module 2 Multi-Bin Sparse Attention Model - Asymmetric Probe Network (exp_012 stage 1)

Network Architecture:
- SharedProbeLayer: Shared learnable probe vectors with RoPE rotation (no bias)
- Module2KeyNetwork: SharedProbe -> dot product (no bias) -> logits (external softmax dim=0)
- Module2QueryNetwork: SharedProbe -> distance-based scoring -> logits (external softmax dim=-1)

Key features (exp_012 stage 1):
- Shared probe vectors between K and Q networks (parameter sharing)
- K-side: dot product scoring without bias
- Q-side: distance-based scoring with -softplus weight mapping
- Total parameters: 24,704 (16,384 shared probes + 8,192 Q weights + 128 Q bias)
- Initialization: randn / sqrt(head_dim) for probes, ln(e-1) for Q weights_raw
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


class SharedProbeLayer(nn.Module):
    """
    Shared Probe Layer with RoPE rotation (no bias).

    Shared by both K and Q networks. Each probe is a learnable vector
    that gets rotated by RoPE to the reference position.

    Total parameters: 16,384 (128 bins * 128 head_dim)
    """

    def __init__(self, num_bins=128, head_dim=128, base=10000, init_probes=None):
        """
        Args:
            num_bins: Number of output bins (default: 128)
            head_dim: Dimension of key/query vectors (default: 128)
            base: RoPE base (default: 10000)
            init_probes: Optional tensor of shape (num_bins, head_dim) for initialization
                         If provided, use this to initialize probes (e.g., K-means centers)
        """
        super().__init__()
        self.num_bins = num_bins
        self.head_dim = head_dim
        self.base = base
        self.num_freqs = head_dim // 2  # For compatibility with reference_angles interface

        # Probe vectors: each row is a probe vector (base vector, unrotated)
        # Shape: (num_bins, head_dim)
        if init_probes is not None:
            # Use provided initialization (e.g., K-means cluster centers)
            assert init_probes.shape == (num_bins, head_dim), \
                f"init_probes shape mismatch: expected ({num_bins}, {head_dim}), got {init_probes.shape}"
            self.probes = nn.Parameter(init_probes.clone())
        else:
            # Default: randn / sqrt(head_dim) for proper variance scaling
            self.probes = nn.Parameter(
                torch.randn(num_bins, head_dim) / math.sqrt(head_dim)
            )

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

    def get_rotated_probes(self, reference_angles):
        """
        Get RoPE-rotated probes for the given reference angles.

        Args:
            reference_angles: Reference angles of shape (num_freqs,)

        Returns:
            rotated_probes: Tensor of shape (num_bins, head_dim)
        """
        # Extract ref_position from reference_angles
        # Since omega[0] = 1.0, reference_angles[0] = ref_position * 1.0 = ref_position
        ref_pos = reference_angles[0].item()

        # Apply RoPE rotation to all probe vectors
        rotated_probes = apply_rope_rotation(self.probes, ref_pos, self.base)

        return rotated_probes


class DistanceBasedQueryScorer(nn.Module):
    """
    Distance-based Query Scorer for exp_012 stage 1.

    Computes Q-side scores using frequency-wise distance features:
    1. Compute per-frequency error: e_{b,f} = P_rot_{b,f} - Q_f
    2. Compute distance: d_{b,f} = sqrt(||e_{b,f}||^2 + eps)
    3. Linear transform: s_Q^(b) = tilde_w_b^T * d_b + c_b
       where tilde_w_b = -softplus(w_raw_b) to ensure negative weights

    Parameters: 128 * 64 (weights_raw) + 128 (bias) = 8,320
    """

    def __init__(self, num_bins=128, num_freqs=64):
        """
        Args:
            num_bins: Number of bins (default: 128)
            num_freqs: Number of frequency components (default: 64)
        """
        super().__init__()
        self.num_bins = num_bins
        self.num_freqs = num_freqs
        self.eps = 1e-8

        # Raw weights for linear transformation (will be mapped via -softplus)
        # Shape: (num_bins, num_freqs)
        # Initialize to ln(e-1) ≈ 0.541 so that -softplus maps to -1
        init_value = math.log(math.e - 1)
        self.q_weights_raw = nn.Parameter(
            torch.full((num_bins, num_freqs), init_value)
        )

        # Bias for each bin
        # Shape: (num_bins,)
        self.q_bias = nn.Parameter(torch.zeros(num_bins))

    def forward(self, Q, rotated_probes):
        """
        Compute distance-based Q scores.

        Args:
            Q: Query vector(s) of shape (head_dim,) or (num_queries, head_dim)
            rotated_probes: RoPE-rotated probe vectors of shape (num_bins, head_dim)

        Returns:
            scores: Shape (num_bins,) or (num_queries, num_bins)
        """
        is_single = Q.dim() == 1
        if is_single:
            Q = Q.unsqueeze(0)  # (1, head_dim)

        num_queries = Q.shape[0]
        head_dim = Q.shape[1]

        # Reshape to frequency pairs: (*, num_freqs, 2)
        Q_freq = Q.view(num_queries, self.num_freqs, 2)  # (num_queries, num_freqs, 2)
        P_freq = rotated_probes.view(self.num_bins, self.num_freqs, 2)  # (num_bins, num_freqs, 2)

        # Compute per-frequency error vectors
        # e_{b,f} = P_rot_{b,f} - Q_f
        # Broadcasting: (num_bins, num_freqs, 2) - (num_queries, num_freqs, 2)
        # Result: (num_queries, num_bins, num_freqs, 2)
        error = P_freq.unsqueeze(0) - Q_freq.unsqueeze(1)  # (num_queries, num_bins, num_freqs, 2)

        # Compute distance (L2 norm with numerical stability)
        # d_{b,f} = sqrt(||e_{b,f}||^2 + eps)
        distance = torch.sqrt(torch.sum(error ** 2, dim=-1) + self.eps)  # (num_queries, num_bins, num_freqs)

        # Apply -softplus mapping to weights
        # tilde_w = -softplus(w_raw) = -ln(1 + exp(w_raw))
        effective_weights = -F.softplus(self.q_weights_raw)  # (num_bins, num_freqs)

        # Ensure dtype compatibility
        effective_weights = effective_weights.to(dtype=Q.dtype)
        bias = self.q_bias.to(dtype=Q.dtype)

        # Linear transformation: s_Q^(b) = tilde_w_b^T * d_b + c_b
        # distance: (num_queries, num_bins, num_freqs)
        # effective_weights: (num_bins, num_freqs)
        # scores: (num_queries, num_bins)
        scores = torch.sum(distance * effective_weights.unsqueeze(0), dim=-1) + bias

        if is_single:
            scores = scores.squeeze(0)  # (num_bins,)

        return scores


class Module2KeyNetwork(nn.Module):
    """
    Module 2 Key Network for bin assignment (exp_012 stage 1).

    Uses shared probe layer (no bias). Computes K-side scores via dot product:
        s_K^(b) = P_rot_b^T * K

    Input: K (post-RoPE key vectors) of shape (num_keys, head_dim)
    Output: logits of shape (num_keys, num_bins)

    Softmax is applied EXTERNALLY over keys (dim=0):
        key_probs = F.softmax(logits, dim=0)  # Each column sums to 1

    Execution timing: Once per round (round_start)
    """

    def __init__(self, shared_probe_layer):
        """
        Args:
            shared_probe_layer: SharedProbeLayer instance (shared with QueryNetwork)
        """
        super().__init__()
        self.probe_layer = shared_probe_layer

    def forward(self, K, reference_angles):
        """
        Forward pass: K -> dot product with rotated probes -> logits

        Args:
            K: Key vectors of shape (num_keys, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)

        Returns:
            logits: Raw logits of shape (num_keys, num_bins)
                Apply softmax(dim=0) externally to get key_probs
        """
        is_single = K.dim() == 1
        if is_single:
            K = K.unsqueeze(0)

        # Get rotated probes
        rotated_probes = self.probe_layer.get_rotated_probes(reference_angles)

        # Ensure dtype compatibility
        rotated_probes = rotated_probes.to(dtype=K.dtype)

        # Dot product (no bias)
        # K: (num_keys, head_dim)
        # rotated_probes.T: (head_dim, num_bins)
        logits = torch.matmul(K, rotated_probes.t())

        if is_single:
            logits = logits.squeeze(0)

        return logits

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.probe_layer._compute_reference_angles(round_start, round_window)


class Module2QueryNetwork(nn.Module):
    """
    Module 2 Query Network for bin routing (exp_012 stage 1).

    Uses shared probe layer with distance-based scoring:
    1. Compute per-frequency distances between Q and rotated probes
    2. Apply linear transformation with -softplus weights
    3. s_Q^(b) = tilde_w_b^T * d_b + c_b

    Input: Q (post-RoPE query vector) of shape (head_dim,) or (num_queries, head_dim)
    Output: logits of shape (num_bins,) or (num_queries, num_bins)

    Softmax is applied EXTERNALLY over bins (dim=-1):
        bin_probs = F.softmax(logits, dim=-1)  # Each row sums to 1

    Execution timing: Every decoding step
    """

    def __init__(self, shared_probe_layer, num_bins=128, num_freqs=64):
        """
        Args:
            shared_probe_layer: SharedProbeLayer instance (shared with KeyNetwork)
            num_bins: Number of bins (default: 128)
            num_freqs: Number of frequency components (default: 64)
        """
        super().__init__()
        self.probe_layer = shared_probe_layer
        self.distance_scorer = DistanceBasedQueryScorer(num_bins, num_freqs)

    def forward(self, Q, reference_angles):
        """
        Forward pass: Q -> distance computation -> linear transform -> logits

        Args:
            Q: Query vector(s) of shape (head_dim,) or (num_queries, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)

        Returns:
            logits: Raw logits of shape (num_bins,) or (num_queries, num_bins)
                Apply softmax(dim=-1) externally to get bin_probs
        """
        # Get rotated probes
        rotated_probes = self.probe_layer.get_rotated_probes(reference_angles)

        # Compute distance-based scores
        logits = self.distance_scorer(Q, rotated_probes)

        return logits

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.probe_layer._compute_reference_angles(round_start, round_window)


class Module2Network(nn.Module):
    """
    Complete Module 2 Network combining Key and Query networks (exp_012 stage 1).

    Uses shared probe layer between K and Q networks.
    - K network: dot product scoring (no bias)
    - Q network: distance-based scoring with -softplus weights

    Total parameters: 24,704
    - Shared probes: 128 * 128 = 16,384
    - Q weights_raw: 128 * 64 = 8,192
    - Q bias: 128
    """

    def __init__(self, num_bins=128, head_dim=128, init_probes=None):
        """
        Args:
            num_bins: Number of bins (default: 128)
            head_dim: Dimension of key/query vectors (default: 128)
            init_probes: Optional tensor of shape (num_bins, head_dim) for probe initialization
        """
        super().__init__()
        self.num_bins = num_bins
        num_freqs = head_dim // 2

        # Shared probe layer (used by both K and Q networks)
        self.shared_probe_layer = SharedProbeLayer(num_bins, head_dim, init_probes=init_probes)

        # Key and Query networks sharing the same probe layer
        self.key_network = Module2KeyNetwork(self.shared_probe_layer)
        self.query_network = Module2QueryNetwork(self.shared_probe_layer, num_bins, num_freqs)

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
        shared_probe_params = sum(p.numel() for p in self.shared_probe_layer.parameters())
        distance_scorer_params = sum(p.numel() for p in self.query_network.distance_scorer.parameters())

        return {
            'shared_probes': shared_probe_params,
            'q_distance_scorer': distance_scorer_params,
            'total': shared_probe_params + distance_scorer_params
        }


def create_model(config, init_probes=None):
    """
    Factory function to create Module 2 network.

    Args:
        config: Configuration dict with model parameters
            - num_bins: Number of bins (default: 128)
            - num_freqs: Ignored (kept for backward compatibility)
            - num_kernels: Ignored (kept for backward compatibility)
        init_probes: Optional tensor of shape (num_bins, head_dim) for probe initialization
                     If provided, use this to initialize probes (e.g., K-means centers)

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
        head_dim=head_dim,
        init_probes=init_probes
    )
    return model
