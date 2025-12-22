"""
Module 2 Multi-Bin Sparse Attention Model - Asymmetric Probe Network (exp_012 stage 3)

Network Architecture:
- SharedProbeLayer: Shared learnable probe vectors with L2 normalization + RoPE rotation
- Module2KeyNetwork: SharedProbe (normalized) -> dot product + magnitude term + bias -> logits (external softmax dim=0)
- Module2QueryNetwork: SharedProbe (normalized) -> distance-based + magnitude term -> logits (external softmax dim=-1)

Stage 3 features (L2 normalization):
- Q-side: L2 normalize Q to unit norm before distance computation
- Q-side: L2 normalize probes before RoPE rotation (for Q network)
- K-side: L2 normalize probes before RoPE rotation (for K network)
- K-side magnitude: Still uses original K (NOT normalized)
- Q-side magnitude: Uses normalized Q

Scoring formulas:
- K-side: s_K^(b) = P_normalized_rot_b^T * K + u_b^T * m^K + k_bias_b
- Q-side: s_Q^(b) = tilde_w_b^T * d_b + v_b^T * m^Q_normalized + c_b
- Distance: d_{b,f} = ||P_normalized_rot_{b,f} - Q_normalized_f||
- Magnitude: m_f = sqrt(x_{2f}^2 + x_{2f+1}^2)

Total parameters: 41,216
- Shared probes: 128 * 128 = 16,384
- Q weights_raw: 128 * 64 = 8,192
- Q bias: 128
- K magnitude weights u: 128 * 64 = 8,192
- Q magnitude weights v: 128 * 64 = 8,192
- K bias: 128
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x, eps=1e-8):
    """
    L2 normalize vectors to unit norm.

    Args:
        x: Tensor of shape (..., dim)
        eps: Small value for numerical stability

    Returns:
        normalized: Tensor of same shape with L2 norm = 1
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return x / (norm + eps)


def compute_magnitude_features(x, num_freqs, eps=1e-8):
    """
    Compute per-frequency magnitude features.

    m_f = sqrt(real_f^2 + imag_f^2 + eps)

    Vector layout: [real_0...real_63, imag_0...imag_63] (front/back, NOT interleaved)

    Args:
        x: Tensor of shape (..., head_dim) where head_dim = num_freqs * 2
        num_freqs: Number of frequency components (F = head_dim // 2)
        eps: Small value for numerical stability

    Returns:
        magnitude: Tensor of shape (..., num_freqs)
    """
    # Split into real and imag parts (front/back layout)
    real_part = x[..., :num_freqs]  # (..., num_freqs)
    imag_part = x[..., num_freqs:]  # (..., num_freqs)

    # Compute magnitude: sqrt(real^2 + imag^2 + eps)
    magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2 + eps)  # (..., num_freqs)

    return magnitude


def rotate_half(x):
    """
    Rotate half of the hidden dims: [a, b] -> [-b, a]
    Vector layout: [real_0...real_63, imag_0...imag_63] (front/back)
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]   # real part
    x2 = x[..., d:]   # imag part
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_rotation(vectors, position, inv_freq=None, base=10000):
    """
    Apply RoPE rotation to vectors using front/back layout.

    Vector layout: [real_0...real_63, imag_0...imag_63] (NOT interleaved)
    This matches transformers/Qwen3RotaryEmbedding.

    Args:
        vectors: Tensor of shape (num_vectors, head_dim) or (head_dim,)
        position: Target position for rotation (scalar)
        inv_freq: Optional inverse frequency tensor from model's RoPE.
                  If None, computes omega from base.
        base: RoPE base (default: 10000), only used if inv_freq is None

    Returns:
        rotated_vectors: Same shape as input
    """
    is_single = vectors.dim() == 1
    if is_single:
        vectors = vectors.unsqueeze(0)

    num_vectors, head_dim = vectors.shape
    num_freqs = head_dim // 2  # 64 for head_dim=128

    # Use provided inv_freq or compute from base
    if inv_freq is not None:
        omega = inv_freq.to(device=vectors.device, dtype=vectors.dtype)
    else:
        dim_indices = torch.arange(num_freqs, device=vectors.device, dtype=vectors.dtype)
        omega = 1.0 / (base ** (2 * dim_indices / head_dim))

    # Compute rotation angles: theta_j = pos * omega_j
    theta = position * omega  # shape: (num_freqs,)

    # Expand to full head_dim: [theta_0...theta_63, theta_0...theta_63]
    theta_full = torch.cat([theta, theta], dim=-1)
    cos_theta = torch.cos(theta_full)
    sin_theta = torch.sin(theta_full)

    # Apply RoPE: y = x * cos + rotate_half(x) * sin
    rotated = vectors * cos_theta + rotate_half(vectors) * sin_theta

    if is_single:
        rotated = rotated.squeeze(0)

    return rotated


def apply_rope_rotation_batched(vectors, positions, inv_freq=None, base=10000):
    """
    Apply RoPE rotation to vectors for multiple positions at once (batched).

    Args:
        vectors: Tensor of shape (num_vectors, head_dim)
        positions: Tensor of shape (batch_size,) - different positions for each batch
        inv_freq: Optional inverse frequency tensor from model's RoPE.
        base: RoPE base (default: 10000)

    Returns:
        rotated_vectors: Tensor of shape (batch_size, num_vectors, head_dim)
    """
    num_vectors, head_dim = vectors.shape
    batch_size = positions.shape[0]
    num_freqs = head_dim // 2

    # Use provided inv_freq or compute from base
    if inv_freq is not None:
        omega = inv_freq.to(device=vectors.device, dtype=vectors.dtype)
    else:
        dim_indices = torch.arange(num_freqs, device=vectors.device, dtype=vectors.dtype)
        omega = 1.0 / (base ** (2 * dim_indices / head_dim))

    # Compute rotation angles for all positions: theta[b, f] = positions[b] * omega[f]
    # positions: (batch_size,) -> (batch_size, 1)
    # omega: (num_freqs,) -> (1, num_freqs)
    theta = positions.unsqueeze(1) * omega.unsqueeze(0)  # (batch_size, num_freqs)

    # Expand to full head_dim: [theta_0...theta_63, theta_0...theta_63]
    theta_full = torch.cat([theta, theta], dim=-1)  # (batch_size, head_dim)
    cos_theta = torch.cos(theta_full)  # (batch_size, head_dim)
    sin_theta = torch.sin(theta_full)  # (batch_size, head_dim)

    # Expand vectors for batch: (num_vectors, head_dim) -> (1, num_vectors, head_dim)
    vectors_expanded = vectors.unsqueeze(0)  # (1, num_vectors, head_dim)

    # Expand cos/sin for vectors: (batch_size, head_dim) -> (batch_size, 1, head_dim)
    cos_theta = cos_theta.unsqueeze(1)  # (batch_size, 1, head_dim)
    sin_theta = sin_theta.unsqueeze(1)  # (batch_size, 1, head_dim)

    # Apply RoPE: y = x * cos + rotate_half(x) * sin
    rotated = vectors_expanded * cos_theta + rotate_half(vectors_expanded) * sin_theta
    # Result: (batch_size, num_vectors, head_dim)

    return rotated


class SharedProbeLayer(nn.Module):
    """
    Shared Probe Layer with RoPE rotation (no bias).

    Shared by both K and Q networks. Each probe is a learnable vector
    that gets rotated by RoPE to the reference position.

    Total parameters: 16,384 (128 bins * 128 head_dim)
    """

    def __init__(self, num_bins=128, head_dim=128, base=10000, init_probes=None, inv_freq=None):
        """
        Args:
            num_bins: Number of output bins (default: 128)
            head_dim: Dimension of key/query vectors (default: 128)
            base: RoPE base (default: 10000), only used if inv_freq is None
            init_probes: Optional tensor of shape (num_bins, head_dim) for initialization
                         If provided, use this to initialize probes (e.g., K-means centers)
            inv_freq: Optional inverse frequency tensor from model's RoPE.
                      If provided, uses this instead of computing from base.
        """
        super().__init__()
        self.num_bins = num_bins
        self.head_dim = head_dim
        self.base = base
        self.num_freqs = head_dim // 2  # For compatibility with reference_angles interface

        # Store inv_freq as a buffer (not a parameter)
        if inv_freq is not None:
            self.register_buffer('inv_freq', inv_freq.clone())
        else:
            self.inv_freq = None

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

    def get_rotated_probes(self, reference_angles, normalize=False):
        """
        Get RoPE-rotated probes for the given reference angles.

        Args:
            reference_angles: Reference angles of shape (num_freqs,)
            normalize: If True, L2 normalize probes before rotation

        Returns:
            rotated_probes: Tensor of shape (num_bins, head_dim)
        """
        # Extract ref_position from reference_angles
        # Since omega[0] = 1.0, reference_angles[0] = ref_position * 1.0 = ref_position
        ref_pos = reference_angles[0].item()

        # Optionally normalize probes to unit norm before rotation
        if normalize:
            probes_to_rotate = l2_normalize(self.probes)
        else:
            probes_to_rotate = self.probes

        # Apply RoPE rotation to all probe vectors
        rotated_probes = apply_rope_rotation(probes_to_rotate, ref_pos, inv_freq=self.inv_freq, base=self.base)

        return rotated_probes

    def get_rotated_probes_batched(self, ref_positions, normalize=False):
        """
        Get RoPE-rotated probes for multiple reference positions at once (batched).

        Args:
            ref_positions: Tensor of shape (batch_size,) - reference positions for each batch item
            normalize: If True, L2 normalize probes before rotation

        Returns:
            rotated_probes_batch: Tensor of shape (batch_size, num_bins, head_dim)
        """
        # Optionally normalize probes to unit norm before rotation
        if normalize:
            probes_to_rotate = l2_normalize(self.probes)
        else:
            probes_to_rotate = self.probes

        # Apply batched RoPE rotation
        rotated_probes_batch = apply_rope_rotation_batched(
            probes_to_rotate, ref_positions,
            inv_freq=self.inv_freq, base=self.base
        )

        return rotated_probes_batch


class DistanceBasedQueryScorer(nn.Module):
    """
    Distance-based Query Scorer for exp_012 stage 3.

    Computes Q-side scores using frequency-wise distance features + magnitude + optional error vector:
    1. L2 normalize Q to unit norm
    2. Compute per-frequency error: e_{b,f} = P_rot_{b,f} - Q_normalized_f
    3. Compute distance: d_{b,f} = sqrt(||e_{b,f}||^2 + eps)
    4. Compute Q magnitude: m^Q_f = sqrt(Q_{2f}^2 + Q_{2f+1}^2 + eps)
    5. Linear transform: s_Q^(b) = tilde_w_b^T * d_b + v_b^T * m^Q + [alpha_b^T * e_b] + c_b
       where tilde_w_b = -softplus(w_raw_b) to ensure negative weights
       and [alpha_b^T * e_b] is optional (use_error_term=True)

    Parameters (use_error_term=False): 128 * 64 + 128 * 64 + 128 = 16,512
    Parameters (use_error_term=True): 128 * 64 + 128 * 64 + 128 * 128 + 128 = 32,896
    """

    def __init__(self, num_bins=128, num_freqs=64, use_l2_norm=True, use_error_term=False):
        """
        Args:
            num_bins: Number of bins (default: 128)
            num_freqs: Number of frequency components (default: 64)
            use_l2_norm: If True (default), L2 normalize Q before distance computation
            use_error_term: If True, add error vector term alpha_b^T * e_b (default: False)
        """
        super().__init__()
        self.num_bins = num_bins
        self.num_freqs = num_freqs
        self.head_dim = num_freqs * 2
        self.eps = 1e-8
        self.use_l2_norm = use_l2_norm
        self.use_error_term = use_error_term

        # Raw weights for distance transformation (will be mapped via -softplus)
        # Shape: (num_bins, num_freqs)
        # Initialize to ln(e-1) ≈ 0.541 so that -softplus maps to -1
        init_value = math.log(math.e - 1)
        self.q_weights_raw = nn.Parameter(
            torch.full((num_bins, num_freqs), init_value)
        )

        # Q magnitude weights (Stage 2)
        # Shape: (num_bins, num_freqs)
        # Initialize to 0 so Stage 2 starts equivalent to Stage 1
        self.q_magnitude_weights = nn.Parameter(
            torch.zeros(num_bins, num_freqs)
        )

        # Error vector weights (Stage 4) - optional
        # Shape: (num_bins, head_dim)
        # Initialize to 0 so it starts equivalent to without this term
        if use_error_term:
            self.q_error_weights = nn.Parameter(
                torch.zeros(num_bins, self.head_dim)
            )

        # Bias for each bin
        # Shape: (num_bins,)
        self.q_bias = nn.Parameter(torch.zeros(num_bins))

    def forward(self, Q, rotated_probes):
        """
        Compute distance-based Q scores with magnitude term.

        Stage 3: Input normalization
        - Q is L2 normalized to unit norm before distance computation
        - rotated_probes should already be normalized
        - Magnitude features computed on normalized Q

        Args:
            Q: Query vector(s) of shape (head_dim,) or (num_queries, head_dim)
            rotated_probes: RoPE-rotated probe vectors of shape (num_bins, head_dim)
                           (should be pre-normalized)

        Returns:
            scores: Shape (num_bins,) or (num_queries, num_bins)
        """
        is_single = Q.dim() == 1
        if is_single:
            Q = Q.unsqueeze(0)  # (1, head_dim)

        num_queries = Q.shape[0]
        head_dim = Q.shape[1]

        # Optionally L2 normalize Q to unit norm
        if self.use_l2_norm:
            Q_processed = l2_normalize(Q, self.eps)
        else:
            Q_processed = Q

        # Split into real and imag parts (front/back layout)
        # Vector layout: [real_0...real_63, imag_0...imag_63]
        Q_real = Q_processed[:, :self.num_freqs]  # (num_queries, num_freqs)
        Q_imag = Q_processed[:, self.num_freqs:]  # (num_queries, num_freqs)
        P_real = rotated_probes[:, :self.num_freqs]  # (num_bins, num_freqs)
        P_imag = rotated_probes[:, self.num_freqs:]  # (num_bins, num_freqs)

        # Compute per-frequency error vectors
        # e_{b,f} = P_rot_{b,f} - Q_f (in complex space)
        # Broadcasting: (num_bins, num_freqs) - (num_queries, num_freqs)
        error_real = P_real.unsqueeze(0) - Q_real.unsqueeze(1)  # (num_queries, num_bins, num_freqs)
        error_imag = P_imag.unsqueeze(0) - Q_imag.unsqueeze(1)  # (num_queries, num_bins, num_freqs)

        # Compute distance (L2 norm with numerical stability)
        # d_{b,f} = sqrt(error_real^2 + error_imag^2 + eps)
        distance = torch.sqrt(error_real ** 2 + error_imag ** 2 + self.eps)  # (num_queries, num_bins, num_freqs)

        # Compute Q magnitude features (on processed Q, normalized or not)
        # m^Q_f = sqrt(real_f^2 + imag_f^2 + eps)
        Q_magnitude = compute_magnitude_features(Q_processed, self.num_freqs, self.eps)  # (num_queries, num_freqs)

        # Apply -softplus mapping to distance weights
        # tilde_w = -softplus(w_raw) = -ln(1 + exp(w_raw))
        effective_weights = -F.softplus(self.q_weights_raw)  # (num_bins, num_freqs)

        # Ensure dtype compatibility
        effective_weights = effective_weights.to(dtype=Q.dtype)
        v_weights = self.q_magnitude_weights.to(dtype=Q.dtype)
        bias = self.q_bias.to(dtype=Q.dtype)

        # Linear transformation:
        # s_Q^(b) = tilde_w_b^T * d_b + v_b^T * m^Q + [alpha_b^T * e_b] + c_b

        # Distance term: (num_queries, num_bins, num_freqs) * (num_bins, num_freqs) -> (num_queries, num_bins)
        distance_term = torch.sum(distance * effective_weights.unsqueeze(0), dim=-1)

        # Magnitude term: (num_queries, num_freqs) @ (num_bins, num_freqs).T -> (num_queries, num_bins)
        magnitude_term = torch.matmul(Q_magnitude, v_weights.t())

        scores = distance_term + magnitude_term + bias

        # Optional error vector term
        if self.use_error_term:
            # Concatenate error vectors to full head_dim: [error_real, error_imag]
            # Shape: (num_queries, num_bins, head_dim)
            error_full = torch.cat([error_real, error_imag], dim=-1)
            error_weights = self.q_error_weights.to(dtype=error_full.dtype)
            # einsum('qbd,bd->qb'): (num_queries, num_bins, head_dim) dot (num_bins, head_dim) -> (num_queries, num_bins)
            error_term = torch.einsum('qbd,bd->qb', error_full, error_weights)
            scores = scores + error_term

        if is_single:
            scores = scores.squeeze(0)  # (num_bins,)

        return scores

    def forward_batched(self, Q_batch, rotated_probes_batch):
        """
        Batched forward pass for multiple rounds with different rotated probes.

        Args:
            Q_batch: Query vectors of shape (batch_size, num_queries, head_dim)
            rotated_probes_batch: RoPE-rotated probes of shape (batch_size, num_bins, head_dim)

        Returns:
            scores: Shape (batch_size, num_queries, num_bins)
        """
        batch_size, num_queries, head_dim = Q_batch.shape
        num_bins = rotated_probes_batch.shape[1]

        # Optionally L2 normalize Q to unit norm
        if self.use_l2_norm:
            Q_processed = l2_normalize(Q_batch, self.eps)
        else:
            Q_processed = Q_batch

        # Split into real and imag parts (front/back layout)
        Q_real = Q_processed[..., :self.num_freqs]  # (batch_size, num_queries, num_freqs)
        Q_imag = Q_processed[..., self.num_freqs:]  # (batch_size, num_queries, num_freqs)
        P_real = rotated_probes_batch[..., :self.num_freqs]  # (batch_size, num_bins, num_freqs)
        P_imag = rotated_probes_batch[..., self.num_freqs:]  # (batch_size, num_bins, num_freqs)

        # Compute per-frequency error vectors
        # For each batch: e_{b,q,bin,f} = P[bin,f] - Q[q,f]
        # Q_real: (batch_size, num_queries, num_freqs) -> (batch_size, num_queries, 1, num_freqs)
        # P_real: (batch_size, num_bins, num_freqs) -> (batch_size, 1, num_bins, num_freqs)
        error_real = P_real.unsqueeze(1) - Q_real.unsqueeze(2)  # (batch_size, num_queries, num_bins, num_freqs)
        error_imag = P_imag.unsqueeze(1) - Q_imag.unsqueeze(2)

        # Compute distance
        distance = torch.sqrt(error_real ** 2 + error_imag ** 2 + self.eps)

        # Compute Q magnitude features
        Q_magnitude = compute_magnitude_features(Q_processed, self.num_freqs, self.eps)  # (batch_size, num_queries, num_freqs)

        # Apply -softplus mapping to distance weights
        effective_weights = -F.softplus(self.q_weights_raw)  # (num_bins, num_freqs)

        # Ensure dtype compatibility
        effective_weights = effective_weights.to(dtype=Q_batch.dtype)
        v_weights = self.q_magnitude_weights.to(dtype=Q_batch.dtype)
        bias = self.q_bias.to(dtype=Q_batch.dtype)

        # Distance term: (batch_size, num_queries, num_bins, num_freqs) * (num_bins, num_freqs)
        # -> (batch_size, num_queries, num_bins)
        distance_term = torch.sum(distance * effective_weights.unsqueeze(0).unsqueeze(0), dim=-1)

        # Magnitude term: (batch_size, num_queries, num_freqs) @ (num_bins, num_freqs).T
        # -> (batch_size, num_queries, num_bins)
        magnitude_term = torch.matmul(Q_magnitude, v_weights.t())

        scores = distance_term + magnitude_term + bias

        # Optional error vector term
        if self.use_error_term:
            # Concatenate error vectors to full head_dim: [error_real, error_imag]
            # Shape: (batch_size, num_queries, num_bins, head_dim)
            error_full = torch.cat([error_real, error_imag], dim=-1)
            error_weights = self.q_error_weights.to(dtype=error_full.dtype)
            # einsum('bqnd,nd->bqn'): (batch_size, num_queries, num_bins, head_dim) dot (num_bins, head_dim)
            # -> (batch_size, num_queries, num_bins)
            error_term = torch.einsum('bqnd,nd->bqn', error_full, error_weights)
            scores = scores + error_term

        return scores


class Module2KeyNetwork(nn.Module):
    """
    Module 2 Key Network for bin assignment (exp_012 stage 3).

    Uses shared probe layer with L2 normalized probes + magnitude term and bias.
    Computes K-side scores:
        s_K^(b) = P_normalized_rot_b^T * K + u_b^T * m^K + k_bias_b

    Note: Only probes are normalized, K is NOT normalized.
    Magnitude features still use original K (not normalized).

    Input: K (post-RoPE key vectors) of shape (num_keys, head_dim)
    Output: logits of shape (num_keys, num_bins)

    Softmax is applied EXTERNALLY over keys (dim=0):
        key_probs = F.softmax(logits, dim=0)  # Each column sums to 1

    Execution timing: Once per round (round_start)

    Parameters: 128 * 64 (u_weights) + 128 (k_bias) = 8,320
    """

    def __init__(self, shared_probe_layer, num_bins=128, num_freqs=64, use_l2_norm=True):
        """
        Args:
            shared_probe_layer: SharedProbeLayer instance (shared with QueryNetwork)
            num_bins: Number of bins (default: 128)
            num_freqs: Number of frequency components (default: 64)
            use_l2_norm: If True (default), L2 normalize probes before rotation
        """
        super().__init__()
        self.probe_layer = shared_probe_layer
        self.num_bins = num_bins
        self.num_freqs = num_freqs
        self.eps = 1e-8
        self.use_l2_norm = use_l2_norm

        # K magnitude weights (Stage 2)
        # Shape: (num_bins, num_freqs)
        # Initialize to 0 so Stage 2 starts equivalent to Stage 1
        self.k_magnitude_weights = nn.Parameter(
            torch.zeros(num_bins, num_freqs)
        )

        # K bias (Stage 2)
        # Shape: (num_bins,)
        # Initialize to 0
        self.k_bias = nn.Parameter(torch.zeros(num_bins))

    def forward(self, K, reference_angles):
        """
        Forward pass: K -> dot product + magnitude term + bias -> logits

        Stage 3: Probes are L2 normalized before rotation.
        K is NOT normalized - magnitude features use original K.

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

        # Get rotated probes (optionally with L2 normalization)
        rotated_probes = self.probe_layer.get_rotated_probes(reference_angles, normalize=self.use_l2_norm)

        # Ensure dtype compatibility
        rotated_probes = rotated_probes.to(dtype=K.dtype)
        u_weights = self.k_magnitude_weights.to(dtype=K.dtype)
        bias = self.k_bias.to(dtype=K.dtype)

        # Dot product term: s_K^(b) = P_normalized_rot_b^T * K (or unnormalized if use_l2_norm=False)
        # K: (num_keys, head_dim)
        # rotated_probes.T: (head_dim, num_bins)
        dot_product_term = torch.matmul(K, rotated_probes.t())  # (num_keys, num_bins)

        # Compute K magnitude features (Stage 2) - still uses original K (NOT normalized)
        # m^K_f = sqrt(K_{2f}^2 + K_{2f+1}^2 + eps)
        K_magnitude = compute_magnitude_features(K, self.num_freqs, self.eps)  # (num_keys, num_freqs)

        # Magnitude term: u_b^T * m^K
        # K_magnitude: (num_keys, num_freqs)
        # u_weights: (num_bins, num_freqs)
        # result: (num_keys, num_bins)
        magnitude_term = torch.matmul(K_magnitude, u_weights.t())

        # Final score: s_K^(b) = P_normalized_rot_b^T * K + u_b^T * m^K + k_bias_b
        logits = dot_product_term + magnitude_term + bias

        if is_single:
            logits = logits.squeeze(0)

        return logits

    def forward_batched(self, K, ref_positions, key_lengths=None):
        """
        Batched forward pass for multiple rounds with different reference positions.

        This computes key logits for multiple rounds in parallel. All rounds share
        the same K data but use different rotated probes (due to different ref_positions).

        Args:
            K: Key vectors of shape (max_keys, head_dim) - shared across all rounds
            ref_positions: Tensor of shape (batch_size,) - reference positions for each round
            key_lengths: Optional tensor of shape (batch_size,) - valid key length for each round
                         If provided, creates a mask for keys beyond each round's valid range.

        Returns:
            logits: Raw logits of shape (batch_size, max_keys, num_bins)
                    Apply masked softmax(dim=1) externally to get key_probs
            key_mask: Bool tensor of shape (batch_size, max_keys)
                      True for valid positions, False for masked positions (if key_lengths provided)
        """
        max_keys, head_dim = K.shape
        batch_size = ref_positions.shape[0]

        # Get batched rotated probes: (batch_size, num_bins, head_dim)
        rotated_probes_batch = self.probe_layer.get_rotated_probes_batched(
            ref_positions, normalize=self.use_l2_norm
        )

        # Ensure dtype compatibility
        rotated_probes_batch = rotated_probes_batch.to(dtype=K.dtype)
        u_weights = self.k_magnitude_weights.to(dtype=K.dtype)
        bias = self.k_bias.to(dtype=K.dtype)

        # Batched dot product using einsum:
        # K: (max_keys, head_dim)
        # rotated_probes_batch: (batch_size, num_bins, head_dim)
        # Result: (batch_size, max_keys, num_bins)
        dot_product_term = torch.einsum('kd,rbd->rkb', K, rotated_probes_batch)

        # Compute K magnitude features (same for all rounds since K is shared)
        # K_magnitude: (max_keys, num_freqs)
        K_magnitude = compute_magnitude_features(K, self.num_freqs, self.eps)

        # Magnitude term: (max_keys, num_bins) - same for all rounds
        magnitude_term = torch.matmul(K_magnitude, u_weights.t())

        # Expand magnitude_term and bias for batch dimension
        # (max_keys, num_bins) -> (1, max_keys, num_bins) -> (batch_size, max_keys, num_bins)
        magnitude_term = magnitude_term.unsqueeze(0).expand(batch_size, -1, -1)
        bias_expanded = bias.unsqueeze(0).unsqueeze(0).expand(batch_size, max_keys, -1)

        # Final score
        logits = dot_product_term + magnitude_term + bias_expanded

        # Create key mask if key_lengths provided
        key_mask = None
        if key_lengths is not None:
            # key_mask[b, k] = True if k < key_lengths[b]
            key_indices = torch.arange(max_keys, device=K.device).unsqueeze(0)  # (1, max_keys)
            key_lengths_expanded = key_lengths.unsqueeze(1)  # (batch_size, 1)
            key_mask = key_indices < key_lengths_expanded  # (batch_size, max_keys)

        return logits, key_mask

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.probe_layer._compute_reference_angles(round_start, round_window)


class Module2QueryNetwork(nn.Module):
    """
    Module 2 Query Network for bin routing (exp_012 stage 3).

    Uses shared probe layer with distance-based scoring + L2 normalization:
    1. L2 normalize Q to unit norm
    2. L2 normalize probes before RoPE rotation
    3. Compute per-frequency distances between normalized Q and normalized probes
    4. Apply linear transformation with -softplus weights
    5. s_Q^(b) = tilde_w_b^T * d_b + v_b^T * m^Q_normalized + [alpha_b^T * e_b] + c_b

    Input: Q (post-RoPE query vector) of shape (head_dim,) or (num_queries, head_dim)
    Output: logits of shape (num_bins,) or (num_queries, num_bins)

    Softmax is applied EXTERNALLY over bins (dim=-1):
        bin_probs = F.softmax(logits, dim=-1)  # Each row sums to 1

    Execution timing: Every decoding step
    """

    def __init__(self, shared_probe_layer, num_bins=128, num_freqs=64, use_l2_norm=True, use_error_term=False):
        """
        Args:
            shared_probe_layer: SharedProbeLayer instance (shared with KeyNetwork)
            num_bins: Number of bins (default: 128)
            num_freqs: Number of frequency components (default: 64)
            use_l2_norm: If True (default), L2 normalize Q and probes
            use_error_term: If True, add error vector term (default: False)
        """
        super().__init__()
        self.probe_layer = shared_probe_layer
        self.use_l2_norm = use_l2_norm
        self.distance_scorer = DistanceBasedQueryScorer(num_bins, num_freqs, use_l2_norm=use_l2_norm, use_error_term=use_error_term)

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
        # Get rotated probes (optionally with L2 normalization)
        rotated_probes = self.probe_layer.get_rotated_probes(reference_angles, normalize=self.use_l2_norm)

        # Compute distance-based scores (Q is normalized inside distance_scorer)
        logits = self.distance_scorer(Q, rotated_probes)

        return logits

    def forward_batched(self, Q_batch, ref_positions):
        """
        Batched forward pass for multiple rounds with different reference positions.

        Args:
            Q_batch: Query vectors of shape (batch_size, num_queries, head_dim)
            ref_positions: Tensor of shape (batch_size,) - reference positions for each round

        Returns:
            logits: Shape (batch_size, num_queries, num_bins)
                Apply softmax(dim=-1) externally to get bin_probs
        """
        # Get batched rotated probes: (batch_size, num_bins, head_dim)
        rotated_probes_batch = self.probe_layer.get_rotated_probes_batched(
            ref_positions, normalize=self.use_l2_norm
        )

        # Compute distance-based scores
        logits = self.distance_scorer.forward_batched(Q_batch, rotated_probes_batch)

        return logits

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.probe_layer._compute_reference_angles(round_start, round_window)


class Module2Network(nn.Module):
    """
    Complete Module 2 Network combining Key and Query networks (exp_012 stage 2).

    Uses shared probe layer between K and Q networks.
    - K network: dot product + magnitude term + bias
    - Q network: distance-based + magnitude term + [error vector term] + bias

    Total parameters (use_error_term=False): 41,216
    Total parameters (use_error_term=True): 57,600 (+16,384)
    - Shared probes: 128 * 128 = 16,384
    - Q weights_raw: 128 * 64 = 8,192
    - Q magnitude weights v: 128 * 64 = 8,192
    - Q error weights alpha: 128 * 128 = 16,384 (optional)
    - Q bias: 128
    - K magnitude weights u: 128 * 64 = 8,192
    - K bias: 128
    """

    def __init__(self, num_bins=128, head_dim=128, init_probes=None, use_l2_norm=True, inv_freq=None, use_error_term=False):
        """
        Args:
            num_bins: Number of bins (default: 128)
            head_dim: Dimension of key/query vectors (default: 128)
            init_probes: Optional tensor of shape (num_bins, head_dim) for probe initialization
            use_l2_norm: If True (default), L2 normalize probes and Q vectors
            inv_freq: Optional inverse frequency tensor from model's RoPE.
            use_error_term: If True, add error vector term to Q network (default: False)
        """
        super().__init__()
        self.num_bins = num_bins
        self.use_l2_norm = use_l2_norm
        self.use_error_term = use_error_term
        num_freqs = head_dim // 2

        # Shared probe layer (used by both K and Q networks)
        self.shared_probe_layer = SharedProbeLayer(num_bins, head_dim, init_probes=init_probes, inv_freq=inv_freq)

        # Key and Query networks sharing the same probe layer
        self.key_network = Module2KeyNetwork(self.shared_probe_layer, num_bins, num_freqs, use_l2_norm=use_l2_norm)
        self.query_network = Module2QueryNetwork(self.shared_probe_layer, num_bins, num_freqs, use_l2_norm=use_l2_norm, use_error_term=use_error_term)

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
        # L2 normalize Q to unit norm as the first step
        Q = l2_normalize(Q)

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

    def forward_keys_batched(self, K, ref_positions, key_lengths=None, return_logits=False):
        """
        Batched forward pass for keys with multiple rounds.

        All rounds share the same K data but use different rotated probes.

        Args:
            K: Key vectors of shape (max_keys, head_dim) - shared across all rounds
            ref_positions: Tensor of shape (batch_size,) - reference positions for each round
            key_lengths: Optional tensor of shape (batch_size,) - valid key length for each round
            return_logits: If True, also return raw logits (for rank-based loss)

        Returns:
            key_probs: Probability distribution of shape (batch_size, max_keys, num_bins)
                       For each round, only positions < key_lengths[i] are valid.
                       Invalid positions have near-zero probability.
            key_mask: Bool tensor of shape (batch_size, max_keys) if key_lengths provided
            key_logits: (optional) Raw logits if return_logits=True
        """
        # Get logits and mask
        logits, key_mask = self.key_network.forward_batched(K, ref_positions, key_lengths)

        # Store raw logits before masking (for rank-based loss)
        raw_logits = logits.clone() if return_logits else None

        # Apply mask before softmax (set invalid positions to -inf)
        if key_mask is not None:
            # key_mask: (batch_size, max_keys) - True for valid, False for invalid
            # Need to expand for num_bins: (batch_size, max_keys, 1)
            logits = logits.masked_fill(~key_mask.unsqueeze(-1), float('-inf'))

        # Softmax over keys (dim=1)
        key_probs = F.softmax(logits, dim=1)

        if return_logits:
            return key_probs, key_mask, raw_logits
        return key_probs, key_mask

    def forward_queries_batched(self, Q_batch, ref_positions, empty_bin_mask_batch=None, return_logits=False):
        """
        Batched forward pass for queries with multiple rounds.

        Args:
            Q_batch: Query vectors of shape (batch_size, num_queries, head_dim)
            ref_positions: Tensor of shape (batch_size,) - reference positions for each round
            empty_bin_mask_batch: Optional bool tensor of shape (batch_size, num_bins)
                                  True for empty bins (will be masked with -inf)
            return_logits: If True, also return raw logits (for rank-based loss)

        Returns:
            bin_probs: Probability distribution of shape (batch_size, num_queries, num_bins)
                       Each row sums to 1 (softmax over bins)
            query_logits: (optional) Raw logits if return_logits=True
        """
        # L2 normalize Q to unit norm as the first step
        Q_batch = l2_normalize(Q_batch)

        # Get logits: (batch_size, num_queries, num_bins)
        logits = self.query_network.forward_batched(Q_batch, ref_positions)

        # Store raw logits before masking (for rank-based loss)
        raw_logits = logits.clone() if return_logits else None

        # Mask empty bins with -inf before softmax
        if empty_bin_mask_batch is not None:
            # empty_bin_mask_batch: (batch_size, num_bins) -> (batch_size, 1, num_bins)
            logits = logits.masked_fill(empty_bin_mask_batch.unsqueeze(1), float('-inf'))

        # Softmax over bins (dim=-1)
        bin_probs = F.softmax(logits, dim=-1)

        if return_logits:
            return bin_probs, raw_logits
        return bin_probs

    def get_param_count(self):
        """Get parameter count breakdown."""
        shared_probe_params = sum(p.numel() for p in self.shared_probe_layer.parameters())
        k_magnitude_params = self.key_network.k_magnitude_weights.numel()
        k_bias_params = self.key_network.k_bias.numel()
        q_distance_weights_params = self.query_network.distance_scorer.q_weights_raw.numel()
        q_magnitude_params = self.query_network.distance_scorer.q_magnitude_weights.numel()
        q_bias_params = self.query_network.distance_scorer.q_bias.numel()

        return {
            'shared_probes': shared_probe_params,
            'k_magnitude_weights': k_magnitude_params,
            'k_bias': k_bias_params,
            'q_distance_weights': q_distance_weights_params,
            'q_magnitude_weights': q_magnitude_params,
            'q_bias': q_bias_params,
            'total': shared_probe_params + k_magnitude_params + k_bias_params + q_distance_weights_params + q_magnitude_params + q_bias_params
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
    import torch
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


def create_model(config, init_probes=None, use_l2_norm=True, inv_freq=None, use_error_term=False):
    """
    Factory function to create Module 2 network.

    Args:
        config: Configuration dict with model parameters
            - num_bins: Number of bins (default: 128)
            - num_freqs: Ignored (kept for backward compatibility)
            - num_kernels: Ignored (kept for backward compatibility)
        init_probes: Optional tensor of shape (num_bins, head_dim) for probe initialization
                     If provided, use this to initialize probes (e.g., K-means centers)
        use_l2_norm: If True (default), L2 normalize probes and Q vectors
        inv_freq: Optional inverse frequency tensor from model's RoPE.
                  If None, will try to load from model_path in config.
        use_error_term: If True, add error vector term to Q network (default: False)

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

    # Load inv_freq if not provided
    if inv_freq is None:
        model_path = model_cfg.get('model_path',
            "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B")
        inv_freq = load_model_inv_freq(model_path)

    model = Module2Network(
        num_bins=num_bins,
        head_dim=head_dim,
        init_probes=init_probes,
        use_l2_norm=use_l2_norm,
        inv_freq=inv_freq,
        use_error_term=use_error_term
    )
    return model
