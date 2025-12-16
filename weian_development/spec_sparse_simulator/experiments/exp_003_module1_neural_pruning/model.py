"""
Neural network model for Module 1 Key Pruning.

Architecture:
1. Kernel Encoding Layer: 128 bins x 64 freqs x 3 von Mises kernels = 73,856 params
2. MLP Layer: 128 -> hidden -> 1
3. Position Scaling Layer: log-scale anchor interpolation
4. Sigmoid activation for drop probability
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
    """

    def __init__(self, num_bins=128, num_freqs=64, num_kernels=3):
        """
        Args:
            num_bins: Number of output bins (default: 128)
            num_freqs: Number of frequency bands (default: 64, from head_dim=128)
            num_kernels: Number of von Mises kernels per frequency band (default: 3)
        """
        super().__init__()
        self.num_bins = num_bins
        self.num_freqs = num_freqs
        self.num_kernels = num_kernels

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
            base: RoPE base (default: 10000)

        Returns:
            reference_angles: Tensor of shape (num_freqs,)
        """
        # Reference position: middle of the round
        ref_position = round_start + round_window // 2

        # Compute RoPE angular frequencies omega
        dim_indices = torch.arange(0, self.num_freqs, device=self.mu.device)
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

        # Reshape K to complex pairs: (num_keys, num_freqs, 2)
        K_complex = K.view(num_keys, self.num_freqs, 2)

        # Compute magnitude: (num_keys, num_freqs)
        magnitude = torch.norm(K_complex, dim=2)

        # Compute angle: (num_keys, num_freqs)
        angle = torch.atan2(K_complex[..., 1], K_complex[..., 0])

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


class PositionScalingLayer(nn.Module):
    """
    Position Scaling Layer using log-scale anchor interpolation.

    Applies position-dependent scaling to logits. Uses 3 anchor positions
    with learnable weights interpolated on log-scale.

    Anchor positions: [1000, 10000, 100000]
    Learnable parameters: 3 anchor weights (softplus-constrained for non-negativity)
    """

    def __init__(self, anchors=[1000, 10000, 100000]):
        """
        Args:
            anchors: List of anchor positions in log-scale
        """
        super().__init__()

        # Register anchor positions as buffer (non-trainable)
        self.register_buffer('anchor_positions', torch.tensor(anchors, dtype=torch.float32))

        # Precompute log10 of anchor positions for interpolation
        self.register_buffer('log_anchors', torch.log10(self.anchor_positions))

        # Initialize learnable anchor weights with softplus constraint
        # softplus(x) = log(1 + exp(x))
        # To get softplus(x) approx 1.0, we need x approx log(exp(1) - 1) approx 0.5413
        init_value = math.log(math.exp(1.0) - 1)
        self.anchor_weights_raw = nn.Parameter(
            torch.full((len(anchors),), init_value, dtype=torch.float32)
        )

    @property
    def anchor_weights(self):
        """
        Apply softplus constraint to ensure non-negative weights.

        Returns:
            Tensor of shape (num_anchors,) with non-negative weights
        """
        return F.softplus(self.anchor_weights_raw)

    def _interpolate_weights(self, positions):
        """
        Interpolate weights in log-scale for given positions.

        Args:
            positions: Tensor of shape (num_keys,) with position indices

        Returns:
            Interpolated weights of shape (num_keys,)
        """
        # Clamp positions to minimum 1 to avoid log(0)
        positions = torch.clamp(positions.float(), min=1.0)

        # Compute log10 of positions
        log_pos = torch.log10(positions)

        # Initialize weights with first anchor weight (for positions < first anchor)
        weights = torch.full_like(log_pos, self.anchor_weights[0].item())

        # Get anchor weights once
        anchor_w = self.anchor_weights

        # Interpolate between anchors
        num_anchors = len(self.log_anchors)
        for i in range(num_anchors - 1):
            log_left = self.log_anchors[i]
            log_right = self.log_anchors[i + 1]
            w_left = anchor_w[i]
            w_right = anchor_w[i + 1]

            # Find positions in current interval [log_left, log_right)
            in_interval = (log_pos >= log_left) & (log_pos < log_right)

            # Compute interpolation coefficient t in [0, 1]
            t = (log_pos - log_left) / (log_right - log_left)

            # Linear interpolation: w = w_left * (1-t) + w_right * t
            interpolated = w_left * (1 - t) + w_right * t

            # Update weights for positions in this interval
            weights = torch.where(in_interval, interpolated, weights)

        # Handle positions >= last anchor (use last anchor weight)
        above_max = log_pos >= self.log_anchors[-1]
        weights = torch.where(above_max, anchor_w[-1], weights)

        return weights

    def forward(self, logits, positions):
        """
        Apply position-dependent scaling to logits.

        The position weights are multiplied element-wise with logits before sigmoid,
        effectively adjusting the sigmoid sharpness based on position.

        Args:
            logits: Tensor of shape (batch_size, num_keys)
            positions: Tensor of shape (batch_size, num_keys)
                Position indices for each key

        Returns:
            Scaled logits of shape (batch_size, num_keys)
        """
        # Get position-specific weights via interpolation
        pos_weights = self._interpolate_weights(positions)

        # Element-wise multiplication of logits and position weights
        scaled_logits = logits * pos_weights

        return scaled_logits


class Module1KeyPruningNetwork(nn.Module):
    """
    Complete Module 1 network for Key Pruning.

    Integrates three components:
    1. KernelEncodingLayer: K -> 128-dim encoding (73,856 params)
    2. MLP: 128 -> hidden_dim -> 1 with ReLU (8,321 params for hidden_dim=64)
    3. PositionScalingLayer: log-scale position weighting (3 params)

    Total parameters: ~82,180 per head

    Forward pass:
        K -> Kernel Encoding (128-dim) -> MLP (1-dim) -> Position Scaling -> Sigmoid

    Output: drop probabilities in [0, 1]
        - p close to 1: Key should be dropped (unlikely to be attended)
        - p close to 0: Key should be retained (likely to be attended)
    """

    def __init__(self, num_bins=128, num_freqs=64, num_kernels=3,
                 mlp_hidden=64, anchor_positions=[1000, 10000, 100000]):
        """
        Args:
            num_bins: Number of kernel encoding bins (default: 128)
            num_freqs: Number of frequency bands (default: 64, from head_dim=128)
            num_kernels: Number of von Mises kernels per frequency (default: 3)
            mlp_hidden: Hidden dimension of MLP (default: 64)
            anchor_positions: Position anchors for scaling (default: [1000, 10000, 100000])
        """
        super().__init__()

        # Component 1: Kernel Encoding Layer
        self.kernel_layer = KernelEncodingLayer(num_bins, num_freqs, num_kernels)

        # Component 2: MLP (128 -> hidden_dim -> 1)
        self.mlp = nn.Sequential(
            nn.Linear(num_bins, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

        # Component 3: Position Scaling Layer
        self.position_scaling = PositionScalingLayer(anchor_positions)

    def forward(self, K, key_positions, reference_angles):
        """
        Forward pass: K -> Kernel Encoding -> MLP -> Position Scaling -> Sigmoid

        Args:
            K: Key vectors of shape (num_keys, head_dim) or (head_dim,)
            key_positions: Position indices of shape (num_keys,) or scalar
            reference_angles: Reference angles of shape (num_freqs,)

        Returns:
            drop_probs: Drop probabilities of shape (num_keys,) or scalar
                Values in [0, 1], where higher values mean higher drop probability
        """
        # Stage 1: Kernel Encoding -> (num_keys, 128)
        encoded = self.kernel_layer(K, reference_angles)

        # Handle single key case for positions
        is_single = K.dim() == 1
        if is_single and not isinstance(key_positions, torch.Tensor):
            key_positions = torch.tensor([key_positions], device=K.device)
        elif is_single and key_positions.dim() == 0:
            key_positions = key_positions.unsqueeze(0)

        # Stage 2: MLP -> (num_keys, 1) -> (num_keys,)
        logits = self.mlp(encoded).squeeze(-1)

        # Stage 3: Position Scaling -> (num_keys,)
        scaled_logits = self.position_scaling(logits, key_positions)

        # Stage 4: Sigmoid -> (num_keys,) in [0, 1]
        drop_probs = torch.sigmoid(scaled_logits)

        # Return scalar if input was single key
        if is_single:
            drop_probs = drop_probs.squeeze()

        return drop_probs

    def get_param_count(self):
        """
        Get parameter count breakdown.

        Returns:
            dict: Parameter counts for each component
                - kernel_layer: ~73,856
                - mlp: ~8,321 (for hidden_dim=64)
                - position_scaling: 3
                - total: ~82,180
        """
        kernel_params = sum(p.numel() for p in self.kernel_layer.parameters())
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        position_params = sum(p.numel() for p in self.position_scaling.parameters())

        return {
            'kernel_layer': kernel_params,
            'mlp': mlp_params,
            'position_scaling': position_params,
            'total': kernel_params + mlp_params + position_params
        }


class Module1Network(nn.Module):
    """
    Complete Module 1 neural network for Key importance prediction.

    Pipeline:
        Query-Key angles -> Kernel Encoding -> MLP -> Position Scaling -> Sigmoid
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration dict with model parameters
        """
        super().__init__()

        # Extract model config
        model_cfg = config['model']

        # Kernel Encoding Layer
        self.kernel_encoding = KernelEncodingLayer(
            num_bins=model_cfg['num_bins'],
            num_kernels=model_cfg['num_kernels']
        )

        # MLP Layer
        input_dim = model_cfg['kernel_encoding_dim']
        hidden_dim = model_cfg['mlp_hidden_dim']
        output_dim = model_cfg['mlp_output_dim']

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Position Scaling Layer
        self.position_scaling = PositionScalingLayer(
            anchors=model_cfg['position_anchors']
        )

        # Output activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_key_angles, positions):
        """
        Forward pass of Module 1 network.

        Args:
            query_key_angles: Tensor of shape (batch_size, num_keys)
            positions: Tensor of shape (batch_size, num_keys)

        Returns:
            Drop probabilities of shape (batch_size, num_keys)
        """
        # 1. Kernel Encoding
        encoded = self.kernel_encoding(query_key_angles)  # (B, N, 128)

        # 2. MLP
        logits = self.mlp(encoded).squeeze(-1)  # (B, N)

        # 3. Position Scaling
        scaled_logits = self.position_scaling(logits, positions)  # (B, N)

        # 4. Sigmoid activation
        drop_probs = self.sigmoid(scaled_logits)  # (B, N)

        return drop_probs


def create_model(config):
    """
    Factory function to create Module 1 network.

    Args:
        config: Configuration dict

    Returns:
        Module1Network instance
    """
    model = Module1Network(config)
    return model
