"""
Module 2 Von Mises Kernel Network (migrated from exp_006)

Network Architecture:
- KernelEncodingLayer: von Mises kernel encoding with frequency bands
- Module2KeyNetwork: KernelEncoding + PositionScaling -> logits (external softmax dim=0)
- Module2QueryNetwork: KernelEncoding + PositionScaling -> logits (external softmax dim=-1)

Adapted for exp_012 with:
- PositionScalingLayer integration
- L2 normalization placeholder (use_l2_norm parameter, default False)
- Batched forward passes for multi-round efficiency

Total parameters: 147,712
- Key network: 73,856
- Query network: 73,856
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Import PositionScalingLayer and apply_rope_rotation from model.py
try:
    from .model import PositionScalingLayer, apply_rope_rotation
except ImportError:
    from model import PositionScalingLayer, apply_rope_rotation


class KernelEncodingLayer(nn.Module):
    """
    Kernel Encoding Layer using von Mises kernels.

    Implements frequency-based encoding with 64 frequency bands and 3 von Mises kernels per band.
    Total parameters: 73,856 (128 bins * 64 freqs * 3 kernels * 3 params + 128 bias)

    Copied from exp_006 model.py lines 18-158 with L2 norm placeholder added.
    """

    def __init__(self, num_bins=128, num_freqs=64, num_kernels=3, use_l2_norm=False):
        """
        Args:
            num_bins: Number of output bins (default: 128)
            num_freqs: Number of frequency bands (default: 64, from head_dim=128)
            num_kernels: Number of von Mises kernels per frequency band (default: 3)
            use_l2_norm: If True, apply L2 normalization to encoding output (default: False)
        """
        super().__init__()
        self.num_bins = num_bins
        self.num_freqs = num_freqs
        self.num_kernels = num_kernels
        self.use_l2_norm = use_l2_norm

        # Learnable parameters: mu, kappa, weight
        # Shape: (num_bins, num_freqs, num_kernels)
        self.mu = nn.Parameter(torch.zeros(num_bins, num_freqs, num_kernels))
        self.kappa = nn.Parameter(torch.ones(num_bins, num_freqs, num_kernels))
        self.weight = nn.Parameter(torch.zeros(num_bins, num_freqs, num_kernels))

        # Bias for each bin
        # Shape: (num_bins,)
        self.bias = nn.Parameter(torch.zeros(num_bins))

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters using exp_006 strategy:
        - mu: Uniform(-1, 1)
        - kappa: Uniform(-1, 1)
        - weight: Xavier uniform
        - bias: zeros
        """
        # mu: Uniform(-1, 1)
        nn.init.uniform_(self.mu, -1.0, 1.0)

        # kappa: Uniform(-1, 1)
        nn.init.uniform_(self.kappa, -1.0, 1.0)

        # weight: Xavier uniform (fan-in = num_freqs * num_kernels)
        fan_in = self.num_freqs * self.num_kernels
        bound = math.sqrt(6.0 / fan_in)
        nn.init.uniform_(self.weight, -bound, bound)

        # bias: zeros (already initialized to zeros)

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

    def _forward_chunk(self, K_chunk, mu_effective):
        """
        Forward pass for a chunk of keys. Used with gradient checkpointing.

        Args:
            K_chunk: Key vectors of shape (chunk_size, head_dim)
            mu_effective: Effective mu of shape (num_bins, num_freqs, num_kernels)

        Returns:
            logits: Encoded features of shape (chunk_size, num_bins)
        """
        chunk_size = K_chunk.shape[0]

        # Reshape K to complex pairs: (chunk_size, num_freqs, 2)
        K_complex = K_chunk.view(chunk_size, self.num_freqs, 2)

        # Compute magnitude: (chunk_size, num_freqs)
        magnitude = torch.norm(K_complex, dim=2)

        # Compute angle: (chunk_size, num_freqs)
        angle = torch.atan2(K_complex[..., 1], K_complex[..., 0])

        # Compute normalized von Mises kernel
        # angle: (chunk_size, num_freqs) -> (chunk_size, 1, num_freqs, 1)
        # mu_effective: (num_bins, num_freqs, num_kernels)
        angle_expanded = angle.view(chunk_size, 1, self.num_freqs, 1)
        mu_expanded = mu_effective.unsqueeze(0)  # (1, num_bins, num_freqs, num_kernels)

        # Normalized von Mises: exp(kappa * (cos(angle - mu) - 1))
        # Shape: (chunk_size, num_bins, num_freqs, num_kernels)
        kernel = torch.exp(
            self.kappa.unsqueeze(0) * (torch.cos(angle_expanded - mu_expanded) - 1)
        )

        # Weighted sum over kernels: (chunk_size, num_bins, num_freqs)
        weighted_kernels = (kernel * self.weight.unsqueeze(0)).sum(dim=3)

        # Multiply by magnitude and sum over frequencies
        # magnitude: (chunk_size, num_freqs) -> (chunk_size, 1, num_freqs)
        magnitude_expanded = magnitude.unsqueeze(1)
        weighted = weighted_kernels * magnitude_expanded  # (chunk_size, num_bins, num_freqs)
        logits = weighted.sum(dim=2)  # (chunk_size, num_bins) - bias added in forward()

        return logits

    def forward(self, K, reference_angles, chunk_size=8192):
        """
        Forward pass of kernel encoding with chunked processing and gradient checkpointing.

        Args:
            K: Key vectors of shape (num_keys, head_dim) or (head_dim,)
            reference_angles: Reference angles of shape (num_freqs,)
                Computed via _compute_reference_angles
            chunk_size: Number of keys to process per chunk (default: 8192)
                Reduces memory by processing keys in chunks with gradient checkpointing.

        Returns:
            logits: Encoded features of shape (num_keys, num_bins) or (num_bins,)
        """
        from torch.utils.checkpoint import checkpoint

        # Handle single vector case
        is_single = K.dim() == 1
        if is_single:
            K = K.unsqueeze(0)  # (1, head_dim)

        num_keys = K.shape[0]

        # Prepare effective mu for this round (shared across all chunks)
        mu_effective = self._prepare_round(reference_angles)

        # Process in chunks with gradient checkpointing
        if num_keys <= chunk_size:
            # Small input: no chunking needed
            logits = self._forward_chunk(K, mu_effective)
        else:
            # Large input: chunk processing with gradient checkpointing
            logits_chunks = []
            for start in range(0, num_keys, chunk_size):
                end = min(start + chunk_size, num_keys)
                K_chunk = K[start:end]

                # Use gradient checkpointing to save memory
                # During forward: compute but don't save intermediates
                # During backward: recompute forward to get gradients
                chunk_logits = checkpoint(
                    self._forward_chunk,
                    K_chunk,
                    mu_effective,
                    use_reentrant=False
                )
                logits_chunks.append(chunk_logits)

            logits = torch.cat(logits_chunks, dim=0)  # (num_keys, num_bins)

        # Add bias (after chunking to avoid redundant additions)
        logits = logits + self.bias

        # Apply L2 normalization if enabled
        if self.use_l2_norm:
            logits = F.normalize(logits, p=2, dim=-1)

        # Return to original shape if input was single vector
        if is_single:
            logits = logits.squeeze(0)  # (num_bins,)

        return logits


class Module2KeyNetwork(nn.Module):
    """
    Module 2 Key Network for bin assignment with von Mises kernels.

    Architecture: KernelEncodingLayer + PositionScaling (optional)

    Input: K (post-RoPE key vectors) of shape (num_keys, head_dim)
    Output: logits of shape (num_keys, num_bins)

    Softmax is applied EXTERNALLY over keys (dim=0):
        key_probs = F.softmax(logits, dim=0)  # Each column sums to 1

    Execution timing: Once per round (round_start)
    """

    def __init__(self, num_bins=128, num_kernels=3, num_freqs=64, head_dim=128, use_l2_norm=False):
        """
        Args:
            num_bins: Number of bins (default: 128)
            num_kernels: Number of von Mises kernels (default: 3)
            num_freqs: Number of frequency bands (default: 64)
            head_dim: Dimension of key vectors (default: 128)
            use_l2_norm: If True, apply L2 normalization to encoding (default: False)
        """
        super().__init__()
        self.kernel_layer = KernelEncodingLayer(num_bins, num_freqs, num_kernels, use_l2_norm)
        self.position_scaling = PositionScalingLayer()

    def forward(self, K, reference_angles, key_positions=None):
        """
        Forward pass: K -> KernelEncoding -> PositionScaling (optional) -> logits

        Args:
            K: Key vectors of shape (num_keys, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)
            key_positions: Optional tensor of shape (num_keys,) for position scaling

        Returns:
            logits: Raw logits of shape (num_keys, num_bins)
                Apply softmax(dim=0) externally to get key_probs
        """
        # KernelEncoding -> logits
        logits = self.kernel_layer(K, reference_angles)

        # Apply position scaling if key_positions provided
        if key_positions is not None:
            logits = self.position_scaling(logits, key_positions)

        return logits

    def forward_batched(self, K, ref_positions, key_lengths=None, key_positions=None):
        """
        Batched forward pass for keys with multiple rounds.

        All rounds share the same K data but use different reference angles.

        Args:
            K: Key vectors of shape (max_keys, head_dim) - shared across all rounds
            ref_positions: Tensor of shape (batch_size,) - reference positions for each round
            key_lengths: Optional tensor of shape (batch_size,) - valid key length for each round
            key_positions: Optional tensor of shape (max_keys,) for position scaling

        Returns:
            logits: Tensor of shape (batch_size, max_keys, num_bins)
            key_mask: Bool tensor of shape (batch_size, max_keys) if key_lengths provided
                      True for valid positions, False for padding
        """
        batch_size = ref_positions.shape[0]
        max_keys = K.shape[0]
        num_bins = self.kernel_layer.num_bins

        # Compute reference angles for all rounds
        # reference_angles: (batch_size, num_freqs)
        num_freqs = self.kernel_layer.num_freqs
        dim_indices = torch.arange(0, num_freqs, device=K.device)
        omega = 1.0 / (10000 ** (2 * dim_indices / (num_freqs * 2)))
        reference_angles = ref_positions.unsqueeze(1) * omega.unsqueeze(0)  # (batch_size, num_freqs)

        # Process each round
        logits_list = []
        for i in range(batch_size):
            ref_angles = reference_angles[i]  # (num_freqs,)
            round_logits = self.forward(K, ref_angles, key_positions=key_positions)
            logits_list.append(round_logits)

        # Stack results: (batch_size, max_keys, num_bins)
        logits = torch.stack(logits_list, dim=0)

        # Create key mask if key_lengths provided
        key_mask = None
        if key_lengths is not None:
            # key_mask[i, j] = True if j < key_lengths[i]
            key_indices = torch.arange(max_keys, device=K.device).unsqueeze(0)  # (1, max_keys)
            key_mask = key_indices < key_lengths.unsqueeze(1)  # (batch_size, max_keys)

        return logits, key_mask

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.kernel_layer._compute_reference_angles(round_start, round_window)


class Module2QueryNetwork(nn.Module):
    """
    Module 2 Query Network for bin routing with von Mises kernels.

    Architecture: KernelEncodingLayer + PositionScaling (optional)
    IDENTICAL architecture to KeyNetwork but SEPARATE parameters.

    Input: Q (post-RoPE query vector) of shape (head_dim,) or (num_queries, head_dim)
    Output: logits of shape (num_bins,) or (num_queries, num_bins)

    Softmax is applied EXTERNALLY over bins (dim=-1):
        bin_probs = F.softmax(logits, dim=-1)  # Each row sums to 1

    Execution timing: Every decoding step
    """

    def __init__(self, num_bins=128, num_kernels=3, num_freqs=64, head_dim=128, use_l2_norm=False):
        """
        Args:
            num_bins: Number of bins (default: 128)
            num_kernels: Number of von Mises kernels (default: 3)
            num_freqs: Number of frequency bands (default: 64)
            head_dim: Dimension of query vectors (default: 128)
            use_l2_norm: If True, apply L2 normalization to encoding (default: False)
        """
        super().__init__()
        self.kernel_layer = KernelEncodingLayer(num_bins, num_freqs, num_kernels, use_l2_norm)
        self.position_scaling = PositionScalingLayer()

    def forward(self, Q, reference_angles, query_positions=None):
        """
        Forward pass: Q -> KernelEncoding -> PositionScaling (optional) -> logits

        Args:
            Q: Query vector(s) of shape (head_dim,) or (num_queries, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)
            query_positions: Optional tensor of shape () or (num_queries,) for position scaling

        Returns:
            logits: Raw logits of shape (num_bins,) or (num_queries, num_bins)
                Apply softmax(dim=-1) externally to get bin_probs
        """
        # KernelEncoding -> logits
        logits = self.kernel_layer(Q, reference_angles)

        # Apply position scaling if query_positions provided
        if query_positions is not None:
            logits = self.position_scaling(logits, query_positions)

        return logits

    def forward_batched(self, Q_batch, ref_positions, query_positions=None):
        """
        Batched forward pass for queries with multiple rounds.

        Args:
            Q_batch: Query vectors of shape (batch_size, num_queries, head_dim)
            ref_positions: Tensor of shape (batch_size,) - reference positions for each round
            query_positions: Optional tensor of shape (batch_size, num_queries) for position scaling

        Returns:
            logits: Tensor of shape (batch_size, num_queries, num_bins)
        """
        batch_size, num_queries, head_dim = Q_batch.shape
        num_bins = self.kernel_layer.num_bins

        # Compute reference angles for all rounds
        # reference_angles: (batch_size, num_freqs)
        num_freqs = self.kernel_layer.num_freqs
        dim_indices = torch.arange(0, num_freqs, device=Q_batch.device)
        omega = 1.0 / (10000 ** (2 * dim_indices / (num_freqs * 2)))
        reference_angles = ref_positions.unsqueeze(1) * omega.unsqueeze(0)  # (batch_size, num_freqs)

        # Process each round
        logits_list = []
        for i in range(batch_size):
            ref_angles = reference_angles[i]  # (num_freqs,)
            Q = Q_batch[i]  # (num_queries, head_dim)
            q_pos = query_positions[i] if query_positions is not None else None
            round_logits = self.forward(Q, ref_angles, query_positions=q_pos)
            logits_list.append(round_logits)

        # Stack results: (batch_size, num_queries, num_bins)
        logits = torch.stack(logits_list, dim=0)

        return logits

    def _compute_reference_angles(self, round_start, round_window=128):
        """Convenience method to compute reference angles."""
        return self.kernel_layer._compute_reference_angles(round_start, round_window)


class Module2Network(nn.Module):
    """
    Complete Module 2 Network combining Key and Query networks with von Mises kernels.

    This is a convenience wrapper that holds both networks and provides
    unified interface for training and inference.

    Total parameters: 147,712
    - Key network: 73,856
    - Query network: 73,856
    """

    def __init__(self, key_network, query_network):
        """
        Args:
            key_network: Module2KeyNetwork instance
            query_network: Module2QueryNetwork instance
        """
        super().__init__()
        self.num_bins = key_network.kernel_layer.num_bins
        self.key_network = key_network
        self.query_network = query_network

    def forward_keys(self, K, reference_angles, key_positions=None):
        """
        Process keys: K -> logits -> key_probs (softmax over keys)

        Args:
            K: Key vectors of shape (num_keys, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)
            key_positions: Optional tensor of shape (num_keys,) for position scaling

        Returns:
            key_probs: Probability distribution of shape (num_keys, num_bins)
                Each column sums to 1 (softmax over keys)
        """
        logits = self.key_network(K, reference_angles, key_positions=key_positions)
        # Softmax over keys (dim=0): each column represents a bin's distribution over keys
        key_probs = F.softmax(logits, dim=0)
        return key_probs

    def forward_queries(self, Q, reference_angles, empty_bin_mask=None, query_positions=None):
        """
        Process queries: Q -> logits -> bin_probs (softmax over bins)

        Args:
            Q: Query vector(s) of shape (head_dim,) or (num_queries, head_dim)
            reference_angles: Reference angles of shape (num_freqs,)
            empty_bin_mask: Optional bool tensor of shape (num_bins,)
                True for empty bins (will be masked with -inf)
            query_positions: Optional tensor of shape () or (num_queries,) for position scaling

        Returns:
            bin_probs: Probability distribution of shape (num_bins,) or (num_queries, num_bins)
                Each row sums to 1 (softmax over bins)
        """
        logits = self.query_network(Q, reference_angles, query_positions=query_positions)

        # Mask empty bins with -inf before softmax
        if empty_bin_mask is not None:
            if logits.dim() == 1:
                logits = logits.masked_fill(empty_bin_mask, float('-inf'))
            else:
                logits = logits.masked_fill(empty_bin_mask.unsqueeze(0), float('-inf'))

        # Softmax over bins (dim=-1): each query selects a bin
        bin_probs = F.softmax(logits, dim=-1)
        return bin_probs

    def forward_keys_batched(self, K, ref_positions, key_lengths=None, return_logits=False, key_positions=None):
        """
        Batched forward pass for keys with multiple rounds.

        Args:
            K: Key vectors of shape (max_keys, head_dim) - shared across all rounds
            ref_positions: Tensor of shape (batch_size,) - reference positions for each round
            key_lengths: Optional tensor of shape (batch_size,) - valid key length for each round
            return_logits: If True, also return raw logits (for rank-based loss)
            key_positions: Optional tensor of shape (max_keys,) for position scaling

        Returns:
            key_probs: Probability distribution of shape (batch_size, max_keys, num_bins)
            key_mask: Bool tensor of shape (batch_size, max_keys) if key_lengths provided
            key_logits: (optional) Raw logits if return_logits=True
        """
        # Get logits and mask
        logits, key_mask = self.key_network.forward_batched(K, ref_positions, key_lengths, key_positions=key_positions)

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

    def forward_queries_batched(self, Q_batch, ref_positions, empty_bin_mask_batch=None, return_logits=False, query_positions=None):
        """
        Batched forward pass for queries with multiple rounds.

        Args:
            Q_batch: Query vectors of shape (batch_size, num_queries, head_dim)
            ref_positions: Tensor of shape (batch_size,) - reference positions for each round
            empty_bin_mask_batch: Optional bool tensor of shape (batch_size, num_bins)
                                  True for empty bins (will be masked with -inf)
            return_logits: If True, also return raw logits (for rank-based loss)
            query_positions: Optional tensor of shape (batch_size, num_queries) for position scaling

        Returns:
            bin_probs: Probability distribution of shape (batch_size, num_queries, num_bins)
            query_logits: (optional) Raw logits if return_logits=True
        """
        # Get logits: (batch_size, num_queries, num_bins)
        logits = self.query_network.forward_batched(Q_batch, ref_positions, query_positions=query_positions)

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


def create_model(num_bins=128, num_kernels=3, num_freqs=64, head_dim=128, use_l2_norm=False):
    """
    Factory function to create Module 2 network with von Mises kernels.

    Args:
        num_bins: Number of bins (default: 128)
        num_kernels: Number of von Mises kernels (default: 3)
        num_freqs: Number of frequency bands (default: 64)
        head_dim: Dimension of key/query vectors (default: 128)
        use_l2_norm: If True, apply L2 normalization to encoding (default: False)

    Returns:
        Module2Network instance
    """
    key_network = Module2KeyNetwork(num_bins, num_kernels, num_freqs, head_dim, use_l2_norm)
    query_network = Module2QueryNetwork(num_bins, num_kernels, num_freqs, head_dim, use_l2_norm)
    return Module2Network(key_network, query_network)
