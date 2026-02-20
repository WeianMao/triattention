"""
Test pruning mode variants.

Validates three pruning granularity modes: per_head, per_layer, per_layer_per_head.
"""

import pytest
import torch


class PruningModeSelector:
    """
    Reference implementation of pruning mode selection logic.

    Implements three granularity modes for token selection.
    """

    @staticmethod
    def select_per_head(scores, budget_per_head, num_layers, num_heads):
        """
        Per-head mode: Each KV head independently selects tokens globally across all layers.

        Args:
            scores: [num_layers, num_heads, num_tokens] - scores for all tokens
            budget_per_head: int - budget for each head
            num_layers: int
            num_heads: int

        Returns:
            torch.Tensor: [num_heads, budget_per_head] - selected token indices per head
        """
        num_tokens = scores.shape[2]

        # Flatten across layers: [num_heads, num_layers * num_tokens]
        scores_per_head = scores.permute(1, 0, 2).reshape(num_heads, -1)

        # Select top-k for each head
        _, top_indices = scores_per_head.topk(budget_per_head, dim=1)

        return top_indices

    @staticmethod
    def select_per_layer(scores, budget_per_layer, num_layers, num_heads):
        """
        Per-layer mode: All heads in same layer share token selection.

        Args:
            scores: [num_layers, num_heads, num_tokens]
            budget_per_layer: int - budget for each layer
            num_layers: int
            num_heads: int

        Returns:
            torch.Tensor: [num_layers, budget_per_layer] - selected token indices per layer
        """
        num_tokens = scores.shape[2]

        # Average scores across heads: [num_layers, num_tokens]
        scores_per_layer = scores.mean(dim=1)

        # Select top-k for each layer
        _, top_indices = scores_per_layer.topk(budget_per_layer, dim=1)

        return top_indices

    @staticmethod
    def select_per_layer_per_head(scores, budget_per_layer_per_head, num_layers, num_heads):
        """
        Per-layer-per-head mode: Each (layer, head) pair independently selects tokens.

        Args:
            scores: [num_layers, num_heads, num_tokens]
            budget_per_layer_per_head: int - budget for each (layer, head) pair
            num_layers: int
            num_heads: int

        Returns:
            torch.Tensor: [num_layers, num_heads, budget_per_layer_per_head] - selected indices
        """
        num_tokens = scores.shape[2]

        # Select top-k for each (layer, head)
        _, top_indices = scores.topk(budget_per_layer_per_head, dim=2)

        return top_indices


# ==================== Test Cases ====================


def test_per_head_mode_output_shape(small_test_config, deterministic_seed):
    """
    Test per_head mode produces correct output shape.
    """
    num_layers = small_test_config["num_layers"]
    num_heads = small_test_config["num_heads"]
    num_tokens = 200
    budget_per_head = 100

    scores = torch.randn(num_layers, num_heads, num_tokens)

    selected = PruningModeSelector.select_per_head(scores, budget_per_head, num_layers, num_heads)

    # Shape should be [num_heads, budget_per_head]
    assert selected.shape == (num_heads, budget_per_head)


def test_per_layer_mode_output_shape(small_test_config, deterministic_seed):
    """
    Test per_layer mode produces correct output shape.
    """
    num_layers = small_test_config["num_layers"]
    num_heads = small_test_config["num_heads"]
    num_tokens = 200
    budget_per_layer = 120

    scores = torch.randn(num_layers, num_heads, num_tokens)

    selected = PruningModeSelector.select_per_layer(scores, budget_per_layer, num_layers, num_heads)

    # Shape should be [num_layers, budget_per_layer]
    assert selected.shape == (num_layers, budget_per_layer)


def test_per_layer_per_head_mode_output_shape(small_test_config, deterministic_seed):
    """
    Test per_layer_per_head mode produces correct output shape.
    """
    num_layers = small_test_config["num_layers"]
    num_heads = small_test_config["num_heads"]
    num_tokens = 200
    budget = 80

    scores = torch.randn(num_layers, num_heads, num_tokens)

    selected = PruningModeSelector.select_per_layer_per_head(scores, budget, num_layers, num_heads)

    # Shape should be [num_layers, num_heads, budget]
    assert selected.shape == (num_layers, num_heads, budget)


def test_per_head_global_selection(deterministic_seed):
    """
    Test per_head mode selects globally across layers.
    """
    num_layers = 3
    num_heads = 2
    num_tokens = 10
    budget_per_head = 5

    # Create scores where layer 0 has highest scores for head 0
    scores = torch.zeros(num_layers, num_heads, num_tokens)
    scores[0, 0, :] = 10.0  # Layer 0, head 0: high scores
    scores[1, 0, :] = 1.0  # Layer 1, head 0: low scores
    scores[2, 0, :] = 1.0  # Layer 2, head 0: low scores

    selected = PruningModeSelector.select_per_head(scores, budget_per_head, num_layers, num_heads)

    # For head 0, should select from layer 0 (indices 0-9 map to layer 0)
    head_0_indices = selected[0]
    assert head_0_indices.max().item() < num_tokens  # Should be from layer 0


def test_per_layer_shared_selection(deterministic_seed):
    """
    Test per_layer mode shares selection across heads.
    """
    num_layers = 2
    num_heads = 3
    num_tokens = 20
    budget_per_layer = 10

    # Create scores where different heads prefer different tokens
    scores = torch.zeros(num_layers, num_heads, num_tokens)
    scores[0, 0, :5] = 10.0  # Head 0 prefers first 5
    scores[0, 1, 5:10] = 10.0  # Head 1 prefers next 5
    scores[0, 2, 10:15] = 10.0  # Head 2 prefers next 5

    selected = PruningModeSelector.select_per_layer(scores, budget_per_layer, num_layers, num_heads)

    # Layer 0 should select based on average scores
    layer_0_selected = selected[0]
    assert layer_0_selected.shape == (budget_per_layer,)

    # Selected indices should be valid
    assert layer_0_selected.min() >= 0
    assert layer_0_selected.max() < num_tokens


def test_per_layer_per_head_independent(deterministic_seed):
    """
    Test per_layer_per_head mode allows independent selection per (layer, head).
    """
    num_layers = 2
    num_heads = 2
    num_tokens = 20
    budget = 5

    # Create distinct scores for each (layer, head)
    scores = torch.zeros(num_layers, num_heads, num_tokens)
    scores[0, 0, :5] = 10.0  # Layer 0, head 0: tokens 0-4
    scores[0, 1, 5:10] = 10.0  # Layer 0, head 1: tokens 5-9
    scores[1, 0, 10:15] = 10.0  # Layer 1, head 0: tokens 10-14
    scores[1, 1, 15:20] = 10.0  # Layer 1, head 1: tokens 15-19

    selected = PruningModeSelector.select_per_layer_per_head(scores, budget, num_layers, num_heads)

    # Each (layer, head) should select its preferred tokens
    assert set(selected[0, 0].tolist()) == set(range(5))
    assert set(selected[0, 1].tolist()) == set(range(5, 10))
    assert set(selected[1, 0].tolist()) == set(range(10, 15))
    assert set(selected[1, 1].tolist()) == set(range(15, 20))


def test_pruning_modes_budget_enforcement(small_test_config, deterministic_seed):
    """
    Test all modes correctly enforce budget constraints.
    """
    num_layers = small_test_config["num_layers"]
    num_heads = small_test_config["num_heads"]
    num_tokens = 150

    scores = torch.randn(num_layers, num_heads, num_tokens)

    # Test per_head
    budget_per_head = 80
    selected_ph = PruningModeSelector.select_per_head(scores, budget_per_head, num_layers, num_heads)
    assert selected_ph.shape[1] == budget_per_head

    # Test per_layer
    budget_per_layer = 100
    selected_pl = PruningModeSelector.select_per_layer(
        scores, budget_per_layer, num_layers, num_heads
    )
    assert selected_pl.shape[1] == budget_per_layer

    # Test per_layer_per_head
    budget_plph = 60
    selected_plph = PruningModeSelector.select_per_layer_per_head(
        scores, budget_plph, num_layers, num_heads
    )
    assert selected_plph.shape[2] == budget_plph


def test_per_head_index_range(deterministic_seed):
    """
    Test per_head mode produces valid global indices.
    """
    num_layers = 4
    num_heads = 2
    num_tokens = 10
    budget_per_head = 8

    scores = torch.randn(num_layers, num_heads, num_tokens)

    selected = PruningModeSelector.select_per_head(scores, budget_per_head, num_layers, num_heads)

    # Indices should be in range [0, num_layers * num_tokens)
    assert selected.min() >= 0
    assert selected.max() < num_layers * num_tokens


def test_per_layer_averaging(deterministic_seed):
    """
    Test per_layer mode correctly averages across heads.
    """
    num_layers = 2
    num_heads = 4
    num_tokens = 20
    budget_per_layer = 10

    scores = torch.zeros(num_layers, num_heads, num_tokens)

    # Set layer 0: token 5 has high score only in head 0
    scores[0, 0, 5] = 100.0
    scores[0, 1:, 5] = 0.0

    # Set layer 0: token 10 has medium score in all heads
    scores[0, :, 10] = 30.0

    selected = PruningModeSelector.select_per_layer(scores, budget_per_layer, num_layers, num_heads)

    layer_0_selected = selected[0]

    # Token 10 should be selected (higher average: 30 vs 25)
    # Token 5 average: 100/4 = 25
    # Token 10 average: 30
    assert 10 in layer_0_selected


def test_per_head_deterministic(deterministic_seed):
    """
    Test per_head mode is deterministic.
    """
    num_layers = 3
    num_heads = 4
    num_tokens = 50
    budget_per_head = 25

    scores = torch.randn(num_layers, num_heads, num_tokens)

    selected_1 = PruningModeSelector.select_per_head(scores, budget_per_head, num_layers, num_heads)
    selected_2 = PruningModeSelector.select_per_head(scores, budget_per_head, num_layers, num_heads)

    assert torch.equal(selected_1, selected_2)


def test_per_layer_per_head_no_sharing(deterministic_seed):
    """
    Test per_layer_per_head mode does not share selections across heads.
    """
    num_layers = 2
    num_heads = 3
    num_tokens = 30
    budget = 10

    # Create scores where each head has different preferences
    scores = torch.randn(num_layers, num_heads, num_tokens)
    scores[0, 0, :10] += 10.0  # Head 0 prefers 0-9
    scores[0, 1, 10:20] += 10.0  # Head 1 prefers 10-19
    scores[0, 2, 20:30] += 10.0  # Head 2 prefers 20-29

    selected = PruningModeSelector.select_per_layer_per_head(scores, budget, num_layers, num_heads)

    # Check that different heads select different tokens
    selected_l0_h0 = set(selected[0, 0].tolist())
    selected_l0_h1 = set(selected[0, 1].tolist())
    selected_l0_h2 = set(selected[0, 2].tolist())

    # Should have minimal overlap (due to different preferences)
    overlap_01 = len(selected_l0_h0 & selected_l0_h1)
    overlap_02 = len(selected_l0_h0 & selected_l0_h2)
    overlap_12 = len(selected_l0_h1 & selected_l0_h2)

    # With strong score differences, overlap should be small
    assert overlap_01 < budget // 2
    assert overlap_02 < budget // 2
    assert overlap_12 < budget // 2


def test_pruning_mode_comparison(small_test_config, deterministic_seed):
    """
    Compare different pruning modes on same scores.
    """
    num_layers = small_test_config["num_layers"]
    num_heads = small_test_config["num_heads"]
    num_tokens = 100
    budget = 50

    scores = torch.randn(num_layers, num_heads, num_tokens)

    # Per-head: global selection per head
    selected_ph = PruningModeSelector.select_per_head(
        scores, budget * num_layers, num_layers, num_heads
    )

    # Per-layer: shared selection per layer
    selected_pl = PruningModeSelector.select_per_layer(scores, budget, num_layers, num_heads)

    # Per-layer-per-head: independent selection
    selected_plph = PruningModeSelector.select_per_layer_per_head(scores, budget, num_layers, num_heads)

    # Verify shapes
    assert selected_ph.shape == (num_heads, budget * num_layers)
    assert selected_pl.shape == (num_layers, budget)
    assert selected_plph.shape == (num_layers, num_heads, budget)


def test_per_head_cross_layer_selection(deterministic_seed):
    """
    Test per_head mode can select tokens from different layers for same head.
    """
    num_layers = 3
    num_heads = 1
    num_tokens = 5
    budget_per_head = 6  # More than one layer

    scores = torch.zeros(num_layers, num_heads, num_tokens)

    # Layer 0: tokens 0-2 high
    scores[0, 0, :3] = 10.0
    # Layer 1: tokens 0-2 high
    scores[1, 0, :3] = 10.0
    # Layer 2: tokens 0-2 low
    scores[2, 0, :3] = 1.0

    selected = PruningModeSelector.select_per_head(scores, budget_per_head, num_layers, num_heads)

    # Should select 3 from layer 0 and 3 from layer 1
    selected_indices = selected[0].tolist()

    # Convert to layer indices
    layer_indices = [idx // num_tokens for idx in selected_indices]

    # Should have tokens from layer 0 and layer 1
    assert 0 in layer_indices
    assert 1 in layer_indices


def test_per_layer_uniform_scores(deterministic_seed):
    """
    Test per_layer mode with uniform scores across heads.
    """
    num_layers = 2
    num_heads = 4
    num_tokens = 30
    budget_per_layer = 15

    scores = torch.randn(num_layers, num_heads, num_tokens)

    # Make all heads have same scores for layer 0
    uniform_scores = torch.randn(num_tokens)
    scores[0, :, :] = uniform_scores.unsqueeze(0).expand(num_heads, -1)

    selected = PruningModeSelector.select_per_layer(scores, budget_per_layer, num_layers, num_heads)

    # All heads should contribute same to average, result should match direct topk on uniform_scores
    _, expected_indices = uniform_scores.topk(budget_per_layer)
    assert set(selected[0].tolist()) == set(expected_indices.tolist())


def test_per_layer_per_head_edge_cases(deterministic_seed):
    """
    Test per_layer_per_head mode edge cases.
    """
    num_layers = 2
    num_heads = 2
    num_tokens = 10

    scores = torch.randn(num_layers, num_heads, num_tokens)

    # Budget equals num_tokens
    selected = PruningModeSelector.select_per_layer_per_head(
        scores, num_tokens, num_layers, num_heads
    )
    assert selected.shape == (num_layers, num_heads, num_tokens)

    # Budget is 1
    selected = PruningModeSelector.select_per_layer_per_head(scores, 1, num_layers, num_heads)
    assert selected.shape == (num_layers, num_heads, 1)


def test_pruning_modes_with_negative_scores(deterministic_seed):
    """
    Test all pruning modes work with negative scores.
    """
    num_layers = 2
    num_heads = 2
    num_tokens = 20
    budget = 10

    scores = torch.randn(num_layers, num_heads, num_tokens) - 5.0  # All negative

    # All modes should work
    selected_ph = PruningModeSelector.select_per_head(
        scores, budget * num_layers, num_layers, num_heads
    )
    selected_pl = PruningModeSelector.select_per_layer(scores, budget, num_layers, num_heads)
    selected_plph = PruningModeSelector.select_per_layer_per_head(scores, budget, num_layers, num_heads)

    # Verify shapes
    assert selected_ph.shape == (num_heads, budget * num_layers)
    assert selected_pl.shape == (num_layers, budget)
    assert selected_plph.shape == (num_layers, num_heads, budget)


@pytest.mark.parametrize("pruning_mode_name", ["per_head", "per_layer", "per_layer_per_head"])
def test_pruning_mode_no_duplicates(pruning_mode_name, small_test_config, deterministic_seed):
    """
    Test that pruning modes don't produce duplicate indices.
    """
    num_layers = small_test_config["num_layers"]
    num_heads = small_test_config["num_heads"]
    num_tokens = 100
    budget = 50

    scores = torch.randn(num_layers, num_heads, num_tokens)

    if pruning_mode_name == "per_head":
        selected = PruningModeSelector.select_per_head(
            scores, budget * num_layers, num_layers, num_heads
        )
        # Check each head has no duplicates
        for h in range(num_heads):
            indices = selected[h].tolist()
            assert len(indices) == len(set(indices))

    elif pruning_mode_name == "per_layer":
        selected = PruningModeSelector.select_per_layer(scores, budget, num_layers, num_heads)
        # Check each layer has no duplicates
        for l in range(num_layers):
            indices = selected[l].tolist()
            assert len(indices) == len(set(indices))

    elif pruning_mode_name == "per_layer_per_head":
        selected = PruningModeSelector.select_per_layer_per_head(scores, budget, num_layers, num_heads)
        # Check each (layer, head) has no duplicates
        for l in range(num_layers):
            for h in range(num_heads):
                indices = selected[l, h].tolist()
                assert len(indices) == len(set(indices))
