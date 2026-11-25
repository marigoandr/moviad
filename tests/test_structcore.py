"""StructCore: structural descriptor of an anomaly map and the score built on it."""

import torch

from moviad.utilities.evaluation.structcore import StructCore


def test_descriptor_of_a_known_map():
    maps = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    # std 0.5; top 1% of 4 pixels = the highest one, 1; total variation (0 vertical + 2 horizontal) / 4 pixels
    assert torch.allclose(StructCore.descriptor(maps), torch.tensor([[0.5, 1.0, 0.5]]))


def test_score_adds_the_distance_with_the_spread_of_the_base_scores():
    torch.manual_seed(0)
    descriptors, base_scores = torch.randn(50, 3) * torch.tensor([1.0, 5.0, 0.1]), torch.rand(50) * 3
    structcore = StructCore(descriptors, base_scores)
    # A descriptor equal to the training mean is at distance 0: the base score is unchanged
    assert torch.allclose(structcore.score(descriptors.mean(0, keepdim=True), torch.tensor([2.0])), torch.tensor([2.0]))
    # On the training images the added term has the same std as the base scores
    added = structcore.score(descriptors, torch.zeros(50))
    assert torch.allclose(added.std(), base_scores.std(), rtol=1e-4)
