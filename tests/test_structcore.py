"""StructCore: structural descriptor of an anomaly map and the score built on it."""

import torch

from moviad.utilities.evaluation.structcore import StructCore


def test_descriptor_of_a_known_map():
    maps = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    # std 0.5; top 1% of 4 pixels = the highest one, 1; total variation (0 vertical + 2 horizontal) / 4 pixels
    assert torch.allclose(StructCore.descriptor(maps, 0.01), torch.tensor([[0.5, 1.0, 0.5]]))


def test_score_adds_the_distance_with_the_spread_of_the_base_scores():
    torch.manual_seed(0)
    maps, base_scores = torch.rand(50, 1, 16, 16) * torch.rand(50, 1, 1, 1), torch.rand(50) * 3
    structcore = StructCore().fit(maps, base_scores)
    # On the training images the added term has the same std as the base scores
    added = structcore.score(maps, torch.zeros(50))
    assert torch.allclose(added.std(), base_scores.std(), rtol=1e-4)
    # With the same base score, a map with an unusual structure scores higher than a training one
    unusual = maps[:1] * 10
    assert structcore.score(unusual, torch.zeros(1)) > structcore.score(maps[:1], torch.zeros(1))
