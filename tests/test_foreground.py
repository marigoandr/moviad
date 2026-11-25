"""Scores restricted to the foreground (the object mask)."""

import torch

from moviad.utilities.foreground import restrict_to_mask


def test_restrict_to_mask_keeps_the_dilated_mask_only():
    scores = torch.ones(1, 1, 8, 8)
    masks = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    masks[..., 12:16, 12:16] = True  # one cell of the 8 x 8 grid: (3, 3)
    assert restrict_to_mask(scores, masks, 0).sum() == 1
    # Elliptical 3 x 3 kernel: the cell and its 4 neighbours
    assert restrict_to_mask(scores, masks, 1).sum() == 5
