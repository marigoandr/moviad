"""Scores restricted to the foreground (the object mask)."""

import torch

from moviad.utilities.foreground import restrict_to_mask


def test_restrict_to_mask_keeps_the_dilated_mask_only():
    scores = torch.ones(1, 1, 8, 8)
    masks = torch.zeros(1, 1, 24, 24, dtype=torch.bool)
    masks[..., 9:12, 9:12] = True  # one cell of the 8 x 8 grid: (3, 3); cells of 3 pixels, centres at 3i + 1
    assert restrict_to_mask(scores, masks, 0).sum() == 1
    # 1 pixel reaches no other centre; 2 pixels reach the centres of the 4 neighbours, 2 pixels off the mask
    assert restrict_to_mask(scores, masks, 1).sum() == 1
    assert restrict_to_mask(scores, masks, 2).sum() == 5
    # 4 pixels also reach the diagonal neighbours, (2, 2) pixels off a corner of the mask
    assert restrict_to_mask(scores, masks, 4).sum() == 9


def test_restrict_to_mask_samples_the_centre_of_each_cell():
    scores = torch.ones(1, 1, 8, 8)
    masks = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    masks[..., 14:16, 14:16] = True  # the bottom-right quarter of cell (3, 3), its centre (14, 14) included
    assert restrict_to_mask(scores, masks, 0)[0, 0, 3, 3] == 1
    masks[..., 14:16, 14:16], masks[..., 12:14, 12:14] = False, True  # the top-left quarter: centre excluded
    assert restrict_to_mask(scores, masks, 0).sum() == 0
