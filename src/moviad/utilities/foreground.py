"""Anomaly scores restricted to the foreground, i.e. to a mask of the object in the image."""

import cv2
import torch
import torch.nn.functional as F


def restrict_to_mask(scores: torch.Tensor, masks: torch.Tensor, dilation: int) -> torch.Tensor:
    """Zero the scores outside the object mask, dilated by `dilation` pixels of the mask.

    The mask is dilated at its own resolution, then sampled at the centre of each cell of the score grid: the
    kept region is the same whatever the grid, up to the size of a cell.

    Args:
        scores: (B, 1, h, w) float scores, on any grid (e.g. a patch grid or a full-size anomaly map).
        masks: (B, 1, H, W) bool masks of the input images.
        dilation: radius of the elliptical dilation, in pixels of the masks; 0 for none.

    Returns:
        (B, 1, h, w) scores, zero outside the dilated mask.
    """
    # Bool to the float dtype and the device of the scores
    masks = masks.to(scores)
    # 1. Dilation at the resolution of the masks
    if dilation > 0:
        kernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)).to(masks)
        masks = (F.conv2d(masks, kernel[None, None], padding=dilation) > 0).to(masks)
    # 2. Down to the score grid: each cell takes the mask pixel at its centre
    # 3. The scores of the cells outside are zeroed
    return scores * F.interpolate(masks, size=scores.shape[-2:], mode="nearest-exact")
