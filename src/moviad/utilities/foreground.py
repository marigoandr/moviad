"""Anomaly scores restricted to the foreground, i.e. to a mask of the object in the image."""

import cv2
import torch
import torch.nn.functional as F


def restrict_to_mask(scores: torch.Tensor, masks: torch.Tensor, dilation: int) -> torch.Tensor:
    """Zero the scores outside the object mask, dilated by `dilation` cells of the score grid.

    Args:
        scores: (B, 1, h, w) float scores, on any grid (e.g. a patch grid or a full-size anomaly map).
        masks: (B, 1, H, W) bool masks of the input images.
        dilation: radius of the elliptical dilation, in cells of the score grid; 0 for none.

    Returns:
        (B, 1, h, w) scores, zero outside the dilated mask.
    """
    # Each cell takes the mask pixel at its centre ("nearest" would take its top-left corner)
    grid = F.interpolate(masks.to(scores), size=scores.shape[-2:], mode="nearest-exact")
    if dilation > 0:
        # Binary dilation as a convolution with the elliptical kernel: on the device, for the whole batch
        kernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)).to(grid)
        grid = (F.conv2d(grid, kernel[None, None], padding=dilation) > 0).to(grid)
    return scores * grid
