"""StructCore: an image score that adds to a base anomaly score how unusual the structure of the anomaly map is."""
from __future__ import annotations

import torch


class StructCore:
    """Image score = base score + weight * D, where D is the distance of the structural descriptor
    of the map from those of the training images: the L2 norm of the descriptor standardised per
    dimension with the mean and std of the training descriptors.

    The weight std(base scores) / std(D) of the training images gives the two terms the same spread.
    The training images are assumed to be mostly normal.

    Args:
        top_ratio: fraction of the highest pixels averaged for the tail mass (see descriptor()).
        eps: added to the standard deviations the descriptors and D are divided by.
    """

    def __init__(self, top_ratio: float = 0.01, eps: float = 1e-6) -> None:
        self.top_ratio = top_ratio
        self.eps = eps

    @staticmethod
    def descriptor(maps: torch.Tensor, top_ratio: float) -> torch.Tensor:
        """Structure of each anomaly map: global dispersion, tail mass and spatial roughness.

        Args:
            maps: (B, 1, H, W) float anomaly maps.
            top_ratio: fraction of the highest pixels averaged for the tail mass.

        Returns:
            (B, 3) float: std of the pixels, mean of the top `top_ratio` pixels, and total variation
            (sum of the absolute differences between vertical and horizontal neighbours) / (H * W).
        """
        s = maps[:, 0]
        flat = s.flatten(1)
        k = max(1, int(flat.shape[1] * top_ratio))
        tv = ((s[:, 1:] - s[:, :-1]).abs().sum((1, 2)) + (s[:, :, 1:] - s[:, :, :-1]).abs().sum((1, 2))) / flat.shape[1]
        return torch.stack([flat.std(1, correction=0), flat.topk(k, 1).values.mean(1), tv], 1)

    def fit(self, maps: torch.Tensor, base_scores: torch.Tensor) -> StructCore:
        """Fit on the training images: their (N, 1, H, W) anomaly maps and (N,) base scores."""
        descriptors = self.descriptor(maps, self.top_ratio)
        self.mean = descriptors.mean(0)
        self.std = descriptors.std(0, correction=0)
        self.weight = base_scores.std(correction=0) / (self._distance(descriptors).std(correction=0) + self.eps)
        return self

    def _distance(self, descriptors: torch.Tensor) -> torch.Tensor:
        return ((descriptors - self.mean) / (self.std + self.eps)).norm(dim=1)

    def score(self, maps: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
        """(B,) StructCore scores from (B, 1, H, W) anomaly maps and (B,) base scores."""
        return base_scores + self.weight * self._distance(self.descriptor(maps, self.top_ratio))
