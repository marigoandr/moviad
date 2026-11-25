"""StructCore: an image score that adds to a base anomaly score how unusual the structure of the anomaly map is."""
import torch


class StructCore:
    """Image score = base score + weight * D, where D is the distance of the structural descriptor
    of the map from those of the training images: the L2 norm of the descriptor standardised per
    dimension with the mean and std of the training descriptors.

    The weight std(base scores) / std(D) of the training images gives the two terms the same spread.
    The training images are assumed to be mostly normal.
    """

    def __init__(self, descriptors: torch.Tensor, base_scores: torch.Tensor, eps: float = 1e-6) -> None:
        """Fit on the training images: (N, 3) descriptor() of their maps, (N,) base scores."""
        self.mean, self.std, self.eps = descriptors.mean(0), descriptors.std(0), eps
        self.weight = base_scores.std() / (self._distance(descriptors).std() + eps)

    @staticmethod
    def descriptor(maps: torch.Tensor, top_ratio: float = 0.01) -> torch.Tensor:
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

    def _distance(self, descriptors: torch.Tensor) -> torch.Tensor:
        return ((descriptors - self.mean) / (self.std + self.eps)).norm(dim=1)

    def score(self, descriptors: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
        """(B,) StructCore scores from (B, 3) descriptors and (B,) base scores."""
        return base_scores + self.weight * self._distance(descriptors)
