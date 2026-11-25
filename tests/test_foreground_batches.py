"""VADModel.predict() and Evaluator.evaluate() on the batch forms: images only (predict() only),
(image, label, mask, path) and (image, label, mask, path, foreground)."""

import torch
from torch.utils.data import DataLoader, Dataset

from moviad.models.vad_model import VADModel
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc


class _Crops(Dataset):
    """Four 1 x 4 x 4 images of constant value i; with_labels: the test tuple, foreground: plus the mask."""

    def __init__(self, with_labels: bool, foreground: bool = False) -> None:
        self.with_labels, self.foreground = with_labels, foreground

    def __len__(self) -> int:
        return 4

    def __getitem__(self, i: int):
        image = torch.full((1, 4, 4), float(i))
        if not self.with_labels:
            return image
        item = (image, int(i >= 2), torch.zeros(1, 4, 4, dtype=torch.int), f"{i}.png")
        return item + (torch.ones(1, 4, 4, dtype=torch.bool),) if self.foreground else item


class _Model(VADModel):
    """Map = the image, score = its mean; with a foreground the map is negated, to show it was passed."""

    def forward(self, images: torch.Tensor, foreground: torch.Tensor | None = None):
        maps = images if foreground is None else -images
        return maps, maps.flatten(1).mean(1)


def test_predict_on_each_batch_form():
    for dataset, sign in ((_Crops(False), 1), (_Crops(True), 1), (_Crops(True, foreground=True), -1)):
        model = _Model().train()
        maps, scores = model.predict(DataLoader(dataset, batch_size=3), "cpu")
        assert not model.training  # predict() switches to eval mode
        assert maps.shape == (4, 1, 4, 4)
        assert torch.equal(scores, sign * torch.arange(4.0))  # dataloader order, all batches


def test_evaluate_passes_the_foreground_to_the_model():
    metric = RocAuc(MetricLvl.IMAGE)
    for dataset, auroc in ((_Crops(True), 1.0), (_Crops(True, foreground=True), 0.0)):
        # Images 2 and 3 are anomalous: ranked highest by the plain scores, lowest by the negated ones
        report = Evaluator.evaluate(_Model(), DataLoader(dataset, batch_size=3), [metric], "cpu")
        assert report[metric.name] == auroc
