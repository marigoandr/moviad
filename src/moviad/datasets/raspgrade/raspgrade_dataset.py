"""Single-raspberry crops of the RaspGrade dataset (Hugging Face FBK-TeV/RaspGrade).

A crops folder, <dataset_path>/<category>/ (e.g. category "yolo_sam3/deblurred/refined"), holds one crop per
raspberry: <file>.png (RGB, zero outside the raspberry) and <file>_mask.png (the raspberry mask, 0/255), and an
index.csv with one row per crop and at least the columns "file", "grade" and "split" ("train" or "test").
The train crops have no grade (unlabelled, assumed normal); the test crops have the grade of their raspberry,
1 to 5, and grades 4 and 5 are anomalous.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.configurations import Split, LabelName

ANOMALOUS_GRADES = (4, 5)  # grades 1-3 are normal


class RaspGradeCropDataset(VADDataset):
    """The crops of one split of a crops folder.

    Args:
        dataset_arguments: the arguments of the dataset, dataset_path the folder of the crops folders.
        category: the crops folder, relative to dataset_path (e.g. "yolo_sam3/deblurred/refined").
        split: "train" or "test".
        foreground: return the raspberry mask too, on both splits (see __getitem__); e.g. to score the
            crops only inside the raspberry, with model(image, foreground=foreground).

    Attributes:
        samples: the rows of index.csv of the split, numbered from 0, in dataset order, plus a "label"
            column, 1 for the anomalous grades and 0 otherwise (the train crops included).
    """

    def __init__(self, dataset_arguments: DatasetArguments, category: str, split: Split | str,
                 foreground: bool = False) -> None:
        super().__init__(dataset_arguments, category, split)
        self.root_category = Path(dataset_arguments.dataset_path) / category
        index_csv = pd.read_csv(self.root_category / "index.csv")
        self.samples = index_csv[index_csv.split == self.split.value].reset_index(drop=True)
        # The train crops have no grade (NaN): normal
        self.samples["label"] = self.samples.grade.isin(ANOMALOUS_GRADES).astype(int)
        self.foreground = foreground

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int) : index of the crop to be returned

        Returns:
            The image (Tensor (3, H, W)) only for the train split, and
            (image, label, mask, path) for the test split: label (LabelName), mask (Tensor (1, H', W'), all
            zero: there is no pixel-level ground truth), path (str) of the crop.
            With foreground, both splits return (image, label, mask, path, foreground), foreground the
            (1, H', W') bool raspberry mask and label NORMAL for the train crops.
            H, W = img_size; H', W' = gt_mask_size.
        """
        row = self.samples.iloc[index]
        path = self.root_category / f"{row.file}.png"
        image = self.transform_image(Image.open(path).convert("RGB"))
        if self.split == Split.TRAIN and not self.foreground:
            return image
        label = LabelName(row.label)
        mask = torch.zeros(1, *self.dataset_arguments.gt_mask_size, dtype=torch.int)
        if not self.foreground:
            return image, label, mask, str(path)
        foreground = self.transform_mask(Image.open(self.root_category / f"{row.file}_mask.png")) > 0.5
        return image, label, mask, str(path), foreground
