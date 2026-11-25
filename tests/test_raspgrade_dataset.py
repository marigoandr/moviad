"""RaspGradeCropDataset on a tiny crops folder: one train crop and one test crop."""

import numpy as np
import pandas as pd
import torch
from PIL import Image

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.raspgrade import RaspGradeCropDataset
from moviad.utilities.configurations import LabelName


def _crops_folder(root):
    crops = root / "yolo" / "deblurred" / "refined"
    crops.mkdir(parents=True)
    mask = np.zeros((16, 16), np.uint8)
    mask[4:12, 4:12] = 255
    for name in ("a_mask0_yolo", "b_obj3_yolo"):
        Image.fromarray(np.full((16, 16, 3), 128, np.uint8)).save(crops / f"{name}.png")
        Image.fromarray(mask).save(crops / f"{name}_mask.png")
    pd.DataFrame({"file": ["a_mask0_yolo", "b_obj3_yolo"], "image_id": ["a", "b"], "obj": ["", 3],
                  "grade": ["", 5], "source": ["yolo", "yolo"], "has_hole": [False, False],
                  "split": ["train", "test"]}).to_csv(crops / "index.csv", index=False)
    return DatasetArguments(dataset_path=str(root), img_size=(8, 8), gt_mask_size=(8, 8))


def test_without_foreground_the_splits_follow_the_moviad_convention(tmp_path):
    args = _crops_folder(tmp_path)
    train = RaspGradeCropDataset(args, "yolo/deblurred/refined", "train")
    assert len(train) == 1 and train[0].shape == (3, 8, 8)  # the image only
    image, label, mask, path = RaspGradeCropDataset(args, "yolo/deblurred/refined", "test")[0]
    assert image.shape == (3, 8, 8) and label == LabelName.ABNORMAL  # grade 5
    assert not mask.any() and mask.shape == (1, 8, 8)  # no pixel-level ground truth
    assert path.endswith("b_obj3_yolo.png")


def test_with_foreground_both_splits_return_the_raspberry_mask(tmp_path):
    args = _crops_folder(tmp_path)
    _, label, _, _, foreground = RaspGradeCropDataset(args, "yolo/deblurred/refined", "train", foreground=True)[0]
    assert label == LabelName.NORMAL  # no grade
    assert foreground.dtype == torch.bool and foreground.sum() == 16  # the 8 x 8 mask, resized to 4 x 4
    _, label, _, _, foreground = RaspGradeCropDataset(args, "yolo/deblurred/refined", "test", foreground=True)[0]
    assert label == LabelName.ABNORMAL and foreground.sum() == 16
