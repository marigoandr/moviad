"""Runtime regressions of PatchCore: imports, devices, saving and fitting."""

import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn


class _FakeExtractor:
    """Stand-in for CustomFeatureExtractor: two feature maps, at 1/4 and 1/8 of the input, from a random conv."""

    quantized = False

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        torch.manual_seed(0)
        self.device = device
        self.conv = nn.Conv2d(3, 8, 3, stride=4, padding=1).to(device)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.conv.to(device)

    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:
        features = self.conv(batch)
        return [features, F.avg_pool2d(features, 2)]


def test_patchcore_imports_without_faiss():
    # faiss is needed only for the product quantization; None in sys.modules makes "import faiss" fail
    code = ("import sys; sys.modules['faiss'] = None\n"
            "from moviad.models.patchcore.patchcore import PatchCore\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _patchcore(device: torch.device = torch.device("cpu"), **kwargs):
    from moviad.models.patchcore.patchcore import PatchCore
    return PatchCore(_FakeExtractor(device), **kwargs)


def test_to_returns_the_model_and_moves_everything():
    model = _patchcore()
    assert model.to("cpu") is model
    assert model.device == torch.device("cpu")
    model.reset_model()  # memory_bank None: nothing to move
    assert model.to("cpu") is model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_to_moves_the_memory_bank_to_the_new_device():
    model = _patchcore()
    model.memory_bank = torch.zeros(3, 8)
    model.to("cuda")
    assert model.memory_bank.device.type == "cuda"
    assert model.coreset_extractor.device.type == "cuda"
    assert model.feature_extractor.device.type == "cuda"
    model.to("cpu")
    assert model.memory_bank.device.type == "cpu"


def test_save_then_load_gives_the_same_memory_bank(tmp_path):
    model = _patchcore()
    model.memory_bank = torch.randn(5, 8)
    model.save(str(tmp_path / "patchcore.pt"))
    loaded = _patchcore()
    loaded.load(str(tmp_path / "patchcore.pt"), None)
    assert torch.equal(loaded.memory_bank, model.memory_bank)


def test_load_accepts_a_memory_bank_saved_as_a_bare_tensor(tmp_path):
    # The format that save() wrote before it wrote a dict
    bank = torch.randn(5, 8)
    torch.save(bank, tmp_path / "patchcore.pt")
    loaded = _patchcore()
    loaded.load(str(tmp_path / "patchcore.pt"), None)
    assert torch.equal(loaded.memory_bank, bank)


def _fit_and_predict(device: torch.device, drop_last: bool = False):
    from torch.utils.data import DataLoader
    from moviad.models.training_args import TrainingArgs

    images = torch.rand(7, 3, 32, 32)
    model = _patchcore(torch.device(device), memory_bank_size=10)
    model.train()
    model.train_epoch(0, DataLoader(images, batch_size=3, drop_last=drop_last), TrainingArgs(batch_size=3, epochs=1))
    model.eval()
    maps, scores = model(images.to(device))
    return model, maps, scores


def test_fit_and_predict_on_cpu():
    model, maps, scores = _fit_and_predict("cpu")
    assert model.memory_bank.dtype == torch.float32
    assert model.memory_bank.shape == (11, 16)  # k-center greedy: the first patch, then memory_bank_size picks
    assert maps.shape == (7, 1, 32, 32) and scores.shape == (7,)


def test_fit_with_a_loader_that_drops_the_last_batch():
    model, _, _ = _fit_and_predict("cpu", drop_last=True)
    assert model.memory_bank.shape == (11, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fit_and_predict_on_cuda():
    model, maps, scores = _fit_and_predict("cuda")
    assert model.memory_bank.dtype == torch.float32
    assert model.memory_bank.device.type == "cuda"
    assert scores.device.type == "cuda"


def test_train_chunk_twice_merges_with_the_memory_bank():
    from torch.utils.data import DataLoader
    from moviad.models.training_args import TrainingArgs

    model = _patchcore(memory_bank_size=10)
    model.train()
    for seed in (0, 1):
        torch.manual_seed(seed)
        model.train_chunk(DataLoader(torch.rand(4, 3, 32, 32), batch_size=2), TrainingArgs(batch_size=2, epochs=1))
    assert model.memory_bank.shape == (11, 16)
