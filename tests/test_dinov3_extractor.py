"""DINOv3 in CustomFeatureExtractor, on a small random DINOv3 in place of the gated hub weights."""

import copy

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor


def test_dinov3_features_are_the_normed_patch_tokens_of_the_blocks(monkeypatch):
    torch.manual_seed(0)
    config = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
                             patch_size=16, num_register_tokens=4)
    reference = DINOv3ViTModel(config).eval()
    model = copy.deepcopy(reference)  # the extractor trims the blocks of its model in place
    monkeypatch.setattr(DINOv3ViTModel, "from_pretrained", lambda name, **kwargs: model)

    extractor = CustomFeatureExtractor("dinov3_vitb16", ["0", "2"], torch.device("cpu"))
    batch = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        features = extractor(batch)

        # The same, block by block on the whole model
        x = reference.embeddings(batch)
        rope = reference.rope_embeddings(batch)
        expected = []
        for i, block in enumerate(reference.model.layer):
            x = block(x, position_embeddings=rope)
            if i in (0, 2):
                expected.append(reference.norm(x[:, 5:]).transpose(1, 2).reshape(2, 32, 4, 6))

    assert len(extractor.model.model.layer) == 3  # trimmed after block 2
    assert [f.shape for f in features] == [(2, 32, 4, 6)] * 2
    for f, e in zip(features, expected):
        torch.testing.assert_close(f, e)
