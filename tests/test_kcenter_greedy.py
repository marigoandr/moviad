"""The chunked random projection of CoresetExtractor gives a k-center greedy coreset of sklearn's
fit_transform() of the same vectors, with the same numpy seed."""

import numpy as np
import torch
from sklearn.random_projection import SparseRandomProjection

from moviad.models.patchcore.kcenter_greedy import CoresetExtractor


def _assert_greedy(x: np.ndarray, chosen: list[int]) -> None:
    """chosen starts from the first vector, and each next one is the farthest from those chosen before it.

    Farthest up to float32 rounding: on random vectors two candidates can be closer than that, and
    then either is a correct pick.
    """
    assert chosen[0] == 0
    min_distances = np.linalg.norm(x - x[0], axis=1)
    for i in chosen[1:]:
        assert min_distances[i] >= min_distances.max() * (1 - 1e-5)
        min_distances = np.minimum(min_distances, np.linalg.norm(x - x[i], axis=1))


def _coreset(x: torch.Tensor, k: int) -> list[int]:
    np.random.seed(0)
    extractor = CoresetExtractor(False, torch.device("cpu"), k=k)
    extractor.PROJECTION_CHUNK = 1000  # several chunks, the last one shorter
    return extractor.get_coreset_idx_randomp(x, float16=False).tolist()


def test_projected_coreset_matches_fit_transform():
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((3000, 512), dtype=np.float32))
    np.random.seed(0)
    projected = SparseRandomProjection(eps=0.9).fit_transform(x.numpy()).astype(np.float64)
    assert projected.shape[1] < x.shape[1]  # the projection does run
    chosen = _coreset(x, 30)
    assert len(chosen) == 30  # exactly k points, the first one included
    _assert_greedy(projected, chosen)
    # Not greedy on the unprojected vectors: the coreset does come from the projection
    try:
        _assert_greedy(x.numpy().astype(np.float64), chosen)
    except AssertionError:
        return
    raise AssertionError("the coreset is also greedy on the unprojected vectors")


def test_unprojected_when_target_dimension_too_large():
    # 3000 vectors at eps 0.9 need 198 dimensions: 64-dimensional vectors are used as they are
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((3000, 64), dtype=np.float32))
    _assert_greedy(x.numpy().astype(np.float64), _coreset(x, 30))


def test_sampling_ratio_sets_the_size_when_k_is_none():
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((3000, 512), dtype=np.float32))
    np.random.seed(0)
    by_ratio = CoresetExtractor(False, torch.device("cpu"), sampling_ratio=0.01, k=None).get_coreset_idx_randomp(x)
    np.random.seed(0)
    by_size = CoresetExtractor(False, torch.device("cpu"), k=30).get_coreset_idx_randomp(x)
    assert torch.equal(by_ratio, by_size)  # 1% of 3000 vectors
