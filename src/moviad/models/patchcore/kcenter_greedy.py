"""k-Center Greedy Method.

Returns points that minimizes the maximum distance of any point to a center.
- https://arxiv.org/abs/1708.00489
"""
from __future__ import annotations

from abc import abstractmethod

import torch
from torch.nn import functional as F
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
from tqdm import tqdm


class CoresetExtractor:
    """Implements k-center-greedy method.

    Args:
        embedding (torch.Tensor): Embedding vector extracted from a CNN
        sampling_ratio (float): Ratio to choose coreset size from the embedding size, used when k is None.
        k (int | None): coreset size.

    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = CoresetExtractor(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    PROJECTION_CHUNK = 65536  # vectors projected at a time: 384 MB of float32 input at 1536 dimensions

    def __init__(self, quantized, device: torch.device, sampling_ratio: float = 0.1,k: int | None = 30000) -> None:

        self.quantized = quantized
        self.projector = SparseRandomProjection(n_components="auto", eps=0.90)
        self.sampling_ratio = sampling_ratio
        self.k = k
        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None

        self.device = device

    def reset_distances(self):
        self.min_distances = None

    def update_distances(self, cluster_centers: list):
        """Update min distances given cluster centers.

        Args:
            cluster_centers (list[int]): indices of cluster centers
        """

        if cluster_centers:
            centers = self.features[cluster_centers]

            distances = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distances
            else:
                self.min_distances = torch.minimum(self.min_distances, distances)

    def get_new_idx(self) -> int:
        """Get index value of a sample.

        Based on minimum distance of the cluster

        Returns:
            int: Sample index
        """

        if isinstance(self.min_distances, torch.Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            msg = f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}"
            raise TypeError(msg)

        return idx

    @abstractmethod
    def get_coreset_idx_randomp(
            self,
            z_lib,
            n: int = 1000,
            k: int = 30000,
            eps: float = 0.90,
            float16: bool = True,
    ):
        """Returns n coreset idx for given z_lib.

        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)

        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.

        Returns:
            coreset indices
        """

        print(z_lib.device)
        print(z_lib.shape)

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        if self.quantized:
            z_lib = torch.int_repr(z_lib)
        num_vectors, dim = z_lib.shape
        projected_dim = johnson_lindenstrauss_min_dim(num_vectors, eps=eps)
        if 0 < projected_dim <= dim:
            # Only the random matrix comes from sklearn, with the same distribution and the same draws from
            # numpy's global RNG as fit_transform(): fit() with an explicit n_components uses just the number of
            # features. The matrix is applied below in chunks on self.device, because fit_transform() on the
            # whole z_lib makes several float64 copies of it in RAM (about 4x its size)
            transformer = SparseRandomProjection(n_components=projected_dim).fit(z_lib[:1])
            components = torch.from_numpy(transformer.components_.toarray()).float().T.to(self.device)
            print(f"   DONE.                 Transformed dim = {(num_vectors, projected_dim)}.")
        else:
            components = None
            print(f"   Not projected: eps={eps} gives {projected_dim} dimensions for {dim}-dimensional vectors. Please increase `eps`.")

        # Projected chunk by chunk on self.device, in float32: neither the projected z_lib nor a float32 copy
        # of it is ever built on the CPU
        dtype = torch.float16 if float16 else torch.float32
        projected = torch.empty(num_vectors, projected_dim if components is not None else dim,
                                dtype=dtype, device=self.device)
        for start in range(0, num_vectors, self.PROJECTION_CHUNK):
            chunk = z_lib[start:start + self.PROJECTION_CHUNK].to(self.device, torch.float32)
            projected[start:start + len(chunk)] = chunk @ components if components is not None else chunk
        z_lib = projected

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)

        k = self.k if self.k is not None else max(1, round(self.sampling_ratio * num_vectors))
        for _ in tqdm(range(k - 1)):  # k points in total: the first one, then k - 1 greedy picks
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            # distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True) # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))

        return torch.stack(coreset_idx)

    @abstractmethod
    def extract_coreset(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Extract coreset from embeddings.

        Args:
            embeddings (torch.Tensor): Embeddings from a CNN.

        Returns:
            torch.Tensor: Coreset embeddings.
        """

        sampled_idxs = self.get_coreset_idx_randomp(embeddings.cpu())
        coreset = embeddings[sampled_idxs]
        return coreset

    def select_coreset_idxs(self, selected_idxs: list[int] | None = None) -> list[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
          indices of samples selected to minimize distance to cluster centers
        """

        if selected_idxs is None:
            selected_idxs = []

        print(f"Fitting random projections. Start dim = {self.embeddings.shape}.")
        if self.embeddings.ndim == 2:
            embedding_np = self.embeddings.cpu().numpy()
            self.projector.fit(embedding_np)
            self.features = torch.from_numpy(self.projector.transform(embedding_np)).to(self.device)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        print(f"DONE: Final dim = {self.features.shape}")

        selected_coreset_idx: list[int] = []
        idx = int(torch.randint(high=self.n_observations, size=(1,)).item())

        for _ in tqdm(range(self.k)):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()

            if idx in selected_idxs:
                msg = "New indices should not be in selected indices."
                raise ValueError(msg)

            self.min_distances[idx] = 0
            selected_coreset_idx.append(idx)

        return selected_coreset_idx
