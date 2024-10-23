import scanpy as sc
from anndata import AnnData
from typing import Optional
import torch
from scvi.model._utils import _init_library_size


def preprocess(
        adata: AnnData,
        n_top_genes: Optional[int] = None,

):
    pass


def get_library_log_means_and_vars(adata, n_batch):
    return _init_library_size(adata,
                              n_batch,
                              )

def gram_matrix(x: torch.Tensor, y: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    """
    Calculate the maximum mean discrepancy gram matrix with multiple gamma values.

    Args:
    ----
        x: Tensor with shape (B, P, M) or (P, M).
        y: Tensor with shape (B, R, M) or (R, M).
        gammas: 1-D tensor with the gamma values.

    Returns
    -------
        A tensor with shape (B, P, R) or (P, R) for the distance between pairs of data
        points in `x` and `y`.
    """
    gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0)

    pairwise_distances_sq = torch.square(pairwise_distances)
    tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape)
    return tmp
