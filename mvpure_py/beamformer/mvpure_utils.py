"""Functions specific to MVPURE beamformers."""
# Author: Julia Jurkowska

import numpy as np

from ..utils import algebra


def get_S_proj_matrix(
        H: np.ndarray,
        R: np.ndarray,
        filter_rank: int | str,
) -> tuple[np.ndarray, int]:
    """
    Compute the orthogonal projection matrix onto subspace spanned by eigenvectors corresponding to the
    ``filter_rank`` number of largest eigenvalues of :math:`S`.

    | :math:`S` is defined as :math:`{H}^t{R}^{-1}{H}` (eq. 21 in [1]_), where :math:`H` is the leadfield matrix
    and :math:`R` is the data covariance matrix.

    | Given projection matrix is used during computing **MVP_R** filter (``beamformer.filters_utils.make_mvp_r``).

    Parameters
    ----------
    H : array-like
        Leadfield matrix.
    R : array-like
        Data covariance matrix
    filter_rank : int | str

        * If ``int``: Defines number of largest eigenvalues of matrix S to use calculating orthogonal projection matrix.
        * If ``full``, it is equal to number of sources.

    Returns
    -------
    proj_matrix : array-like
        Projection matrix onto :math:`S` subspace
    filter_rank : int
        Rank of the used filter

    References
    -----------
    """
    S = algebra._get_S(H, R)
    s, u = np.linalg.eigh(S)

    if isinstance(filter_rank, int):
        # get eigenvectors corresponding to r largest eigenvalues
        u = u[:, -filter_rank:]

    if filter_rank == "full":
        # filter rank as int: number of eigenvalues
        filter_rank = len(s)

    proj_matrix = np.matmul(u, u.T)
    return proj_matrix, filter_rank


def get_G_proj_matrix(
        H: np.ndarray,
        N: np.ndarray,
        filter_rank: int | str,
) -> tuple[np.ndarray, int]:
    """
    Compute the orthogonal projection matrix onto subspace spanned by eigenvectors corresponding to the
    ``filter_rank`` number of largest eigenvalues of :math:`G`.

    | :math:`G` is defined as :math:`{H}^t{N}^{-1}{H}` (eq. 20 in [1]_), where :math:`H` is the leadfield matrix
    and :math:`N` is the noise covariance matrix.

    | Given projection matrix is used during computing **MVP_N** filter (``beamformer.filters_utils.make_mvp_n``).

    Parameters
    ----------
    H : array-like
        Leadfield matrix.
    N : array-like
        Noise covariance matrix
    filter_rank : int | str

        * If ``int``: Defines number of largest eigenvalues of matrix S to use calculating orthogonal projection matrix.
        * If ``full``, it is equal to number of sources.

    Returns
    -------
    proj_matrix : array-like
        Projection matrix onto :math:`G` subspace
    filter_rank : int
        Rank of the used filter

    References
    -----------
    """
    G = algebra._get_G(H, N)
    s, u = np.linalg.eigh(G)

    if isinstance(filter_rank, int):
        # get eigenvectors corresponding to r largest eigenvalues
        u = u[:, -filter_rank:]

    if filter_rank == "full":
        # filter rank as int: number of eigenvalues
        filter_rank = len(s)

    proj_matrix = np.matmul(u, u.T)
    return proj_matrix, filter_rank
