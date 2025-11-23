"""Computing MVPURE beamformers (MVP_R, MVP_N) weights."""
# Author: Julia Jurkowska

import numpy as np

from .mvpure_utils import (
    get_G_proj_matrix,
    get_S_proj_matrix
)


def make_mvp_r(
        Gk: np.ndarray,
        bf_numer: np.ndarray,
        bf_denom_inv: np.ndarray,
        filter_rank: str | int,
        n_orient: int,
        R: np.ndarray) -> np.ndarray:
    """
    Implements MVP_R filter defined in Eq. 43 of [1]_.

    | It is defined as: :math:`{W}^{(r)}_{MVP_R}=P^{(r)}_{S_0}{W}_{LCMV_R}`,
    where :math:`P^{(r)}_{S_0}` is the orthogonal projection matrix onto subspace spanned by eigenvectors
    corresponding to the ``filter_rank`` number of largest eigenvalues of :math:`S_0`.
    :math:`S_0` is defined as: :math:`{H}_0^t{R}^{-1}{H}_0` (eq. 24), where :math:`H_0` is the leadfield matrix
    and :math:`R` is the data covariance matrix.

    Parameters
    -----------
    Gk : ndarray, shape (n_sources, n_channels, 1)
        Leadfield transposition.
    bf_numer : ndarray, shape (n_sources, 1, n_channels)
        bf_numer obtained from ``mne.beamformer._compute_beamformer._compute_bf_terms`` function
    bf_denom_inv : ndarray, shape (n_sources, n_orient, n_orient)
        inversion of bf_denom obtained from ``mne.beamformer._compute_beamformer._compute_bf_terms`` function
    filter_rank : int
        Defines number of eigenvectors corresponding to the largest eigenvalues of matrix S to take under consideration.
    n_orient: int
        number of dipole orientations defined at each source point
        in case of using MV-PURE should be equal to 1
    R : array-like
        Data covariance matrix

    Returns
    -----------
    W : ndarray, shape (n_sources, n_channels)
        Beamformer weights for 'MVP_R'.

    """

    print("MVP_R computation - in progress...")
    P_L2, frank = get_S_proj_matrix(H=Gk, R=R, filter_rank=filter_rank)

    print(f"Filter rank: {frank}")
 
    W_LCMV = np.matmul(bf_denom_inv, bf_numer)
    # W_LCMV = W_LCMV.reshape(W_LCMV.shape[0], W_LCMV.shape[2])
 
    W = np.matmul(P_L2, W_LCMV)
    W = W.reshape(W.shape[0], n_orient, W.shape[1])

    return W


def make_mvp_n(
        Gk: np.ndarray,
        bf_numer: np.ndarray,
        bf_denom_inv: np.ndarray,
        filter_rank: str | int,
        n_orient: int,
        N: np.ndarray
) -> np.ndarray:
    """
    Implements MVP_R filter defined in Eq. 44 of [1]_.

    | It is defined as: :math:`{W}^{(r)}_{MVP_N}=P^{(r)}_{G_0}{W}_{LCMV_N}`,
    where :math:`P^{(r)}_{G_0}` is the orthogonal projection matrix onto subspace spanned by
    eigenvectors corresponding to the ``filter_rank`` number of largest eigenvalues of :math:`G_0`.
    :math:`G_0` is defined as: :math:`{H}_0^t{N}^{-1}{H}_0` (eq. 23), where :math:`H_0` is lead field matrix and :math:`N` is
    noise covariance matrix.

    Parameters
    -----------
    Gk : ndarray, shape (n_sources, n_channels, 1)
        Leadfield transposition.
    bf_numer : ndarray, shape (n_sources, 1, n_channels)
        bf_numer obtained from ``mne.beamformer._compute_beamformer._compute_bf_terms`` function
    bf_denom_inv : ndarray, shape (n_sources, n_orient, n_orient)
        inversion of bf_denom obtained from ``mne.beamformer._compute_beamformer._compute_bf_terms`` function
    filter_rank : int
        Defines number of eigenvectors corresponding to the largest eigenvalues of matrix S to take under consideration.
    n_orient: int
        number of dipole orientations defined at each source point
        in case of using MVPURE should be equal to 1
    N : array-like
        Noise covariance matrix

    Returns
    -----------
    W : ndarray, shape (n_sources, n_channels)
        Beamformer weights for 'MVP_N'.
    """

    print("MVP_N computation - in progress...")
    P_L3, frank = get_G_proj_matrix(H=Gk, N=N, filter_rank=filter_rank)

    print(f"Filter rank: {frank}")
    
    W_LCMV = np.matmul(bf_denom_inv, bf_numer)
    # W_LCMV = W_LCMV.reshape(W_LCMV.shape[0], W_LCMV.shape[2])

    W = np.matmul(P_L3, W_LCMV)
    W = W.reshape(W.shape[0], n_orient, W.shape[1])

    return W
