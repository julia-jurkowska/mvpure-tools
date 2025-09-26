""" Functions used during sources localization. """

# Author: Julia Jurkowska

import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm

from ..viz import plot_RN_eigenvalues
from ..utils import algebra


def suggest_n_sources_and_rank(R: np.ndarray,
                               N: np.ndarray,
                               show_plot: bool = True,
                               subject: str = None,
                               n_sources_threshold: float = 1,
                               rank_threshold: float = 1.5,
                               **kwargs) -> tuple[int, int]:
    """
    Automatically propose number of sources to localize and rank based on Proposition 3 in [1]_.

    Parameters
    ----------
    R : array-like
        Data covariance matrix
    N : array-like
        Noise covariance matrix
    show_plot : bool
        Whether to display a graph of the eigenvalues of the :math:`RN^{-1}` matrix.
        Default to True.
    subject : str
        Subject name the analysis is performed for. Optional.
    n_sources_threshold : float
        Number of eigenvalues of the :math:`RN^{-1}` matrix below this threshold corresponds to the suggested
        number of sources to localize.
        Default to 1.0. For more details see Observation 1 in [1]_.
    rank_threshold : float
        Number of eigenvalues of the :math:`RN^{-1}` matrix below this threshold corresponds to the
        suggested rank optimization parameter.
        Default to 1.5. For more details see Proposition 3 in [1]_.

    Returns
    -------
    n_sources : int
        Suggested number of sources to localize.
    rank : int
        Suggested rank optimization parameter.

    References
    ----------

    """
    if show_plot:
        _, eigvals = plot_RN_eigenvalues(R=R,
                                         N=N,
                                         subject=subject,
                                         return_eigvals=True,
                                         **kwargs)
    else:
        eigvals = algebra.get_pinv_RN_eigenvals(R=R, N=N)

    # Suggesting number of sources
    n_sources_temp = np.where(eigvals > n_sources_threshold)[0]
    if n_sources_temp.size == 0:
        raise ValueError(
            f"All eigenvalues of $\\mathrm{{RN}}^{{-1}}$ are smaller than {n_sources_threshold}."
        )
    else:
        n_sources = n_sources_temp[-1] + 1

    # Suggesting rank
    rank_temp = np.where(eigvals > rank_threshold)[0]
    if rank_temp.size == 0:
        raise ValueError(
            f"All eigenvalues of $\\mathrm{{RN}}^{{-1}}$ are smaller than {rank_threshold}."
        )
    else:
        rank = rank_temp[-1] + 1

    print(f"Suggested number of sources to localize: {n_sources}")
    print(f"Suggested rank is: {rank}")
    return int(n_sources), int(rank)


import os

os.environ["OMP_NUM_THREADS"] = "8"  # you can tune this

import numpy as np
from tqdm import tqdm
from typing import List, Tuple

# ---------------- Block builders (CPU, vectorized) ----------------


def _build_S_blocks_batch_cpu(H_sel: np.ndarray, S_sel: np.ndarray,
                              A_R_batch: np.ndarray,
                              H_batch: np.ndarray) -> np.ndarray:
    k = S_sel.shape[0]
    B = A_R_batch.shape[1]
    m = k + 1
    blocks = np.empty((B, m, m), dtype=A_R_batch.dtype)

    if k == 0:
        s_batch = np.sum(A_R_batch * H_batch, axis=0)  # (B,)
        blocks[:] = s_batch[:, None, None]
        return blocks

    a_batch = H_sel[:, :k].T @ A_R_batch
    s_batch = np.sum(A_R_batch * H_batch, axis=0)

    blocks[:, :k, :k] = S_sel[None, :, :]
    blocks[:, :k, k:k + 1] = a_batch.T[:, :, None]
    blocks[:, k:k + 1, :k] = a_batch.T[:, None, :]
    blocks[:, k, k] = s_batch
    return blocks


def _build_G_blocks_batch_cpu(H_sel: np.ndarray, G_sel: np.ndarray,
                              A_N_batch: np.ndarray,
                              H_batch: np.ndarray) -> np.ndarray:
    k = G_sel.shape[0]
    B = A_N_batch.shape[1]
    m = k + 1
    blocks = np.empty((B, m, m), dtype=A_N_batch.dtype)

    if k == 0:
        s_batch = np.sum(A_N_batch * H_batch, axis=0)
        blocks[:] = s_batch[:, None, None]
        return blocks

    a_batch = H_sel[:, :k].T @ A_N_batch
    s_batch = np.sum(A_N_batch * H_batch, axis=0)

    blocks[:, :k, :k] = G_sel[None, :, :]
    blocks[:, :k, k:k + 1] = a_batch.T[:, :, None]
    blocks[:, k:k + 1, :k] = a_batch.T[:, None, :]
    blocks[:, k, k] = s_batch
    return blocks


def _build_T_blocks_batch_cpu(H_sel: np.ndarray, T_sel: np.ndarray,
                              A_TR_batch: np.ndarray,
                              H_batch: np.ndarray) -> np.ndarray:
    k = T_sel.shape[0]
    B = A_TR_batch.shape[1]
    m = k + 1
    blocks = np.empty((B, m, m), dtype=A_TR_batch.dtype)

    if k == 0:
        s_batch = np.sum(A_TR_batch * H_batch, axis=0)
        blocks[:] = s_batch[:, None, None]
        return blocks

    a_batch = H_sel[:, :k].T @ A_TR_batch
    s_batch = np.sum(A_TR_batch * H_batch, axis=0)

    blocks[:, :k, :k] = T_sel[None, :, :]
    blocks[:, :k, k:k + 1] = a_batch.T[:, :, None]
    blocks[:, k:k + 1, :k] = a_batch.T[:, None, :]
    blocks[:, k, k] = s_batch
    return blocks


# ---------------- Main function (CPU-only) ----------------


def get_activity_index(
    localizer_to_use: str,
    H: np.ndarray,
    R: np.ndarray,
    N: np.ndarray,
    n_sources_to_localize: int,
    r: int,
    batch_size: int = 16384,
    show_progress: bool = True
) -> Tuple[List[int], np.ndarray, int, np.ndarray]:
    """
    CPU-only version of the hybrid-fast algorithm.
    All computations are NumPy-based (no GPU).
    """
    # Heavy precomputations (once)
    R_inv = np.linalg.pinv(R)
    N_inv = np.linalg.pinv(N)

    A_R = R_inv @ H
    A_N = N_inv @ H
    A_TR = (R_inv @ N @ R_inv) @ H

    H_cpu = H.copy()
    A_R_cpu = A_R
    A_N_cpu = A_N
    A_TR_cpu = A_TR

    n_channels, n_sources = H_cpu.shape

    # Selection buffers
    max_k = n_sources_to_localize
    H_sel = np.zeros((n_channels, max_k), dtype=H_cpu.dtype)
    S_sel = np.zeros((0, 0), dtype=H_cpu.dtype)
    G_sel = np.zeros((0, 0), dtype=H_cpu.dtype)
    T_sel = np.zeros((0, 0), dtype=H_cpu.dtype)

    index_max: List[int] = []
    act_index_values: List[float] = []

    func = _choose_activity_index(localizer_to_use)  # must be CPU function

    selected_mask = np.zeros(n_sources, dtype=bool)
    selected_count = 0
    all_indices = np.arange(n_sources, dtype=np.int64)

    for outer_iter in range(n_sources_to_localize):
        best_val = -np.inf
        best_idx = None

        if show_progress:
            pbar = tqdm(total=n_sources,
                        desc=f"iter{outer_iter+1}/{n_sources_to_localize}")

        selected_mask[:] = False
        for idx in index_max:
            selected_mask[idx] = True

        need_G = ("mai" in localizer_to_use) or (localizer_to_use == "mpz_mvp")
        need_T = ("mpz" in localizer_to_use)

        for start in range(0, n_sources, batch_size):
            end = min(n_sources, start + batch_size)
            batch_idx = all_indices[start:end]
            batch_idx = batch_idx[~selected_mask[batch_idx]]
            if batch_idx.size == 0:
                if show_progress:
                    pbar.update(end - start)
                continue

            # Current batch (CPU)
            A_R_batch = A_R_cpu[:, batch_idx]
            H_batch = H_cpu[:, batch_idx]

            blocks_S = _build_S_blocks_batch_cpu(H_sel, S_sel, A_R_batch,
                                                 H_batch)

            if need_G:
                A_N_batch = A_N_cpu[:, batch_idx]
                blocks_G = _build_G_blocks_batch_cpu(H_sel, G_sel, A_N_batch,
                                                     H_batch)
            else:
                blocks_G = None

            if need_T:
                A_TR_batch = A_TR_cpu[:, batch_idx]
                blocks_T = _build_T_blocks_batch_cpu(H_sel, T_sel, A_TR_batch,
                                                     H_batch)
            else:
                blocks_T = None

            # Evaluate localizer
            for j_i, j in enumerate(batch_idx):
                if "mai" in localizer_to_use:
                    val = func(blocks_G[j_i],
                               blocks_S[j_i],
                               current_iter=outer_iter + 1,
                               r=r)
                elif "mpz" in localizer_to_use:
                    if localizer_to_use == "mpz_mvp":
                        Q_cpu = np.linalg.pinv(blocks_S[j_i]) - np.linalg.pinv(
                            blocks_G[j_i])
                        val = func(blocks_S[j_i],
                                   blocks_T[j_i],
                                   Q_cpu,
                                   r=r,
                                   current_iter=outer_iter + 1)
                    else:
                        val = func(blocks_S[j_i],
                                   blocks_T[j_i],
                                   current_iter=outer_iter + 1,
                                   r=r)
                else:
                    val = func(blocks_S[j_i], current_iter=outer_iter + 1, r=r)

                if val > best_val:
                    best_val = val
                    best_idx = int(j)

            if show_progress:
                pbar.update(end - start)

        if show_progress:
            pbar.close()

        if best_idx is None:
            break

        index_max.append(int(best_idx))
        act_index_values.append(float(best_val))

        # Update selection state
        H_sel[:,
              selected_count:selected_count + 1] = H_cpu[:,
                                                         best_idx:best_idx + 1]

        # Update reference blocks
        aR_chosen = A_R_cpu[:, best_idx:best_idx + 1]
        S_sel = _build_S_blocks_batch_cpu(H_sel, S_sel, aR_chosen,
                                          H_cpu[:, best_idx:best_idx + 1])[0]
        if need_G:
            aN_chosen = A_N_cpu[:, best_idx:best_idx + 1]
            G_sel = _build_G_blocks_batch_cpu(H_sel, G_sel, aN_chosen,
                                              H_cpu[:,
                                                    best_idx:best_idx + 1])[0]
        if need_T:
            aTR_chosen = A_TR_cpu[:, best_idx:best_idx + 1]
            T_sel = _build_T_blocks_batch_cpu(H_sel, T_sel, aTR_chosen,
                                              H_cpu[:,
                                                    best_idx:best_idx + 1])[0]

        selected_count += 1

    H_res = None
    H_res = H[:, index_max].copy()

    print(index_max, r)
    return index_max, np.array(act_index_values, dtype=float), r, H_res


def _choose_activity_index(localizer_to_use: str):
    """
    Specify activity index to use.

    Parameters:
    -----------
    localizer_to_use: str
        name of the localizer to use

    Returns:
    -----------
        func: function _get_xxx_xxx specific for given activity index
    """
    if localizer_to_use == "mai":
        func = _get_simple_mai
    elif localizer_to_use == "mpz":
        func = _get_simple_mpz
    elif localizer_to_use == "mai_mvp":
        func = _get_mai_mvp
    elif localizer_to_use == "mpz_mvp":
        func = _get_mpz_mvp
    else:
        raise ValueError(
            "Only possible localizers to specify are: 'mai', 'mpz', "
            "'mai_mvp', 'mpz_mvp'")

    return func


def _get_simple_mai(G: np.ndarray,
                    S: np.ndarray,
                    current_iter: int,
                    r: int = None) -> float:
    """
    Using eq. 39 from [1]_.

    MAI = tr{GS^(-1)} - l
        where G = H.T @ N^(-1) @ H [eq. 8]
              S = H.T @ R^(-1) @ H [eq. 5]

    Parameters:
    -----------
    G : array-like, shape (n_sources, n_sources)
        matrix defined in eq. 8, calculated in function _get_G()
    S : array-like, shape (n_sources, n_sources)
        matrix defined in eq. 5, calculated in function _get_S()
    current_iter: int
        Number of dipole being localized in current iteration.

    Returns:
    -----------
    mai: float
        MAI activity index as defined in [2]_.

    References
    ----------
    .. [1] Piotrowski, T., Nikadon, J., & Moiseev, A. (2021).
           Localization of brain activity from EEG/MEG using MV-PURE framework.
           *Biomedical Signal Processing and control, 64*, 102243.
    .. [2] Moiseev, A., Gaspar, J. M., Schneider, J. A., & Herdman, A. T. (2011).
           Application of multi-source minimum variance beamformers for reconstruction of correlated neural activity.
           *NeuroImage, 58*(2), 481-496.

    """
    if isinstance(G, float) and isinstance(S, float):
        mai = G / S - current_iter
    else:
        # Following implementation from source article we should useL
        mai = np.trace(np.matmul(G, np.linalg.pinv(S))) - current_iter
    return mai


def _get_simple_mpz(S: np.ndarray,
                    T: np.ndarray,
                    current_iter: int,
                    r: int = None) -> float:
    """
    Using eq. 42 from [1]_.
    MPZ = tr{S@T^(-1)} - l
        where S = H.T @ R^(-1) @ H  [eq. 5]
              T = H.T @ R^(-1) @ N @ R^(-1) @ H  [eq. 12]

    Parameters:
    -----------
    S : array-like, shape (n_sources, n_sources)
        matrix defined in eq. 5, calculated in function _get_S()
    T : array-like, shape (n_sources, n_sources)
        matrix defined in eq. 12, calculated in function _get_T()
    current_iter: int
        number of dipole being localized

    Returns:
    -----------
    mpz: float
        MPZ activity index as defined in [2]_.

    References
    ----------
    .. [1] Piotrowski, T., Nikadon, J., & Moiseev, A. (2021).
           Localization of brain activity from EEG/MEG using MV-PURE framework.
           *Biomedical Signal Processing and control, 64*, 102243.
    .. [2] Moiseev, A., Gaspar, J. M., Schneider, J. A., & Herdman, A. T. (2011).
           Application of multi-source minimum variance beamformers for reconstruction of correlated neural activity.
           *NeuroImage, 58*(2), 481-496.
    """
    if isinstance(S, float) and isinstance(T, float):
        mpz = S / T - current_iter
    else:
        # Following implementation from source article we should useL
        mpz = np.trace(np.matmul(S, np.linalg.pinv(T))) - current_iter

    return mpz


def _get_mai_mvp(G: np.ndarray, S: np.ndarray, r: int,
                 current_iter: int) -> float:
    """
    MAI_MVP = sum_to_l(eigenval(G @ S^(-1)) - l for 1 <= l <= r
            = sum_to_r(eigenval(G @ S^(-1)) - r for l > r
        where G = H.T @ N^(-1) @ H  [eq. 8]
              S = H.T @ R^(-1) @ H  [eq. 5]
            and H - leadfield matrix, N - noise covariance, and R - data covariance.
    Note: Using full-rank MAI_MVP localizer (r = n_sources_to_localize) replicates results for MAI (_get_simple_mai).
    Parameters
    ----------
    G : array-like
        Matrix defined in eq. 8, computed in function _get_G()
    S : array-like
        Matrix defined in eq. 5, computed in function _get_S()
    r : int
        Optimization parameter
    current_iter : int
        Number of dipole being localized in current iteration.

    Returns
    -------
    mai_mvp : float
        MAI_MVP activity index

    """
    # same as mai_ext -- maybe just refer to _get_mai_extension?
    # if current_iter [l] <= r - use simple MAI from _get_simple_mai()
    if current_iter <= r:
        mai_mvp = _get_simple_mai(G, S, current_iter)
    else:
        s, u = np.linalg.eig(np.matmul(G, np.linalg.pinv(S)))
        sorted_idx = np.argsort(s)[::-1]
        s = s[sorted_idx]
        mai_mvp = np.sum(s[:r]) - r
    return mai_mvp


def _get_mpz_mvp(S: np.ndarray, T: np.ndarray, Q: np.ndarray, r: int,
                 current_iter: int) -> float:
    """
    MPZ_MVP = trace{S@T^(-1)} - r             for r <= l
            = trace(S @ T^(-1) @ P_SQ_r) - r  for l > r
        where S = H.T @ R^(-1) @ H  [eq. 5]
              T = H.T @ R^(-1) @ N @ R^(-1) @ H  [eq. 12]
        and P_SQ_r is the oblique projection matrix onto the subspace of S @ Q,
            H - leadfield matrix,
            N - noise covariance, and R - data covariance.

    Note: Using full-rank MPZ_MVP localizer (r = n_sources_to_localize) replicates results for MPZ (_get_simple_mpz).

    Parameters
    ----------
    S : array-like
        Matrix defined in eq. 5, calculated in function _get_S()
    T : array-like
        Matrix defined in eq. 12, calculated in function _get_T()
    Q : array-like
        Matrix Q calculated as Q = S^(-1) - G^(-1) in function _get_Q().
    r : int
        Optimization parameter
    current_iter : int
        Number of dipole being localized in current iteration.

    Returns
    -------
    mpz_mvp : float
        MPZ_MVP activity index
    """
    # if current_iter [l] <= r - use simple MPZ from _get_simple_mpz()
    if current_iter <= r:
        mpz_mvp = _get_simple_mpz(S, T, current_iter)
    else:
        s, u = np.linalg.eig(np.matmul(S, Q))
        sorted_idx = np.argsort(s)[::-1]
        u = u[:, sorted_idx]
        proj_matrix = u @ np.block([[
            np.eye(r), np.zeros((r, current_iter - r))
        ], [np.zeros((current_iter - r, current_iter))]]) @ np.linalg.pinv(u)
        mpz_mvp = np.trace(S @ np.linalg.pinv(T) @ proj_matrix) - r

    return mpz_mvp
