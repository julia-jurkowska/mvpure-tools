""" Functions used during sources localization. """

# Author: Julia Jurkowska

import os
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg as sla
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve, pinv
from tqdm import tqdm

from ..utils import algebra
from ..viz import plot_RN_eigenvalues

# Number of threads used by BLAS/LAPACK - ADJUSTABLE
os.environ["OMP_NUM_THREADS"] = "10"


# ============================================================
# ---------------- Helper: build batch blocks ----------------
# ============================================================

def _build_blocks_batch(H_sel: np.ndarray,
                        M_sel: Optional[np.ndarray],
                        A_M_batch: np.ndarray,
                        H_batch: np.ndarray) -> np.ndarray:
    """
    Construct block matrices for a batch of candidate sources.

    Parameters
    ----------
    H_sel : np.ndarray
        Already selected columns of H.
    M_sel : np.ndarray or None
        Current accumulated block matrix (None if empty).
    A_M_batch : np.ndarray
        Batch of transformed candidate sources (e.g., R⁻¹ H).
    H_batch : np.ndarray
        Corresponding candidate columns from H.

    Returns
    -------
    blocks : np.ndarray
        Batch of block matrices, shape (B, m, m).
    """
    k = 0 if M_sel is None else M_sel.shape[0]   # size of current block
    B = A_M_batch.shape[1]                       # number of candidates in the batch
    m = k + 1                                    # new block size
    dtype = A_M_batch.dtype

    if k == 0:
        # First iteration: build scalar blocks (1x1 matrices)
        s_batch = np.sum(A_M_batch * H_batch, axis=0)
        return s_batch.reshape(B, 1, 1).astype(dtype, copy=False)

    # Cross-terms with already selected sources
    a_batch = H_sel[:, :k].T @ A_M_batch
    s_batch = np.sum(A_M_batch * H_batch, axis=0)

    # Assemble full block matrices for the batch
    blocks = np.empty((B, m, m), dtype=dtype)
    blocks[:, :k, :k] = M_sel[None, :, :]             # top-left block (previous selection)
    blocks[:, :k, k:k+1] = a_batch.T.reshape(B, k, 1) # last column
    blocks[:, k:k+1, :k] = a_batch.T.reshape(B, 1, k) # last row
    blocks[:, k, k] = s_batch                         # bottom-right element

    return blocks


# ============================================================
# ---------------- Activity index: simplified ----------------
# ============================================================

def _simple_mai_batch(blocks_G: np.ndarray,
                      blocks_S: np.ndarray,
                      current_iter: int) -> np.ndarray:
    """
    Compute the MAI activity index for a batch (simplified form) in parallel.

    Definition:
        MAI = trace(G S⁻¹) - current_iter
    """
    B, m, _ = blocks_S.shape

    # Handle 1x1 blocks quickly
    if m == 1:
        return blocks_G[:, 0, 0] / blocks_S[:, 0, 0] - current_iter

    I = np.eye(m)

    def _process_block(i):
        # Inverse of S
        try:
            invS = cho_solve(cho_factor(blocks_S[i], lower=False, check_finite=False),
                             I, check_finite=False)
        except np.linalg.LinAlgError:
            invS = pinv(blocks_S[i])

        # Activity index for this block
        return np.trace(blocks_G[i] @ invS) - current_iter

    # Parallel execution across all blocks
    results = Parallel(n_jobs=-1)(delayed(_process_block)(i) for i in range(B))
    return np.array(results, dtype=float)


def _simple_mpz_batch(blocks_S: np.ndarray,
                      blocks_T: np.ndarray,
                      current_iter: int) -> np.ndarray:
    """
    Compute the MPZ activity index for a batch (simplified form) in parallel.

    Definition:
        MPZ = trace(S T⁻¹) - current_iter
    """
    B, m, _ = blocks_S.shape

    # Handle 1x1 blocks quickly
    if m == 1:
        return blocks_S[:, 0, 0] / blocks_T[:, 0, 0] - current_iter

    I = np.eye(m)

    def _process_block(i):
        # Inverse of T
        try:
            invT = cho_solve(
                cho_factor(blocks_T[i], lower=False, check_finite=False),
                I, check_finite=False
            )
        except np.linalg.LinAlgError:
            invT = np.linalg.pinv(blocks_T[i])

        return np.trace(blocks_S[i] @ invT) - current_iter

    # Parallel execution across blocks
    results = Parallel(n_jobs=-1)(delayed(_process_block)(i) for i in range(B))
    return np.array(results, dtype=float)


# ============================================================
# ---------------- Activity index: full versions --------------
# ============================================================

def _mai_mvp_batch(blocks_G: np.ndarray,
                   blocks_S: np.ndarray,
                   r: int,
                   current_iter: int) -> np.ndarray:
    """
    Compute the MAI-MVP activity index for a batch in parallel.

    This version matches the eigenvalue-based definition:
        MAI-MVP = sum of top-r eigenvalues of (G S⁻¹) - r
    """
    B, m, _ = blocks_S.shape

    # For early iterations (<= r), fall back to simplified version
    if current_iter <= r:
        return _simple_mai_batch(blocks_G, blocks_S, current_iter)

    I = np.eye(m)

    def _process_block(i):
        if m == 1:
            return blocks_G[i, 0, 0] / blocks_S[i, 0, 0] - r

        # Inverse of S
        try:
            invS = cho_solve(cho_factor(blocks_S[i], lower=False, check_finite=False),
                             I, check_finite=False)
        except np.linalg.LinAlgError:
            invS = pinv(blocks_S[i])

        # Eigen-decomposition of G S⁻¹
        M = blocks_G[i] @ invS
        s, _ = np.linalg.eig(M)
        s_sorted = np.sort(s)[::-1].real  # descending order

        return np.sum(s_sorted[:r]) - r

    # Parallel execution across blocks (order preserved)
    results = Parallel(n_jobs=-1)(delayed(_process_block)(i) for i in range(B))
    return np.array(results, dtype=float)


def _mpz_mvp_batch(blocks_S: np.ndarray,
                   blocks_T: np.ndarray,
                   blocks_G: np.ndarray,
                   r: int,
                   current_iter: int) -> np.ndarray:
    """
    Compute the MPZ-MVP activity index for a batch.

    Exact CPU version with block projection:
        MPZ-MVP = trace(S * T⁻¹ P) - r
    where P is the oblique projection onto the top-r eigenspace of SQ.
    """
    B, m, _ = blocks_S.shape
    I = np.eye(m)

    if current_iter <= r:
        return _simple_mpz_batch(blocks_S, blocks_T, current_iter)

    def _process_block(i):
        # Inverse of S
        try:
            invS = cho_solve(cho_factor(blocks_S[i], lower=False, check_finite=False),
                             I, check_finite=False)
        except np.linalg.LinAlgError:
            invS = pinv(blocks_S[i])

        # Inverse of G
        try:
            invG = cho_solve(cho_factor(blocks_G[i], lower=False, check_finite=False),
                             I, check_finite=False)
        except np.linalg.LinAlgError:
            invG = pinv(blocks_G[i])

        # Q = S⁻¹ - G⁻¹
        Q = invS - invG

        # Eigen-decomposition of S Q
        s, u = np.linalg.eig(blocks_S[i] @ Q)
        sorted_idx = np.argsort(s)[::-1]  # sort eigenvalues descending
        u = u[:, sorted_idx]

        # Projection matrix onto top-r subspace
        proj_matrix = u @ np.block([
            [np.eye(r), np.zeros((r, current_iter - r))],
            [np.zeros((current_iter - r, current_iter))]
        ]) @ pinv(u)

        # Apply T⁻¹ to projection
        try:
            temp = cho_solve(cho_factor(blocks_T[i], lower=False, check_finite=False),
                             proj_matrix, check_finite=False)
        except np.linalg.LinAlgError:
            temp = pinv(blocks_T[i]) @ proj_matrix

        return np.trace(blocks_S[i] @ temp) - r

    # Parallel execution across all B blocks
    results = Parallel(n_jobs=-1)(delayed(_process_block)(i) for i in range(B))
    return np.array(results, dtype=float)

# ============================================================
# ---------------- Dispatcher: choose function ---------------
# ============================================================

def _choose_activity_index_batch(localizer_to_use: str):
    """
    Return the appropriate activity index function depending on the localizer name.
    """
    if localizer_to_use == "mai":
        return lambda Gs, Ss, Ts, current_iter, r: _simple_mai_batch(Gs, Ss, current_iter)
    elif localizer_to_use == "mpz":
        return lambda Gs, Ss, Ts, current_iter, r: _simple_mpz_batch(Ss, Ts, current_iter)
    elif localizer_to_use == "mai_mvp":
        return lambda Gs, Ss, Ts, current_iter, r: _mai_mvp_batch(Gs, Ss, r, current_iter)
    elif localizer_to_use == "mpz_mvp":
        return lambda Gs, Ss, Ts, current_iter, r: _mpz_mvp_batch(Ss, Ts, Gs, r, current_iter)
    else:
        raise ValueError("Allowed: 'mai', 'mpz', 'mai_mvp', 'mpz_mvp'")


# ============================================================
# ---------------- Main function -----------------------------
# ============================================================

def get_activity_index(localizer_to_use: str,
                       H: np.ndarray,
                       R: np.ndarray,
                       N: np.ndarray,
                       n_sources_to_localize: int,
                       r: int,
                       batch_size: int = 16384,
                       show_progress: bool = True
                       ) -> Tuple[List[int], np.ndarray, int, np.ndarray]:
    """
    Greedy source localization algorithm.

    Parameters
    ----------
    localizer_to_use : str
        Which localizer to use: 'mai', 'mpz', 'mai_mvp', 'mpz_mvp'.
    H : np.ndarray
        Leadfield matrix (channels x sources).
    R : np.ndarray
        Measurement covariance matrix.
    N : np.ndarray
        Noise covariance matrix.
    n_sources_to_localize : int
        Number of sources to select.
    r : int
        Rank parameter.
    batch_size : int
        Number of candidates processed per batch.
    show_progress : bool
        Whether to show a progress bar.

    Returns
    -------
    index_max : List[int]
        Indices of selected sources.
    act_values : np.ndarray
        Activity index values for each selected source.
    r : int
        Rank parameter (unchanged).
    H_res : np.ndarray
        Selected columns of H.
    """
    # Precompute transformed matrices with Cholesky or fallback to pseudoinverse
    try:
        choR = sla.cho_factor(R, lower=False, check_finite=False)
        A_R = sla.cho_solve(choR, H, check_finite=False)
    except:
        A_R = np.linalg.pinv(R) @ H

    try:
        choN = sla.cho_factor(N, lower=False, check_finite=False)
        A_N = sla.cho_solve(choN, H, check_finite=False)
    except:
        A_N = np.linalg.pinv(N) @ H

    try:
        A_TR = sla.cho_solve(choR, N @ A_R, check_finite=False)
    except:
        A_TR = (np.linalg.pinv(R) @ N @ np.linalg.pinv(R)) @ H

    # Initialization
    n_channels, n_sources = H.shape
    H_sel = np.zeros((n_channels, n_sources_to_localize), dtype=H.dtype)
    S_sel = G_sel = T_sel = None
    index_max: List[int] = []
    act_values: List[float] = []

    func = _choose_activity_index_batch(localizer_to_use)
    selected_mask = np.zeros(n_sources, dtype=bool)
    all_indices = np.arange(n_sources, dtype=int)

    need_G = ("mai" in localizer_to_use) or (localizer_to_use == "mpz_mvp")
    need_T = ("mpz" in localizer_to_use)

    # Greedy selection loop
    for outer_iter in range(n_sources_to_localize):
        best_val = -np.inf
        best_idx = None

        if show_progress:
            pbar = tqdm(total=n_sources, desc=f"iter {outer_iter+1}/{n_sources_to_localize}")

        # Process candidates in batches
        for start in range(0, n_sources, batch_size):
            end = min(n_sources, start + batch_size)
            batch_idx = all_indices[start:end]
            valid = ~selected_mask[batch_idx]
            if not np.any(valid):
                if show_progress:
                    pbar.update(end - start)
                continue
            batch_idx = batch_idx[valid]

            # Build block structures
            A_R_batch = A_R[:, batch_idx]
            H_batch = H[:, batch_idx]
            blocks_S = _build_blocks_batch(H_sel[:, :len(index_max)], S_sel, A_R_batch, H_batch)

            if need_G:
                A_N_batch = A_N[:, batch_idx]
                blocks_G = _build_blocks_batch(H_sel[:, :len(index_max)], G_sel, A_N_batch, H_batch)
            else:
                blocks_G = np.zeros_like(blocks_S)

            if need_T:
                A_TR_batch = A_TR[:, batch_idx]
                blocks_T = _build_blocks_batch(H_sel[:, :len(index_max)], T_sel, A_TR_batch, H_batch)
            else:
                blocks_T = np.zeros_like(blocks_S)

            # Compute activity index values for the batch
            vals = func(blocks_G, blocks_S, blocks_T, current_iter=outer_iter + 1, r=r)
            local_best_idx = int(np.argmax(vals))
            local_best_val = float(vals[local_best_idx])

            # Update global best
            if local_best_val > best_val:
                best_val = local_best_val
                best_idx = int(batch_idx[local_best_idx])

            if show_progress:
                pbar.update(end - start)

        if show_progress:
            pbar.close()

        if best_idx is None:
            break

        # Save chosen index and its value
        index_max.append(best_idx)
        act_values.append(float(best_val))
        sel_pos = len(index_max) - 1
        H_sel[:, sel_pos:sel_pos+1] = H[:, best_idx:best_idx+1]
        selected_mask[best_idx] = True

        # Incrementally update block matrices S_sel / G_sel / T_sel
        aR_chosen = A_R[:, best_idx:best_idx+1]
        S_sel = _build_blocks_batch(H_sel[:, :sel_pos+1], S_sel, aR_chosen,
                                    H[:, best_idx:best_idx+1])[0]

        if need_G:
            aN_chosen = A_N[:, best_idx:best_idx+1]
            G_sel = _build_blocks_batch(H_sel[:, :sel_pos+1], G_sel, aN_chosen,
                                        H[:, best_idx:best_idx+1])[0]

        if need_T:
            aTR_chosen = A_TR[:, best_idx:best_idx+1]
            T_sel = _build_blocks_batch(H_sel[:, :sel_pos+1], T_sel, aTR_chosen,
                                        H[:, best_idx:best_idx+1])[0]

    # Collect selected sources
    H_res = H[:, index_max].copy() if index_max else np.zeros((n_channels, 0), dtype=H.dtype)

    # Final summary
    print("\n[Activity Index Result]")
    print(f"  Selected indices (index_max): {index_max}")
    print(f"  Rank parameter (r): {r}")

    return index_max, np.array(act_values, dtype=float), r, H_res

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
