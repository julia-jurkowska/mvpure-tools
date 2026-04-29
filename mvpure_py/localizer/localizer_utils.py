""" Functions used during sources localization. """

# Authors: Julia Jurkowska, Tomasz Piotrowski

import os
from typing import List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve, pinv
from tqdm import tqdm

# Number of threads used by BLAS/LAPACK - ADJUSTABLE
os.environ["OMP_NUM_THREADS"] = "10"


# ============================================================
# -------- PSD low-rank pseudo-inverse (apply-only) ----------
# ============================================================

class PSDPinvApply:
    """
    Robust pseudo-inverse operator for PSD (possibly low-rank) matrix A:
        A^+ ≈ U diag(1/lam) U^T  (truncated eigenpairs)

    Use: op.apply(X) = A^+ @ X
    """
    __slots__ = ("U", "lam", "cutoff")

    def __init__(self, U: np.ndarray, lam: np.ndarray, cutoff: float):
        self.U = U
        self.lam = lam
        self.cutoff = cutoff

    def apply(self, X: np.ndarray) -> np.ndarray:
        if self.U.size == 0:
            return np.zeros_like(X)
        UtX = self.U.T @ X
        return self.U @ (UtX / self.lam[:, None])


def psd_pinv_operator(A: np.ndarray,
                      rmax: Optional[int] = None,
                      eps_rel: float = 1e-10) -> PSDPinvApply:
    """
    Build a robust pseudo-inverse *operator* for PSD A (low-rank supported).

    Parameters
    ----------
    A : (m,m) PSD matrix (may be singular)
    rmax : Optional[int]
        Keep at most rmax eigenpairs (largest).
    eps_rel : float
        Keep eigenvalues > eps_rel * max_eig.

    Returns
    -------
    PSDPinvApply
        .apply(X) computes A^+ @ X
    """
    As = 0.5 * (A + A.T)  # enforce symmetry numerically
    lam_all, U_all = np.linalg.eigh(As)  # ascending
    lam_max = lam_all[-1] if lam_all.size else 0.0

    if lam_max <= 0:
        return PSDPinvApply(
            U=np.zeros((A.shape[0], 0), dtype=A.dtype),
            lam=np.zeros((0,), dtype=A.dtype),
            cutoff=0.0
        )

    cutoff = eps_rel * float(lam_max)
    keep = lam_all > cutoff

    if rmax is not None and int(np.sum(keep)) > int(rmax):
        idx = np.argsort(lam_all)[-int(rmax):]
        keep2 = np.zeros_like(keep, dtype=bool)
        keep2[idx] = True
        keep = keep & keep2

    lam = lam_all[keep].astype(A.dtype, copy=False)
    U = U_all[:, keep].astype(A.dtype, copy=False)

    return PSDPinvApply(U=U, lam=lam, cutoff=cutoff)


# ============================================================
# ---------------- Helper: build batch blocks ----------------
# ============================================================

def _build_blocks_batch(H_sel: np.ndarray,
                        M_sel: Optional[np.ndarray],
                        A_M_batch: np.ndarray,
                        H_batch: np.ndarray) -> np.ndarray:
    """
    Construct block matrices for a batch of candidate sources.
    """
    k = 0 if M_sel is None else M_sel.shape[0]
    B = A_M_batch.shape[1]
    m = k + 1
    dtype = A_M_batch.dtype

    if k == 0:
        s_batch = np.sum(A_M_batch * H_batch, axis=0)
        return s_batch.reshape(B, 1, 1).astype(dtype, copy=False)

    a_batch = H_sel[:, :k].T @ A_M_batch
    s_batch = np.sum(A_M_batch * H_batch, axis=0)

    blocks = np.empty((B, m, m), dtype=dtype)
    blocks[:, :k, :k] = M_sel[None, :, :]
    blocks[:, :k, k:k+1] = a_batch.T.reshape(B, k, 1)
    blocks[:, k:k+1, :k] = a_batch.T.reshape(B, 1, k)
    blocks[:, k, k] = s_batch

    return blocks


# ============================================================
# ---------------- Activity index: simplified ----------------
# ============================================================

def _simple_mai_batch(blocks_G: np.ndarray,
                      blocks_S: np.ndarray,
                      current_iter: int) -> np.ndarray:
    """
    MAI = trace(G S⁻¹) - current_iter
    """
    B, m, _ = blocks_S.shape

    if m == 1:
        return blocks_G[:, 0, 0] / blocks_S[:, 0, 0] - current_iter

    I = np.eye(m)

    def _process_block(i):
        try:
            invS = cho_solve(cho_factor(blocks_S[i], lower=False, check_finite=False),
                             I, check_finite=False)
        except np.linalg.LinAlgError:
            invS = pinv(blocks_S[i])

        return np.trace(blocks_G[i] @ invS) - current_iter

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
    MAI-MVP = sum of top-r eigenvalues of (G S⁻¹) - r
    """
    B, m, _ = blocks_S.shape

    if current_iter <= r:
        return _simple_mai_batch(blocks_G, blocks_S, current_iter)

    I = np.eye(m)

    def _process_block(i):
        if m == 1:
            return blocks_G[i, 0, 0] / blocks_S[i, 0, 0] - r

        try:
            invS = cho_solve(cho_factor(blocks_S[i], lower=False, check_finite=False),
                             I, check_finite=False)
        except np.linalg.LinAlgError:
            invS = pinv(blocks_S[i])

        M = blocks_G[i] @ invS
        s, _ = np.linalg.eig(M)
        s_sorted = np.sort(s)[::-1].real

        return np.sum(s_sorted[:r]) - r

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
    elif localizer_to_use == "mai_mvp":
        return lambda Gs, Ss, Ts, current_iter, r: _mai_mvp_batch(Gs, Ss, r, current_iter)
    else:
        raise ValueError("Allowed: 'mai' and 'mai_mvp'")


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
    """

    # --------------------------------------------------------
    # Robust precompute for PSD low-rank R and N (NO Cholesky)
    # --------------------------------------------------------
    # A_R  = R^+ H
    # A_N  = N^+ H
    # A_TR = R^+ N R^+ H   (same object as: pinv(R) @ N @ pinv(R) @ H)
    # --------------------------------------------------------

    R_pinv = psd_pinv_operator(R, rmax=None, eps_rel=1e-10)
    N_pinv = psd_pinv_operator(N, rmax=None, eps_rel=1e-10)

    A_R = R_pinv.apply(H)
    A_N = N_pinv.apply(H)
    A_TR = R_pinv.apply(N @ A_R)

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
    H_res = H[:, sorted(index_max)].copy() if index_max else np.zeros((n_channels, 0), dtype=H.dtype)

    # Final summary
    print("\n[Activity Index Result]")
    print(f"  Selected indices (index_max): {index_max}")
    print(f"  Index max values: {np.array(act_values, dtype=float)}")
    print(f"  Rank parameter (r): {r}\n")

    return index_max, np.array(act_values, dtype=float), r, H_res