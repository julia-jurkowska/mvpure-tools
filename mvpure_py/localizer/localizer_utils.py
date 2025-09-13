""" Functions used during sources localization. """

# Author: Julia Jurkowska

import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm

from ..viz import plot_RN_eigenvalues
from ..utils import algebra


def suggest_n_sources_and_rank(
    R: np.ndarray,
    N: np.ndarray,
    show_plot: bool = True,
    subject: str = None,
    n_sources_threshold: float = 1,
    rank_threshold: float = 1.5,
    **kwargs
) -> tuple[int, int]:
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
        _, eigvals = plot_RN_eigenvalues(R=R, N=N, subject=subject, return_eigvals=True, **kwargs)
    else:
        eigvals = algebra.get_pinv_RN_eigenvals(R=R, N=N)
        
    # Suggesting number of sources
    n_sources_temp = np.where(eigvals > n_sources_threshold)[0]
    if n_sources_temp.size == 0:
        raise ValueError(f"All eigenvalues of $\\mathrm{{RN}}^{{-1}}$ are smaller than {n_sources_threshold}.")
    else:
        n_sources = n_sources_temp[-1] + 1
    
    # Suggesting rank
    rank_temp = np.where(eigvals > rank_threshold)[0]
    if rank_temp.size == 0:
        raise ValueError(f"All eigenvalues of $\\mathrm{{RN}}^{{-1}}$ are smaller than {rank_threshold}.")
    else:
        rank = rank_temp[-1] + 1
    
    print(f"Suggested number of sources to localize: {n_sources}")
    print(f"Suggested rank is: {rank}")
    return int(n_sources), int(rank)


def get_activity_index(localizer_to_use: str,
                       H: np.ndarray,
                       R: np.ndarray,
                       N: np.ndarray,
                       n_sources_to_localize: int,
                       r: int) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Calculate activity index specified in ``localizer_to_use``.

    Parameters
    -----------
    localizer_to_use: str
        activity index to use.
        Options: "mai", "mpz", "mai_mvp", "mpz_mvp".
    H : array-like, shape (n_channels, n_sources)
        leadfield matrix
    R : array-like, shape (n_channels, n_channels)
        data covariance matrix
    N : array-like, shape (n_channels, n_channels)
        noise covariance matrix
    n_sources_to_localize: int
        number of sources to localize
    r : int
        optimization parameter

    Returns
    -----------
    index_max: array-like, shape (n_sources_to_localize,)
        indices of localized dipoles
    act_index_values: array-like, shape (n_sources_to_localize,)
        activity index value
    r : int
        optimization parameter used
    H_res: array-like, shape (n_channels, n_sources_to_localize)
        subset of leadfield matrix for localized dipoles
    """
    # placeholder for localizer activity index
    index_max = np.zeros(n_sources_to_localize)
    act_index_values = np.zeros(n_sources_to_localize)
    # leadfield of dipoles corresponding to max index
    H_res = None
    # iterate through number of sources to localize
    for n in range(1, n_sources_to_localize + 1):
        # placeholder for activity index for each source
        act = np.zeros(H.shape[1])
        # iterate through all sources
        for l in tqdm(range(H.shape[1]), total=H.shape[1]):
            # current dipole leadfield
            H_curr = H[:, l]
            if H_res is not None:
                if len(H_res.shape) == 1:
                    H_res = H_res.reshape(-1, 1)
                # if some source was selected in previous loop(s) [ n > 1 ]
                # stack current leadfield with selected dipole(s) leadfield(s)
                H_curr = np.hstack((H_res, H_curr.reshape(-1, 1)))

            # choose activity index
            func = _choose_activity_index(localizer_to_use)
            S_curr = algebra._get_S(H_curr, R)

            # MAI family
            if "mai" in localizer_to_use:
                G_curr = algebra._get_G(H_curr, N)
                act[l] = func(G_curr, S_curr, current_iter=n, r=r)

            # MPZ family
            elif "mpz" in localizer_to_use:
                T_curr = algebra._get_T(H_curr, R, N)
                if localizer_to_use == "mpz_mvp":
                    G_curr = algebra._get_G(H_curr, N)
                    Q_curr = algebra._get_Q(S_curr, G_curr)
                    act[l] = func(S_curr, T_curr, Q_curr, r=r, current_iter=n)
                else:
                    act[l] = func(S_curr, T_curr, current_iter=n, r=r)

        # find for which dipole index is max
        index_max[n-1] = int(np.argmax(act))
        act_index_values[n-1] = np.max(act)

        # create H_res with leadfields of selected dipoles
        if H_res is None:
            H_res = H[:, int(index_max[n-1])]
        else:
            H_res = np.hstack((H_res, H[:, int(index_max[n-1])].reshape(-1, 1)))

    if H_res.ndim == 1:
        H_res = H_res[:, np.newaxis]
    index_max = [int(i) for i in index_max]
    print(f"Leadfield indices corresponding to localized sources: {index_max}")

    return index_max, act_index_values, r, H_res


def _choose_activity_index(
        localizer_to_use: str
):
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
        raise ValueError("Only possible localizers to specify are: 'mai', 'mpz', "
                         "'mai_mvp', 'mpz_mvp'")

    return func


def _get_simple_mai(G: np.ndarray, S: np.ndarray, current_iter: int, r: int = None) -> float:
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


def _get_simple_mpz(S: np.ndarray, T: np.ndarray, current_iter: int, r: int = None) -> float:
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


def _get_mai_mvp(G: np.ndarray, S: np.ndarray, r: int, current_iter: int) -> float:
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


def _get_mpz_mvp(S: np.ndarray, T: np.ndarray, Q: np.ndarray, r: int, current_iter: int) -> float:
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
        proj_matrix = u @ np.block(
            [[np.eye(r), np.zeros((r, current_iter - r))], [np.zeros((current_iter - r, current_iter))]]
        ) @ np.linalg.pinv(u)
        mpz_mvp = np.trace(S @ np.linalg.pinv(T) @ proj_matrix) - r

    return mpz_mvp
