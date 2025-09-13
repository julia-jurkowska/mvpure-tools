""" Functions for computing algebraic expressions. """
# Author: Julia Jurkowska

import numpy as np


def get_pinv_RN_eigenvals(
        R: np.ndarray,
        N: np.ndarray
):
    eigvals = np.real(
        np.sort(np.linalg.eigvals(R @ np.linalg.pinv(N)))
    )[::-1]
    return eigvals


def _get_S(H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute the matrix S based on the given transformation.
    Using eq. 21.

    S is defined as: S := H^t * R^(-1) * H
    Where:
    - H is a leadfield for given sources
    - R is data covariance matrix

    Parameters:
    -----------
    H : array-like
        Leadfield matrix
    R : array-like
        Data covariance matrix

    Returns
    -------
    array-like: Computed matrix S
    """
    if H.ndim == 3 and H.shape[2] == 1:
        H = np.squeeze(H, axis=-1)
    if H.shape[0] == R.shape[0]:
        return np.transpose(H) @ np.linalg.pinv(R) @ H
    elif H.shape[1] == R.shape[0]:
        return H @ np.linalg.pinv(R) @ np.transpose(H)


def _get_G(H: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Compute the matrix G based on a given transformation.
    Using eq. 8.

    G is defined as: G := H^t * N^(-1) * H
    Where:
    - H is a leadfield for given sources
    - N is noise covariance matrix

    Parameters:
    -----------
    H : array-like
        Leadfield matrix
    N : array-like
        Noise covariance matrix

    Returns
    -------
    float | array-like : Computed matrix G
    """
    if H.ndim == 3 and H.shape[2] == 1:
        H = np.squeeze(H, axis=-1)
    if H.shape[0] == N.shape[0]:
        return np.transpose(H) @ np.linalg.pinv(N) @ H
    elif H.shape[1] == N.shape[0]:
        return H @ np.linalg.pinv(N) @ np.transpose(H)


def _get_T(H: np.ndarray, R: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Compute the matrix T based on a given transformation.
    Using eq. 22.

    T is defined as: T := H^t * R^(-1) * N * R^(-1) * H * S^*(-1)
    Where:
    S := H^t * R^(-1) * H
    G := H^t * N^(-1) * H
    - H is a leadfield for given sources
    - R is data covariance matrix
    - N is noise covariance matrix

    Parameters
    ----------
    H : array-like
        Leadfield matrix
    R : array-like
        Data covariance matrix
    N : array-like
        Noise covariance matrix

    Returns
    -------
    float | array-like: Computed matrix T
    """
    R_pinv = np.linalg.pinv(R)
    return np.transpose(H) @ R_pinv @ N @ R_pinv @ H


def _get_Q(S: float | np.ndarray, G: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the matrix Q for a given transformation.

    The matrix Q is defined as:
    Q = (H^t * R^(-1) * H^(-1))
        - (H^t * N^(-1) * H^(-1))

    Note that S = H^t * R^(-1) * H^(-1)  [eq. 5]
        and G = H^t * N^(-1) * H^(-1)  [eq. 8]

    Therefore:
    Q = S^(-1) - G^(-1)

    for 1 ≤ l ≤ l0.

    Parameters
    ----------
    S : array-like
        Matrix as in eq. 5.
        Can be obtained using function _get_S.
    G : array-like
        Matrix as in eq. 8.
        Can be obtained using function _get_G.

    Returns
    -------
    float | array-like: Computed matrix Q.
    """
    if isinstance(G, float) and isinstance(S, float):
        G = np.array([[G]])
        S = np.array([[S]])
    Q = np.linalg.pinv(S) - np.linalg.pinv(G)
    if Q.size == 1:
        return float(Q)
    return Q
