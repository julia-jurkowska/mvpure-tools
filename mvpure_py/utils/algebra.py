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
