"""Tests for mvpure_py.utils.algebra."""

# Author: sonoAESS

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mvpure_py.utils.algebra import get_pinv_RN_eigenvals


def test_get_pinv_RN_eigenvals_identity():
    """Test that identity matrices give eigenvalues equal to one."""
    identity = np.eye(4)
    eigvals = get_pinv_RN_eigenvals(identity, identity)

    assert_allclose(eigvals, np.ones(4))


def test_get_pinv_RN_eigenvals_known_values():
    """Test eigenvalues against a case with known pseudo-inverse spectrum."""
    # N is invertible, so pinv(N) == inv(N) and R @ N^-1 = R / 2
    n_inv = 0.5 * np.array([[1.0, 0.0], [0.0, 2.0]])
    r = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected = np.sort(np.real(np.linalg.eigvals(r @ n_inv)))[::-1]

    eigvals = get_pinv_RN_eigenvals(r, np.linalg.inv(n_inv))

    assert_allclose(eigvals, expected)


def test_get_pinv_RN_eigenvals_singular_n():
    """Test that a singular N is handled through the pseudo-inverse."""
    # A rank-deficient N whose pseudo-inverse is still well defined
    n = np.array([[1.0, 1.0], [1.0, 1.0]])
    r = np.eye(2)

    eigvals = get_pinv_RN_eigenvals(r, n)

    # R @ pinv(N) is singular, so one eigenvalue is zero
    assert_allclose(eigvals[-1], 0.0, atol=1e-12)
    assert np.isrealobj(eigvals)


def test_get_pinv_RN_eigenvals_sorted_descending():
    """Test that returned eigenvalues are sorted in descending order."""
    rng = np.random.default_rng(42)
    r = rng.standard_normal((5, 5))
    n = rng.standard_normal((5, 5))
    n = n @ n.T + np.eye(5)  # make N symmetric positive definite

    eigvals = get_pinv_RN_eigenvals(r, n)

    assert np.all(np.diff(eigvals) <= 1e-12)


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (6, 6)])
def test_get_pinv_RN_eigenvals_output_shape(shape):
    """Test that the number of eigenvalues matches the matrix size."""
    n = shape[0]
    identity = np.eye(n)

    eigvals = get_pinv_RN_eigenvals(identity, identity)

    assert eigvals.shape == (n,)