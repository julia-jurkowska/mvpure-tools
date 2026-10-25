"""Tests for mvpure_py.utils._utils."""

# Author: sonoAESS

import pytest

from mvpure_py.utils._utils import (
    _check_hemi_and_vertices_matching,
    _check_hemi_param,
)


@pytest.mark.parametrize("hemi", ["lh", "rh", "both"])
def test_check_hemi_param_valid(hemi):
    """Test that the accepted hemisphere names pass the check."""
    assert _check_hemi_param(hemi) is None


@pytest.mark.parametrize("hemi", ["left", "LH", "", None, "both ", 0])
def test_check_hemi_param_invalid(hemi):
    """Test that an unknown hemisphere name raises a ValueError."""
    with pytest.raises(ValueError, match="'hemi' should be one of"):
        _check_hemi_param(hemi)


@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_check_hemi_and_vertices_matching_single_valid(hemi):
    """Test a single-hemisphere call with a nested single-group vertex list."""
    assert _check_hemi_and_vertices_matching(hemi, [[1, 2, 3]]) is None


@pytest.mark.parametrize("hemi", ["lh", "rh"])
@pytest.mark.xfail(
    reason="Flat vertex lists are documented as valid for a single hemisphere, "
           "but the check requires len(vertices) == 1",
)
def test_check_hemi_and_vertices_matching_single_flat(hemi):
    """Test a single-hemisphere call with a flat vertex list."""
    assert _check_hemi_and_vertices_matching(hemi, [1, 2, 3]) is None


@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_check_hemi_and_vertices_matching_single_invalid(hemi):
    """Test that a single hemisphere with two vertex groups raises."""
    with pytest.raises(ValueError, match="one brain hemisphere"):
        _check_hemi_and_vertices_matching(hemi, [[1, 2], [3, 4]])


def test_check_hemi_and_vertices_matching_both_valid():
    """Test that hemi='both' accepts two vertex groups."""
    assert _check_hemi_and_vertices_matching("both", [[1, 2], [3, 4]]) is None


def test_check_hemi_and_vertices_matching_both_invalid():
    """Test that hemi='both' with a flat vertex list raises."""
    with pytest.raises(ValueError, match="two brain hemispheres"):
        _check_hemi_and_vertices_matching("both", [1, 2, 3])