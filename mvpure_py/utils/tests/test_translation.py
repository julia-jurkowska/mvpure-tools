"""Tests for mvpure_py.utils.translation."""

# Author: sonoAESS

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mvpure_py.utils.translation import (
    transform_leadfield_indices_to_vertices,
    transform_vertices_to_leadfield_indices,
    vertices_to_coordinates,
)

# Left hemisphere has 4 vertices and right hemisphere has 3, so leadfield
# indices 0-3 belong to 'lh' and 4-6 belong to 'rh'.
LH_VERTNO = [10, 20, 30, 40]
RH_VERTNO = [100, 200, 300]


def _make_src(lh_vertno=None, rh_vertno=None):
    """Build a minimal stand-in for an mne.SourceSpaces object."""
    lh_vertno = np.array(LH_VERTNO if lh_vertno is None else lh_vertno)
    rh_vertno = np.array(RH_VERTNO if rh_vertno is None else rh_vertno)
    return [
        {
            "vertno": lh_vertno,
            "rr": np.arange(lh_vertno.size * 3, dtype=float).reshape(-1, 3),
            "inuse": np.ones(lh_vertno.size, dtype=int),
            "nuse": int(lh_vertno.size),
        },
        {
            "vertno": rh_vertno,
            "rr": np.arange(rh_vertno.size * 3, dtype=float).reshape(-1, 3),
            "inuse": np.ones(rh_vertno.size, dtype=int),
            "nuse": int(rh_vertno.size),
        },
    ]


@pytest.fixture
def src():
    """Fixture providing a fresh SourceSpaces stand-in for every test."""
    return _make_src()


def test_transform_leadfield_indices_to_vertices_lh(src):
    """Test mapping left-hemisphere leadfield indices to vertex numbers."""
    lh_mapped, lh_to_use = transform_leadfield_indices_to_vertices(
        [0, 2], src, hemi="lh")

    assert lh_mapped == [10, 30]
    assert_array_equal(lh_to_use, [0, 2])


def test_transform_leadfield_indices_to_vertices_rh(src):
    """Test that right-hemisphere indices are offset by the left length."""
    rh_mapped, rh_to_use = transform_leadfield_indices_to_vertices(
        [4, 6], src, hemi="rh")

    assert rh_mapped == [100, 300]
    assert_array_equal(rh_to_use, [4, 6])


def test_transform_leadfield_indices_to_vertices_both(src):
    """Test that 'both' returns the two vertex lists and the two index lists."""
    lh_mapped, lh_to_use, rh_mapped, rh_to_use = (
        transform_leadfield_indices_to_vertices([0, 1, 4, 5], src, hemi="both")
    )

    assert lh_mapped == [10, 20]
    assert rh_mapped == [100, 200]
    assert_array_equal(lh_to_use, [0, 1])
    assert_array_equal(rh_to_use, [4, 5])


def test_transform_leadfield_indices_to_vertices_no_mapping(src):
    """Test that include_mapping=False returns only the vertex numbers."""
    result = transform_leadfield_indices_to_vertices(
        [0, 4], src, hemi="both", include_mapping=False)

    assert result == ([10], [100])


def test_transform_leadfield_indices_to_vertices_invalid_hemi(src):
    """Test that an unsupported hemisphere raises a ValueError."""
    with pytest.raises(ValueError, match="'hemi' should be one of"):
        transform_leadfield_indices_to_vertices([0], src, hemi="left")


def test_transform_vertices_to_leadfield_indices_flat_lh(src):
    """Test mapping a flat left-hemisphere vertex list to leadfield indices."""
    lf_idx = transform_vertices_to_leadfield_indices([20, 40], src, hemi="lh")

    assert_array_equal(lf_idx, [1, 3])


def test_transform_vertices_to_leadfield_indices_flat_rh(src):
    """Test that right-hemisphere indices are offset by the left length."""
    lf_idx = transform_vertices_to_leadfield_indices([200], src, hemi="rh")

    assert_array_equal(lf_idx, [5])


def test_transform_vertices_to_leadfield_indices_nested(src):
    """Test mapping a nested [lh, rh] vertex list to leadfield indices."""
    lf_idx = transform_vertices_to_leadfield_indices([[10, 30], [300]], src,
                                                     hemi="both")

    assert_array_equal(lf_idx, [0, 2, 6])


def test_transform_vertices_to_leadfield_indices_include_mapping(src):
    """Test that include_mapping=True also returns the vertex-to-index dicts."""
    lf_idx, lh_map, rh_map = transform_vertices_to_leadfield_indices(
        [[10, 30], [300]], src, hemi="both", include_mapping=True)

    assert_array_equal(lf_idx, [0, 2, 6])
    assert lh_map == {10: 0, 30: 2}
    assert rh_map == {300: 6}


def test_transform_vertices_to_leadfield_indices_skips_unknown(src):
    """Test that vertices absent from the source space are ignored."""
    lf_idx = transform_vertices_to_leadfield_indices([[10, 999], [200]], src,
                                                     hemi="both")

    assert_array_equal(lf_idx, [0, 5])


def test_translation_roundtrip(src):
    """Test that indices -> vertices -> indices is the identity."""
    lf_idx = np.array([0, 2, 4, 6])
    vertices = transform_leadfield_indices_to_vertices(
        lf_idx, src, hemi="both", include_mapping=False)
    roundtrip = transform_vertices_to_leadfield_indices(vertices, src,
                                                        hemi="both")

    assert_array_equal(roundtrip, lf_idx)


def test_vertices_to_coordinates(src):
    """Test that vertex numbers are converted to their 3D coordinates."""
    coords = vertices_to_coordinates([[0, 2], [1]], src)

    assert coords.shape == (3, 3)
    assert_array_equal(coords[0], src[0]["rr"][0])
    assert_array_equal(coords[2], src[1]["rr"][1])


def test_vertices_to_coordinates_empty_hemisphere(src):
    """Test that an empty hemisphere contributes no rows."""
    coords = vertices_to_coordinates([[0], []], src)

    assert coords.shape == (1, 3)
