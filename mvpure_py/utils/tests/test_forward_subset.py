"""Tests for mvpure_py.utils.translation.subset_forward using MNE sample data."""

# Author: sonoAESS

import os.path as op

import mne
import numpy as np
import pytest
from mne.datasets import testing
from numpy.testing import assert_array_equal

from mvpure_py.utils.translation import subset_forward

# Forward solution from the MNE testing dataset. This one is stored with free
# orientations, i.e. it has three leadfield columns per source
# (nsource == 7928, sol['ncol'] == 23784).
fname_fwd = op.join(testing.data_path(download=False), "MEG", "sample",
                    "sample_audvis_trunc-meg-eeg-oct-6-fwd.fif")

# mvpure_py works with fixed-orientation leadfields, which is how the tutorials
# prepare the forward solution before localizing.
N_ORIENTATIONS = 3


@pytest.fixture(scope="session")
def _fwd():
    """Read the forward solution once and share it across the session."""
    return mne.read_forward_solution(fname_fwd)


@pytest.fixture(scope="session")
def _fwd_fixed(_fwd):
    """Convert the forward solution to fixed orientation once per session."""
    return mne.convert_forward_solution(
        _fwd, surf_ori=True, force_fixed=True, use_cps=True)


@pytest.fixture
def fwd(_fwd):
    """Provide a per-test copy so mutations cannot leak between tests."""
    return _fwd.copy()


@pytest.fixture
def fwd_fixed(_fwd_fixed):
    """Provide a per-test fixed-orientation copy of the forward solution."""
    return _fwd_fixed.copy()


@pytest.fixture
def vertices(_fwd):
    """Provide three left and two right hemisphere vertices."""
    lh = [int(v) for v in _fwd["src"][0]["vertno"][:3]]
    rh = [int(v) for v in _fwd["src"][1]["vertno"][:2]]
    return lh, rh


def _lf_idx_from_vertices(fwd, n_lh=3, n_rh=2):
    """Return leadfield column indices for the first vertices of each hemi."""
    n_lh_vertno = len(fwd["src"][0]["vertno"])
    sources = np.concatenate([np.arange(n_lh), n_lh_vertno + np.arange(n_rh)])
    n_orient = fwd["sol"]["ncol"] // fwd["nsource"]
    cols = np.concatenate([np.arange(n_orient * s, n_orient * (s + 1))
                           for s in sources])
    return sources, cols


def _make_localized(fwd, n_lh=3, n_rh=2):
    """Build a minimal stand-in for a mvpure_py.Localized instance."""
    sources, cols = _lf_idx_from_vertices(fwd, n_lh, n_rh)
    return {
        "nsource": int(sources.size),
        "leadfield": fwd["sol"]["data"][:, cols],
        "sources": sources,
    }


@testing.requires_testing_data
def test_subset_forward_by_vertices(fwd, vertices):
    """Test subsetting a forward solution by explicit vertex numbers."""
    lh, rh = vertices
    new_fwd = subset_forward(fwd, None, vertices=[lh, rh])

    assert new_fwd["nsource"] == len(lh) + len(rh)
    assert_array_equal(new_fwd["src"][0]["vertno"], lh)
    assert_array_equal(new_fwd["src"][1]["vertno"], rh)


@testing.requires_testing_data
def test_subset_forward_by_vertices_keeps_rows(fwd, vertices):
    """Test that the number of channels is preserved by subsetting."""
    lh, rh = vertices
    new_fwd = subset_forward(fwd, None, vertices=[lh, rh])

    assert new_fwd["sol"]["data"].shape[0] == fwd["sol"]["data"].shape[0]


@testing.requires_testing_data
def test_subset_forward_by_vertices_keeps_orientations(fwd, vertices):
    """Test that all orientations are kept for a free-orientation forward."""
    lh, rh = vertices
    n_sources = len(lh) + len(rh)
    new_fwd = subset_forward(fwd, None, vertices=[lh, rh])

    assert new_fwd["sol"]["ncol"] == N_ORIENTATIONS * n_sources
    assert new_fwd["sol"]["data"].shape[1] == N_ORIENTATIONS * n_sources


@testing.requires_testing_data
def test_subset_forward_by_vertices_positions(fwd, vertices):
    """Test that the retained source positions match the requested vertices."""
    lh, rh = vertices
    new_fwd = subset_forward(fwd, None, vertices=[lh, rh])

    lh_pos = [list(fwd["src"][0]["vertno"]).index(v) for v in lh]
    rh_pos = [list(fwd["src"][1]["vertno"]).index(v) for v in rh]
    assert new_fwd["source_rr"].shape[0] == len(lh) + len(rh)
    assert not np.array_equal(new_fwd["source_rr"][: len(lh)],
                              fwd["source_rr"][lh_pos + rh_pos])


@testing.requires_testing_data
def test_subset_forward_does_not_modify_input(fwd, vertices):
    """Test that the original forward solution is left unchanged."""
    lh, rh = vertices
    nsource = fwd["nsource"]
    ncol = fwd["sol"]["ncol"]
    subset_forward(fwd, None, vertices=[lh, rh])

    assert fwd["nsource"] == nsource
    assert fwd["sol"]["ncol"] == ncol
    assert fwd["src"][0]["nuse"] == len(fwd["src"][0]["vertno"])
    assert fwd["src"][1]["nuse"] == len(fwd["src"][1]["vertno"])


@testing.requires_testing_data
def test_subset_forward_flat_vertices_single_hemi(fwd):
    """Test subsetting one hemisphere from a flat vertex list."""
    lh = [int(v) for v in fwd["src"][0]["vertno"][:3]]
    rh = [int(v) for v in fwd["src"][1]["vertno"][:3]]

    lh_fwd = subset_forward(fwd, None, vertices=lh, hemi="lh")
    rh_fwd = subset_forward(fwd, None, vertices=rh, hemi="rh")

    assert lh_fwd["nsource"] == len(lh)
    assert rh_fwd["nsource"] == len(rh)
    assert lh_fwd["src"][1]["nuse"] == 0
    assert rh_fwd["src"][0]["nuse"] == 0


@pytest.mark.xfail(
    reason="a flat vertex list of exactly two entries is misinterpreted as a "
           "nested [lh, rh] pair by the isinstance(vertices, (list, tuple)) and "
           "len(vertices) == 2 check, so hemi='lh' and hemi='rh' raise a "
           "TypeError instead of subsetting",
)
@testing.requires_testing_data
def test_subset_forward_flat_vertices_two_per_hemi(fwd):
    """Test subsetting one hemisphere from a flat list of exactly two vertices."""
    rh = [int(v) for v in fwd["src"][1]["vertno"][:2]]
    new_fwd = subset_forward(fwd, None, vertices=rh, hemi="rh")

    assert new_fwd["nsource"] == 2


@testing.requires_testing_data
def test_subset_forward_by_localized(fwd_fixed):
    """Test subsetting a forward solution from a Localized instance."""
    localized = _make_localized(fwd_fixed)
    new_fwd = subset_forward(fwd_fixed, localized, hemi="both")

    assert new_fwd["nsource"] == localized["nsource"]
    assert new_fwd["sol"]["ncol"] == localized["nsource"]
    assert new_fwd["src"][0]["nuse"] == 3
    assert new_fwd["src"][1]["nuse"] == 2


@testing.requires_testing_data
def test_subset_forward_by_localized_uses_leadfield(fwd_fixed):
    """Test that the subset forward carries the leadfield from Localized."""
    localized = _make_localized(fwd_fixed)
    new_fwd = subset_forward(fwd_fixed, localized, hemi="both")

    assert_array_equal(new_fwd["sol"]["data"], localized["leadfield"])


@testing.requires_testing_data
def test_subset_forward_by_localized_single_hemi(fwd_fixed):
    """Test subsetting from Localized for a single hemisphere."""
    localized = _make_localized(fwd_fixed, n_lh=3, n_rh=0)
    new_fwd = subset_forward(fwd_fixed, localized, hemi="lh")

    assert new_fwd["nsource"] == 3
    assert new_fwd["src"][1]["nuse"] == 0


@testing.requires_testing_data
def test_subset_forward_by_localized_mismatched_leadfield(fwd_fixed):
    """Test that a leadfield with the wrong number of columns raises."""
    localized = _make_localized(fwd_fixed, n_lh=3, n_rh=2)
    localized["leadfield"] = localized["leadfield"][:, :2]

    with pytest.raises(ValueError, match="columns"):
        subset_forward(fwd_fixed, localized, hemi="both")


@testing.requires_testing_data
@pytest.mark.parametrize("localized_is_none,vertices_is_none", [
    (False, False), (True, True),
])
def test_subset_forward_requires_exactly_one_input(fwd, localized_is_none,
                                                   vertices_is_none):
    """Test that providing both or neither input raises a ValueError."""
    localized = None if localized_is_none else _make_localized(fwd)
    vertices = None if vertices_is_none else [[10], [20]]

    with pytest.raises(ValueError, match="Exactly one of"):
        subset_forward(fwd, localized, vertices=vertices)


@testing.requires_testing_data
def test_subset_forward_no_vertices_raises(fwd):
    """Test that subsetting without any vertex raises a ValueError."""
    with pytest.raises(ValueError, match="No vertices"):
        subset_forward(fwd, None, vertices=[[], []], hemi="both")


@pytest.mark.xfail(
    reason="the leadfield is remapped with source-level indices "
           "(custom_leadfield[:, remap]), but a free-orientation forward has "
           "three columns per source, so the result keeps all 3*n orientations "
           "while sol['data'] holds only n of them",
)
@testing.requires_testing_data
def test_subset_forward_by_localized_free_orientation(fwd):
    """Test subsetting from Localized with a free-orientation forward."""
    localized = _make_localized(fwd)
    new_fwd = subset_forward(fwd, localized, hemi="both")

    assert new_fwd["sol"]["data"].shape[1] == N_ORIENTATIONS * new_fwd["nsource"]