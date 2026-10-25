"""Tests for mvpure_py.utils._helper."""

# Author: sonoAESS

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mvpure_py.utils._helper import split_kwargs


def _func_a(param_a, param_b=1, **kwargs):
    """Dummy function used to inspect valid parameter names."""
    return param_a, param_b, kwargs


def _func_b(param_a=0, param_c=2, **kwargs):
    """Dummy function used to inspect valid parameter names."""
    return param_a, param_c, kwargs


def test_split_kwargs_basic():
    """Test that kwargs are routed to the function accepting them."""
    split = split_kwargs({"param_a": 1, "param_b": 2, "param_c": 3},
                         {"a": _func_a, "b": _func_b})

    assert split == {"a": {"param_a": 1, "param_b": 2}, "b": {"param_c": 3}}


def test_split_kwargs_all_keys_present():
    """Test that every function gets a dict even when it receives no kwargs."""
    split = split_kwargs({"param_a": 1}, {"a": _func_a, "b": _func_b})

    assert split == {"a": {"param_a": 1}, "b": {}}


def test_split_kwargs_empty():
    """Test that empty kwargs produce empty dicts for every function."""
    split = split_kwargs({}, {"a": _func_a, "b": _func_b})

    assert split == {"a": {}, "b": {}}


def test_split_kwargs_overlap_goes_to_first_match():
    """Test that a parameter shared by several functions goes to the first one."""
    split = split_kwargs({"param_a": 7}, {"a": _func_a, "b": _func_b})

    assert split == {"a": {"param_a": 7}, "b": {}}


def test_split_kwargs_var_keyword_is_never_matched():
    """Test that ``**kwargs`` in a signature does not swallow unknown params."""
    # Both functions accept **kwargs, but "param_b" is explicit in _func_a only
    split = split_kwargs({"param_b": 1}, {"a": _func_a})

    assert split == {"a": {"param_b": 1}}


def test_split_kwargs_unknown_raises():
    """Test that an unknown parameter raises a TypeError."""
    with pytest.raises(TypeError, match="Unknown parameter: param_z"):
        split_kwargs({"param_z": 1}, {"a": _func_a, "b": _func_b})


def test_split_kwargs_real_functions():
    """Test the helper with the package's own functions."""
    from mvpure_py.utils.algebra import get_pinv_RN_eigenvals

    split = split_kwargs({"R": np.eye(2)}, {"algebra": get_pinv_RN_eigenvals})

    assert list(split) == ["algebra"]
    assert_array_equal(split["algebra"]["R"], np.eye(2))


def test_split_kwargs_isolates_kwargs():
    """Test that the returned dicts can be used as kwargs without collision."""
    split = split_kwargs({"param_a": 1, "param_b": 2, "param_c": 3},
                         {"a": _func_a, "b": _func_b})

    # Would raise TypeError on duplicate arguments if the split were wrong
    assert _func_a(**split["a"]) == (1, 2, {})
    assert _func_b(**split["b"]) == (0, 3, {})