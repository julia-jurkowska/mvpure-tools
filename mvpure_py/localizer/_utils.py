from ..utils import _check_hemi_param

import mne
import numpy as np
from termcolor import colored
import logging
logging.basicConfig(
    level=logging.WARNING,
    format='[%(levelname)s] %(message)s'
)


def _check_localize_params(
        forward: mne.Forward,
        leadfield: np.array) -> None:
    """
    Checking user input for mvpure_py.localizer.localize() function.
    """
    if forward is None and leadfield is None:
        raise ValueError("One of the parameters 'forward' and 'leadfield' should not be None")

    if forward is not None and leadfield is not None:
        if np.array_equal(forward['sol']['data'], leadfield):
            raise ValueError("Both parameters 'forward' and 'leadfield' were specified. "
                             "They should be equal, but they are not.")


def _prepare_localize_params(
        forward: mne.Forward,
        leadfield: np.array) -> np.array:
    """
    Adjusting input parameters in mvpure_py.localizer.localize() function.
    Handling forward/leadfield options.
    """
    if forward is not None:
        leadfield = forward["sol"]["data"]

    return leadfield


def _check_n_sources_param(n_sources: int) -> None:
    """
    Checks type of 'n_sources_to_localize' from mvpure_py.localizer.localize() function.
    Should be an integer.
    """
    if not isinstance(n_sources, (int, np.integer)):
        raise TypeError(f"Parameter 'n_sources_to_localize' should be an integer. Got type: {type(n_sources)} instead.")


def _check_rank(
        r: int | str | float,
        n_sources_to_localize: int):
    """
    Checking correctness of rank optimization ('r') parameter.
    """
    # 'r' should be less or equal to number of sources to localize
    if isinstance(r, (int, np.integer)) and r > n_sources_to_localize:
        raise ValueError(f"Optimization parameter 'r' (set to {r}) should be equal or less than"
                         f" number of sources to localize ('n_sources_to_localize' set to {n_sources_to_localize}).")

    # only option for 'r' as string is 'full'
    if isinstance(r, str) and r not in ["full"]:
        raise ValueError(f"If parameter 'r' is a string it should be set to 'full'. "
                         f"Got {r} instead.")

    # if 'r' == 0, use full rank
    if isinstance(r, (int, np.integer)) and r == 0:
        return "full"

    # if 'r' is smaller than zero, take n_sources_to_localize - abs(r)
    if isinstance(r, (int, np.integer)) and r < 0 and (n_sources_to_localize - abs(r)) > 0:
        adjusted_value = n_sources_to_localize - abs(r)
        logging.warning(
            f"Got rank lower than zero. Taking n_sources_to_localize - {abs(r)} = {adjusted_value} instead."
        )
        return adjusted_value

    # if 'r' is smaller than zero and n_sources_to_localize - abs(r) == 0, use full rank
    if isinstance(r, (int, np.integer)) and r < 0 and (n_sources_to_localize - abs(r)) == 0:
        return "full"

    # if 'r' is smaller than zero and n_sources_to_localize - abs(r) == 0, raise error
    if isinstance(r, (int, np.integer)) and r < 0 and (n_sources_to_localize - abs(r)) < 0:
        raise ValueError(f"Optimization parameter 'r' (set to {r}) should be equal or higher than 0.")

    # if using float and float can not be converted to integer - raise error
    if isinstance(r, (float, np.floating)) and int(r) != r:
        raise TypeError(f"Parameter 'r' should be type an integer (if numeric). Got float instead.")


def _prepare_localizer_to_use(localizer_to_use: str | list[str]):
    """
    Checking and adjusting 'localizer_to_use' parameter from mvpure_py.localizer.localize() function.
    """
    # list of all localizers options
    all_localizers = ['mai', 'mpz', 'mai_mvp', 'mpz_mvp']
    # if localizer_to_use is a string
    if isinstance(localizer_to_use, str):
        # if localizer_to_use == 'all' use all localizers
        if localizer_to_use == 'all':
            param_localizer_to_use = all_localizers
        # can be single name of localizers
        elif localizer_to_use in all_localizers:
            param_localizer_to_use = [localizer_to_use]
        # otherwise raise ValueError
        else:
            raise ValueError(f"If 'localizer_to_use' is a string it should be either 'all' or "
                             f"name of one of the localizers to use. Possible options are: {all_localizers}. "
                             f"Got '{localizer_to_use}' instead.")
    elif isinstance(localizer_to_use, list):
        param_localizer_to_use = [loc for loc in localizer_to_use if loc in all_localizers]
        if len(param_localizer_to_use) < len(localizer_to_use):
            print(colored(
                f"WARNING: One or more of localizers' names are not valid."
                f" Possible options are: {all_localizers}, got {localizer_to_use}.",
                "yellow"
            ))
        if len(param_localizer_to_use) == 0:
            raise ValueError(f"None of the localizers' name are valid."
                             f" Possible options are: {all_localizers}, got {localizer_to_use}")
    else:
        raise ValueError(f"'localizer_to_use' should be either list of valid localizers' names "
                         f"or string with single localizer name")
    return param_localizer_to_use


def _define_hemi_based_on_vertices(localized) -> str:
    """
    Define hemi (`both`, `rh`, `lh`) based on vertices stored in `localized` object.
    """
    if hasattr(localized['stc'], 'lh_vertno') and hasattr(localized['stc'], 'rh_vertno'):
        hemi = "both"
    elif hasattr(localized['stc'], 'lh_vertno'):
        hemi = "lh"
    elif hasattr(localized['stc'], 'rh_vertno'):
        hemi = "rh"

    return hemi


def _check_norm_param(norm):
    if norm not in ["max", "sum"]:
        raise ValueError(f"'norm' can be equal to one of the following: "
                         f" 'max' (max normalization), 'sum' (L1 normalization. "
                         f"Got {norm} instead.")


def _check_vertices_and_indices(lh_vertices: list, lh_indices: list, rh_vertices: list, rh_indices: list):
    if (lh_indices is not None and lh_vertices is None) or (lh_indices is None and lh_vertices is not None):
        raise ValueError(f"Both 'lh_vertices' and 'lh_indices' should be specified. "
                         f"Got lh_vertices={lh_vertices} and lh_indices={lh_indices}.")

    if (rh_indices is not None and rh_vertices is None) or (rh_indices is None and rh_vertices is not None):
        raise ValueError(f"Both 'rh_vertices' and 'rh_indices' should be specified. "
                         f"Got rh_vertices={rh_vertices} and rh_indices={rh_indices}.")

    if lh_vertices is not None and lh_indices is not None and len(lh_vertices) != len(lh_indices):
        raise ValueError(f"Can not add vertices to Localized object because length of vertices list ({len(lh_vertices)})"
                         f"does not match length of indices list ({len(lh_indices)}).")

    if rh_vertices is not None and rh_indices is not None and len(rh_vertices) != len(rh_indices):
        raise ValueError(f"Can not add vertices to Localized object because length of vertices list ({len(rh_vertices)})"
                         f"does not match length of indices list ({len(rh_indices)}).")
