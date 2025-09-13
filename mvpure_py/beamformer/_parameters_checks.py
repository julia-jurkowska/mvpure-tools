import numpy as np


def _prepare_parameters(mvpure_params: dict, n_sources: int):
    """
    Checking MVPURE specific parameters.

    Parameters:
    -----------
    mvpure_params: dict
        dictionary with MVPURE specific parameters
    n_sources: int
        number of dipole sources
    """

    if mvpure_params is None:
        # if mvpure_params is not specified: compute MCMV filter for full rank
        print(f"'filter_type' is not defined. Using 'MVP_R' as default.")
        print(f"'filter_rank' is not defined. Using 'full' as default.")
        filter_type = 'MVP_R'
        filter_rank = 'full'
        return filter_type, filter_rank

    if 'filter_type' not in mvpure_params.keys() or 'filter_type' is None:
        # if filter_type not specified: compute "MVP_R" filter
        print(f"'filter_type' is not defined. Using 'MVP_R' as default.")
        filter_type = "MVP_R"
    else:
        filter_type = mvpure_params['filter_type']

    if 'filter_rank' not in mvpure_params.keys() or mvpure_params['filter_rank'] is None:
        # if filter_rank not specified: compute filter for full rank
        filter_rank = 'full'
        print(f"'filter_rank' is not defined. Using 'full' as default.")
    else:
        filter_rank = mvpure_params['filter_rank']
    _check_filter_rank(filter_rank, n_sources)

    return filter_type, filter_rank


def _check_filter_rank(filter_rank, n_sources):
    """Checking if filter_rank parameter is valid."""
    if isinstance(filter_rank, str) and filter_rank != "full":
        raise ValueError("Only possible string values for 'filter rank' is 'full'")

    if isinstance(filter_rank, (int, np.integer)) and filter_rank > n_sources:
        raise ValueError("Given filter rank value is higher then number of sources.")

