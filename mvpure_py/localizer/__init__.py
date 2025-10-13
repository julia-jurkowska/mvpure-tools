"""
References
----------
.. [1] Jurkowska, J., Dreszer, J., Lewandowska, M., Tolpa, K., & Piotrowski, T. J. (2025).
        Multi-Source Neural Activity Indices and Spatial Filters for EEG/MEG Inverse Problem: An Extension to MNE-Python. bioRxiv, 2025-09.
"""
from .mvpure_localizer import (
    localize,
    Localized,
    read_localized
)
from .localizer_utils import (
    get_activity_index,
    suggest_n_sources_and_rank
)

from ._utils import (
    _check_localize_params,
    _prepare_localize_params,
    _check_vertices_and_indices,
    _check_n_sources_param
)
