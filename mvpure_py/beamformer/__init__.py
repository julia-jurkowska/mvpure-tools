"""
References
----------
.. [1] Jurkowska, J., Dreszer, J., Lewandowska, M., Tolpa, K., & Piotrowski, T. J. (2025).
        Multi-Source Neural Activity Indices and Spatial Filters for EEG/MEG Inverse Problem: An Extension to MNE-Python. bioRxiv, 2025-09.
"""

from .mvp_filter import (
    make_filter,
    apply_filter,
    apply_filter_cov,
    apply_filter_epochs,
    apply_filter_raw)

from ._compute_beamformer import _compute_beamformer

from ._parameters_checks import (
    _prepare_parameters,
    _check_filter_rank
)

from .mvpure_utils import (
    get_S_proj_matrix,
    get_G_proj_matrix
)

from .filters_utils import (
    make_mvp_n,
    make_mvp_r
)

from .mvp_filter import (
    apply_filter,
    apply_filter_cov,
    apply_filter_epochs,
    apply_filter_raw
)
