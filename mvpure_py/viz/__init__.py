"""
References
----------
.. [1] Jurkowska, J., Dreszer, J., Lewandowska, M., Tolpa, K., & Piotrowski, T. J. (2025).
        Multi-Source Neural Activity Indices and Spatial Filters for EEG/MEG Inverse Problem: An Extension to MNE-Python. bioRxiv, 2025-09.
"""

from .viz import (
    plot_RN_eigenvalues,
    plot_sources_with_activity,
    plot_localized_sources,
    _save_brain_object,
    _assign_color_mapping
)

from .group_viz import (
    group_plot_regions,
    group_plot_add_foci
)
