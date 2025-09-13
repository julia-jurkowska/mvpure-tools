""" Functions for group results visualizations. """
# Author: Julia Jurkowska

import mne
import numpy as np
import pandas as pd

from mvpure_py import viz
from mvpure_py.utils._helper import split_kwargs


def group_plot_add_foci(
        data_dict: dict,
        localizer: str,
        hemi: str = "both",
        surf: str = "inflated",
        delta: float = None,
        subjects_dir: str = None,
        **kwargs
):
    """
    Create group plot presenting localized dipoles using ``mne.viz.Brain.add_foci()``.

    Parameters
    ----------
    data_dict : dict
        Dictionary with data from localization analysis in a form:

        .. code-block:: python

            {
                'subject': {
                    'localizer_1': mvpure_py.Localized,
                    ...
                    'localizer_n': mvpure_py.Localized
                }
            }
    localizer : str
        Localizer type to plot results for
    hemi : str
        Hemisphere to plot.
    surf : str
        FreeSurfer surface mesh name
    delta : float
        Optimization parameter used during analysis.
        Is only used for setting title of the plot.
    subjects_dir : str
        Directory where `subject` folder is being stored
    **kwargs : dict
        Additional keyword arguments passed to:
            - ``mne.viz.Brain`` constructor
            - ``Brain.add_foci`` method
    """
    splitted_kwargs = split_kwargs(kwargs=kwargs,
                                   func_map={"brain": mne.viz.Brain,
                                             "foci": mne.viz.Brain.add_foci})
    if delta is not None:
        title = f"{localizer} for delta: {delta}"
    else:
        title = f"{localizer}"

    brain = mne.viz.Brain(subject="fsaverage",
                          surf=surf,
                          hemi=hemi,
                          title=title,
                          subjects_dir=subjects_dir,
                          **splitted_kwargs["brain"])

    subjects = list(data_dict.keys())
    for subject in subjects:
        current = data_dict[subject][localizer]
        sorted_vertices = dict(
            sorted(current['vertices'].items(),
                   key=lambda item: item[1]['activity_index_order'])
        )
        for i, vert in enumerate(sorted_vertices):
            _norm_factor = 1 - i / len(sorted_vertices)
            _c = viz._assign_color_mapping(_norm_factor, cmap="Reds")
            _scale_factor = np.linspace(0.8, 0.3, num=len(sorted_vertices))[i]
            brain.add_foci(coords=current['vertices'][vert]["morphed_to_fs"],
                           coords_as_verts=True,
                           hemi=current['vertices'][vert]["hemi"],
                           color=_c,
                           scale_factor=_scale_factor,
                           **splitted_kwargs["foci"])


def group_plot_regions(
        data_dict: dict,
        localizer: str,
        n_sources_to_loc: int,
        hemi: str,
        parc: str,
        subjects_dir: str = None,
        surf: str = "inflated",
        cmap: str = "Reds",
        none_color: str = "gainsboro",
        return_df_info: bool = False,
        **kwargs
):
    """
    Plot the fsaverage brain with results from group-level EEG source localization.

    Function visualizes the fsaverage brain and highlights cortical regions based on their accumulated scores from
    a group-level source localization analysis. The regions are scored as follows:

    - The region corresponding to the strongest source gets `n_sources_to_loc` points,
    - The second-strongest gets `n_sources_to_loc - 1` points,
    - ...
    - The weakest among the top `n_sources_to_loc` gets 1 point.

    Points are accumulated across subjects. Regions with more points are shown
    with more intense colors on the brain surface using cmap.

    Parameters
    ----------
    data_dict : dict
        Dictionary with data from localization analysis in a form:

        .. code-block:: python

            {
                'subject': {
                    'localizer_1': mvpure_py.Localized,
                    ...
                    'localizer_n': mvpure_py.Localized
                }
            }

    localizer : str
        Localizer type to plot results for
    n_sources_to_loc : int
         Number of sources that were localized
    hemi : str
        Hemisphere to plot.
    parc : str
        Name of parcellation to was performed and should be read for fsaverage brain
    subjects_dir : str
        Directory where ``subject`` folder is being stored
    surf : str
        FreeSurfer surface mesh name
    cmap : str
        Matplotlib cmap to use.
    none_color : str
        Matplotlib-valid color to use for regions that were not identified in single-subjects
    return_df_info : bool
        Whether to return pandas dataframe with information about regions scoring
    kwargs :
        Additional keyword arguments passed to ``mne.viz.Brain`` constructor

    Returns
    -------

    """
    subjects = list(data_dict.keys())
    # Get region ranking
    # n points for first localized source where n is number of localized sources
    # n-1 points for second localized source etc.
    # 1 point for n localized source
    region_ranking = dict()
    for subject in subjects:
        current = data_dict[subject][localizer]
        for vert in current['vertices']:
            order = current['vertices'][vert]['activity_index_order']
            points = n_sources_to_loc - order + 1
            if current['vertices'][vert]['brain_region'] in list(region_ranking.keys()):
                region_ranking[current['vertices'][vert]['brain_region']] += points
            else:
                region_ranking[current['vertices'][vert]['brain_region']] = points

    sorted_ranking = dict(sorted(region_ranking.items(), key=lambda item: item[1], reverse=True))
    max_points = list(sorted_ranking.values())[0]

    # Plotting on 'fsaverage' brain
    labels = mne.read_labels_from_annot(
        subject="fsaverage",
        parc=parc,
        subjects_dir=subjects_dir,
        hemi=hemi
    )

    brain = mne.viz.Brain(
        subject="fsaverage",
        surf=surf,
        subjects_dir=subjects_dir,
        hemi=hemi,
        **kwargs
    )
    # Add each label to the brain visualization
    for i, label in enumerate(labels):
        if label.name in list(region_ranking.keys()):
            _norm_factor = region_ranking[label.name] / max_points
            _c = viz._assign_color_mapping(_norm_factor, cmap=cmap)
            brain.add_label(label, hemi=label.hemi, color=_c, borders=False)
        else:
            brain.add_label(label, hemi=label.hemi, color=none_color, borders=False)

    if return_df_info:
        sorted_ranking_df = pd.DataFrame.from_dict({"region": list(sorted_ranking.keys()),
                                                    "points": list(sorted_ranking.values())})
        return sorted_ranking_df
