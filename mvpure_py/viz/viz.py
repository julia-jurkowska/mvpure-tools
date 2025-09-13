import mne
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import numpy as np

from ..utils import (
    _check_hemi_param,
    get_pinv_RN_eigenvals
)
from ..utils._helper import split_kwargs


def plot_RN_eigenvalues(
    R: np.ndarray,
    N: np.ndarray,
    figsize: tuple = (12, 6),
    return_eigvals: bool = False,
    n_sources_threshold: float = 1.0,
    rank_threshold: float = 1.5,
    subject: str = None,
    **kwargs
) -> tuple[plt.Figure, np.ndarray] | plt.Figure:
    """
    Plot eigenvalues of matrix :math:`RN^{-1}` where R is data covariance and N is noise covariance in descending order.
    Adding horizontal lines for the cut-off values: ``n_sources_to_localize`` and ``rank_threshold``.

    Parameters
    ----------
    R : array-like
        Data covariance matrix
    N : array-like
        Noise covariance matrix
    figsize : tuple
        Size of the figure in the format (width, height) in inches.
        Default to (12,6).
    return_eigvals : bool
        Whether to rerun array with eigenvalues sorted in descending order.
        Default to False.
    n_sources_threshold : float
        Number of eigenvalues of the :math:`RN^{-1}` matrix below this threshold corresponds to the suggested
        number of sources to localize.
        Default to 1.0. For more details see Observation 1 in [1]_..
    rank_threshold : float
        Number of eigenvalues of the :math:`RN^{-1}` matrix below this threshold corresponds to the
        suggested rank optimization parameter.
        Default to 1.5. For more details see Proposition 3 in [1]_.
    subject : str
        Subject name the analysis is performed for. Optional.

    References
    ----------
    """
    eigval = np.real(
        np.sort(
            np.linalg.eigvals(R @ np.linalg.pinv(N))
        )
    )[::-1]
    
    fig = plt.figure(figsize=figsize)
    plt.scatter(x=range(1, len(eigval) + 1), y=eigval, **kwargs)
    plt.hlines(
        y=n_sources_threshold,
        xmin=1,
        xmax=len(eigval) + 1,
        label=f"$\\mathrm{{RN}}^{{-1}}$=1",
        color="firebrick")
    plt.hlines(
        y=rank_threshold,
        xmin=1,
        xmax=len(eigval) + 1,
        label=f"$\\mathrm{{RN}}^{{-1}}$=3/2",
        color="darkseagreen")
    
    if subject is not None:
        plt.title(f"Subject: {subject} - $\\mathrm{{RN}}^{{-1}}$ eigenvalues")
    else:
        plt.title(f"$\\mathrm{{RN}}^{{-1}}$ eigenvalues")
    
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()

    if return_eigvals:
        return fig, eigval
    return fig
    

def plot_sources_with_activity(
        subject: str,
        stc: mne.SourceEstimate,
        hemi: str = "both",
        show_source_locations: bool = True,
        **kwargs
) -> mne.viz.Brain:
    """
    Display ``mne.SourceEstimate.plot()`` with additional option to preview localized sources as black dots.
    This functionality helps when the user wants to view time series for individual sources, because user knows
    where to click.
    If ``show_source_locations`` is set to ``False``, it is equivalent to ``mne.SourceEstimate.plot()``.

    Parameters
    ----------
    subject : str
        Subject name the analysis is performed for. Optional.
    stc : mne.SourceEstimate
        Source estimate obtained from signal reconstruction.
    hemi : str
        Hemisphere to show.
        Default to 'both'
    show_source_locations : bool
        Whether to preview positions of localized sources as black dots.
        Default to True.

    Returns
    -------
    mne.viz.Brain : mne.viz.Brain object
    """
    splitted_kwargs = split_kwargs(kwargs=kwargs,
                                   func_map={"stc_plot": mne.SourceEstimate.plot,
                                             "foci": mne.viz.Brain.add_foci})
    _check_hemi_param(hemi)
    brain = stc.plot(
        subject=subject,
        hemi=hemi,
        **splitted_kwargs["stc_plot"]
    )
    if show_source_locations:
        if hemi in ['both', 'lh'] and stc.lh_vertno.size != 0:
            brain.add_foci(stc.lh_vertno, coords_as_verts=True, hemi="lh", color='black', scale_factor=0.4,
                        **splitted_kwargs["foci"])
        if hemi in ['both', 'rh'] and stc.rh_vertno.size != 0:
            brain.add_foci(stc.rh_vertno, coords_as_verts=True, hemi="rh", color='black', scale_factor=0.4,
                        **splitted_kwargs["foci"])
    return brain


def plot_localized_sources(
        subject: str,
        lh_vertices: list = None,
        rh_vertices: list = None,
        hemi: str = "both",
        color="crimson",
        **kwargs) -> mne.viz.Brain:
    """
    Display ``mne.viz.Brain`` with foci corresponding to positions of localized sources.

    Parameters
    ----------
    subject : str
        Subject name the analysis is performed for.
    lh_vertices : list (optional)
        List of localized vertices in left hemisphere.
    rh_vertices : list (optional)
        List of localized vertices in right hemisphere.
    hemi : str (default: "both")
        Hemisphere to show.
    color : default ("crimson")
        Color to use for foci plotting. Can be anything matplotlib accepts.

    Returns
    -------
    mne.viz.Brain : ``mne.viz.Brain`` object with foci added on localized sources coordinates.

    """
    splitted_kwargs = split_kwargs(kwargs=kwargs,
                                   func_map={"brain": mne.viz.Brain,
                                             "foci": mne.viz.Brain.add_foci})
    brain = mne.viz.Brain(subject=subject, surf="inflated", hemi=hemi, **splitted_kwargs["brain"])
    if hemi in ["both", "lh"]:
        brain.add_foci(
            lh_vertices, coords_as_verts=True, hemi='lh', color=color, scale_factor=0.8, **splitted_kwargs["foci"])
    if hemi in ["both", "rh"]:
        brain.add_foci(
            rh_vertices, coords_as_verts=True, hemi='rh', color=color, scale_factor=0.8, **splitted_kwargs["foci"])
    return brain


def _assign_color_mapping(norm_power: float, cmap: str):
    cmap = cm.get_cmap(cmap)
    rgba = cmap(norm_power)
    rgb = tuple(int(255 * x) for x in rgba[:3])
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _save_brain_object(brain: mne.viz.Brain, output_path: str, output_name: str):
    """
    Save mne.viz.Brain object to .html.

    Parameters
    ----------
    brain : mne.viz.Brain
        mne.viz.Brain object to be saved to .html
    output_path : str
        Directory where 'brain' object should be saved.
    output_name : str
        Name of the .html file to be saved.
    """
    if not os.path.exists(output_path):
        print(f"Output directory {output_path} does not exists. Creating...")
        os.makedirs(output_path)

    if len(output_name.split('.')) == 2 and output_name.split('.')[-1] != "html":
        raise ValueError(f"Only saving to .html file is possible. Got {output_name.split('.')[-1]}.")
    elif len(output_name.split('.')) == 1:
        output_name = f"{output_name}.html"
    # Get the pyvista plotter
    plotter = brain._renderer.plotter
    # Save as html
    plotter.export_html(os.path.join(output_path, output_name))
    print(f"Saved to {os.path.join(output_path, output_name)}.")
