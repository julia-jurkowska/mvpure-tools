"""Validation of results achieved on synthetic data."""
# Author: Julia Jurkowska
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error

from ..utils import (
    vertices_to_coordinates,
    transform_leadfield_indices_to_vertices,
    subset_forward
)

from ..localizer import localize


def localization_error(
        vertices_true: list[list[int]],
        localized_vertices: list[list[int]],
        src: mne.SourceSpaces
) -> dict:
    """
    Compute localization error between true and estimated source vertices.

    The function converts vertex indices to 3D coordinated using the provided source space and computes pairwise
    Euclidean distances between true and localized sources. The Hungarian algorithm is used to determine the optimal
    one-to-one assignment that minimizes total distance error.

    Parameters
    ----------
    vertices_true : list[list[int]]
        Ground-truth vertices for each hemisphere in the form ``[lh_vertices, rh_vertices]``
    localized_vertices : list[list[int]]
        Localized vertices for each hemisphere in the form ``[lh_vertices, rh_vertices]``
    src : mne.SourceSpaces
        Source space used to map vertices to 3D coordinates

    Returns
    -------
    results : dict
        Dictionary containing localization error statistics:

        - ``mean_error`` : float
          Mean distance error (mm).
        - ``median_error`` : float
          Median distance error (mm).
        - ``sum_error`` : float
          Sum of all matched vertex errors (mm).
        - ``max_error`` : float
          Maximum error among matched vertices (mm).
        - ``min_error`` : float
          Minimum error among matched vertices (mm).
        - ``errors_per_vertex`` : ndarray
          Distance error for each matched vertex (mm).
    """
    pos_true = vertices_to_coordinates(vertices_true, src)
    pos_loc = vertices_to_coordinates(localized_vertices, src)

    if len(pos_true) == 0 or len(pos_loc) == 0:
        raise ValueError("No active vertices found in ground truth or estimated set.")

    # Pairwise distance matrix
    dist = cdist(pos_true, pos_loc, metric="euclidean")

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(dist)
    matched_distances = dist[row_ind, col_ind]
    # Unit conversion to mm
    matched_distances = matched_distances * 1000.0

    results = {
        "mean_error": float(np.mean(matched_distances)),
        "median_error": float(np.median(matched_distances)),
        "sum_error": float(np.sum(matched_distances)),
        "max_error": float(np.max(matched_distances)),
        "min_error": float(np.min(matched_distances)),
        "errors_per_vertex": matched_distances,
    }
    return results


def evaluate_localization_for_each_rank(
        subject: str,
        subjects_dir: str,
        n_sources: int,
        R: mne.Covariance,
        N: mne.Covariance,
        forward: mne.Forward,
        true_vertices: list[list[int]],
        localizer_to_use: list | str,
        plot_correct_sources_by_rank: bool = True,
        plot_sum_error_by_rank: bool = False,
        plot_mean_error_by_rank: bool = True,
        show_progress: bool = True
):
    """
    Evaluate source localization performance for different rank parameters.

    The function iteratively runs a source localization method for ranks ``1 ... n_sources`` and compared the localized
    vertices with ground-truth vertices. For each rank, the number of correctly localized sources and localization error
    metrics are computed.

    Parameters
    ----------
    subject : str
        Subject identifier the analysis is performed for
    subjects_dir : str
        Directory where ``subject`` folder is being stored
    n_sources : int
        Number of sources to localize
    R : mne.Covariance
        Data covariance matrix
    N : mne.Covariance
        Noise covariance matrix
    forward : mne.Forward
        Forward solution used for source localization.
    true_vertices : list[list[int]]
        Ground-truth vertices in the format ``[lh_vertices, rh_vertices]``.
    localizer_to_use : list | str
        Type of localizer index to use.
        Options are: 'mai', 'mpz', 'mai_mvp', 'mpz_mvp'.
    plot_correct_sources_by_rank : bool, optional
        Whether to plot the number of correctly localized sources per rank.
        Default is True.
    plot_sum_error_by_rank : bool, optional
        Whether to plot the sum of localization errors per rank.
        Default is False.
    plot_mean_error_by_rank : bool, optional
        Whether to plot the mean localization error per rank.
        Default is True.
    show_progress : bool, optional
        Whether to display progress during localization. Default is True.

    Returns
    -------
    summary_dict : dict
        Dictionary summarizing results for each rank. Each entry contains:

        - ``localized`` : localization result object
        - ``n_correctly_localized`` : int
          Number of correctly identified vertices.
        - ``correctly_localized`` : list[list[int]]
          Intersection between estimated and true vertices.
        - ``error_info`` : dict
          Localization error statistics returned by
          :func:`localization_error`.
    """
    summary_dict = {}
    for r in range(1, n_sources + 1):
        rank_label = f"rank_{r}"
        summary_dict[rank_label] = {}
        # Localize for rank r
        locs_temp = localize(
            subject=subject,
            subjects_dir=subjects_dir,
            localizer_to_use=localizer_to_use,
            n_sources_to_localize=n_sources,
            R=R,
            N=N,
            forward=forward,
            r=r,
            show_progress=show_progress
        )
        lh_vert, lh_idx, rh_vert, rh_idx = transform_leadfield_indices_to_vertices(
            lf_idx=locs_temp["sources"],
            src=forward['src'],
            hemi="both",
            include_mapping=True
        )
        locs_temp.add_vertices_info(
            lh_vertices=lh_vert,
            lh_indices=lh_idx,
            rh_vertices=rh_vert,
            rh_indices=rh_idx
        )
        summary_dict[rank_label]['localized'] = locs_temp

        lh_intersection = list(set(lh_vert).intersection(true_vertices[0]))
        rh_intersection = list(set(rh_vert).intersection(true_vertices[1]))

        summary_dict[rank_label]["n_correctly_localized"] = len(lh_intersection) + len(rh_intersection)
        summary_dict[rank_label]['correctly_localized'] = [sorted(lh_intersection), sorted(rh_intersection)]
        summary_dict[rank_label]["error_info"] = localization_error(
            vertices_true=true_vertices,
            localized_vertices=locs_temp['vertno_in_use'],
            src=forward['src']
        )

    if plot_correct_sources_by_rank:
        _plot_correct_sources_by_rank(summary_dict)

    if plot_sum_error_by_rank:
        _plot_sum_error_by_rank(summary_dict)

    if plot_mean_error_by_rank:
        _plot_mean_error_by_rank(summary_dict)
    return summary_dict


def evaluate_reconstruction(summary_dict: dict,
                            signal,
                            sim_stc: mne.SourceEstimate,
                            fwd: mne.Forward,
                            R: mne.Covariance,
                            N: mne.Covariance,
                            reg=0.05,
                            pick_ori=None,
                            rank=None,
                            **kwargs):
    """
    Evaluate reconstruction quality using LCMV beamforming.

    For each rank entr in the localization summary, two reconstructions are computed:

    - LCMV using the full forward model
    - LCMV using a forward model restricted to the localized sources.

    Reconstruction performance is evaluated using the mean squared error (MSE)
    between the simulated source activity and the reconstructed activity for
    the correctly localized vertices.

    Parameters
    ----------
    summary_dict : dict
        Localization summary returned by
        :func:`evaluate_localization_for_each_rank`.
    signal : mne.Epochs
        Sensor-space data used for beamforming.
    sim_stc : mne.SourceEstimate
        Simulated ground-truth source time course.
    fwd : mne.Forward
        Full forward solution.
    R : mne.Covariance
        Data covariance
    N : mne.Covariance
        Noise covariance
    reg : float, optional
        Regularization parameter for LCMV. Default is 0.05.
    pick_ori : str | None, optional
        Orientation selection for beamformer weights.
    rank : int | None, optional
        Rank used in beamformer computation.
    **kwargs
        Additional arguments passed to ``mne.beamformer.make_lcmv``.
    """
    for key_rank, summary in summary_dict.items():
        print(f"{key_rank}")
        # LCMV on full (unchanged) mne.Forward
        lcmv_org = mne.beamformer.make_lcmv(
            signal.info,
            fwd,
            data_cov=R,
            reg=reg,
            noise_cov=N,
            pick_ori=pick_ori,
            weight_norm=None,
            rank=rank,
            **kwargs
        )

        stc_lcmv_org = mne.beamformer.apply_lcmv(signal, lcmv_org)

        # LCMV only on correctly localized sources
        new_fwd = subset_forward(
            old_fwd=fwd,
            localized=summary['localized'],
            hemi="both"
        )
        lcmv_subset = mne.beamformer.make_lcmv(
            signal.info,
            new_fwd,
            data_cov=R,
            reg=reg,
            noise_cov=N,
            pick_ori=pick_ori,
            weight_norm=None,
            rank=rank,
            **kwargs
        )

        stc_lcmv_subset = mne.beamformer.apply_lcmv(signal, lcmv_subset)

        # Crop Source Estimate from simulation
        sim_stc_cropped = sim_stc.copy().crop(tmin=stc_lcmv_subset.times[0],
                                              tmax=stc_lcmv_subset.times[-1])
        # check
        assert np.array_equal(stc_lcmv_subset.times, stc_lcmv_org.times)
        assert np.array_equal(stc_lcmv_subset.times, sim_stc_cropped.times)

        sim_stc_soi, _ = get_stc_data_for_vertices(sim_stc_cropped, vertices=summary['correctly_localized'])
        stc_lcmv_org_soi, _ = get_stc_data_for_vertices(stc_lcmv_org, vertices=summary['correctly_localized'])
        stc_lcmv_subset_soi, _ = get_stc_data_for_vertices(stc_lcmv_subset, vertices=summary['correctly_localized'])

        mse_sim_lcmv_org = mean_squared_error(sim_stc_soi, stc_lcmv_org_soi)
        print(f"MSE simulation <-> full LCMV: {mse_sim_lcmv_org}")

        mse_sim_lcmv_subset = mean_squared_error(sim_stc_soi, stc_lcmv_subset_soi)
        print(f"MSE simulation <-> subset LCMV: {mse_sim_lcmv_subset}")


def get_stc_data_for_vertices(stc: mne.SourceEstimate,
                              vertices: list[list[int]]):
    """
    Extract source time courses for specific vertices from a SourceEstimate.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate containing source time courses.
    vertices : [list[list[int]]
        Vertices of interest in the format ``[lh_vertices, rh_vertices]``

    Returns
    -------
    data : ndarray
        Source time courses corresponding to the selected vertices
        with shape ``(n_vertices, n_times)``.
    indices : ndarray
        Indices of the selected vertices within ``stc.data``.

    """
    lh_vertices, rh_vertices = vertices

    lh_vertno, rh_vertno = stc.vertices
    lh_len = len(lh_vertno)

    lh_map = {v: i for i, v in enumerate(lh_vertno)}
    rh_map = {v: i + lh_len for i, v in enumerate(rh_vertno)}

    indices = []

    for v in lh_vertices:
        if v in lh_map:
            indices.append(lh_map[v])

    for v in rh_vertices:
        if v in rh_map:
            indices.append(rh_map[v])

    indices = np.array(indices)

    return stc.data[indices], indices


def compare_with_strongest_sources_lcmv(
        signal,
        n_sources: int,
        true_vertices: list[list[int]],
        forward: mne.Forward,
        R: mne.Covariance,
        N: mne.Covariance,
        reg: float = 0.05,
        pick_ori=None,
):
    """
    Identify the strongest sources based on LCMV reconstruction and compare them to ground-truth sources.

    Parameters
    ----------
    signal :
        Sensor space data
    n_sources : int
        Number of sources to localize.
    true_vertices : list[list[int]]
        Ground-truth vertices in the format ``[lh_vertices, rh_vertices]``.
    forward : mne.Forward
        Forward solution used for creating ground-truth data.
    R : mne.Covariance
        Data covariance matrix
    N : mne.Covariance
        Noise covariance matrix
    reg : float
        The regularization for the whitened data covariance. Default to 0.05.
    pick_ori :
        Orientation specification. Default to None.
    """
    lcmv = mne.beamformer.make_lcmv(
        signal.info,
        forward,
        R,
        reg=reg,
        noise_cov=N,
        pick_ori=pick_ori,
        weight_norm="nai",
        rank=None
    )
    stc = mne.beamformer.apply_lcmv(signal, lcmv)

    # Maximum NAI value for each source across time
    peak_nai_per_source = np.max(np.abs(stc.data), axis=1)
    # Indices of strongest sources
    top_idx = np.argsort(peak_nai_per_source)[-n_sources:][::-1]

    top_vertices = [[], []]  # [[lh], [rh]]
    n_lh = len(stc.vertices[0])

    for idx in top_idx:
        if idx < n_lh:
            top_vertices[0].append(stc.vertices[0][idx])
        else:
            top_vertices[1].append(stc.vertices[1][idx - n_lh])

    lh_intersection = list(set(top_vertices[0]).intersection(true_vertices[0]))
    rh_intersection = list(set(top_vertices[1]).intersection(true_vertices[1]))

    n_correctly_localized = len(lh_intersection) + len(rh_intersection)
    error_info = localization_error(
        vertices_true=true_vertices,
        localized_vertices=top_vertices,
        src=forward['src']
    )

    return top_vertices, n_correctly_localized, error_info


def _plot_correct_sources_by_rank(
        evaluation_summary_dict: dict,
        color: str = "indigo"
):
    """
    Plot the number of correctly localized sources as a function of rank.
    """
    correctly_localized = [evaluation_summary_dict[key]["n_correctly_localized"]
                           for key in list(evaluation_summary_dict.keys())]
    plt.scatter(
        x=np.arange(1, len(correctly_localized)+1),
        y=correctly_localized,
        color=color
    )
    plt.xlim((0, len(correctly_localized)+1))
    plt.xticks(np.arange(1, len(correctly_localized)+1, 1),
               np.arange(1, len(correctly_localized)+1, 1))
    plt.ylim((0, len(correctly_localized) + 1))
    plt.yticks(np.arange(1, len(correctly_localized) + 1, 1),
               np.arange(1, len(correctly_localized) + 1, 1))
    plt.grid(alpha=0.3)
    plt.xlabel("Rank parameter")
    plt.ylabel("Number of correctly localized sources")
    plt.show()


def _plot_sum_error_by_rank(
        evaluation_summary_dict: dict,
        color: str = "indigo"
):
    """
    Plot the sum of localization errors as a function of rank.
    """
    error_sum = [evaluation_summary_dict[key]["error_info"]["sum_error"]
                 for key in list(evaluation_summary_dict.keys())]
    plt.scatter(
        x=np.arange(1, len(error_sum) + 1),
        y=error_sum,
        color=color
    )
    plt.xlim((0, len(error_sum) + 1))
    plt.xticks(np.arange(1, len(error_sum) + 1, 1),
               np.arange(1, len(error_sum) + 1, 1))
    plt.grid(alpha=0.3)
    plt.xlabel("Rank parameter")
    plt.ylabel("Sum of distance error [mm]")
    plt.show()


def _plot_mean_error_by_rank(
        evaluation_summary_dict: dict,
        color: str = "indigo"
):
    """
    Plot the mean localization error as a function of rank.
    """
    mean_sum = [evaluation_summary_dict[key]["error_info"]["mean_error"]
                for key in list(evaluation_summary_dict.keys())]
    plt.scatter(
        x=np.arange(1, len(mean_sum) + 1),
        y=mean_sum,
        color=color
    )
    plt.xlim((0, len(mean_sum) + 1))
    plt.xticks(np.arange(1, len(mean_sum) + 1, 1),
               np.arange(1, len(mean_sum) + 1, 1))
    plt.grid(alpha=0.3)
    plt.xlabel("Rank parameter")
    plt.ylabel("Mean of distance error [mm]")
    plt.show()
