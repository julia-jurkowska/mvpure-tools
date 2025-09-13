"""
Computing MVPURE beamformers.
Based on mne.beamformer._lcmv.py (https://github.com/mne-tools/mne-python/blob/main/mne/beamformer/_lcmv.py)
"""
# Author: Julia Jurkowska

import numpy as np
import mne

from mne._fiff.meas_info import _simplify_info
from mne._fiff.pick import pick_channels_cov, pick_info
from mne.forward import _subject_from_forward
from mne.minimum_norm.inverse import _check_depth, combine_xyz
from mne.rank import compute_rank
from mne.source_estimate import _get_src_type, _make_stc
from mne.utils import (
    _check_info_inv,
    _check_one_ch_type,
    logger,
    verbose,
)

from mne.beamformer._lcmv import (
    apply_lcmv,
    apply_lcmv_epochs,
    apply_lcmv_raw,
    apply_lcmv_cov
    
)
from mne.beamformer._compute_beamformer import (
    Beamformer,
    _prepare_beamformer_input,
)

from ._compute_beamformer import _compute_beamformer

from ._parameters_checks import _prepare_parameters


@verbose
def make_filter(
    info,
    forward,
    data_cov,
    reg=0.05,
    noise_cov=None,
    label=None,
    pick_ori=None,
    rank="info",
    weight_norm="unit-noise-gain-invariant",
    reduce_rank=False,
    depth=None,
    inversion="matrix",
    verbose=None,
    mvpure_params: dict = None,  # if None: standard path from MNE
):
    """
    Compute spatial filter from MVPURE package.
    Based on mne.beamformer.make_lcmv() function with additional parameter ``mvpure_params`` that specifies
    MVPURE-specific parameters:
    - 'filter_type' (with options MVP_R and MVP_N. If None, MVP_R will be used)
    - 'filter_rank' (as integer lower or equal to number of sources or 'full'; if None 'full' will be used)

    mvpure_params = {
        'filter_type': str,
        'filter_rank': str | int
    }

    Parameters
    ----------
    info :
        Specifies the channels to include. Bad channels (in ``info['bads']``)
        are not used.
    forward : mne.Forward
        Forward operator.
    data_cov : mne.Covariance
        Data covariance instance.
    reg : float
        The regularization for the whitened data covariance.
    noise_cov : mne.Covariance
        Noise covariance instance.
    label : mne.Label
        Restricts the solution to a given label.
    pick_ori :
        The source orientation to compute the beamformer in.
    rank : dict | None | 'full' | 'info'
        See compute_rank.
    weight_norm : None | 'unit-noise-gain' | 'nai'
        The weight normalization scheme to use.
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
    depth :
    inversion : 'matrix' | 'single'
        The inversion scheme to compute the weights.
    verbose : bool | str | int | None
        Control verbostiy of the loggin output. If None, use the default verbostiy level.
    mvpure_params :  dict
        Dictionary with MVPURE-py specific parameters.
        mvpure_params = {
            'filter_type': str,
            'filter_rank': str | int
        }
        - 'filter_type' (with options MVP_R and MVP_N. If None, MVP_R will be used)
        - 'filter_rank' (as integer lower or equal to number of sources or 'full'; if None 'full' will be used)

    Returns
    -------
    mne.Beamformer :
        Dictionary containing filter weights from MVPURE beamformer.
        Contains the same set of keys as mne.Beamformer.
    """
    # check number of sensor types present in the data and ensure a noise cov
    info = _simplify_info(info, keep=("proc_history",))
    noise_cov, _, allow_mismatch = _check_one_ch_type(
        None, info, forward, data_cov, noise_cov
    )
    # check MV-PURE specific parameters
    filter_type, filter_rank = _prepare_parameters(mvpure_params, forward['nsource'])
    # XXX we need this extra picking step (can't just rely on minimum norm's
    # because there can be a mismatch. Should probably add an extra arg to
    # _prepare_beamformer_input at some point (later)
    picks = _check_info_inv(info, forward, data_cov, noise_cov)
    info = pick_info(info, picks)
    data_rank = compute_rank(data_cov, rank=rank, info=info)
    noise_rank = compute_rank(noise_cov, rank=rank, info=info)
    for key in data_rank:
        if (
            key not in noise_rank or data_rank[key] != noise_rank[key]
        ) and not allow_mismatch:
            raise ValueError(
                f"{key} data rank ({data_rank[key]}) did not match the noise rank ("
                f"{noise_rank.get(key, None)})"
            )
    del noise_rank
    rank = data_rank
    logger.info(f"Making {filter_type} beamformer with rank {rank} (note: MNE-Python rank)")
    del data_rank
    depth = _check_depth(depth, "depth_sparse")
    if inversion == "single":
        depth["combine_xyz"] = False

    (
        is_free_ori,
        info,
        proj,
        vertno,
        G,
        whitener,
        nn,
        orient_std,
    ) = _prepare_beamformer_input(
        info,
        forward,
        label,
        pick_ori,
        noise_cov=noise_cov,
        rank=rank,
        pca=False,
        **depth,
    )
    ch_names = list(info["ch_names"])

    data_cov = pick_channels_cov(data_cov, include=ch_names)
    noise_cov = pick_channels_cov(noise_cov, include=ch_names)
    Cm = data_cov._get_square()
    Nm = noise_cov._get_square()
    if "estimator" in data_cov:
        del data_cov["estimator"]
    rank_int = sum(rank.values())
    del rank

    # compute spatial filter
    n_orient = 3 if is_free_ori else 1
    W, max_power_ori = _compute_beamformer(
        G,
        Cm,
        Nm,
        reg,
        n_orient,
        weight_norm,
        pick_ori,
        reduce_rank,
        rank_int,
        inversion=inversion,
        nn=nn,
        orient_std=orient_std,
        whitener=whitener,
        mvpure_params=mvpure_params,
    )

    # get src type to store with filters for _make_stc
    src_type = _get_src_type(forward["src"], vertno)

    # get subject to store with filters
    subject_from = _subject_from_forward(forward)

    # Is the computed beamformer a scalar or vector beamformer?
    is_free_ori = is_free_ori if pick_ori in [None, "vector"] else False
    is_ssp = bool(info["projs"])

    filters = Beamformer(
        kind=filter_type,
        weights=W,
        data_cov=data_cov,
        noise_cov=noise_cov,
        whitener=whitener,
        weight_norm=weight_norm,
        pick_ori=pick_ori,
        ch_names=ch_names,
        proj=proj,
        is_ssp=is_ssp,
        vertices=vertno,
        is_free_ori=is_free_ori,
        n_sources=forward["nsource"],
        src_type=src_type,
        source_nn=forward["source_nn"].copy(),
        subject=subject_from,
        rank=rank_int,
        max_power_ori=max_power_ori,
        inversion=inversion,
    )

    return filters


def apply_filter(evoked: mne.Evoked,
                 filters: mne.beamformer.Beamformer,
                 verbose=None):
    """
    Apply MVPURE beamformer weights on evoked data.

    Parameters
    ----------
    evoked : mne.Evoked
        Evoked data to invert.
    filters : mne.beamformer.Beamformer
        MVPURE spatial filter (beamformer weights) returned from :func:`make_filter`.

    Returns
    -------
    mne.SourceEstimate: Source time courses
    """
    return apply_lcmv(evoked, filters, verbose=verbose)
    
    
def apply_filter_epochs(epochs: mne.Epochs,
                        filters: mne.beamformer.Beamformer,
                        return_generator=False,
                        verbose=None):
    """
    Apply MVPURE beamformer weights on single trial data.

    Parameters
    ----------
    epochs : mne.Epochs
        Single trial epochs.
    filters : mne.beamformer.Beamformer
        MVPURE spatial filter (beamformer weights) returned from :func:`make_filter`.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating over the stcs
        without having to keep them all in memory.

    Returns
    -------
    list | generator of mne.SourceEstimate
        The source estimated for all epochs
    """
    return apply_lcmv_epochs(epochs, filters, return_generator=return_generator, verbose=verbose)


def apply_filter_raw(raw: mne.io.Raw, 
                     filters: mne.beamformer.Beamformer,
                     start=None,
                     stop=None,
                     verbose=None):
    """
    Apply MVPURE beamformer weights on raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    filters : mne.beamformer.Beamformer
        MVPURE spatial filter (beamformer weights) returned from :func:`make_filter`.
    start : int
        Index of first time sample.
    stop : int
        Index of first time sample not to include

    Returns
    -------
    mne.SourceEstimate : Source time courses.

    """
    return apply_lcmv_raw(raw, filters, start, stop, verbose=None)


def apply_filter_cov(data_cov: mne.Covariance,
                     filters: mne.beamformer.Beamformer,
                     verbose=None):
    """
    Apply MVPURE beamformer weights to a data covariance matrix to estimate source power.

    Parameters
    ----------
    data_cov : mne.Covariance
        Data covariance matrix.
    filters : mne.beamformer.Beamformer
        MVPURE spatial filter (beamformer weights) returned from :func:`make_filter`.

    Returns
    -------
    mne.SourceEstimate : Source power.

    """
    return apply_lcmv_cov(data_cov, filters, verbose=verbose)

