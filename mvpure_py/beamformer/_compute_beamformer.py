""" Modification to MNE-Python _compute_beamformer
(https://github.com/mne-tools/mne-python/blob/main/mne/beamformer/_compute_beamformer.py)
 in order to adapt to MVPURE requirements"""
# Author: Julia Jurkowska

from copy import deepcopy

import numpy as np

from scipy.linalg import cho_factor, cho_solve

from mne.minimum_norm.inverse import _prepare_forward
from mne.utils import logger

from mne.beamformer._compute_beamformer import (
    _check_proj_match,
    _check_src_type,
    _prepare_beamformer_input,
    _reduce_leadfield_rank,
    _sym_inv_sm
)

from ._mne_legacy import (
    _compute_mne_legacy,
    _compute_weighted_filter,
    _compute_bf_terms
)

from .filters_utils import (
    make_mvp_n,
    make_mvp_r
)

from ._parameters_checks import _prepare_parameters


def _compute_beamformer(
        G,
        Cm,
        Nm,
        reg,
        n_orient,
        weight_norm,
        pick_ori,
        reduce_rank,
        rank,
        inversion,
        nn,
        orient_std,
        whitener,
        mvpure_params: dict = None
):
    """
    Computing beamformer.
    Based on mne.beamformer._compute_beamformer._compute_beamformer() function with additional parameter 'mvpure_params'
    that specifies MVPURE-specific parameters:
    - 'filter_type' (with options MVP_R and MVP_N. If None, MVP_R will be used)
    - 'filter_rank' (as integer lower or equal to number of sources or 'full'; if None 'full' will be used)

    mvpure_params = {
        'filter_type': str,
        'filter_rank': str | int
    }
    """
    (
        Gk,
        Cm,
        Cm_inv,
        Nm,
        Nm_inv,
        n_orient,
        n_sources,
        n_channels,
        max_power_ori,
        sk,
        loading_factor
    ) = _compute_mne_legacy(G,
                            Cm,
                            Nm,
                            reg,
                            n_orient,
                            weight_norm,
                            pick_ori,
                            reduce_rank,
                            rank,
                            inversion,
                            nn,
                            orient_std,
                            whitener)

    bf_numer, bf_denom = _compute_bf_terms(Gk, Cm_inv)
    # assert bf_denom.shape == (n_sources,) + (n_orient,) * 2
    # assert bf_numer.shape == (n_sources, n_orient, n_channels)
    # check MVPURE specific parameters
    filter_type, filter_rank = _prepare_parameters(mvpure_params, n_sources)
    if filter_type == 'MVP_R':
        # bf_denom_inv = _sym_inv_sm(bf_denom, reduce_rank, inversion, sk)
        # bf_denom_inv = np.linalg.pinv(bf_denom)
        c, lower = cho_factor(bf_denom, lower=True, check_finite=True)
        bf_denom_inv = cho_solve((c, lower), np.eye(bf_denom.shape[0]))
        # assert bf_denom_inv.shape == (n_sources, n_orient, n_orient)

        W = make_mvp_r(Gk, bf_numer, bf_denom_inv, filter_rank=filter_rank,
                       n_orient=n_orient, R=Cm)
    elif filter_type == 'MVP_N':
        bf_numer_noise, bf_denom_noise = _compute_bf_terms(Gk, Nm_inv)
        # bf_denom_noise_inv = _sym_inv_sm(bf_denom_noise, reduce_rank, inversion, sk)
        # bf_denom_noise_inv = np.linalg.pinv(bf_denom_noise)
        c, lower = cho_factor(bf_denom_noise, lower=True, check_finite=True)
        bf_denom_noise_inv = cho_solve((c, lower), np.eye(bf_denom.shape[0]))
        
        W = make_mvp_n(Gk, bf_numer_noise, bf_denom_noise_inv, filter_rank=filter_rank,
                       n_orient=n_orient, N=Nm)
    else:
        raise ValueError(f"Only filters available in MVPURE package are: MVP_R, MVP_N. Got {filter_type} instead.")

    W = _compute_weighted_filter(W,
                                 Cm,
                                 bf_numer,
                                 n_orient,
                                 weight_norm,
                                 rank,
                                 n_sources,
                                 loading_factor,
                                 )

    W = W.reshape(n_sources * n_orient, n_channels)
    logger.info("Filter computation complete")
    return W, max_power_ori
