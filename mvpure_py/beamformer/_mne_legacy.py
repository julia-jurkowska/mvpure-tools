"""
Split function _compute_beamformer from MNE for use in MV-PURE.
Based on ``mne.beamformer.compute_beamformer``. See source `here
(https://github.com/mne-tools/mne-python/blob/main/mne/beamformer/_compute_beamformer.py)`_.
"""
# Author: Julia Jurkowska

import numpy as np

from mne.beamformer._compute_beamformer import (
    _reduce_leadfield_rank,
    _sym_inv_sm
)

from mne.utils import (
    _check_option,
    _pl,
    _reg_pinv,
    _sym_mat_pow,
    logger,
    warn,
)


def _compute_bf_terms(Gk, Cm_inv):
    # bf_numer = np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv)
    # bf_denom = np.matmul(bf_numer, Gk)

    Gk=np.squeeze(Gk)
    bf_numer = np.matmul(Gk, Cm_inv)
    bf_denom = np.matmul(bf_numer, Gk.T)

    return bf_numer, bf_denom


def _compute_mne_legacy(
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
):
    """
    First part of split _compute_beamformer function for MVPURE usage.
    For more details see original documentation from MNE-Python.

    Parameters
    ----------
    G : ndarray, shape (n_dipoles, n_channels)
        The leadfield.
    Cm : ndarray, shape (n_channels, n_channels)
        The data covariance matrix.
    reg : float
        Regularization parameter.
    n_orient : int
        Number of dipole orientations defined at each source point
    weight_norm : None | 'unit-noise-gain' | 'nai'
        The weight normalization scheme to use.
    pick_ori : None | 'normal' | 'max-power'
        The source orientation to compute the beamformer in.
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
    rank : dict | None | 'full' | 'info'
        See compute_rank.
    inversion : 'matrix' | 'single'
        The inversion scheme to compute the weights.
    nn : ndarray, shape (n_dipoles, 3)
        The source normals.
    orient_std : ndarray, shape (n_dipoles,)
        The std of the orientation prior used in weighting the lead fields.
    whitener : ndarray, shape (n_channels, n_channels)
        The whitener.
    """
    _check_option(
        "weight_norm",
        weight_norm,
        ["unit-noise-gain-invariant", "unit-noise-gain", "nai", None],
    )

    # Whiten the data covariance
    Cm = whitener @ Cm @ whitener.T.conj()
    # Whiten the noise covariance
    Nm = whitener @ Nm @ whitener.T.conj()
    # Restore to properly Hermitian as large whitening coefs can have bad
    # rounding error
    Cm[:] = (Cm + Cm.T.conj()) / 2.0
    Nm[:] = (Nm + Nm.T.conj()) / 2.0

    assert Cm.shape == (G.shape[0],) * 2
    assert Nm.shape == (G.shape[0],) * 2
    s, _ = np.linalg.eigh(Cm)
    if not (s >= -s.max() * 1e-7).all():
        # This shouldn't ever happen, but just in case
        warn(
            "data covariance does not appear to be positive semidefinite, "
            "results will likely be incorrect"
        )
    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity
    # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
    Cm_inv, loading_factor, rank = _reg_pinv(Cm, reg, rank)
    Nm_inv, _, _ = _reg_pinv(Nm, reg, rank)

    assert orient_std.shape == (G.shape[1],)
    n_sources = G.shape[1] // n_orient
    assert nn.shape == (n_sources, 3)

    logger.info(f"Computing beamformer filters for {n_sources} source{_pl(n_sources)}")
    n_channels = G.shape[0]
    assert n_orient in (3, 1)
    Gk = np.reshape(G.T, (n_sources, n_orient, n_channels)).transpose(0, 2, 1)
    assert Gk.shape == (n_sources, n_channels, n_orient)
    sk = np.reshape(orient_std, (n_sources, n_orient))
    del G, orient_std

    _check_option("reduce_rank", reduce_rank, (True, False))

    # inversion of the denominator
    _check_option("inversion", inversion, ("matrix", "single"))
    if (
            inversion == "single"
            and n_orient > 1
            and pick_ori == "vector"
            and weight_norm == "unit-noise-gain-invariant"
    ):
        raise ValueError(
            'Cannot use pick_ori="vector" with inversion="single" and '
            'weight_norm="unit-noise-gain-invariant"'
        )
    if reduce_rank and inversion == "single":
        raise ValueError(
            'reduce_rank cannot be used with inversion="single"; '
            'consider using inversion="matrix" if you have a '
            "rank-deficient forward model (i.e., from a sphere "
            "model with MEG channels), otherwise consider using "
            "reduce_rank=False"
        )
    if n_orient > 1:
        _, Gk_s, _ = np.linalg.svd(Gk, full_matrices=False)
        assert Gk_s.shape == (n_sources, n_orient)
        if not reduce_rank and (Gk_s[:, 0] > 1e6 * Gk_s[:, 2]).any():
            raise ValueError(
                "Singular matrix detected when estimating spatial filters. "
                "Consider reducing the rank of the forward operator by using "
                "reduce_rank=True."
            )
        del Gk_s

    #
    # 1. Reduce rank of the lead field
    #
    if reduce_rank:
        Gk = _reduce_leadfield_rank(Gk)

    #
    # 2. Reorient lead field in direction of max power or normal
    #
    if pick_ori == "max-power":
        assert n_orient == 3
        _, bf_denom = _compute_bf_terms(Gk, Cm_inv)
        if weight_norm is None:
            ori_numer = np.eye(n_orient)[np.newaxis]
            ori_denom = bf_denom
        else:
            # compute power, cf Sekihara & Nagarajan 2008, eq. 4.47
            ori_numer = bf_denom
            # Cm_inv should be Hermitian so no need for .T.conj()
            ori_denom = np.matmul(
                np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv @ Cm_inv), Gk
            )
        ori_denom_inv = _sym_inv_sm(ori_denom, reduce_rank, inversion, sk)
        ori_pick = np.matmul(ori_denom_inv, ori_numer)
        assert ori_pick.shape == (n_sources, n_orient, n_orient)

        # pick eigenvector that corresponds to maximum eigenvalue:
        eig_vals, eig_vecs = np.linalg.eig(ori_pick.real)  # not Hermitian!
        # sort eigenvectors by eigenvalues for picking:
        order = np.argsort(np.abs(eig_vals), axis=-1)
        # eig_vals = np.take_along_axis(eig_vals, order, axis=-1)
        max_power_ori = eig_vecs[np.arange(len(eig_vecs)), :, order[:, -1]]
        assert max_power_ori.shape == (n_sources, n_orient)

        # set the (otherwise arbitrary) sign to match the normal
        signs = np.sign(np.sum(max_power_ori * nn, axis=1, keepdims=True))
        signs[signs == 0] = 1.0
        max_power_ori *= signs

        # Compute the lead field for the optimal orientation,
        # and adjust numer/denom
        Gk = np.matmul(Gk, max_power_ori[..., np.newaxis])
        n_orient = 1
    else:
        max_power_ori = None
        if pick_ori == "normal":
            Gk = Gk[..., 2:3]
            n_orient = 1

    return Gk, Cm, Cm_inv, Nm, Nm_inv, n_orient, n_sources, n_channels, max_power_ori, sk, loading_factor


def _compute_weighted_filter(W,
                             Cm,
                             bf_numer,
                             n_orient,
                             weight_norm,
                             rank,
                             n_sources,
                             loading_factor,
                             ):
    """
    Second  part of split _compute_beamformer function for MVPURE usage - after computing MVP_R/MVP_N filter.
    For more details see original documentation from MNE-Python and _compute_beamformer.

    Parameters
    ----------
    W : np.ndarray
        The beamformer filter weights.
    Cm : np.ndarray
        Data covariance matrix
    bf_numer : np.ndarray
        First result of _compute_bf_terms()
    n_orient : int
        Number of dipole orientations defined at each source point
    weight_norm : None | 'unit-noise-gain' | 'nai'
        The weight normalization scheme to use.
    rank : dict | None | 'full' | 'info'
        See compute_rank.
    n_sources : int
        Number of sources to localize

    Returns
    -------
    W : np.ndarray
        Beamformer filter weights after weighting.
    """
    if weight_norm is not None:
        # Three different ways to calculate the normalization factors here.
        # Only matters when in vector mode, as otherwise n_orient == 1 and
        # they are all equivalent.
        #
        # In MNE < 0.21, we just used the Frobenius matrix norm:
        #
        #    noise_norm = np.linalg.norm(W, axis=(1, 2), keepdims=True)
        #    assert noise_norm.shape == (n_sources, 1, 1)
        #    W /= noise_norm
        #
        # Sekihara 2008 says to use sqrt(diag(W_ug @ W_ug.T)), which is not
        # rotation invariant:
        if weight_norm in ("unit-noise-gain", "nai"):
            noise_norm = np.matmul(W, W.swapaxes(-2, -1).conj()).real
            noise_norm = np.reshape(  # np.diag operation over last two axes
                noise_norm, (n_sources, -1, 1)
            )[:, :: n_orient + 1]
            np.sqrt(noise_norm, out=noise_norm)
            noise_norm[noise_norm == 0] = np.inf
            assert noise_norm.shape == (n_sources, n_orient, 1)
            W /= noise_norm
        else:
            assert weight_norm == "unit-noise-gain-invariant"
            # Here we use sqrtm. The shortcut:
            #
            #    use = W
            #
            # ... does not match the direct route (it is rotated!), so we'll
            # use the direct one to match FieldTrip:
            use = bf_numer
            inner = np.matmul(use, use.swapaxes(-2, -1).conj())
            W = np.matmul(_sym_mat_pow(inner, -0.5), use)
            noise_norm = 1.0

        if weight_norm == "nai":
            # Estimate noise level based on covariance matrix, taking the
            # first eigenvalue that falls outside the signal subspace or the
            # loading factor used during regularization, whichever is largest.
            if rank > len(Cm):
                # Covariance matrix is full rank, no noise subspace!
                # Use the loading factor as noise ceiling.
                if loading_factor == 0:
                    raise RuntimeError(
                        "Cannot compute noise subspace with a full-rank "
                        "covariance matrix and no regularization. Try "
                        "manually specifying the rank of the covariance "
                        "matrix or using regularization."
                    )
                noise = loading_factor
            else:
                noise, _ = np.linalg.eigh(Cm)
                noise = noise[-rank]
                noise = max(noise, loading_factor)
            W /= np.sqrt(noise)

    return W
