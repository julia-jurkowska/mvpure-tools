# Author: Julia Jurkowska

import mne
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean
from typing import Tuple, List, Dict


def simulate_source_epochs(
        n_epochs: int,
        lf_subset_indices: np.ndarray,
        n_times: int,
        poststimuli_mask: np.ndarray,
        bg_lf_subset_indices: np.ndarray,
        poststimuli_lf_subset_indices: np.ndarray,
        lf_to_label: Dict,
        sfreq: float,
        erp_factor: float = None,
        snr_db: float = 10.0,
        order_bg: int = 6,
        order: int = 6,
        target_std_bg: float = 10e-9,
        target_std: float = 30e-9,
        coupling_bg: float = 0.1,
        coupling: float = 0.4,
        noise_bg: float = 1.0,
        noise: float = 0.3,
        gaussian_filter_sigma_bg: float = 1.0,
        gaussian_filter_sigma: float = 4.0,
        n_dominant_eigvals: int = 3,
        seed=None
):
    """
    Simulate source-space EEG activity for multiple epochs (``n_epochs``).

    The simulation generates background activity using a multivariate autoregressive (MVAR) process
    and adds post-stimulus activity within a defined time window. Post-stimulus activity is generated
    from a low-rank latent MVAR process and projected onto selected cortical sources.

    Optionally, an ERP waveform (P1-N1-P2) can be added to the active sources with amplitude and latency jitter.

    Parameters
    ----------
    n_epochs : int
        Number of epochs to simulate.
    lf_subset_indices : array-like
        Leadfield indices corresponding to the simulated sources.
    n_times : int
        Number of time samples per epoch.
    poststimuli_mask : ndarray (bool)
        Boolean mask indicating the time window where stimulus-locked activity occurs.
    bg_lf_subset_indices : ndarray
        Leadfield indices of sources that should contain background activity.
    poststimuli_lf_subset_indices : array-like
        Leadfield indices of sources that should contain post-stimulus activity.
    lf_to_label : dict
        Mapping from leadfield index to anatomical label name.
    sfreq : float
        Sampling frequency in Hz.
    erp_factor : float | None
        Scaling factor for ERP waveform amplitude. If None, no ERP is added.
    snr_db : float
        Desired signal-to-noise ratio (in dB) between post-stimulus activity and background activity.
    order_bg : int
        MVAR order for background activity.
    order : int
        MVAR order for post-stimulus activity.
    target_std_bg : float
        Target standard deviation of background source activity (Am).
    target_std : float
        Target standard deviation of post-stimulus activity (Am).
    coupling_bg : float
        Cross-source coupling strength for background MVAR model.
    coupling : float
        Cross-source coupling strength for stimulus-driven MVAR model.
    noise_bg : float
        Innovation noise amplitude for background activity.
    noise : float
        Innovation noise amplitude for stimulus-driven activity.
    gaussian_filter_sigma_bg : float
        Temporal smoothing (Gaussian sigma) for background activity.
    gaussian_filter_sigma : float
        Temporal smoothing for stimulus activity.
    n_dominant_eigvals :int
        Dimensionality of latent sources used to generated correlated stimulus-locked activity.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    X_epochs: ndarray, shape (n_epochs, n_sources, n_times)
        Simulated source-space time series for all epochs.

    """
    rng = np.random.default_rng(seed)
    n_sources = len(lf_subset_indices)
    X_epochs = np.zeros((n_epochs, n_sources, n_times))

    U = rng.normal(size=(len(poststimuli_lf_subset_indices), n_dominant_eigvals))
    U /= np.linalg.norm(U, axis=0, keepdims=True)

    for epoch in range(n_epochs):
        seed_bg = rng.integers(1e6)
        seed_act = rng.integers(1e6)

        # Background activity
        X_bg = np.zeros((n_sources, n_times))
        X_bg_subset = simulate_mvar_for_dipoles(
            n_sources=len(bg_lf_subset_indices),
            n_samples=n_times,
            order=order_bg,
            target_std=target_std_bg,
            coupling_strength=coupling_bg,
            noise_level=noise_bg,
            gaussian_filter_sigma=gaussian_filter_sigma_bg,
            seed=seed_bg
        )

        for i, idx in enumerate(bg_lf_subset_indices):
            src_idx = lf_subset_indices.tolist().index(idx)
            X_bg[src_idx] = X_bg_subset[i]

        # Poststimuli activity
        X_post = np.zeros((n_sources, n_times))

        # Modify target std and cross-source coupling of main activity to make it more diverse across epochs
        target_std_epoch = target_std * rng.uniform(0.9, 1.1)
        coupling_epoch = coupling * rng.uniform(0.9, 1.1)

        # Simulate only for n_dominant_eigvals sources
        latent = simulate_mvar_for_dipoles(
            n_sources=n_dominant_eigvals,
            n_samples=poststimuli_mask.sum(),
            order=order,
            target_std=target_std_epoch,
            coupling_strength=coupling_epoch,
            gaussian_filter_sigma=gaussian_filter_sigma,
            noise_level=noise,
            seed=seed_act
        )
        X_post_subset = U @ latent

        for i, idx in enumerate(poststimuli_lf_subset_indices):
            X_post[lf_subset_indices.tolist().index(idx), poststimuli_mask] = X_post_subset[i]

        if erp_factor:
            t = np.arange(poststimuli_mask.sum()) / sfreq

            latency_jitter_global = rng.normal(0, 0.005)
            erp_base = erp_waveform(t - latency_jitter_global)

            for i in range(X_post_subset.shape[0]):
                amp_jitter = rng.normal(1.0, 0.05)

                # Adding visual boost depending on label
                label = lf_to_label[poststimuli_lf_subset_indices[i]].split('-')[0]
                if label in ["cuneus", "lateraloccipital", "inferiorparietal"]:
                    visual_boost = 1.6
                elif label == "superiorparietal":
                    visual_boost = 1.3
                else:
                    visual_boost = 0.8

                src_idx = lf_subset_indices.tolist().index(
                    poststimuli_lf_subset_indices[i]
                )
                X_post[src_idx, poststimuli_mask] += (
                        visual_boost
                        * erp_factor
                        * amp_jitter
                        * erp_base
                )

        # SNR control
        bg_pow = np.mean(X_bg[:, poststimuli_mask] ** 2)
        act_pow = np.mean(X_post[:, poststimuli_mask] ** 2)
        X_post *= np.sqrt(bg_pow * 10 ** (snr_db / 10) / (act_pow + 1e-15))
        X_epochs[epoch] = X_bg + X_post

    return X_epochs


def simulate_mvar_for_dipoles(
        n_sources: int,
        n_samples: int,
        order: int = 6,
        target_std: float = 10e-9,
        coupling_strength: float = 0.2,
        noise_level: float = 1.0,
        gaussian_filter_sigma: float = 2.0,
        seed=None
) -> np.ndarray:
    """
    Simulate multivariate autoregresive (MVAR) signals for cortical dipoles.

    A MVAR process is generated with random coeefficients and optional cross-source coupling.
    The resulting time series are smoothed and scaled to a target physical amplitude.

    Parameters
    ----------
    n_sources : int
        Number of simulated sources.
    n_samples : int
        Number of time samples.
    order : int
        MVAR order
    target_std : float
        Target standard deviation of simulated signals.
    coupling_strength : float
        Scaling factor for cross-source coupling coefficients.
    noise_level : float
        Innovation noise amplitude.
    gaussian_filter_sigma : float
        Temporal smoothing applied to each source.
    seed : int | None
        Random seed.

    Returns
    -------
    X: ndarray, shape (n_sources, n_samples)
        Simulated source activity.
    """
    rng = np.random.default_rng(seed)

    # Generate random MVAR coefficients
    A = rng.normal(scale=0.03, size=(order, n_sources, n_sources))

    # Add coupling structure
    for k in range(order):
        for i in range(n_sources):
            for j in range(n_sources):
                if i != j:
                    A[k, i, j] *= coupling_strength

    # Stabilize system (scale eigenvalues)
    companion = np.zeros((order * n_sources, order * n_sources))
    companion[:n_sources, :] = np.hstack(A)
    companion[n_sources:, :-n_sources] = np.eye((order - 1) * n_sources)

    # Stability margin
    max_ev = np.max(np.abs(np.linalg.eigvals(companion)))
    target_radius = 0.9
    if max_ev >= target_radius:
        A *= target_radius / (max_ev + 1e-12)

    # Generate innovations
    noise = rng.normal(scale=noise_level, size=(n_sources, n_samples))

    # Simulate MVAR process
    X = np.zeros((n_sources, n_samples))
    for t in range(order, n_samples):
        for k in range(1, order + 1):
            X[:, t] += A[k - 1] @ X[:, t - k]
        X[:, t] += noise[:, t]

    X = gaussian_filter1d(X, sigma=gaussian_filter_sigma, axis=1)

    # Scale to target physical amplitude
    current_std = np.std(X, axis=1, keepdims=True)
    current_std[current_std == 0] = 1.0
    X = X / current_std * target_std

    return X


def simulate_sensor_epochs(
        X_epochs: np.ndarray,
        leadfield: np.ndarray,
        lf_subset_indices: np.ndarray,
        src: mne.SourceSpaces,
        tmin: float,
        sfreq: float,
        noise_factor: float = 0.1,
        info: mne.Info =None,
        seed=None
) -> mne.EpochsArray:
    """
    Project simulated source activity to EEG sensors.

    The function inserts simulated sources into the full cortical source space,
    applied the forward model, and adds sensor noise and slow drift to produce EEG epochs.

    Parameters
    ----------
    X_epochs : ndarray
        Source-space activity with shape (n_epochs, n_sources, n_times)
    leadfield : ndarray
        Forward model matrix mapping sources to sensors.
    lf_subset_indices : ndarray
        Leadfield indices corresponding to the simulated sources.
    src : mne.SourceSpaces
        MNE source space used for the forward model.
    tmin : float
        Epoch start time in seconds.
    sfreq : float
        Sampling frequency.
    noise_factor : float
        Scaling factor controlling sensor noise amplitude.
    info : mne.Info
        MNE measurement info structure for creating EpochsArray.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    sim_epochs: mne.EpochsArray
        Simulated EEG sensor data.
    """
    rng = np.random.default_rng(seed)
    n_epochs, n_sources, n_times = X_epochs.shape
    # Number of all vertices for current subject
    n_vertices_total = sum(len(s['vertno']) for s in src)

    # Placeholder for sensors activity
    Y_epochs = np.zeros((n_epochs, leadfield.shape[0], n_times))

    for epoch in range(n_epochs):
        # Map sources to full vertices from source space
        X_full = np.zeros((n_vertices_total, n_times))
        X_full[lf_subset_indices, :] = X_epochs[epoch]

        # Project to sensors
        Y_clean = leadfield @ X_full

        drift = gaussian_filter1d(
            rng.normal(size=Y_clean.shape),
            sigma=sfreq,
            axis=1
        )
        drift *= 0.2 * np.std(Y_clean)
        Y_clean = Y_clean + drift

        # Add Gaussian noise
        noise = np.random.randn(*Y_clean.shape)
        noise *= np.std(Y_clean) * noise_factor

        Y_epochs[epoch] = Y_clean + noise

    sim_epochs = mne.EpochsArray(
        data=Y_epochs,
        info=info.copy() if info is not None else None,
        tmin=tmin,
        verbose=False
    )
    return sim_epochs


def add_simulated_epochs_to_stc(
        X_epochs: np.ndarray,
        src: mne.SourceSpaces,
        n_times: int,
        lf_subset_indices,
        tmin: float,
        sfreq: float,
        subject: str
) -> mne.SourceEstimate:
    """
    Convert averaged simulated source activity to an MNE Source Estimate.

    Parameters
    ----------
    X_epochs : ndarray
        Simulated source activity with shape (n_epochs, n_sources, n_times)
    src : mne.SourceSpaces
        Source space used in the forward model.
    n_times : int
        Number of time points.
    lf_subset_indices : ndarray
        Leadfield indices corresponding to simulated sources.
    tmin : float
        Start time of the data.
    sfreq : float
        Sampling frequency
    subject : str
        Subject name used by MNE.

    Returns
    -------
    stc_evoked: mne.SourceEstimate
        Source estimate containing the average simulated activity.

    """
    X_avg = X_epochs.mean(axis=0)
    X_full_avg = np.zeros((sum(len(s['vertno']) for s in src), n_times))
    X_full_avg[lf_subset_indices, :] = X_avg

    stc_evoked = mne.SourceEstimate(
        data=X_full_avg,
        vertices=[src[0]['vertno'], src[1]['vertno']],
        tmin=tmin,
        tstep=1 / sfreq,
        subject=subject
    )

    return stc_evoked


def erp_waveform(t: np.ndarray) -> np.ndarray:
    """
    Generate a synthetic ERP waveform (P1-N1-P2 complex).

    Parameters
    ----------
    t : ndarray
        Time vector in seconds.

    Returns
    -------
    waveform : ndarray
        ERP waveform evaluated at times `t`.

    """
    p1 = 0.15 * np.exp(-0.5 * ((t - 0.085 + 0.05) / 0.025) ** 2)
    n1 = -1.0 * np.exp(-0.5 * ((t - 0.12 + 0.05) / 0.03) ** 2)
    p2 = 0.3 * np.exp(-0.5 * ((t - 0.22 + 0.05) / 0.06) ** 2)
    return p1 + n1 + p2


def split_vertices(
        labels_info: Dict,
        poststimuli_labels: List
) -> Tuple[List, List]:
    """
    Separate vertices by hemisphere and by stimulus activity role.

    Parameters
    ----------
    labels_info : dict
        Dictionary describing labels and selected vertices.
    poststimuli_labels : list
        Labels designated as stimulus-active regions.

    Returns
    -------
    noise_vertices with noise activity split into left/right hemisphere.
    post_vertices: list
        Vertices belonging only to stimulus-active labels.
    """
    # Placeholder for vertices
    lh_vertices_noise = []
    rh_vertices_noise = []
    lh_vertices_post = []
    rh_vertices_post = []

    for label, info in labels_info.items():
        verts = info['random_vertice']
        hemi = label.split('-')[1]
        for vert in verts:
            if hemi == 'lh':
                if info['poststimuli']:
                    lh_vertices_post.append(vert)
                if info['noise']:
                    lh_vertices_noise.append(vert)
            else:
                if info['poststimuli']:
                    rh_vertices_post.append(vert)
                if info['noise']:
                    rh_vertices_noise.append(vert)

    noise_vertices = [sorted(np.array(lh_vertices_noise)), sorted(np.array(rh_vertices_noise))]
    post_vertices = [sorted(np.array(lh_vertices_post)), sorted(np.array(rh_vertices_post))]

    return noise_vertices, post_vertices


def get_random_vertices(
        n_vertices_per_label_bg: int,
        n_vertices_per_poststimuli_label: int,
        noise_labels: List,
        poststimuli_labels: List,
        subject: str,
        subjects_dir: str,
        src: mne.SourceSpaces,
        find_close: bool = False,
        max_distance_cm: float = 2.0,
        seed=None
) -> Dict:
    """
    Randomly select cortical vertices from anatomical labels.

    Vertices are drawn from FreeSurfer annotation labels and
    filtered to include only those present in the current source space.

    Parameters
    ----------
    n_vertices_per_label_bg : int
        Number of vertices selected per background label.
    n_vertices_per_poststimuli_label : int
        Number of vertices selected per stimulus-active label.
    noise_labels : List
        Labels used for activity.
    poststimuli_labels : List
        Labels generating stimulus-locked activity.
    subject : str
        Subject name.
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR.
    src : mne.SourceSpaces
        Source space used for simulation.
    find_close : bool
        If True and `n_vertices_per_poststimuli_label` is at least 2, selected sources are within `max_distance_cm`.
    max_distance_cm : float
        Maximum allowed Euclidean distance in centimeters between the first randomly selected vertex
        and all remaining vertices.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    labels_info : dict
        Dictionary containing selected vertices and metadata.
    """
    rng = np.random.default_rng(seed)

    labels_info = {}
    # Read current label
    for label in list(set(noise_labels) | set(poststimuli_labels)):
        labels_info[label] = {}
        # Read current label
        temp_label = mne.read_labels_from_annot(
            subject, regexp=label, subjects_dir=subjects_dir
        )[0]

        # Determine hemisphere
        hemi = temp_label.name.split("-")[1]
        if hemi == "lh":
            hemi_idx = 0
        elif hemi == "rh":
            hemi_idx = 1
        else:
            raise ValueError("Handling only left ('lh') and right ('rh') hemispheres.")
        subject_vertices = [
            v for v in temp_label.vertices if v in src[hemi_idx]['vertno']
        ]

        if label in noise_labels and label in poststimuli_labels:
            n_random_vertices = n_vertices_per_poststimuli_label
            labels_info[label]['noise'] = True
            labels_info[label]['poststimuli'] = True
        elif label in poststimuli_labels and label not in noise_labels:
            n_random_vertices = n_vertices_per_poststimuli_label
            labels_info[label]['noise'] = False
            labels_info[label]['poststimuli'] = True
        elif label in noise_labels and label not in poststimuli_labels:
            n_random_vertices = n_vertices_per_label_bg
            labels_info[label]['noise'] = True
            labels_info[label]['poststimuli'] = False
        else:
            raise ValueError(f"Unknown label: {label}")

        # Randomly select vertices from current label
        if (
                find_close
                and label in poststimuli_labels
                and n_random_vertices >= 2
        ):
            selected_vertices = _get_random_close_vertices(
                n_vertices=n_random_vertices,
                src=src,
                subject_vertices=subject_vertices,
                hemi_idx=hemi_idx,
                max_distance_cm=max_distance_cm,
                rng=rng
            )
        else:
            selected_vertices= rng.choice(
                subject_vertices, size=n_random_vertices, replace=False
            ).tolist()

        labels_info[label]["random_vertice"] = selected_vertices

    return labels_info


def add_leadfield_indices_info(
        labels_info: Dict,
        lh_vert_to_lf: Dict,
        rh_vert_to_lf: Dict
) -> Dict:
    """
    Add leadfield indices corresponding to selected vertices for each label.

    Parameters
    ----------
    labels_info : dict
        Dictionary describing selected vertices for each label.
    lh_vert_to_lf : dict
        Mapping from left-hemisphere vertex to leadfield index.
    rh_vert_to_lf : dict
        Mapping from right-hemisphere vertex to leadfield index

    Returns
    -------
    labels_info : dict
        Updated dictionary where each label entry additionally contains
        a ``'leadfield_indices'`` field listing the leadfield indices
        corresponding to the selected vertices.
    """
    for label in labels_info:
        if label.endswith("-lh"):
            labels_info[label]['leadfield_indices'] = [lh_vert_to_lf[v] for v in labels_info[label]['random_vertice']]
        elif label.endswith("-rh"):
            labels_info[label]['leadfield_indices'] = [rh_vert_to_lf[v] for v in labels_info[label]['random_vertice']]

    return labels_info


def assign_label_to_leadfield_index(
        labels_info: Dict,
        lh_vert_to_lf: Dict,
        rh_vert_to_lf: Dict
) -> Dict:
    """
    Create a mapping from leadfield indices to anatomical labels.

    Parameters
    ----------
    labels_info : dict
        Dictionary describing selected vertices for each label.
    lh_vert_to_lf : dict
        Mapping from left-hemisphere vertex to leadfield index.
    rh_vert_to_lf : dict
        Mapping from right-hemisphere vertex to leadfield index.

    Returns
    -------
    lf_to_label : dict
        Mapping from leadfield index to label name.
    """
    lf_to_label = {}
    for label in labels_info:
        if label.endswith("-lh"):
            vert_to_lf = lh_vert_to_lf
        elif label.endswith("-rh"):
            vert_to_lf = rh_vert_to_lf

        verts = labels_info[label]['random_vertice']
        lfs = [vert_to_lf[v] for v in verts]
        for lf_idx in lfs:
            lf_to_label[lf_idx] = label

    return lf_to_label


def _get_random_close_vertices(
        n_vertices: int,
        src: mne.SourceSpaces,
        subject_vertices: list,
        hemi_idx: int,
        max_distance_cm: float,
        rng: np.random.Generator
) -> list:
    """
    Randomly select source-space vertices located within a local neighbourhood.

    The first vertex is randomly selected from the provided `subject_vertices`.
    The remaining `n_vertices - 1` vertices are then randomly chosen from vertices which
    Euclidean distance from the first vertex does not exceed `max_distance_cm`.

    Parameters
    ----------
    n_vertices : int
        Total number of vertices to return.
    src : mne.SourceSpaces
        Source space containing vertex coordinates and indices.
    subject_vertices : list
        List of valid vertex indices available for selection within the current anatomical label and hemisphere.
    hemi_idx : int
        Hemisphere index in the source space:
        `0` for left hemisphere, `1` for right hemisphere
    max_distance_cm : float
        Maximum allowed Euclidean distance in centimeters between the first randomly selected vertex
        and all remaining vertices.
    rng : np.random.Generator
        NumPy random generator used for reproducible sampling.

    Returns
    -------
    list
        Sorted list of selected vertex indices of length `n_vertices`.

    Raises
    -------
    ValueError
        If no vertices are found within the specified distance threshold.
    ValueError
        If fewer than `n_vertices - 1` nearby vertices are available.
    """
    rr = src[hemi_idx]['rr']
    vertno = src[hemi_idx]['vertno']

    vertex_to_idx = {v: i for i, v in enumerate(vertno)}

    # Select first vertice randomly
    first_vertex = rng.choice(subject_vertices)
    first_coord = rr[vertex_to_idx[first_vertex]]

    # Find vertices within distance threshold
    close_candidates = []
    for v in subject_vertices:
        if v == first_vertex:
            continue
        dist = euclidean(rr[vertex_to_idx[v]], first_coord)
        if dist <= max_distance_cm / 100.0:
            close_candidates.append(v)

    if len(close_candidates) == 0:
        raise ValueError(
            f"No vertices within {max_distance_cm} cm found in label'"
        )

    # Randomly choose all other vertices
    if (n_vertices - 1) <= len(close_candidates):
        remain_vertices = rng.choice(close_candidates, size=n_vertices - 1, replace=False)
    else:
        raise ValueError(
            f"Too many vertices requested ({n_vertices - 1}."
            f" Found only {len(close_candidates)} within {max_distance_cm} cm for label."
        )

    return sorted([int(first_vertex)] + [int(v) for v in remain_vertices])
