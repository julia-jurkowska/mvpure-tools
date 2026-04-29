# IMPORTS
import mne
import os
import numpy as np
from scipy.spatial import cKDTree
from termcolor import colored

from mvpure_py import localizer, utils
from tutorials import preprocessing_utils


def full_analysis(
        subjects: list,
        subjects_dir: str,
        localizer_to_use: list,
        find_peak: bool = True,
        n_sources_to_localize: int = 10,
        r="auto",
        delta: float = 0.8,
        adaptive_r: bool = False,
        task_type: str = "target",
        parc: str = "Yeo2011_17Networks_N1000",
        tmin: float = 0.4,
        tmax: float = 0.6,
        noise_cov_tmin: float = -0.200,
        ptp_value: float = 100,
        detrend_value: int = 0
):
    result = dict()
    for i, subject in enumerate(subjects):
        if isinstance(tmin, list) and isinstance(tmax, list):
            tmin_current = float(f"0.{tmin[i]}")
            tmax_current = float(f"0.{tmax[i]}")
        else:
            tmin_current = tmin
            tmax_current = tmax
        print(colored(
            f"Subject {subject} -- using time in range ({tmin_current}, {tmax_current})",
            "green"
        ))
        try:
            result[subject] = single_subject_analysis(
                subject=subject,
                subjects_dir=subjects_dir,
                localizer_to_use=localizer_to_use,
                find_peak=find_peak,
                n_sources_to_localize=n_sources_to_localize,
                r=r,
                delta=delta,
                adaptive_r=adaptive_r,
                task_type=task_type,
                parc=parc,
                tmin=tmin_current,
                tmax=tmax_current,
                noise_cov_tmin=noise_cov_tmin,
                ptp_value=ptp_value,
                detrend_value=detrend_value
            )
        except Exception as e:
            print(
                colored(
                    text=f"SUBJECT {subject} ERROR: {e}",
                    color="red"
                )
            )

    return result


def single_subject_analysis(
        subject: str,
        subjects_dir: str,
        localizer_to_use: list,
        find_peak: bool,
        n_sources_to_localize: int = 10,
        r="auto",
        delta: float = 0.8,
        adaptive_r: bool = False,
        task_type: str = "target",
        parc: str = "Yeo2011_17Networks_N1000",
        tmin: float = 0.4,
        tmax: float = 0.6,
        noise_cov_tmin: float = -0.200,
        ptp_value: float = None,
        detrend_value: int = 0
):
    raw_path = os.path.join(subjects_dir, subject, "_eeg", "_pre", "_real", f"{subject}_oddball_preprocessed_raw.fif")
    forward_path = os.path.join(subjects_dir, subject, "forward", f"{subject}_ico4-fwd.fif")
    trans_path = os.path.join(subjects_dir, subject, "_eeg", "trans", f"{subject}-fit_trans.fif")

    # Reading data
    print(colored(f"Reading data for subject: {subject}", "magenta"))
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    trans = mne.read_trans(trans_path)
    fwd_vector = mne.read_forward_solution(forward_path)
    # Converting vector to scalar
    # converting to scalar
    fwd = mne.convert_forward_solution(fwd_vector, surf_ori=True, force_fixed=True, use_cps=True)
    # leadfield matrix
    leadfield = fwd["sol"]["data"]
    # source positions extracted from forward model
    src = fwd["src"]

    # Final check for Bad Samples Using TRIMOUTLIER
    channel_sd_lower_bound = -np.inf
    channel_sd_upper_bound = np.inf
    amplitude_threshold = 50 * 1e-6  # *1e-6
    point_spread_width = 1000

    raw = preprocessing_utils.trimoutlier(raw, channel_sd_lower_bound, channel_sd_upper_bound, amplitude_threshold,
                                          point_spread_width)

    # re-reference linked mastoids
    raw = raw.copy().set_eeg_reference(ref_channels=['TP9', 'TP10'])

    all_events, all_event_id = mne.events_from_annotations(raw, event_id={'Stimulus/S  5': 3, 'Stimulus/S  6': 4,
                                                                          'Stimulus/S  7': 5})
    if ptp_value is None:
        epoched = mne.Epochs(raw, all_events, event_id=dict(standard=3, deviant=4, target=5), tmin=-0.2, tmax=0.8,
                             baseline=(None, 0), reject_by_annotation=True, detrend=detrend_value, proj=False,
                             reject=ptp_value, preload=True)
    else:
        reject_criteria = dict(
            eeg=ptp_value,  # V
        )
        epoched = mne.Epochs(raw, all_events, event_id=dict(standard=3, deviant=4, target=5), tmin=-0.2, tmax=0.8,
                             baseline=(None, 0), reject_by_annotation=True, detrend=detrend_value, proj=False,
                             reject=reject_criteria, preload=True)  # , reject=reject_criteria

    epoched.drop_bad()

    if task_type is not None:
        target = epoched[task_type]
    else:
        target = epoched
    target_erp = target.average()

    sel_epoched = target.copy()
    sel_epoched = sel_epoched.set_eeg_reference('average', projection=True)
    sel_epoched.apply_proj()

    if find_peak:
        ch_target_pz, lat_target_pz, amp_target_pz = target_erp.copy().pick('Pz').get_peak(
            tmin=tmin, tmax=tmax, mode="pos", return_amplitude=True, strict=False
        )
        lat_evoked = lat_target_pz.copy()
        print(colored(
            f"Data covariance using time in range ({np.round(lat_evoked - 0.1, 3)}, {np.round(lat_evoked + 0.1, 3)})",
            "green"
        ))
        data_cov = mne.compute_covariance(
            sel_epoched, tmin=lat_evoked - 0.1, tmax=lat_evoked + 0.1, method="empirical")

    else:
        print(colored(
            f"Data covariance using time in range ({tmin}, {tmax})",
            "green"
        ))
        data_cov = mne.compute_covariance(
            sel_epoched, tmin=tmin, tmax=tmax, method="empirical")

    # sel_evoked = sel_epoched.average()

    # Computing noise covariances
    noise_cov = mne.compute_covariance(
        sel_epoched, tmin=noise_cov_tmin, tmax=0, method='empirical')
    # fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, sel_epoched.info)

    # Only fragment that was taken for data covariance
    # reconstruction_evoked = sel_evoked.crop(tmin=lat_evoked - 0.1, tmax=lat_evoked + 0.1)

    # LOCALIZATION
    locs_dict = localizer.mvpure_localizer.localize(subject=subject,
                                                    subjects_dir=subjects_dir,
                                                    localizer_to_use=localizer_to_use,
                                                    n_sources_to_localize=n_sources_to_localize,
                                                    R=data_cov.data,
                                                    N=noise_cov.data,
                                                    forward=fwd,
                                                    r=r,
                                                    delta=delta,
                                                    adaptive_r=adaptive_r)

    if isinstance(locs_dict, localizer.Localized):
        locs_dict = {locs_dict['loc_type']: locs_dict}

    result = dict()
    for loc_type in locs_dict:
        result[loc_type] = single_localizer_analysis(subject=subject, subjects_dir=subjects_dir,
                                                     locs=locs_dict[loc_type], forward=fwd, parc=parc)

    return result


def single_localizer_analysis(
        subject: str,
        subjects_dir: str,
        locs: localizer.Localized,
        forward: mne.Forward,
        parc: str
):
    lh_vertices, lh_ldx, rh_vertices, rh_idx = utils.transform_leadfield_indices_to_vertices(
        lf_idx=locs["sources"],
        src=forward["src"],
        hemi='both',
        include_mapping=True
    )
    locs.add_vertices_info(lh_vertices=lh_vertices, lh_indices=lh_ldx,
                           rh_vertices=rh_vertices, rh_indices=rh_idx)
    locs.assign_brain_regions(parc=parc)
    for hemi in ["lh", "rh"]:
        map_vertices_to_fsaverage(locs=locs, hemi=hemi, subject_from=subject, subjects_dir=subjects_dir)

    return locs


def map_vertices_to_fsaverage(
        locs: localizer.Localized,
        hemi: str,
        subject_from: str,
        subjects_dir: str,
        surf: str = "white"
):
    # vertices
    vert = [vert for vert in list(locs['vertices'].keys()) if locs['vertices'][vert]['hemi'] == hemi]
    if len(vert) == 0:
        return None
    # hemi int
    hemis_int = 0 if hemi == 'lh' else 1

    # Load the target fsaverage surface
    surf_path = os.path.join(subjects_dir, 'fsaverage', 'surf', f'{hemi}.{surf}')
    fs_vertices, _ = mne.read_surface(surf_path)
    # In fsaverage vertices are from 0 to (n_vertices - 1)
    fs_vertex_indices = np.arange(fs_vertices.shape[0])

    # Convert vertices from subject_from into MNI coordinates
    mni_coords = mne.vertex_to_mni(vert, hemis=hemis_int,
                                   subject=subject_from, subjects_dir=subjects_dir)
    # Convert vertices from fsaverage into MNI coordinates
    fs_mni_coords = mne.vertex_to_mni(fs_vertex_indices, hemis=hemis_int,
                                      subject="fsaverage", subjects_dir=subjects_dir)
    # Using KDTree to find the closest fsaverage vertex for each input MNI coordinates
    tree = cKDTree(fs_mni_coords)
    _, idx = tree.query(mni_coords)
    mapped_vertices = fs_vertex_indices[idx]

    # Add to instance
    for i, v in enumerate(vert):
        locs['vertices'][v]["morphed_to_fs"] = int(mapped_vertices[i])
