"""Functions for checking parameters of MVPURE-PY package functions"""
# Author: Julia Jurkowska

import os
import mne


def _check_hemi_param(hemi: str):
    """
    Check if parameter ``hemi`` is one of the following: 'both`, 'rh', 'lh'.
    """
    if hemi not in ['both', 'rh', 'lh']:
        raise ValueError("Parameter 'hemi' should be one of the following options: 'both', 'rh', 'lh'")


def _check_hemi_and_vertices_matching(hemi: str, vertices: list[int] | list[list[int]]):
    """
    Check if ``hemi`` and ``vertices`` match basen on ``vertices`` length.-
    """
    if hemi in ['rh', 'lh'] and len(vertices) != 1:
        raise ValueError("'hemi' defined as one brain hemisphere but 'vertices' implicates that both of hemispheres "
                         "are used")
    if hemi == 'both' and len(vertices) != 2:
        raise ValueError("'hemi' defined as two brain hemispheres but 'vertices' implicates only one is being used.")


def _check_parc_subject_params(subject: str, subjects_dir: str, parc: str):
    """
    Check if subject related parameters are correctly specified.
    """
    # First - try FreeSurfer parcellation for given subject
    avail_labels = mne.label._read_annot_cands(dir_name=os.path.join(subjects_dir, subject, "label"))
    if parc in avail_labels:
        return subject
    else:
        print(f"Parcellation {parc} can't be found for given subject. Trying fsaverage instead...")
        if os.path.exists(os.path.join(subjects_dir, "fsaverage")):
            subject = "fsaverage"
            avail_labels = mne.label._read_annot_cands(dir_name=os.path.join(subjects_dir, subject, "label"))
            if parc in avail_labels:
                print(f"Parcellation will be morphed to fsaverage.")
                return subject
            else:
                raise ValueError("Given 'parc' parameter is not valid for given subject and can not be morphed from "
                                 "fsaverage.")
        else:
            raise ValueError(f"Desired parcellation can not be found neither in subject folder. "
                             f"An attempt was made to morph labels from 'fsaverage' but 'fsaverage' can not be found in "
                             f"given subjects_dir {subjects_dir}.")
