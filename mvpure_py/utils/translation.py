"""Operations on ``mne.Forward``, ``mne.SourceSpaces`` and mapping between vertices and leadfield indices."""
# Author: Julia Jurkowska

import mne
import os
import numpy as np

from scipy.spatial import cKDTree

import mvpure_py
from ._utils import (
    _check_hemi_param,
    _check_hemi_and_vertices_matching
)


def transform_leadfield_indices_to_vertices(lf_idx, src: mne.SourceSpaces, hemi: str = "both",
                                            include_mapping: bool = True):
    """
    Transform leadfield 1-axis indices into vertices number using ``mne.SourceSpaces``

    Parameters
    -----------
    lf_idx: iterable
        iterable with leadfield indices
    src: mne.SourceSpace
        source space from forward with corresponding leadfield
    hemi: str (Default to "both")
        vertices from which hemisphere should be returned.
        Options are: 'lh', 'rh' and 'both'.

    Returns
    -----------
    vertices: list | tuple(list, list)
        markings of vertices which correspond to leadfield matrices
    """
    # get left and right indices of leadfield
    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']
    lh_len = len(lh_vertno)

    lf_idx = np.array(lf_idx)

    # boolean mask to separate left and right hemisphere dipoles
    is_lh = lf_idx < lh_len
    is_rh = ~is_lh

    # map indices to vertex numbers
    lh_to_use = lf_idx[is_lh]
    lh_mapped = lh_vertno[lh_to_use].astype(int).tolist()
    rh_to_use = lf_idx[is_rh]
    rh_mapped = rh_vertno[rh_to_use - lh_len].astype(int).tolist()

    # return left vertno if left hemisphere chosen
    if hemi == "lh":
        if include_mapping:
            return lh_mapped, lh_to_use
        else:
            return lh_mapped
    # return right vertno if right hemisphere chosen
    elif hemi == "rh":
        if include_mapping:
            return rh_mapped, rh_to_use
        else:
            return rh_mapped
    # return tuple with left and right vertno if both hemispheres chosen
    elif hemi == "both":
        if include_mapping:
            return lh_mapped, lh_to_use, rh_mapped, rh_to_use
        else:
            return lh_mapped, rh_mapped
    else:
        raise ValueError(f"Possible options for 'hemi' parameter are: 'lh', 'rh', 'both'. "
                         f"Got {hemi} instead.")


def transform_vertices_to_leadfield_indices(vertices: list, src: mne.SourceSpaces) -> list:
    """
    Transform vertices numbers (from ``mne.SourceSpace``) to leadfield 1-axis indices.

    Parameters
    -----------
    vertices: list
        Markings of vertices which correspond to leadfield indices.
    src: mne.SourceSpace
        Source space from forward with corresponding leadfield.

    Returns
    -----------
    lf_idx: list
        iterable with leadfield indices
    """
    # get all vertices markings
    all_vertno = []
    for i in range(len(src)):
        all_vertno += list(src[i]['vertno'])
    # transform from vertices to leadfield indices
    lf_idx = list(np.where(np.isin(np.array(all_vertno), vertices)))

    return lf_idx


def subset_forward(old_fwd: mne.Forward,
                   localized,
                   vertices: list[int] = None,
                   hemi: str = "both") -> mne.Forward:
    """
    Subset ``old_fwd`` (mne.Forward) for it to contain information only for certain vertices' numbers.

    Parameters
    -----------
    old_fwd: mne.Forward
        ``mne.Forward`` to get subset from
    hemi: str
        hemisphere(s) containing the dipoles that are expected to be present in ``mne.Forward`` subset

    Returns
    -----------
    mne.Forward: subset of mne.Forward for given vertices

    """
    if localized is not None:
        new_fwd = _subset_forward_by_localized(old_fwd, localized, hemi)
    elif vertices is not None:
        new_fwd = _subset_forward_by_vertices(old_fwd, vertices, hemi)
    return new_fwd


def _subset_forward_by_localized(old_fwd: mne.Forward, localized, #: mvpure_py.Localized,
                                 hemi: str = "both") -> mne.Forward:
    """
    Subset mne.Forward for it to contain information only for certain dipoles
    that were previously localized via mvpure.localizer localizers

    Parameters
    -----------
    old_fwd: mne.Forward
        mne.Forward to et subset from
    localized: mvpure_py.Localized
        instance with information about localized sources
    hemi: str
        hemisphere(s) containing the dipoles that are expected to be present in mne.Forward subset

    Returns
    -----------
    mne.Forward: subset of mne.Forward for given vertices

    """
    # create a copy of old_fwd as a draft for new one
    new_fwd = old_fwd.copy()
    new_fwd['nsource'] = localized['nsource']
    new_fwd['sol']['ncol'] = localized['nsource']
    new_fwd['sol']['data'] = localized['leadfield']
    new_fwd['source_rr'] = old_fwd['source_rr'][list(localized['sources']), :]
    new_fwd['source_nn'] = old_fwd['source_nn'][list(localized['sources']), :]

    # transform source ordinal number (leadfield index) to vertice number
    trans_sources = transform_leadfield_indices_to_vertices(localized['sources'],
                                                            old_fwd['src'],
                                                            hemi=hemi,
                                                            include_mapping=False)
    new_fwd['src'] = _subset_src(old_fwd['src'], trans_sources, hemi)

    return new_fwd


def _subset_forward_by_vertices(old_fwd: mne.Forward, vertices: list[int], hemi: str = "both") -> mne.Forward:
    """
    Subset mne.Forward for it to contain information only for certain dipoles
    that were previously localized via mvpure.localizer localizators

    Parameters:
    -----------
    old_fwd: mne.Forward
        mne.Forward to et subset from
    vertices: list[int]
        list of vertices indices to include in mne.Forward subset
    hemi: str
        hemisphere(s) containing the dipoles that are expected to be present in mne.Forward subset

    Returns:
    -----------
    mne.Forward: subset of mne.Forward for given vertices

    """
    # create a copy of old_fwd as a draft for new one
    new_fwd = old_fwd.copy()
    new_fwd['nsource'] = len(vertices)
    new_fwd['sol']['ncol'] = len(vertices)
    lf_idx = transform_vertices_to_leadfield_indices(vertices, old_fwd['src'])
    new_fwd['sol']['data'] = old_fwd['leadfield'][:, lf_idx]
    new_fwd['source_rr'] = old_fwd['source_rr'][lf_idx, :]
    new_fwd['source_nn'] = old_fwd['source_nn'][lf_idx, :]
    new_fwd['src'] = _subset_src(old_fwd['src'], vertices, hemi)

    return new_fwd


def _subset_src(old_src: mne.SourceSpaces,
                vertices: list[int] | list[list[int]],
                hemi: str) -> mne.SourceSpaces:
    """
    Subset mne.SourceSpaces for it to contain information only for certain vertices' numbers.
    Parameters:
    -----------
    old_src: mne.SourceSpaces
        source space to get subset from
    vertices: list[int] | list[list[int]]
        vertices to be included in subset of mne.SourceSpaces
        If hemi == 'both' ```vertices``` should be list containing two lists with integers.
        If hemi == 'lh' or hemi == 'rh' ```vertices``` should be a list of integers.
    hemi: str (Options: 'both', 'rh', 'lh')
        hemispheres from which vertices are.

    Returns:
    -----------
    mne.SourceSpaces: subset of mne.SourceSpaces for given vertices
    """
    # check correctness of 'hemi' parameter an its consistence to 'vertices' parameter
    _check_hemi_param(hemi)
    _check_hemi_and_vertices_matching(hemi, vertices)
    # create a copy of old_src as a draft for new one
    new_src = old_src.copy()
    # iterate through hemispheres
    for i in range(len(vertices)):
        new_src[i]['nuse'] = len(vertices[i]) if len(vertices) == 1 else len(vertices)
        new_src[i]['inuse'] = np.where(np.isin(np.arange(len(old_src[i]['inuse'])),
                                               vertices[i]), 1, 0)
        new_src[i]['vertno'] = np.sort(vertices[i])

    return new_src


def map_vertices_to_fsaverage(locs,
                              hemi: list[str] | str,
                              surf: str = "white"):
    """
    Map vertices from subject-specific surface to fsaverage coordinates for one or both hemispheres.
    This function wraps `_map_vertices_to_fsaverage_for_hemi` and allows mapping vertices
    for either a single hemisphere ('lh' or 'rh') or both hemispheres simultaneously.

    Parameters
    ----------
    locs : mvpure_py.Localized
        ``mvpure_py.Localized`` object to map vertices from.
        After mapping, each vertex in the specified hemisphere(s) will have a new key
        ``"morphed_to_fs"`` storing the corresponding fsaverage vertex index.
    hemi : list[str] | str
        Which hemisphere to map.
    surf : str (Default: 'white')
        Surface which coordinates use during mapping.

    Raises
    -------
    ValueError
        If ``hemi`` is not one of ['lh', 'rh', 'both'].
    """
    _check_hemi_param(hemi)
    if hemi == "both":
        hemi = ["lh", "rh"]
    elif hemi in ["lh", "rh"]:
        hemi = [hemi]
    else:
        raise ValueError("Only possible options for 'hemi' parameter are: ['lh', 'rh', 'both'].")

    for h in hemi:
        _map_vertices_to_fsaverage_for_hemi(locs=locs, hemi=h, surf=surf)


def _map_vertices_to_fsaverage_for_hemi(locs, hemi: str, surf: str):
    """
    Map vertices from a subject-specific brain surface to the fsaverage surface for a given hemisphere.

    This function:
    - takes vertices from mvpure_py.Localized object for a specific subject and hemisphere,
    - converts their coordinates to MNI space,
    - finds the closest corresponding vertices on the fsaverage surface.

    Parameters
    ----------
    locs : mvpure_py.Localized
        mvpure_py.Localized object containing vertex information and subject metadata.
    hemi : str
        Hemisphere to process. Needs to be 'lh' or 'rh'.
    surf : str
        Surface type to use for fsaverage mapping.

    Notes
    ----------
    - The mapping is hemisphere-specific and assumes fsaverage has a standard FreeSurfer directory structure.

    """
    # Read vertices from mvpure_py.Localized object (only for given hemi)
    vert = [vert for vert in list(locs['vertices'].keys()) if locs['vertices'][vert]['hemi'] == hemi]
    # If there are no vertices for given hemisphere - return None
    if len(vert) == 0:
        return None
    hemis_int = 0 if hemi == 'lh' else 1

    # Load the target fsaverage surface
    surf_path = os.path.join(locs['subjects_dir'], 'fsaverage', 'surf', f'{hemi}.{surf}')
    fs_vertices, _ = mne.read_surface(surf_path)
    fs_vertex_indices = np.arange(fs_vertices.shape[0])  # from 0 to (n_vertices - 1)

    # Convert vertices from subject_from into MNI coordinates
    mni_coords = mne.vertex_to_mni(vert,
                                   hemis=hemis_int,
                                   subject=locs['subject'],
                                   subjects_dir=locs['subjects_dir'])
    # Convert vertices from fsaverage into MNI coordinates
    fs_mni_coords = mne.vertex_to_mni(fs_vertex_indices,
                                      hemis=hemis_int,
                                      subject="fsaverage",
                                      subjects_dir=locs['subjects_dir'])
    # Using KDTree to find the closest fsaverage vertex for each input MNI coordinates
    tree = cKDTree(fs_mni_coords)
    _, idx = tree.query(mni_coords)
    mapped_vertices = fs_vertex_indices[idx]

    # Add to instance
    for i, v in enumerate(vert):
        locs['vertices'][v]["morphed_to_fs"] = int(mapped_vertices[i])

