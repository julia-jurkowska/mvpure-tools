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
        Iterable with leadfield (column) indices
    src: mne.SourceSpace
        source space from forward with corresponding leadfield
    hemi: str (Default to "both")
        vertices from which hemisphere should be returned.
        Options are: 'lh', 'rh' and 'both'.
    include_mapping : bool
        If True, return also the indices in the original leadfield.
        (Useful to know which columns were used).

    Returns
    -----------
    vertices: list | tuple(list, list)
        markings of vertices which correspond to leadfield matrices
    """
    _check_hemi_param(hemi)
    # get left and right indices of leadfield
    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']
    lh_len = len(lh_vertno)

    lf_idx = np.array(lf_idx)

    # boolean mask to separate left and right hemisphere dipoles
    is_lh = lf_idx < lh_len
    is_rh = lf_idx >= lh_len

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


def transform_vertices_to_leadfield_indices(
        vertices: list[int] | list[list[int]],
        src: mne.SourceSpaces,
        hemi: str,
        include_mapping: bool = False):
    """
    Transform vertex numbers (from ``mne.SourceSpace``) to leadfield 1-axis indices.

    Parameters
    -----------
    vertices: list[int] | list[list[int]]
        Either a flat list of vertex numbers or [lh_vertices, rh_vertices] when hemi='both'.
    src: mne.SourceSpace
        Source space from forward with corresponding leadfield.
    hemi: {'lh', 'rh', 'both'}
    include_mapping: bool
        If True, also return mapping dicts {vertex -> lf_index} for each hemisphere.

    Returns
    -----------
    lf_idx: np.ndarray
        Leadfield column indices.
    OR (when include_mapping=True)
        (lf_idx_array, lh_vert_to_lf, rh_vert_to_lf)
    """
    _check_hemi_param(hemi)

    lf_idx = []

    # always treat vertices as [lh, rh]
    if isinstance(vertices[0], (list, tuple, np.ndarray)):
        lh_vertices, rh_vertices = vertices
    else:
        if hemi == 'lh':
            lh_vertices, rh_vertices = vertices, []
        elif hemi == "rh":
            lh_vertices, rh_vertices = [], vertices
        else:
            raise ValueError("Parameter hemi can be only equal to one of the following: ['lh', 'rh', 'both']")

    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']
    lh_len = len(lh_vertno)

    lh_map = {v: i for i, v in enumerate(lh_vertno)}
    rh_map = {v: i for i, v in enumerate(rh_vertno)}

    for v in lh_vertices:
        if v in lh_map:
            lf_idx.append(lh_map[v])

    for v in rh_vertices:
        if v in rh_map:
            lf_idx.append(rh_map[v] + lh_len)

    if include_mapping:
        lh_vert_to_lf = {v: lh_map[v] for v in lh_vertices}
        rh_vert_to_lf = {v: rh_map[v] + lh_len for v in rh_vertices}
        return np.array(lf_idx, dtype=int), lh_vert_to_lf, rh_vert_to_lf

    return np.array(lf_idx, dtype=int)


def subset_forward(old_fwd: mne.Forward,
                   localized,
                   vertices: list[list[int]] | list[int] = None,
                   hemi: str = "both") -> mne.Forward:
    """
    Subset ``old_fwd`` (mne.Forward) so that it contains information only for certain vertices.

    Parameters
    -----------
    old_fwd: mne.Forward
        ``mne.Forward`` to get subset from
    localized: mvpure_py.Localized
        mvpure_py.Localized object containing information about localized sources.
    vertices : list[list[int]] | list[int]
        list of vertices indices to include in mne.Forward subset.
    hemi: str
        hemisphere(s) containing the dipoles that are expected to be present in ``mne.Forward`` subset

    Returns
    -----------
    mne.Forward: subset of mne.Forward for given vertices

    """
    if localized is not None and vertices is None:
        new_fwd = _subset_forward_by_localized(old_fwd, localized, hemi)
    elif vertices is not None and localized is None:
        new_fwd = _subset_forward_by_vertices(old_fwd, vertices, hemi)
    else:
        raise ValueError("Exactly one of `localized` or `vertices` must be provided.")
    return new_fwd


def _subset_forward_by_localized(old_fwd: mne.Forward, localized, #: mvpure_py.Localized,
                                 hemi: str = "both") -> mne.Forward:
    """
    Subset mne.Forward for it to contain information only for certain dipoles
    that were previously localized via mvpure_py.localizer localizers

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
    mne.Forward: subset of mne.Forward for localized vertices identified from mvpure_py.Localized

    """
    # Translate leadfield (column) indices of localized sources into vertex numbers, split by hemisphere
    lh_vertices, lh_lf_idx, rh_vertices, rh_lf_idx = transform_leadfield_indices_to_vertices(
        localized['sources'],
        old_fwd['src'],
        hemi="both",
        include_mapping=True
    )

    lh_vertices = np.array(sorted(lh_vertices), dtype=int)
    rh_vertices = np.array(sorted(rh_vertices), dtype=int)

    if hemi == "lh":
        rh_vertices = np.array([], dtype=int)
    elif hemi == "rh":
        lh_vertices = np.array([], dtype=int)
    elif hemi != "both":
        raise ValueError(
            f"Possible options for 'hemi' parameter are: 'lh', 'rh', 'both'. Got {hemi} instead."
        )

    n_sources = len(lh_vertices) + len(rh_vertices)
    if n_sources == 0:
        raise ValueError("No vertices found for the requested hemisphere(s) in `localized`.")

    # Dummy STC for proper vertex selection
    stc = mne.SourceEstimate(
        data=np.zeros((n_sources, 1)),
        vertices=[lh_vertices, rh_vertices],
        tmin=0.0,
        tstep=1.0,
        subject=old_fwd['src'][0].get('subject_his_id', None),
    )

    new_fwd = mne.forward.restrict_forward_to_stc(old_fwd, stc)

    custom_leadfield = np.asarray(localized['leadfield'])
    if custom_leadfield.shape[1] != new_fwd['sol']['data'].shape[1]:
        raise ValueError(
            f"localized['leadfield'] has {custom_leadfield.shape[1]} columns, "
            f"but the restricted forward has {new_fwd['sol']['data'].shape[1]} sources. "
            "Check that 'localized' and 'hemi' refer to the same vertex set."
        )

    new_lh_vertno = new_fwd['src'][0]['vertno']
    new_rh_vertno = new_fwd['src'][1]['vertno']
    new_order_verts = np.concatenate([new_lh_vertno, new_rh_vertno])

    old_order_verts = np.concatenate([lh_vertices, rh_vertices])
    # position of each new-order vertex within the original (localized) ordering
    old_pos = {v: i for i, v in enumerate(old_order_verts)}
    try:
        remap = np.array([old_pos[v] for v in new_order_verts])
    except KeyError as e:
        raise ValueError(
            f"Vertex {e} present after restriction but not found in localized sources; "
            "leadfield cannot be reliably remapped."
        )

    new_fwd['sol']['data'] = custom_leadfield[:, remap]
    return new_fwd


def _subset_forward_by_vertices(old_fwd: mne.Forward, vertices: list[list[int]], hemi: str = "both") -> mne.Forward:
    """
    Subset mne.Forward for it to contain information only for certain dipoles listed as `vertices`.

    Parameters:
    -----------
    old_fwd: mne.Forward
        mne.Forward to et subset from
    vertices: list[list[int]]
        list of vertices indices to include in mne.Forward subset.
    hemi: str
        hemisphere(s) containing the dipoles that are expected to be present in mne.Forward subset

    Returns:
    -----------
    mne.Forward: subset of mne.Forward for given vertices

    """
    # new_fwd = old_fwd.copy()
    src = old_fwd["src"]

    # Vertices should be stored as list in a form [[left vertices], [right vertices]]
    if isinstance(vertices, (list, tuple)) and len(vertices) == 2:
        lh_vertices = np.array(sorted(vertices[0]), dtype=int)
        rh_vertices = np.array(sorted(vertices[1]), dtype=int)
    else:
        flat = np.array(vertices, dtype=int)
        if hemi == "lh":
            lh_vertices = np.array(sorted(flat), dtype=int)
            rh_vertices = np.array([], dtype=int)
        elif hemi == "rh":
            lh_vertices = np.array([], dtype=int)
            rh_vertices = np.array(sorted(flat), dtype=int)
        else:
            raise ValueError(
                "Flat vertex list requires specifying hemi='lh' or 'rh'"
            )

    if lh_vertices.size + rh_vertices.size == 0:
        raise ValueError("No vertices provided to subset the forward with.")

    # Build dummy STC for proper vertex selection
    n_verts = lh_vertices.size + rh_vertices.size
    stc = mne.SourceEstimate(
        data=np.zeros((n_verts, 1)),
        vertices=[lh_vertices, rh_vertices],
        tmin=0.0,
        tstep=1.0,
        subject=src[0].get('subject_his_id', None),
    )

    new_fwd = mne.forward.restrict_forward_to_stc(old_fwd, stc)

    return new_fwd


def vertices_to_coordinates(vertices: list[list[int]],
                            src: mne.SourceSpaces):
    """
    Convert vertex indices to 3D coordinates using mne.SourceSpace.

    Parameters
    ----------
    vertices : list[list[int]]
        List of vertices for which coordinates are being obtained.
        First list needs to contain vertices from the left hemisphere, and second list from the right hemisphere.
    src : mne.SourceSpace
        Source space to use.
    """
    lh_vertices, rh_vertices = vertices
    lh_coords = src[0]['rr'][lh_vertices] if len(lh_vertices) > 0 else np.empty((0, 3))
    rh_coords = src[1]['rr'][rh_vertices] if len(rh_vertices) > 0 else np.empty((0, 3))

    return np.vstack([lh_coords, rh_coords])


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
    vert = [vert[1] for vert in list(locs['vertices'].keys()) if locs['vertices'][vert]['hemi'] == hemi]
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

