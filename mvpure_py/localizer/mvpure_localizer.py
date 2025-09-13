""" Localizing sources. """

# Author: Julia Jurkowska

from termcolor import colored
import mne
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    level=logging.WARNING,
    format='[%(levelname)s] %(message)s'
)

from .localizer_utils import (
    get_activity_index
)

from ._utils import (_check_localize_params,
                     _prepare_localize_params,
                     _check_rank,
                     _prepare_localizer_to_use,
                     _define_hemi_based_on_vertices,
                     _check_norm_param,
                     _check_vertices_and_indices,
                     _check_n_sources_param
                     )

from ..utils import (_check_parc_subject_params,
                     split_kwargs,
                     get_pinv_RN_eigenvals)

from ..viz.viz import (_assign_color_mapping,
                       _save_brain_object)


def localize(
        subject: str,
        subjects_dir: str,
        localizer_to_use: str | list,
        n_sources_to_localize: int,
        R: np.ndarray,
        N: np.ndarray,
        forward: mne.Forward = None,
        leadfield: np.ndarray = None,
        r: str | int = 'full'
) -> dict:
    """
    Localize brain activity.

    Parameters
    ----------
    subject : str
        Subject name the analysis is performed for
    subjects_dir : str
        Directory where ``subject`` folder is being stored
    localizer_to_use : str | list
        Type of localizer index to use.
        Options are: 'mai', 'mpz', 'mai_mvp', 'mpz_mvp'.
    n_sources_to_localize : int
        Number of sources to localize
    R : array-like
        Data covariance matrix
    N : array-like
        Noise covariance matrix
    forward : mne.Forward
        The forward operator.
        Can be None if ``leadfield`` has valid value.
    leadfield : array-like
        Leadfield matrix with shape (n_channels, n_sources).
        Can be None if ``forward`` has valid value.
    r : str | int
        Optimization parameter.

        - If int, should be equal or less to ``n_sources_to_localize``.
        - If equal to ``"full"``: ``r`` will be equal to ``n_sources_to_localize``.

    Returns
    -------
    mvpure_py.Localized : Instance of ``mvpure_py.Localized`` containing information of localizer type, localized
    sources and their leadfield.
    """
    # check and prepare parameters
    _check_localize_params(forward, leadfield)
    _check_n_sources_param(n_sources=n_sources_to_localize)
    temp_r = _check_rank(r, n_sources_to_localize)
    if temp_r is not None:
        r = temp_r

    leadfield = _prepare_localize_params(forward, leadfield)

    # dictionary with results
    RES = {}

    if isinstance(r, str) and r == "full":
        opt_rank = n_sources_to_localize
    else:
        opt_rank = r

    # preprocess localizer to use
    localizer_to_use = _prepare_localizer_to_use(localizer_to_use)

    # get eigenvalues of RN^{-1}
    eigvals = get_pinv_RN_eigenvals(R, N)

    # iterate through localizers
    for loc in localizer_to_use:
        print(colored(f"Calculating activity index for localizer: {loc}", "cyan"))
        act_idx, act_val, output_r, lf = get_activity_index(
            localizer_to_use=loc,
            H=leadfield,
            R=R,
            N=N,
            n_sources_to_localize=n_sources_to_localize,
            r=opt_rank
        )
        sources_dict = {}
        for i, source in enumerate(act_idx):
            sources_dict[source] = {
                "activity_index_order": i + 1
            }
        RES[loc] = Localized(
            subject=subject,
            subjects_dir=subjects_dir,
            loc_type=loc,
            sources=act_idx,
            activity_index_values=act_val,
            leadfield=lf,
            activity_index_order=sources_dict,
            filter_rank=int(output_r),
            _eigvals=eigvals
        )

    # if only one localizer used - delete dictionary as output
    if len(RES.keys()) == 1:
        RES = RES[list(RES.keys())[0]]

    return RES


class Localized(dict):
    """
    Class to handle localized sources.
    """

    def __init__(self, subject: str, subjects_dir: str, **kwargs):
        kwargs["subject"] = subject
        kwargs["subjects_dir"] = subjects_dir
        super().__init__(**kwargs)

    def __repr__(self):
        entr = "<Localized"
        loc_type = f"{self['loc_type']}"
        entr += f" | Localizer type: {loc_type}"
        nsource = self["nsource"]
        entr += f" | Number of localized sources: {nsource}"
        entr += ">"

        return entr

    def __getitem__(self,
                    item):
        if item == "nsource" and "sources" in self:
            self[item] = len(self["sources"])
        return super().__getitem__(item)

    def add_stc(self,
                stc: mne.SourceEstimate):
        """
        Add ``mne.SourceEstimate`` to class keys in order to perform power analysis.

        Parameters
        ----------
        stc : mne.SourceEstimate
            source estimate for given subject
        """
        if not isinstance(stc, mne.SourceEstimate):
            raise TypeError(f"Expected 'stc' to be of type mne.SourceEstimate, but got {type(stc)}.")
        self['stc'] = stc

    def add_vertices_info(self,
                          lh_vertices: list = None,
                          lh_indices: list = None,
                          rh_vertices: list = None,
                          rh_indices: list = None):
        """
        Add key ``vertices`` that saves the information about:

        - hemisphere of vertex [``hemi``]
        - original leadfield index [``lf_idx``]
        - ordinal position of the activity index for given vertex [``activity_index_order``]
        - sub-leadfield index for data of given source [``lf_order``]

        It will be in a form:

        .. code-block:: python

            self['vertices'] = dict(
                {vertex_number}: dict(
                    'hemi': 'lh' | 'rh',
                    'lf_idx': {lf_idx},
                    'activity_index_order': {activity_index_order},
                    'lf_order': {activity_index_order - 1}
                )
            )

        .. note::

            After using ``assign_brain_regions()`` function there will be key 'brain_region' with the appropriate
            brain region assigned based on the atlas.

        Parameters
        ----------
        lh_vertices : list | None
            vertex numbers of localized sources in the left hemisphere
        lh_indices : list | None
            original leadfield indies of localized sources in the left hemisphere
        rh_vertices : list | None
            vertex numbers of localized sources in the right hemisphere
        rh_indices : list | None
            original leadfield indies of localized sources in the right hemisphere
        """
        _check_vertices_and_indices(lh_vertices, lh_indices, rh_vertices, rh_indices)
        self["vertices"] = {}
        # left hemisphere
        lf_indx = None
        if lh_vertices is not None:
            for lf_vert, lf_indx in zip(lh_vertices, lh_indices):
                self["vertices"][int(lf_vert)] = {
                    "hemi": "lh",
                    "lf_idx": int(lf_indx),
                    "activity_index_order": self['activity_index_order'][lf_indx]["activity_index_order"],
                    "lf_order": self['activity_index_order'][lf_indx]["activity_index_order"] - 1,
                    "activity_index_values": float(
                        self['activity_index_values'][self['activity_index_order'][lf_indx]["activity_index_order"] - 1]
                    )
                }

        # right hemisphere
        if rh_vertices is not None:
            for rh_vert, rh_indx in zip(rh_vertices, rh_indices):
                self["vertices"][int(rh_vert)] = {
                    "hemi": "rh",
                    "lf_idx": int(rh_indx),
                    "activity_index_order": self['activity_index_order'][rh_indx]["activity_index_order"],
                    "lf_order": self['activity_index_order'][rh_indx]["activity_index_order"] - 1,
                    "activity_index_values": float(
                        self['activity_index_values'][self['activity_index_order'][rh_indx]["activity_index_order"] - 1]
                    )
                }

    def add_power_of_reconstructed(self,
                                   stc: mne.SourceEstimate = None):
        """
        Compute power of reconstructed sources.
        By default, adds key ``power_of_reconstructed`` with power of localized sources.
        If ``stc`` (type: ``mne.SourceEstimate``) has data for specific hemisphere (``lh_data``/``rh_data``),
        computes also power just for given hemisphere.

        Parameters
        ----------
        stc : mne.SourceEstimate
            source estimate for given subject
            Can be None if key `stc` already exists in Localized object, then uses source estimate object saved there.
        """
        if "stc" not in self:
            self.add_stc(stc)
        # Compute total power
        self['power_of_reconstructed'] = np.linalg.norm(self['stc'].data, axis=1) ** 2
        # Compute power for one or both hemispheres.
        if hasattr(self['stc'], 'lh_data'):
            self['lh_power_of_reconstructed'] = np.linalg.norm(self['stc'].lh_data, axis=1) ** 2
        if hasattr(self['stc'], 'rh_data'):
            self['rh_power_of_reconstructed'] = np.linalg.norm(self['stc'].rh_data, axis=1) ** 2

    def assign_brain_regions(self,
                             parc: str):
        """
        Assign label from given parcellation ``parc`` to each source in source estimate.
        If requested parcellation is present in FreeSurfer folder for given subject, labels from there are being read.
        If not code checks presence of  'fsaverage' folder in ``subjects_dir`` directory and try to read labels
        from there. If successful, labels are then morphed to given subject.

        Parameters
        ----------
        parc : str
            Name of parcellation to be performed
        """
        sub_for_parc = _check_parc_subject_params(self["subject"], self["subjects_dir"], parc)
        if "vertices" not in self:
            raise ValueError("This operation can be only performed if 'vertices' has been added to Localize subject."
                             "Use mvpure_py.localizer.Localize.add_vertices_info() first.")

        for vertex in self['vertices']:
            self._assign_brain_region_to_vertex(vertex,
                                                hemi=self['vertices'][vertex]['hemi'],
                                                sub_for_parc=sub_for_parc,
                                                parc=parc)
        self["parc"] = parc

    def _assign_brain_region_to_vertex(self,
                                       vertex: int,
                                       hemi: str,
                                       sub_for_parc: str,
                                       parc: str):
        """
        Assign label from given parcellation `parc` to given vertex `vertex`.
        If requested parcellation is present in FreeSurfer folder for given `subject`, labels from there are being read.
        If not code checks presence of 'fsaverage' folder in `subjects_dir` directory and tried to read labels from
        there. If successful, labels are then morphed to given subject.

        Parameters
        ----------
        vertex : int
            Number of vertex for which brain region should be assigned to
        hemi : str
            Hemisphere which vertex is in
        subject : str
            Subject name the analysis is performed for
        sub_for_parc : str
            Subject used for parcellation.
            Should be equal to `subject` if requested parcellation is in `subject`'s FreeSurfer folder, otherwise should
            be equal to `fsaverage`.
        subjects_dir : str
            Directory where `subject` (and `fsaverage` if needed) folder is being stored
        parc : str
            Name of parcellation to be performed
        """
        labels = mne.read_labels_from_annot(
            subject=sub_for_parc,
            parc=parc,
            hemi=hemi,
            surf_name="white",
            subjects_dir=self["subjects_dir"]
        )
        # Morph if subject is fsaverage
        if sub_for_parc == "fsaverage":
            labels = mne.morph_labels(labels,
                                      subject_to=self["subject"],
                                      subject_from=sub_for_parc,
                                      subjects_dir=self["subjects_dir"],
                                      surf_name="white")
        for label in labels:
            if label.name.split('-')[-1] == hemi:
                if vertex in label.vertices:
                    self['vertices'][vertex]['brain_region'] = label.name

    def plot_sources_power(self,
                           stc: mne.SourceEstimate = None,
                           norm: str = "max",
                           cmap: str = "Reds",
                           **kwargs):
        if "power_of_reconstructed" not in self:
            self.add_power_of_reconstructed(stc)
        hemi = _define_hemi_based_on_vertices(self)
        _check_norm_param(norm)

        splitted_kwargs = split_kwargs(kwargs=kwargs,
                                       func_map={"brain": mne.viz.Brain,
                                                 "foci": mne.viz.Brain.add_foci})

        if hemi == "both":
            if norm == "sum":
                normalized_powers = [self['lh_power_of_reconstructed'] / np.sum(self['power_of_reconstructed']),
                                     self['rh_power_of_reconstructed'] / np.sum(self['power_of_reconstructed'])]
            elif norm == "max":
                normalized_powers = [self['lh_power_of_reconstructed'] / np.max(self['power_of_reconstructed']),
                                     self['rh_power_of_reconstructed'] / np.max(self['power_of_reconstructed'])]

            brain = mne.viz.Brain(subject=self["subject"], surf="inflated", hemi=hemi, **splitted_kwargs["brain"])
            for i in range(len(normalized_powers)):
                for j in range(len(normalized_powers[i])):
                    c = _assign_color_mapping(normalized_powers[i][j], cmap)
                    brain.add_foci(self["stc"].vertices[i][j],
                                   coords_as_verts=True,
                                   hemi=("lh" if i == 0 else "rh"),
                                   color=c,
                                   scale_factor=0.8,
                                   **splitted_kwargs["foci"])

    def plot_localized_sources(self,
                               hemi: str = "both",
                               color_mapping: bool = True,
                               color: str = "crimson",
                               cmap: str = "Reds",
                               scale_mapping: bool = True,
                               scale_factor: float = 1.0,
                               save_html: bool = False,
                               **kwargs):

        splitted_kwargs = split_kwargs(kwargs=kwargs,
                                       func_map={"brain": mne.viz.Brain,
                                                 "foci": mne.viz.Brain.add_foci})
        brain = mne.viz.Brain(subject=self['subject'], surf="inflated", hemi=hemi, **splitted_kwargs["brain"])
        # sort dict with vertices info with respect to 'activity_index_order'
        sorted_vertices = dict(
            sorted(self['vertices'].items(), key=lambda item: item[1]['activity_index_order'])
        )
        for i, vert in enumerate(sorted_vertices):
            # Add color mapping
            if color_mapping:
                _norm_factor = 1 - i / len(sorted_vertices)
                _c = _assign_color_mapping(_norm_factor, cmap)
            else:
                _c = color

            # Add scale factor mapping
            if scale_mapping:
                _scale_factor = np.linspace(scale_factor, 0.5, num=len(sorted_vertices))[i]
            else:
                _scale_factor = scale_factor

            # Plot
            brain.add_foci(coords=vert,
                           coords_as_verts=True,
                           hemi=self['vertices'][vert]['hemi'],
                           color=_c,
                           scale_factor=_scale_factor,
                           **splitted_kwargs["foci"])

        if save_html:
            _save_brain_object(brain=brain,
                               output_path=os.path.join(self['subjects_dir'], self['subject'], "html"),
                               output_name=f"{self['subject']}_localized.html")
        return brain

    def plot_localized_brain_regions(self,
                                     hemi: str,
                                     parc: str,
                                     subjects_dir: str = None,
                                     surf: str = "inflated",
                                     cmap: str = "Reds",
                                     none_color: str = "gainsboro",
                                     return_df_info: bool = False,
                                     **kwargs):
        if "vertices" not in self:
            raise ValueError("It is necessary to run mvpure_py.Localized.add_vertices_info() before trying to plot"
                             "sources by brain region")
        if "brain_region" not in self["vertices"]:
            self.assign_brain_regions(parc)

        region_ranking = dict()
        for vert in self['vertices']:
            order = self['vertices'][vert]['activity_index_order']
            points = len(self['vertices']) - order + 1
            if self['vertices'][vert]['brain_region'] in list(region_ranking.keys()):
                region_ranking[self['vertices'][vert]['brain_region']] += points
            else:
                region_ranking[self['vertices'][vert]['brain_region']] = points

        sorted_ranking = dict(sorted(region_ranking.items(), key=lambda item: item[1], reverse=True))
        max_points = list(sorted_ranking.values())[0]

        # Plotting on 'fsaverage' brain
        labels = mne.read_labels_from_annot(
            subject="fsaverage",
            parc=parc,
            subjects_dir=subjects_dir,
            hemi=hemi
        )

        brain = mne.viz.Brain(
            subject="fsaverage",
            surf=surf,
            subjects_dir=subjects_dir,
            hemi=hemi,
            **kwargs
        )

        # Add each label to the brain visualization
        for i, label in enumerate(labels):
            if label.name in list(region_ranking.keys()):
                _norm_factor = region_ranking[label.name] / max_points
                _c = _assign_color_mapping(_norm_factor, cmap=cmap)
                brain.add_label(label, hemi=label.hemi, color=_c, borders=False)
            else:
                brain.add_label(label, hemi=label.hemi, color=none_color, borders=False)

        if return_df_info:
            sorted_ranking_df = pd.DataFrame.from_dict({"region": list(sorted_ranking.keys()),
                                                        "points": list(sorted_ranking.values())})
            return sorted_ranking_df

    def plot_by_brain_region(self,
                             hemi: str = "both",
                             cmap: str = "PiYG",
                             scale_mapping: bool = True,
                             scale_factor: float = 1.0,
                             save_html: bool = False,
                             parc: str = None,
                             **kwargs):
        """
        Visualize positions of localized sources taking into consideration brain region of each vertex.

        .. note::

            It is necessary to perform ``add_vertices_info()`` before trying to plot brain regions.

        Parameters
        ----------
        hemi : str (Default to 'both'.)
            Hemisphere which vertex is in
        cmap : str (Default to 'PiYG'.)
            matplotlib colormap to use for plotting
        scale_mapping : bool (Default to True)
            Whether to take into account the order of source locations using the foci scale for visualization (the
            largest foci corresponds to the first located source etc.).
        scale_factor : float (Default to 1.0)
            Scaling factor in the range [0,1] for the first located source.
        save_html : bool (Default to False)
            Whether to save 3D output plot to html file.
        parc : str (Default to None)
            Name of parcellation to be performed.
            Needed only if 'brain_region' key is not present in Localized['vertices'] yet.
        """
        if "vertices" not in self:
            raise ValueError("It is necessary to run mvpure_py.Localized.add_vertices_info() before trying to plot"
                             "sources by brain region")
        if "brain_region" not in self["vertices"]:
            # self.assign_brain_regions(subject, subjects_dir, parc)
            self.assign_brain_regions(parc)

        splitted_kwargs = split_kwargs(kwargs=kwargs,
                                       func_map={"brain": mne.viz.Brain,
                                                 "foci": mne.viz.Brain.add_foci})

        brain = mne.viz.Brain(subject=self['subject'], surf="inflated", hemi=hemi, **splitted_kwargs["brain"])
        unique_brain_regions = set(item['brain_region'].split("-")[0] for item in self['vertices'].values())
        for i, reg in enumerate(unique_brain_regions):
            _c = _assign_color_mapping(i / len(unique_brain_regions), cmap)
            for j, vert in enumerate(self['vertices']):
                if self['vertices'][vert]['brain_region'].split("-")[0] == reg:
                    if scale_mapping:
                        _scale_factor = np.linspace(scale_factor, 0.5,
                                                    num=len(self['vertices']))[
                            self['vertices'][vert]['activity_index_order'] - 1]
                    else:
                        _scale_factor = scale_factor

                    # Plot
                    brain.add_foci(coords=vert,
                                   coords_as_verts=True,
                                   hemi=self['vertices'][vert]['hemi'],
                                   color=_c,
                                   scale_factor=_scale_factor,
                                   **splitted_kwargs["foci"])
        if save_html:
            _save_brain_object(brain=brain,
                               output_path=os.path.join(self['subjects_dir'], self['subject'], "html"),
                               output_name=f"{self['subject']}_brain_regions.html")

    def save(self, output_path: str):
        """
        Save ``mvpure_py.Localized`` object to pickle file.

        Parameters
        ----------
        output_path : str
            Directory where ``mvpure_py.Localized`` data should be stored.

        """
        if output_path.split('.')[-1] not in ['pkl', 'pickle']:
            raise ValueError(f"mvpure_py.Localized object can only be saved to '.pkl' or 'pickle' files. \
                             Got {output_path.split('.')[-1]} instead.")
        
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)


def read_localized(fname: str) -> Localized:
    """
    Read a ``mvpure_py.Localized`` object from pickle file.

    Parameters
    ----------
    fname : str
        Path to file where pickle file with ``mvpure_py.Localized`` object is stored

    Return
    ----------
    mvpure_py.Localized: read ``mvpure_py.Localized`` object
    """
    if fname.split('.')[-1] not in ['pkl', 'pickle']:
        raise ValueError(f"Loading mvpure_py.Localized object possible only from '.pkl' and '.pickle' files. \
                         Got {fname.split('.')[-1]} instead.")
    
    with open(fname, 'rb') as f:
        loaded_localized = pickle.load(f)
    
    return loaded_localized
        
