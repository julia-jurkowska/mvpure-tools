from itertools import chain
import mne
import numpy as np

import mne
import numpy as np
import matplotlib.pyplot as plt
import os

from mne import pick_types


def plot_rejchanspec(selectedspec, raw_info_ch_names, tmp_rmchans, lowerthres, upperthres, save=False, path_to_save='',
                     title=''):
    fig, ax = plt.subplots(figsize=(10, 0.1 * len(raw_info_ch_names)))

    ax.scatter(selectedspec, raw_info_ch_names, label='Selected Channels', color='blue')

    ax.scatter(selectedspec[tmp_rmchans], np.array(raw_info_ch_names)[tmp_rmchans],
               label='Channels to be removed', color='red')

    ax.axvline(lowerthres, color='green', linestyle='--', label='Lower threshold')
    ax.axvline(upperthres, color='orange', linestyle='--', label='Upper threshold')

    ax.set_xlabel('Selected Spectrum Values')
    ax.set_ylabel('Channel Names')

    ax.set_yticks(np.arange(0, len(raw_info_ch_names)))
    ax.set_yticklabels(np.array(raw_info_ch_names), fontsize=8)

    ax.grid(True, axis='y', which='both', linestyle='--', color='gray', alpha=0.7)

    plt.yticks(rotation=0)

    ax.legend()
    ax.set_title(title, fontsize=12)
    if save:
        plt.savefig(path_to_save, dpi=200)
        plt.close()
    else:
        plt.show()


def create_annotations(reject_data_intervals, sfreq, meas_date):
    """
    Creates MNE annotations from a list of intervals.
    :param reject_data_intervals: List of intervals' (start, end) in samples.
    :param sfreq: Sampling frequency of the data.
    :return: MNE annotations object.
    """
    # Calculate onset times in seconds
    onsets = reject_data_intervals[:, 0] / sfreq

    # Calculate durations in seconds
    durations = (reject_data_intervals[:, 1] - reject_data_intervals[:, 0] + 1) / sfreq

    # Create annotations
    annotations = mne.Annotations(onset=onsets, duration=durations, description='BAD', orig_time=meas_date)

    return annotations


def rejchanspec(raw, freqlims, stdlims, sample_rate=None, plot=False):
    """
    Removes noisy EEG channels from raw MNE data, based on Z-scores within specified frequency bands.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data.
    freqlims : list of tuples
        Frequency bands for analysis, e.g., [(0.5, 5), (5, 70)].
    stdlims : list of tuples
        Z-score thresholds for each frequency band, e.g., [(-3, 3), (-2.5, 2.5)].
    sample_rate : float, optional
        Sampling frequency. If None, it is taken from raw.info['sfreq'].
    plot : bool, optional
        Whether to generate diagnostic plots (requires `plot_rejchanspec` function).

    Returns:
    --------
    raw_rem : mne.io.Raw
        A copy of raw with bad EEG channels marked in `raw_rem.info['bads']`.
    specdata : ndarray
        Power spectral density values for EEG channels.
    """

    if sample_rate is None:
        sample_rate = raw.info['sfreq']

    # Pick EEG channel indices and names
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]

    # Get EEG data only
    eeg_data = raw.get_data(picks=eeg_picks)

    # Compute power spectral density (log scale)
    specdata, specfreqs = mne.time_frequency.psd_array_welch(
        eeg_data,
        sfreq=sample_rate,
        fmin=0,
        fmax=sample_rate / 2,
        n_fft=int(sample_rate),
        n_overlap=0,
        n_per_seg=int(sample_rate),
        verbose='ERROR',
        window='hamming',
        n_jobs=1
    )
    specdata = 10 * np.log10(specdata)

    rmchans = []

    for index in range(len(freqlims)):
        # Frequency range for the current band
        fbeg = np.where(specfreqs == freqlims[index][0])[0][0]
        fend = np.where(specfreqs == freqlims[index][1])[0][0]

        # Average PSD across frequency band
        selectedspec = np.mean(specdata[:, fbeg:fend + 1], axis=1)
        m = np.median(selectedspec)
        s = np.std(selectedspec, ddof=1)

        lowerthres = m + s * stdlims[index][0]
        upperthres = m + s * stdlims[index][1]

        tmp_rmchans = np.where((selectedspec <= lowerthres) | (selectedspec >= upperthres))[0]
        rejected_eeg_names = [eeg_ch_names[i] for i in tmp_rmchans]
        rmchans.append(rejected_eeg_names)

        if plot:
            if not os.path.exists('images'):
                os.mkdir('images')
            plot_rejchanspec(
                selectedspec,
                eeg_ch_names,
                tmp_rmchans,
                lowerthres,
                upperthres,
                save=False,
                path_to_save=f'images/{freqlims[index][0]}_{freqlims[index][1]}_rejchanspec.png',
                title=f'Criteria:\n{freqlims[index][0]} - {freqlims[index][1]} Hz\n'
                      f'Z-score thresholds: below {stdlims[index][0]}, above {stdlims[index][1]}'
            )

    # Flatten and deduplicate list of bad channels
    rmchans = list(set(chain.from_iterable(rmchans)))

    # Mark channels as bad (do not drop yet)
    raw_rem = raw.copy()
    raw_rem.info['bads'] = rmchans

    print('Removed EEG channels:', rmchans)

    return raw_rem, specdata


def rejchanspec_inter(raw, freqlims, stdlims, sample_rate=None):
    """
    Removes noisy channels from raw MNE data, based on the Z scores specified in stdlims for frequency bands specified in freqlims
    :param raw: raw MNE object
    :param freqlims: list of tumples with 2 values defining range for the Z score rejection, ex. [(0.5,5),(5,70)]
    :param stdlims: list of tuples with 2 values defining range of zscores within which signal is not considered as noisy, ex. [(-3,3),(-2.5,2.5)]
    :param sample_rate: sampling frequency
    :return: raw MNE object with interpolated noisy channels
    """
    if sample_rate is None:
        sample_rate = raw.info['sfreq']
    specdata, specfreqs = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=sample_rate, fmin=0,
                                                             fmax=sample_rate / 2, n_fft=1000, n_overlap=0,
                                                             n_per_seg=1000, verbose='Error', window='hamming',
                                                             n_jobs=-1)  # change fmin to 1?

    specdata = 10 * np.log10(specdata)

    rmchans = []
    for index in range(len(freqlims)):
        fbeg = np.where(specfreqs == freqlims[index][0])[0][0]
        fend = np.where(specfreqs == freqlims[index][1])[0][0]
        selectedspec = np.mean(specdata[:, fbeg:fend + 1], axis=1)
        m = np.median(selectedspec)
        s = np.std(selectedspec)

        tmp_rmchans = \
        np.where(np.logical_or(selectedspec <= m + s * stdlims[index][0], selectedspec >= m + s * stdlims[index][1]))[0]
        rmchans.append([raw.info['ch_names'][i] for i in tmp_rmchans])

    rmchans = list(chain.from_iterable(rmchans))
    raw.info['bads'] = rmchans
    raw_inter = raw.copy().interpolate_bads(reset_bads=True)
    print('Interpolated channels: ', rmchans)

    return raw_inter


def get_masked_data(raw, chans_to_pick):
    # Assuming 'raw' is your MNE Raw object and you already have raw.get_data()
    data = raw.get_data(picks=chans_to_pick)

    # Step 1: Get the annotations marked as 'BAD'
    bad_annotations = [annot for annot in raw.annotations if 'BAD' in annot['description']]

    # Step 2: Create a boolean mask for 'BAD' segments
    n_samples = raw.n_times
    sfreq = raw.info['sfreq']  # sampling frequency

    # Initialize a boolean mask of False (no bad data initially)
    bad_mask = np.zeros(n_samples, dtype=bool)

    # Mark 'BAD' segments in the mask
    for annot in bad_annotations:
        # Convert annotation start and duration to sample indices
        start_sample = int(annot['onset'] * sfreq)
        end_sample = start_sample + int(annot['duration'] * sfreq)

        # Set the mask to False for 'BAD' segments
        bad_mask[start_sample:end_sample] = True

    # Step 3: Apply the mask to the data using numpy.ma.masked_array
    # The mask needs to be broadcasted to match the shape of the data (n_channels, n_samples)
    masked_data = np.ma.masked_array(data, mask=np.broadcast_to(bad_mask, data.shape))

    return masked_data


def trimoutlier(raw, amplitude_threshold, point_spread_width, channel_sd_lower_bound=-np.inf,
                channel_sd_upper_bound=np.inf):
    meas_date = raw.info['meas_date']
    if not (channel_sd_lower_bound and channel_sd_upper_bound and amplitude_threshold and point_spread_width):
        raise ValueError('trimoutlier() requires 5 input arguments.')

    # Select only EEG channels (and not marked as bad)
    eeg_picks = pick_types(raw.info, eeg=True, exclude='bads')
    chans_to_pick = [raw.ch_names[i] for i in eeg_picks]

    print(f"Selected {len(chans_to_pick)} EEG channels (excluding bads)")

    data = get_masked_data(raw, chans_to_pick)  # make sure this only loads EEG data
    std_all_pnts = np.std(data, axis=1, ddof=1)

    # Use EEG-only indices to find bad channels
    bad_chan_mask = (std_all_pnts < channel_sd_lower_bound) | (std_all_pnts > channel_sd_upper_bound)
    bad_chan_idx = np.where(bad_chan_mask)[0]

    if len(bad_chan_idx) > 0:
        # Map EEG indices back to raw channel names
        bad_chan_names = [chans_to_pick[i] for i in bad_chan_idx]
        raw.info['bads'] = raw.info['bads'] + bad_chan_names
        raw.drop_channels(bad_chan_names)

        clean_channel_mask = ~bad_chan_mask

        print('The following channels were removed:')
        print(bad_chan_names)
    else:
        clean_channel_mask = np.ones(len(chans_to_pick), dtype=bool)
        print('No channel removed.')

    # Return if data is epoched
    if raw.get_data().ndim == 3:
        print('Epoched data detected: datapoint rejection is skipped.')
        raw.info['clean_channel_mask'] = clean_channel_mask
        return raw

    # Remove bad datapoints
    window_size = point_spread_width  # milliseconds
    window_size_in_frame = int(round(window_size / (1000 / raw.info['sfreq'])))

    # Use EEG data for amplitude thresholding
    abs_min_max_all_chan = np.max(np.abs(data), axis=0)
    bad_points = (abs_min_max_all_chan.data > amplitude_threshold) & (~abs_min_max_all_chan.mask)

    if np.any(bad_points):
        bad_points_expanded = np.convolve(bad_points.astype(int), np.ones(window_size_in_frame), 'same') > 0
        bad_points_expanded[abs_min_max_all_chan.mask] = False

        reject_data_intervals = np.where(np.diff(np.concatenate(([False], bad_points_expanded, [False]))))[0].reshape(
            -1, 2)

        if len(raw.annotations) > 0:
            new_annotations = create_annotations(reject_data_intervals, raw.info['sfreq'], meas_date)
            raw_annot = raw.copy().set_annotations(raw.annotations + new_annotations)
        else:
            annotations = create_annotations(reject_data_intervals, raw.info['sfreq'], meas_date)
            raw_annot = raw.copy().set_annotations(annotations)

        bad_points_in_sec = len(np.where(bad_points_expanded)[0]) / raw.info['sfreq']
        print(
            f'\n{amplitude_threshold}uV threshold with {window_size}ms spreading rejected {bad_points_in_sec:.1f} sec data, added {reject_data_intervals.shape[0]} boundaries.')
    else:
        raw_annot = raw.copy()
        print('No datapoint rejected.')

    print('trimOutlier done. The masks for clean channels and data points are stored in the raw object.')

    return raw_annot

