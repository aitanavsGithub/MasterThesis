from pathlib import Path

import numpy as np
import pandas as pd

from spikeinterface import load_sorting_analyzer
from sklearn.preprocessing import normalize
from scipy.signal import find_peaks

from preprocess_brw import find_amplitude, extract_average_waveform


# =============================================================================
# Helper functions
# =============================================================================

def get_waveform_length(analyzer):
    """Determine waveform length from first unit."""

    waveforms = analyzer.get_extension("waveforms")
    test_wave = waveforms.get_waveforms_one_unit(unit_id=0)

    return test_wave.shape[1]


def compute_unit_waveforms(analyzer, frames):
    """
    Extract average waveforms and number of single waveforms
    for all units in an analyzer.

    Returns
    -------
    average_waveforms : ndarray, shape (n_units, frames)
    waveform_counts : ndarray, shape (n_units,)
        Number of individual waveforms contributing to each average.
    """
    n_units = analyzer.get_num_units()

    average_waveforms = np.zeros((n_units, frames))
    waveform_counts = np.zeros(n_units, dtype=int)
    noise_std = np.zeros(n_units)

    for unit_id in range(n_units):
        avg_waveform, singles = extract_average_waveform(analyzer,u_id=unit_id,)

        average_waveforms[unit_id] = avg_waveform
        waveform_counts[unit_id] = singles.shape[0]

        noise_std[unit_id] = np.mean(np.std(singles, axis=0))

    return average_waveforms, waveform_counts, noise_std


def filter_waveforms(average_waveforms,waveform_counts,noise_std,amp_range=(50, 500),singles_min=10, min_snr=3):
    """
    Filter units by amplitude, number of single waveforms, and SNR.
    """
    amplitudes = find_amplitude(average_waveforms)
    snrs = np.zeros_like(amplitudes)

    for unit_id, (amplitude, noise_std) in enumerate(zip(amplitudes, noise_std)):
        if noise_std > 0:
            snrs[unit_id] = amplitude / noise_std
        else:
            snrs[unit_id] = np.inf

    valid = (
        (amplitudes >= amp_range[0])
        & (amplitudes <= amp_range[1])
        & (waveform_counts >= singles_min)
        & (snrs >= min_snr)
    )

    filtered_waveforms = average_waveforms[valid].copy()
    unit_ids = np.where(valid)[0]

    return filtered_waveforms, unit_ids


def orient_waveforms_negative(waveforms):
    """
    Flip waveforms so the dominant peak is negative.
    """
    if len(waveforms) == 0:
        return waveforms

    max_vals = np.max(waveforms, axis=1)
    min_vals = np.min(waveforms, axis=1)

    flip_mask = max_vals > np.abs(min_vals)
    waveforms[flip_mask] *= -1

    return waveforms


def filter_by_peak_position(waveforms,unit_ids,min_idx=13,max_idx=23,):
    """
    Keep only waveforms whose negative peak lies within
    [min_idx, max_idx].

    Returns
    -------
    filtered_waveforms
    filtered_unit_ids
    """
    peak_idx = np.argmin(waveforms, axis=1)

    keep_mask = ((peak_idx >= min_idx)& (peak_idx <= max_idx))

    return (waveforms[keep_mask],unit_ids[keep_mask])


def filter_single_negative_peak(waveforms,unit_ids,prominence=0.3):
    """
    Keep only waveforms with a single prominent negative peak.

    Parameters
    ----------
    waveforms : ndarray (n_waveforms, T)
    unit_ids : ndarray
    prominence : float
        Minimum prominence of peaks (in normalized units of waveform scale)
    min_distance : int
        Minimum distance between peaks (samples)

    Returns
    -------
    filtered_waveforms, filtered_unit_ids
    """

    if waveforms.shape[0] == 0:
        return waveforms, unit_ids

    keep_mask = []

    for wf in waveforms:

        # invert so negative peaks become positive peaks
        inv_wf = -wf

        # detect peaks with prominence that depends on the range of the waveform
        peaks, properties = find_peaks(inv_wf, prominence=prominence)

        # keep only if exactly one strong peak
        keep_mask.append(len(peaks) == 1)

    keep_mask = np.array(keep_mask)

    return waveforms[keep_mask], unit_ids[keep_mask]


def normalize_waveforms(waveforms, norm='max'):
    """
    Mean-center and normalize each waveform individually.

    waveforms : np.ndarray
        Shape (n_waveforms, waveform_length)
    norm : str
        Normalization type passed to sklearn.preprocessing.normalize

    """

    waveforms = np.asarray(waveforms)

    if waveforms.ndim != 2:
        raise ValueError("Expected input shape (n_waveforms, waveform_length)")
    
    if waveforms.shape[0] == 0:
        print("Warning: no waveforms remaining after filtering")
        return waveforms

    # 1. remove per-waveform mean
    mean_subtracted = waveforms - np.mean(waveforms, axis=1, keepdims=True)

    # 2. normalize each waveform
    normalized = normalize(mean_subtracted, norm=norm, axis=1)

    return normalized


def process_patch(analyzer_path,patch_id,frames,amp_range=(50, 500),singles_min=10,min_snr=3,peak_window=(13, 23),peak_prominence=0.3,normalize=True):
    """
    Process a single analyzer patch.
    """
    analyzer = load_sorting_analyzer(analyzer_path)

    average_waveforms, waveform_counts, noise_std = compute_unit_waveforms(analyzer,frames)

    # filter by amplitude and number of single waveforms
    filtered_waveforms, unit_ids = filter_waveforms(average_waveforms,waveform_counts,noise_std,amp_range=amp_range,singles_min=singles_min,min_snr=min_snr,)

    # flip positive waveforms to negative so the dominant peak is consistently negative
    filtered_waveforms = orient_waveforms_negative(filtered_waveforms)

    # filter by peak position
    filtered_waveforms, unit_ids = filter_by_peak_position(filtered_waveforms,unit_ids,min_idx=peak_window[0],max_idx=peak_window[1],)

    # normalize waveforms
    if normalize:
        filtered_waveforms = normalize_waveforms(filtered_waveforms)

    # filter by number of prominent peaks
    filtered_waveforms, unit_ids = filter_single_negative_peak(filtered_waveforms,unit_ids,prominence=peak_prominence)

    print(f"Patch {patch_id}: {len(unit_ids)} units passed filtering")

    return filtered_waveforms, unit_ids


# =============================================================================
# Main function
# =============================================================================

def extract_filtered_waveforms(analyzer_folder,skip_first_n_patches=0,save_waveforms_csv=None,save_metadata_csv=None, amp_range=(50, 500),singles_min=10,min_snr=3,peak_window=(13, 23),peak_prominence=0.3,normalize=True):
    """
    Process all analyzer_patch_* folders.

    Parameters
    ----------
    analyzer_folder : str or Path
        Folder containing analyzer_patch_* directories.

    skip_first_n_patches : int
        Number of initial patches to skip.

    save_waveforms_csv : str or None
        Optional CSV path for waveform matrix.

    save_metadata_csv : str or None
        Optional CSV path for patch/unit metadata.

    amp_range : tuple
        (min_amplitude, max_amplitude)

    singles_min : int
        Minimum number of single waveforms required.

    Returns
    -------
    waveform_df : pd.DataFrame
        Waveform matrix only.
        Shape = (n_units_kept, n_samples)

    metadata_df : pd.DataFrame
        Corresponding metadata:
        columns = ['patch', 'unit']
    """
    analyzer_folder = Path(analyzer_folder)

    # create a list of analyzer paths
    analyzer_paths = sorted(analyzer_folder.glob("analyzer_patch_*"), key=lambda p: int(p.name.split("_")[-1]))
    if skip_first_n_patches > 0:
        analyzer_paths = analyzer_paths[skip_first_n_patches:]

    if len(analyzer_paths) == 0:
        raise ValueError(f"No analyzer_patch_* folders found in {analyzer_folder}")

    reference_analyzer = load_sorting_analyzer(analyzer_paths[0])
    frames = get_waveform_length(reference_analyzer)

    waveform_rows = []
    metadata_rows = []

    for patch_path in analyzer_paths:

        patch_id = int(patch_path.name.split("_")[-1])

        waveforms, unit_ids = process_patch(patch_path,patch_id,frames,amp_range=amp_range,singles_min=singles_min,min_snr=min_snr,peak_window=peak_window,peak_prominence=peak_prominence,normalize=normalize)

        waveform_rows.extend(waveforms)

        metadata_rows.extend(
            [
                {
                    "patch": patch_id,
                    "unit": unit_id,
                }
                for unit_id in unit_ids
            ]
        )

    waveform_df = pd.DataFrame(waveform_rows)
    metadata_df = pd.DataFrame(metadata_rows)

    print("\nFinished processing")
    print(f"Total units retained: {len(metadata_df)}")
    print(f"Waveform shape: {waveform_df.shape}")

    if save_waveforms_csv is not None:
        waveform_df.to_csv(save_waveforms_csv,index=False,header=False)

    if save_metadata_csv is not None:
        metadata_df.to_csv(save_metadata_csv,index=False,)

    return waveform_df, metadata_df

# Usage

if __name__ == "__main__":
    analyzer_folder = "analyzer_output/test_data"

    waveforms, metadata = extract_filtered_waveforms(
    analyzer_folder="analyzer_output/test_data",
    skip_first_n_patches=1,
    save_waveforms_csv="NEW_BRW_average_waveforms_filtered_dissorg.csv",
    save_metadata_csv="NEW_BRW_waveform_metadata_dissorg.csv",
    amp_range=(50, 500),
    singles_min=25,
    min_snr=3,
    peak_window=(18, 20),
    peak_prominence=0.2,
    normalize=True)

    analyzer_folder = "analyzer_output/excit_data"

    waveforms, metadata = extract_filtered_waveforms(
    analyzer_folder="analyzer_output/excit_data",
    skip_first_n_patches=1,
    save_waveforms_csv="NEW_BRW_average_waveforms_filtered_excit.csv",
    save_metadata_csv="NEW_BRW_waveform_metadata_excit.csv",
    amp_range=(50, 500),
    singles_min=100,
    min_snr=3,
    peak_window=(18, 20),
    peak_prominence=0.2,
    normalize=True)   # Maybe choose not to do this, should we normalize??