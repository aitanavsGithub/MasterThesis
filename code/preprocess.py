import h5py
import numpy as np
import os
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import pandas as pd


# Extract waveforms from .bxr files and save in dictionary. Based on BrainWave code.
def extract_waveforms_from_bxr(fileDirectory, fileName, chIdx, wellID='Well_A1', dataStartPositionSec=2, dataDurationSec=10):

    # --- open file ---
    file = h5py.File(fileDirectory + fileName, 'r')

    # --- metadata ---
    samplingRate = file.attrs['SamplingRate']
    waveformLength = file[wellID + '/SpikeForms'].attrs['Wavelength']
    minDigitalValue = file.attrs['MinDigitalValue']
    maxDigitalValue = file.attrs['MaxDigitalValue']
    minAnalogValue = file.attrs['MinAnalogValue']
    maxAnalogValue = file.attrs['MaxAnalogValue']

    dacFactor = (maxAnalogValue - minAnalogValue) / (maxDigitalValue - minDigitalValue)
    offsetValue = minAnalogValue - dacFactor * minDigitalValue

    # --- time window ---
    startFrame = int(dataStartPositionSec * samplingRate)
    endFrame = int((dataStartPositionSec + dataDurationSec) * samplingRate)

    # --- load data ---
    spikeTimes = np.array(file[wellID + '/SpikeTimes'])
    spikeChIdxs = np.array(file[wellID + '/SpikeChIdxs'])
    waveformsFlat = np.array(file[wellID + '/SpikeForms'])

    has_units = wellID + '/SpikeUnits' in file
    if has_units:
        spikeUnits = np.array(file[wellID + '/SpikeUnits'])

    # --- reshape ---
    waveformsAll = waveformsFlat.reshape(-1, waveformLength)

    # --- filter by time only (ALL channels kept) ---
    mask = ((spikeTimes >= startFrame) & (spikeTimes < endFrame))
    waveforms = waveformsAll[mask]
    channels = spikeChIdxs[mask]

    # --- convert to analog ---
    waveforms = offsetValue + dacFactor * waveforms

    # --- group by channel -> unit ---
    result = {}

    if has_units:
        units = spikeUnits[mask]

        for ch in np.unique(channels):
            ch_mask = (channels == ch)
            result[ch] = {}

            ch_waveforms = waveforms[ch_mask]
            ch_units = units[ch_mask]

            for u in np.unique(ch_units):
                result[ch][u] = ch_waveforms[ch_units == u]

    else:
        for ch in np.unique(channels):
            result[ch] = waveforms[channels == ch]

    # --- close file ---
    file.close()

    return result


#### Preprocessing helper functions!

# Align negative peaks. NOT USED ANYMORE!!
def align_to_negative_peak(waveforms, target_idx=None):
    """
    Align waveforms so their negative peak (min) is at target_idx.
    Uses circular shift (wrap-around).
    
    waveforms: np.array (n_spikes, waveformLength)
    target_idx: index to align to (default = center)
    """
    n_spikes, L = waveforms.shape
    
    if target_idx is None:
        target_idx = L // 2  # center
    
    aligned = np.zeros_like(waveforms)
    
    for i in range(n_spikes):
        wf = waveforms[i]
        min_idx = np.argmin(wf)
        
        shift = target_idx - min_idx
        
        # circular shift
        aligned[i] = np.roll(wf, shift)
    
    return aligned

# remove units that only contain a single waveform, likely noise/artifacts.
def remove_single_waveform_units(data, thresh = 1):
    """
    Remove units that contain only a single waveform.

    data : dict
        Structure: {channel: {unit: waveforms (n_spikes, waveform_length)}}
    """
    cleaned = {}

    for ch, units in data.items():
        if not isinstance(units, dict):
            continue

        new_units = {}

        for u, wf in units.items():
            if wf.shape[0] > thresh:
                new_units[u] = wf

        if new_units:
            cleaned[ch] = new_units

    return cleaned

# Flip channels whose most prominent peak is positive, so that all waveforms have a negative peak.
def enforce_negative_peak(waveforms):
    """
    Ensure all waveforms have a dominant negative peak.
    
    waveforms: (n_spikes, waveformLength)
    """
    corrected = np.copy(waveforms)
    
    for i, wf in enumerate(corrected):
        max_val = np.max(wf)
        min_val = np.min(wf)
        
        # compare absolute amplitudes
        if max_val > abs(min_val):
            corrected[i] = -wf  # invert
    
    return corrected

# Filter out any waveforms whose negative peak is not within a certain index range
def filter_by_peak_position(waveforms, min_idx=19, max_idx=21):
    """
    Keep only waveforms whose minimum lies within [min_idx, max_idx]
    
    waveforms: (n_spikes, waveformLength)
    """
    keep = []
    
    for wf in waveforms:
        peak_idx = np.argmin(wf)
        if min_idx <= peak_idx <= max_idx:
            keep.append(wf)
    
    return np.array(keep)

# Normalise each waveform individually by mean-centering and scaling to unit norm.
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

    # 1. remove per-waveform mean
    mean_subtracted = waveforms - np.mean(waveforms, axis=1, keepdims=True)

    # 2. normalize each waveform
    normalized = normalize(mean_subtracted, norm=norm, axis=1)

    return normalized

# Moving average smoothing
def moving_average(x, window=5):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')

# Smooth waveforms using either moving average, Gaussian filter, or Savitzky-Golay filter.
def smooth_waveforms(data, method="sg", axis=-1, **kwargs):
    """
    Smooth waveform data using moving average, Gaussian, or Savitzky-Golay filtering.

    Parameters
    ----------
    data : np.ndarray
        Input waveform array.
        Example shapes:
            (n_waveforms, n_samples)
            (n_samples,)

    method : str
        Smoothing method:
            "ma" -> moving average
            "gf" -> gaussian filter
            "sg" -> Savitzky-Golay filter

    axis : int
        Axis along which to smooth.

    **kwargs
        Additional parameters passed to the selected smoother.

        Moving Average ("ma"):
            window : int (default=5)

        Gaussian Filter ("gf"):
            sigma : float (default=1)

        Savitzky-Golay ("sg"):
            window_length : int (default=11)
            polyorder : int (default=3)

    Returns
    -------
    np.ndarray
        Smoothed array with same shape as input.
    """

    if method == "ma":
        window = kwargs.get("window", 5)

        return np.apply_along_axis(
            lambda x: moving_average(x, window=window),
            axis=axis,
            arr=data
        )

    elif method == "gf":
        sigma = kwargs.get("sigma", 1)

        return gaussian_filter1d(
            data,
            sigma=sigma,
            axis=axis
        )

    elif method == "sg":
        window_length = kwargs.get("window_length", 11)
        polyorder = kwargs.get("polyorder", 3)

        return savgol_filter(
            data,
            window_length=window_length,
            polyorder=polyorder,
            axis=axis
        )

    else:
        raise ValueError(
            "method must be one of: 'ma', 'gf', 'sg'"
        )

### Export preprocessed waveforms to csv

def export_mean_waveforms_to_csv(data, filename="mean_waveforms.csv"):
    """
    Export mean waveform of each unit to CSV.
    
    Each row = one mean waveform
    Each column = one waveform sample point
    """

    rows = []

    for ch, units in data.items():

        for u, wf in units.items():

            mean_wf = np.mean(wf, axis=0)

            rows.append(mean_wf)

    df = pd.DataFrame(rows)

    df.to_csv(filename, index=False, header=False)

    print(f"Saved {len(df)} mean waveforms to: {filename}")

    return df


#####################################################################################
########################## Preprocessing Master function!! ##########################
#####################################################################################

def preprocess_waveforms(raw_data,peak_min=19,peak_max=21,min_waveforms=20,smoothing_method="sg",smoothing_kwargs=None,export_csv=False,export_filename="mean_waveforms.csv"):
    """
    Full waveform preprocessing pipeline.

    Processing steps:
        1. Enforce negative peaks
        2. Filter by peak position
        3. Remove weak units
        4. Smooth waveforms (optional)
        5. Normalize waveforms

    Parameters
    ----------
    raw_data : dict
        Nested waveform dictionary:
            {
                channel: {
                    unit: np.ndarray(n_waveforms, n_samples)
                }
            }

    peak_min : int
        Minimum allowed peak index.

    peak_max : int
        Maximum allowed peak index.

    min_waveforms : int
        Minimum number of waveforms required per unit.

    smoothing_method : str
        Smoothing method:
            "ma"   -> moving average
            "gf"   -> gaussian filter
            "sg"   -> Savitzky-Golay
            "none" -> skip smoothing

    smoothing_kwargs : dict or None
        Extra parameters passed to smooth_waveforms().

        Examples:
            {"window": 5}
            {"sigma": 2}
            {"window_length": 11, "polyorder": 3}

    Returns
    -------
    dict
        Fully preprocessed waveform dictionary.
    """

    if smoothing_kwargs is None:
        smoothing_kwargs = {}

    # =========================================================
    # Step 1: Enforce negative peaks + peak position filtering
    # =========================================================

    filtered = {}

    for ch, units in raw_data.items():

        new_units = {}

        for u, wf in units.items():

            # enforce negative peak polarity
            wf_processed = enforce_negative_peak(wf)

            # filter by peak position
            wf_processed = filter_by_peak_position(wf_processed,peak_min,peak_max)

            # keep non-empty units only
            if wf_processed.shape[0] > 0:
                new_units[u] = wf_processed

        if new_units:
            filtered[ch] = new_units

    # =========================================================
    # Step 2: Remove weak units
    # =========================================================

    cleaned = remove_single_waveform_units(
        filtered,
        thresh=min_waveforms
    )

    # =========================================================
    # Step 3: Smooth waveforms (optional)
    # =========================================================

    smoothed = {}

    for ch, units in cleaned.items():

        new_units = {}

        for u, wf in units.items():

            # skip smoothing
            if smoothing_method == "none":
                wf_smooth = wf

            else:
                wf_smooth = smooth_waveforms(wf,method=smoothing_method,axis=-1,**smoothing_kwargs)

            if wf_smooth.shape[0] > 0:
                new_units[u] = wf_smooth

        if new_units:
            smoothed[ch] = new_units

    # =========================================================
    # Step 4: Normalize waveforms
    # =========================================================

    normalized = {}

    for ch, units in smoothed.items():

        new_units = {}

        for u, wf in units.items():

            wf_norm = normalize_waveforms(wf)

            if wf_norm.shape[0] > 0:
                new_units[u] = wf_norm

        if new_units:
            normalized[ch] = new_units

    # =========================================================
    # Step 5: export mean waveforms to csv (optional)
    # =========================================================
    if export_csv:
        export_mean_waveforms_to_csv(normalized,filename=export_filename)

    return normalized

