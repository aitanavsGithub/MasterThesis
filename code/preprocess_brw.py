import os                                        # file path handling
import numpy as np
import matplotlib.pyplot as plt  
from pprint import pprint 

import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface as sa
from spikeinterface.preprocessing import unsigned_to_signed, bandpass_filter, whiten, detect_and_remove_bad_channels
import spikeinterface.widgets as sw
from spikeinterface import load_sorting_analyzer

def generate_patch(start):
    result = []
    for _ in range(16):
        result.extend(range(start, start + 16))  # take 16 numbers
        start += 16 + 48                         # skip 48 numbers
    return [str(i) for i in result]

# should add function to extract waveform

def find_amplitude(waveform):
    if waveform.ndim == 1:
        return np.max(waveform) - np.min(waveform)
    elif waveform.ndim == 2:
        return np.max(waveform, axis=1) - np.min(waveform, axis=1)
    else:
        raise ValueError("Input waveform must be either 1D or 2D.")

def extract_average_waveform(analyzer, u_id):
    """
    Extracts the average waveform and single waveforms for a given unit from a spikeinterface analyzer.
    The waveform extension for one unit is of shape (n_spikes, n_channels, n_frames). 
    The average waveform is computed across individual spikes for each channel, resulting in a shape of (n_channels, n_frames).
    The channel with the maximum variance of the average waveform is picked, since this is likely the one that shows the waveform best. 
    (Similarly, the one with the largest amplitude could be picked)

    parameters:
    - analyzer: a spikeinterface sorting analyzer object containing the waveform extension
    - u_id: the unit id for which to extract the waveforms
    returns:
    - av_wave: the average waveform for the unit, of shape (n_frames,)
    - single_waveforms: the single waveforms for the unit, of shape (n_spikes, n_frames)
    """

    waveforms = analyzer.get_extension(extension_name="waveforms") 
    wave0 = waveforms.get_waveforms_one_unit(unit_id=u_id)                  # load the waveform for unit i

    av_wave = np.mean(wave0, axis=0)                                        # find the average wave per channel
    av_wave_var = np.var(av_wave, axis=0)                                   # compute variance per channel
    e_max = np.argmax(av_wave_var)                                          # find the channel with max variance - assuming it's the one that shows
                                                                            # the waveform best
    AV_WAVE = av_wave[:, e_max]                                             # pick that channel to plot
    SINGLE_WAVES = wave0[:, :, e_max]                                       # get the single channel waveforms

    # save the average and single waveforms in arrays
    return AV_WAVE, SINGLE_WAVES

def remove_single_waveform_units(data, thresh = 10):
    """
    Remove units that have less than a certain number of single waveforms, as these are likely to be noise. 

    parameters:
    - data: a dictionary containing the average waveforms, single waveforms, and other details for each unit
    - thresh: the minimum number of single waveforms required for a unit to be kept
    returns:
    - filtered_data: a dictionary containing only the units that have at least thresh single waveforms
    """

    return 

