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
    return np.max(waveform, axis=1) - np.min(waveform, axis=1)

def extract_average_waveform(analyzer, u_id):
    """
    Should explain this a little here
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

