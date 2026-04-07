import os                                        # file path handling
import numpy as np
import matplotlib.pyplot as plt  
from pprint import pprint 
import pandas as pd

import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface as sa
from spikeinterface.preprocessing import unsigned_to_signed, bandpass_filter, whiten, detect_and_remove_bad_channels
import spikeinterface.widgets as sw
from spikeinterface import load_sorting_analyzer
from helper import find_amplitude, extract_average_waveform


############################## load data/set params ##############################
num_patches = 16
datapath = "data\\excit_data\\20240627-iPCS-1806-div35-iNeurons_00.brw"
recording_biocam = se.read_biocam(datapath, fill_gaps_strategy="zeros", )

############################## extract waveforms for all patches ##############################

# extract baseline values
analyzer_i = load_sorting_analyzer("analyzer_output\\excit_data\\analyzer_patch_6")        # load analyzer from a centre patch
waveforms = analyzer_i.get_extension(extension_name="waveforms")               # load extension
test_wave = waveforms.get_waveforms_one_unit(unit_id=0)
sampling_frequency = recording_biocam.get_sampling_frequency()
frames = test_wave.shape[1]
frame_axis = np.arange(frames)                                     # define time axis in frames
time_axis = frame_axis / sampling_frequency * 1000                             # time axis in ms

# loop over patches and extract waveforms
all_waveforms = []
wave_deets = pd.DataFrame(columns=["unit", "patch"])

for i in range(num_patches):
    # load analyzer for patch i
    analyzer_i = load_sorting_analyzer("analyzer_output\\excit_data\\analyzer_patch_" + str(i))       

    # define arrays to save waveforms in, dimensions: n_units x frames for average waveforms, n_units x n_spikes x frames for single waveforms
    n_units = analyzer_i.get_num_units()
    average_waveforms = np.zeros((n_units, frames))
    # single_waveforms = np.zeros((n_units, test_wave.shape[0], frames))

    # extract waveforms for all units in patch i
    for j in range(n_units):
        av_wave, singles = extract_average_waveform(analyzer_i, u_id=j)  # extract waveforms for unit j
        average_waveforms[j, :] = av_wave
        # single_waveforms[j, :, :] = singles    

    amplitudes = find_amplitude(average_waveforms)  # find the amplitude of each average waveform
    average_waveforms_filtered = average_waveforms[amplitudes > 50, :] 
    units = np.where(amplitudes > 50)[0]   

    # To add to dataframe   
    all_waveforms.append(average_waveforms_filtered)
    patch_deets = pd.DataFrame({"unit": units, "patch": i})
    wave_deets = pd.concat([wave_deets, patch_deets], ignore_index=True)

    print(f"Patch {i} done, {len(units)} units with amplitude > 50µV found!")
        

############################### Save both waveforms and details as .csv ################################
  
all_waveforms_concatenated = np.concatenate(all_waveforms, axis=0)
print(all_waveforms_concatenated.shape)
print(f"frames: {frames}, total units: {len(wave_deets)}")

np.savetxt("average_waveforms_excit_filtered.csv", all_waveforms_concatenated, delimiter=",")
wave_deets.to_csv("waveform_details_excit.csv", index=True)

