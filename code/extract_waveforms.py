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
#datapath = "data\\test_data\\testdata_raw.brw"
recording_biocam = se.read_biocam(datapath, fill_gaps_strategy="zeros", )
var_thresh = 500  # variance threshold for filtering units?? 

############################## extract waveforms for all patches ##############################

# extract baseline values
analyzer_i = load_sorting_analyzer("analyzer_output\\excit_data\\analyzer_patch_6")        # load analyzer from a centre patch
#analyzer_i = load_sorting_analyzer("analyzer_output\\test_data\\analyzer_patch_6")        # load analyzer from a centre patch
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
    #analyzer_i = load_sorting_analyzer("analyzer_output\\test_data\\analyzer_patch_" + str(i))   

    # define arrays to save waveforms in, dimensions: n_units x frames for average waveforms, n_units x n_spikes x frames for single waveforms
    n_units = analyzer_i.get_num_units()
    average_waveforms = np.zeros((n_units, frames))
    waveform_singles_var = np.zeros(n_units)
    # single_waveforms = np.zeros((n_units, test_wave.shape[0], frames))

    # extract waveforms for all units in patch i
    for j in range(n_units):
        av_wave, singles = extract_average_waveform(analyzer_i, u_id=j)  # extract waveforms for unit j
        average_waveforms[j, :] = av_wave
        waveform_singles_var[j] = np.mean(np.var(singles, axis=0))  # compute variance   
        # single_waveforms[j, :, :] = singles  


    ### Filter out units
    # calculate amplitudes
    amplitudes = find_amplitude(average_waveforms) 

    # apply filtering criteria: amplitude > 50µV and variance < var_thresh
    valid_indices = (amplitudes > 50) & (amplitudes < 500) & (waveform_singles_var < var_thresh)
    average_waveforms_filtered = average_waveforms[valid_indices, :] 
    units = np.where(valid_indices)[0]

    # Invert the flipped waveforms if needed
    max_vals = np.max(average_waveforms_filtered, axis=1)
    min_vals = np.min(average_waveforms_filtered, axis=1)
    average_waveforms_filtered[max_vals > np.abs(min_vals)] *= -1 


    # To add to dataframe   
    all_waveforms.append(average_waveforms_filtered)
    patch_deets = pd.DataFrame({"unit": units, "patch": i})
    wave_deets = pd.concat([wave_deets, patch_deets], ignore_index=True)

    print(f"Patch {i} done, {len(units)} units with 500µV > amplitude > 50µV and var < {var_thresh} found!")
############################### Save both waveforms and details as .csv ################################
  
all_waveforms_concatenated = np.concatenate(all_waveforms, axis=0)
print(all_waveforms_concatenated.shape)
print(f"frames: {frames}, total units: {len(wave_deets)}")

np.savetxt("analyzer_output\\excit_data\\average_waveforms_excit_filtered3.csv", all_waveforms_concatenated, delimiter=",")
wave_deets.to_csv("analyzer_output\\excit_data\\waveform_details_excit_filtered3.csv", index=True)

